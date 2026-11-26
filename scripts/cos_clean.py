#!/usr/bin/env python3
"""C0 cleaning of COS scenes v1.1 -> scenes v2 (docs/cos_planner_reflex.md, C0).

  python scripts/cos_clean.py --manifest /l/.../cos/scenes_v1/manifest.json --out /l/.../cos/scenes_v2 --jobs 12

Per window (motions/<window>.npz, M = 50 Hz G1 qpos, chord frame): kinematic trick / uneven-floor / foot-penetration tests
and clip-name keywords. Kept scenes are re-verified with synth.verify. Writes OUT/<folder>/{scenes.jsonl, removed.jsonl,
window_flags.jsonl}, OUT/manifest.json and OUT/clean_summary.json (counts per reason x family).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DT = 0.02
TH = dict(flight_h=0.05, flight_frames=5, vz=1.0, roll_z=0.87, angvel=8.0, floor_var=0.05, pen=-0.02, block_s=0.5)
TRICK = re.compile(r"flip|cartwheel|cart_wheel|roll|handstand|hand_stand|parkour|vault|acrobat|somersault|tumbl|jump|"
                   r"(?<![a-z])hop|leap|skip|breakdanc|trick|gymnast|spin|twirl|pirouette|(?<![a-z])dive", re.I)
TERRAIN = re.compile(r"stair|step|climb|ramp|slope|platform|ladder|(?<![a-z])hill|incline|curb|ledge|ascend|descend", re.I)
STEP_OK = re.compile(r"step_?over|side_?step|stepping_?over|stepover|sidestep", re.I)
ORDER = ["trick_name", "flight", "pelvis_vz", "acrobatic", "terrain_name", "uneven_floor", "foot_penetration"]
G = FF = None


def init():
    global G, FF
    import torch
    torch.set_num_threads(1)
    from cat_mjlab.cos.features import FrameFeatures, _extent
    FF = FrameFeatures(); G = FF.g


def name_of(clip):
    return clip.replace("stageii", "").replace("_poses", "").replace("Subject", "")


def window_flags(q, clip):
    """Flags of one window (q [T, 36] float)."""
    import torch
    from cat_mjlab.cos.features import _extent
    qt = torch.as_tensor(q, dtype=torch.float32)
    world, xpos = G.primitives(qt)
    xmat = G.fk(qt)[1]
    lo, hi = _extent(world, G.compiled)
    lo = lo.numpy(); xpos = xpos.numpy(); xmat = xmat.numpy()
    fz = np.stack([lo[:, FF.left_foot, 2].min(1), lo[:, FF.right_foot, 2].min(1)], 1)      # [T, 2]
    fmin = fz.min(1)
    T = len(q)
    f = {}
    nm = name_of(clip)
    f["trick_name"] = bool(TRICK.search(nm))
    tn = bool(TERRAIN.search(nm))
    f["step_exception"] = bool(tn and STEP_OK.search(nm) and not re.search(r"stair|climb|ramp|slope|platform|ladder", nm, re.I))
    f["terrain_name"] = tn and not f["step_exception"]
    floor = float(np.percentile(fmin, 5))
    air = (fz > floor + TH["flight_h"]).all(1)
    run = best = 0
    for v in air:
        run = run + 1 if v else 0; best = max(best, run)
    f["flight_frames"] = int(best); f["flight"] = best >= TH["flight_frames"]
    pel = xpos[:, G.body["pelvis"]]
    k = 2 if T > 5 else 1
    vz = np.abs(pel[2 * k:, 2] - pel[:-2 * k, 2]) / (2 * k * DT) if T > 2 * k else np.zeros(1)
    f["max_vz"] = float(vz.max()); f["pelvis_vz"] = f["max_vz"] > TH["vz"]
    R = xmat[:, G.body["pelvis"]]                                     # [T, 3, 3] columns = body axes in world
    upz = R[:, 2, 2]; latz = np.abs(R[:, 2, 1])
    hz = hi.numpy()[:, FF.groups["head"], 2].max(1)                 # top of the head below the pelvis = inverted
    head_below = bool((hz < pel[:, 2]).any())
    if T > 2 * k:
        Ra, Rb = R[:-2 * k], R[2 * k:]
        rel = np.einsum("tji,tjk->tik", Ra, Rb)
        ang = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)) / (2 * k * DT)
    else:
        ang = np.zeros(1)
    f["min_upz"] = float(upz.min()); f["max_latz"] = float(latz.max()); f["max_angvel"] = float(ang.max())
    f["acrobatic"] = bool(upz.min() < 0 or latz.max() > TH["roll_z"] or head_below or ang.max() > TH["angvel"])
    nb = max(1, int(round(TH["block_s"] / DT)))
    blocks = [fmin[i:i + nb].min() for i in range(0, T, nb) if len(fmin[i:i + nb]) >= nb // 2]
    f["floor_range"] = float(max(blocks) - min(blocks)); f["uneven_floor"] = f["floor_range"] > TH["floor_var"]
    f["min_foot"] = float(fmin.min()); f["foot_penetration"] = f["min_foot"] < TH["pen"]
    other = [i for g in ("hands",) for i in FF.groups.get(g, [])] + list(FF.knee_shapes)
    f["min_other"] = float(lo[:, other, 2].min()) if other else 0.
    f["trick_frames"] = np.flatnonzero(air | np.r_[np.zeros(k), vz > TH["vz"], np.zeros(k)][:T].astype(bool)).tolist()
    return f


def one(job):
    folder, window, scenes = job
    d = np.load(Path(folder) / "motions" / f"{window}.npz")
    M = d["M"].astype(np.float64); D = d["D"].astype(np.float64)
    f = window_flags(M, scenes[0]["clip"])
    reasons = [r for r in ORDER if f[r]]
    tf = set(f.pop("trick_frames"))
    trick_event = []
    for s in scenes:
        hit = any(any((a - 25) <= t < (b + 25) for t in tf) for a, b in s["event_frames"]) if tf else False
        trick_event.append(hit)
    ver = []
    if not reasons:
        from cat_mjlab.cos.synth import verify
        for s in scenes:
            v = verify(G, M, D, s["boxes"])
            ver.append(dict(ok=v["ok"], ok_pen=v["ok_pen"], m_clear=v["m_clear"]))
    return folder, window, f, reasons, trick_event, ver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=12); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    man = json.load(open(a.manifest)); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = []; S = {}
    for folder in man["scene_folders"]:
        byw = collections.OrderedDict()
        for l in open(Path(folder) / "scenes.jsonl"):
            s = json.loads(l)
            if not s.get("verified"):
                continue
            if s["source"] in man.get("dropped_sources", []):
                continue
            byw.setdefault(s["window"], []).append(s)
        S[folder] = byw
        jobs += [(folder, w, ss) for w, ss in byw.items()]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} windows, {sum(len(j[2]) for j in jobs)} verified scenes", flush=True)
    import multiprocessing as mp
    cnt = collections.defaultdict(collections.Counter)      # reason -> family -> scenes (non-exclusive)
    first = collections.defaultdict(collections.Counter)    # first reason -> family -> scenes
    wcnt = collections.Counter(); kept = collections.Counter(); kept_feas = collections.Counter()
    kept_fc = collections.Counter(); before = collections.Counter(); before_feas = collections.Counter()
    agree = n_ver = 0; trick_ev = 0; step_exc = 0; other_low = 0
    files = {}
    t = time.time()
    with mp.get_context("fork").Pool(a.jobs, initializer=init) as pool:
        for k, (folder, window, f, reasons, tev, ver) in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            name = Path(folder).name
            if name not in files:
                (out / name).mkdir(exist_ok=True)
                files[name] = {x: open(out / name / f"{x}.jsonl", "w") for x in ("scenes", "removed", "window_flags")}
            F = files[name]
            F["window_flags"].write(json.dumps(dict(window=window, reasons=reasons, **f)) + "\n")
            ss = S[folder][window]
            step_exc += int(f["step_exception"]); other_low += int(f["min_other"] < -0.02)
            for r in reasons:
                wcnt[r] += 1
            for i, s in enumerate(ss):
                fam = s["family"]; feas = bool(s.get("tracked"))
                before[fam] += 1; before_feas[fam] += feas
                if reasons:
                    for r in reasons:
                        cnt[r][fam] += 1
                    first[reasons[0]][fam] += 1
                    trick_ev += int(tev[i] and any(r in ("flight", "pelvis_vz") for r in reasons))
                    F["removed"].write(json.dumps(dict(scene=s["scene"], window=window, family=fam, source=s["source"],
                                                       reasons=reasons)) + "\n")
                    continue
                v = ver[i]; n_ver += 1; agree += int(v["ok"] == bool(s["verified"]))
                s = dict(s); s["reverify"] = v; s["clean_v2"] = True
                F["scenes"].write(json.dumps(s) + "\n")
                if v["ok"]:
                    kept[fam] += 1; kept_feas[fam] += feas
                    kept_fc[fam] += int(feas and s.get("tracked_clear") is not None and s["tracked_clear"] >= 0)
            if k % 1000 == 0:
                print(f"{k}/{len(jobs)} windows {time.time() - t:.0f}s", flush=True)
    for F in files.values():
        for h in F.values():
            h.close()
    fams = sorted(before)
    summ = dict(thresholds=TH, windows=len(jobs), windows_removed_by_reason=dict(wcnt),
                scenes_before=dict(before), scenes_before_feasible=dict(before_feas),
                removed_by_reason=dict((r, dict(cnt[r])) for r in ORDER), removed_first_reason=dict((r, dict(first[r])) for r in ORDER),
                kept_verified=dict(kept), kept_feasible=dict(kept_feas), kept_feasible_clear=dict(kept_fc),
                reverify_agreement=agree / max(1, n_ver), reverified=n_ver, trick_inside_event_scenes=trick_ev,
                step_name_exceptions_windows=step_exc, windows_hand_or_knee_below_2cm=other_low,
                total=dict(before=sum(before.values()), kept=sum(kept.values()), kept_feasible=sum(kept_feas.values()),
                           kept_feasible_clear=sum(kept_fc.values())))
    json.dump(summ, open(out / "clean_summary.json", "w"), indent=1)
    m2 = dict(man); m2.update(schema="cos-scenes-v2", version="scenes v2 (C0 cleaned v1.1)",
                              scene_folders=[str(out / Path(f).name) for f in man["scene_folders"]],
                              motions_from={str(out / Path(f).name): f for f in man["scene_folders"]},
                              scenes=sum(before.values()), verified=sum(kept.values()), clean=summ, log="docs/cos_planner_reflex.md")
    json.dump(m2, open(out / "manifest.json", "w"), indent=1)
    print(json.dumps(summ["total"]), "reverify", round(summ["reverify_agreement"], 4), f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
