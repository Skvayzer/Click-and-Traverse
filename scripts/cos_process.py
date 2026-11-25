#!/usr/bin/env python3
"""COS scenes v1 for one motion library (docs/cos_obstacles.md, data preparation).

  python scripts/cos_process.py events --lib LIBDIR
  python scripts/cos_process.py synth  --lib LIBDIR --gait-bank G.pkl --out SCENEDIR [--feasibility F.jsonl] [--jobs 12]

events: per-frame proxy features, G0 detectors + clean tier on every clip -> LIBDIR/events.jsonl, LIBDIR/windows.jsonl
        (clean events +- 1.5 s, merged per clip, 3-12 s; feasibility input format), LIBDIR/events_summary.json.
synth:  per window with chord >= 0.5 m: chord frame, default motion D(M), for every event up to K = 6 diverse fits
        (cat_mjlab/cos/kinds.py: several obstacle kinds per family, near-duplicates rejected), plus up to 3 multi-obstacle
        scenes for windows with several events; each scene gets distractors and room walls and is verified at 100 Hz
        against all boxes. Writes SCENEDIR/scenes.jsonl (boxes in the chord frame, quality metrics, feasibility flag,
        tracked-trajectory clearance), SCENEDIR/motions/<window>.npz (M, D and the SONIC-tracked trajectory, chord frame),
        SCENEDIR/attempts.jsonl and SCENEDIR/summary.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DT = 0.02


def _j(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def take_of(c):
    import re
    s = re.sub(r"(__r\d+)$", "", c["clip_id"])
    return s


# ------------------------------------------------------------------------------------------------------------- events
def cmd_events(a):
    from cat_mjlab.cos.events import FAMILIES, clean, detect
    from cat_mjlab.cos.features import FrameFeatures
    lib = Path(a.lib)
    clips = [json.loads(l) for l in open(lib / "clips.jsonl")]
    FF = FrameFeatures()
    shards = {}
    ev_out = open(lib / "events.jsonl", "w")
    win_out = open(lib / "windows.jsonl", "w")
    S = collections.defaultdict(lambda: collections.Counter())
    hours = collections.Counter(); t = time.time()
    for ci, c in enumerate(clips):
        if c["shard"] not in shards:
            shards = {c["shard"]: np.load(lib / f"qpos_{c['shard']:03d}.npy", mmap_mode="r")}
        q = np.asarray(shards[c["shard"]][c["start"]:c["start"] + c["frames"]])
        hours[c["source"]] += c["seconds"] / 3600
        F = FF(q)
        ev = detect(F)
        kept = []
        for fam, lst in ev.items():
            for a0, b0, mag in lst:
                ok = bool(clean(fam, a0, b0, F, c["clip_id"] + " " + c.get("member", ""), c["source"]))
                rec = dict(family=fam, clip=c["clip_id"], source=c["source"], subset=c["subset"], take=take_of(c),
                           f0=int(a0), f1=int(b0), dur=round((b0 - a0) * DT, 2), mag=round(mag, 3), clean=ok)
                ev_out.write(json.dumps(rec) + "\n")
                key = (c["source"], fam)
                S[key]["raw"] += 1
                if ok:
                    S[key]["clean"] += 1; S[key]["clean_s"] += (b0 - a0) * DT
                    kept.append(rec)
        # windows: clean events +- 1.5 s, merged, 3-12 s
        kept.sort(key=lambda x: x["f0"]); groups = []
        n = c["frames"]
        for x in kept:
            a0, b0 = max(0, x["f0"] - 75), min(n, x["f1"] + 75)
            if groups and a0 <= groups[-1]["b"] and b0 - groups[-1]["a"] <= 600:
                g = groups[-1]; g["b"] = max(g["b"], b0); g["events"].append([x["family"], x["f0"], x["f1"], x["mag"]])
            else:
                groups.append(dict(a=a0, b=b0, events=[[x["family"], x["f0"], x["f1"], x["mag"]]]))
        for k, g in enumerate(groups):
            if g["b"] - g["a"] < 150:
                need = 150 - (g["b"] - g["a"]); g["a"] = max(0, g["a"] - need // 2); g["b"] = min(n, g["b"] + need - need // 2)
            g["b"] = min(g["b"], g["a"] + 600)
            win_out.write(json.dumps(dict(id=f"{c['clip_id']}__w{k}", clip=c["clip_id"], source=c["source"],
                                          subset=c["subset"], take=take_of(c), licence=c["licence"],
                                          qpos_file=str(lib / f"qpos_{c['shard']:03d}.npy"),
                                          a=c["start"] + g["a"], b=c["start"] + g["b"], clip_frame0=g["a"],
                                          events=[[f, e0 - g["a"], e1 - g["a"], m] for f, e0, e1, m in g["events"]],
                                          families=sorted({e[0] for e in g["events"]}))) + "\n")
        if ci % 500 == 0:
            print(f"{ci}/{len(clips)} clips {time.time() - t:.0f}s", flush=True)
    summ = dict(hours={k: round(v, 2) for k, v in hours.items()},
                events={f"{s}/{f}": dict(v) for (s, f), v in sorted(S.items())})
    (lib / "events_summary.json").write_text(json.dumps(summ, indent=1))
    print("EVENTS DONE", json.dumps(summ)[:2000], flush=True)


# -------------------------------------------------------------------------------------------------------------- synth
ARGS = G = BANK = FEAS = None


def init(a):
    global ARGS, G, BANK, FEAS, FF
    import torch
    torch.set_num_threads(1)
    from cat_mjlab.exact_geometry import ExactGeometry
    from cat_mjlab.cos.features import FrameFeatures
    ARGS = a
    G = ExactGeometry(); FF = FrameFeatures(G)
    BANK = pickle.load(open(a.gait_bank, "rb"))
    FEAS = {}
    if a.feasibility and Path(a.feasibility).exists():
        for l in open(a.feasibility):
            m = json.loads(l); FEAS[m["id"]] = m


def realism(kind, boxes):
    """Dimension checks against common real objects (flag, not a filter)."""
    m = [b for b in boxes if b["role"] == "main"]
    lo = min(b["center"][2] - b["half"][2] for b in m); hi = max(b["center"][2] + b["half"][2] for b in m)
    if kind in ("beam", "bar", "shelf", "low_ceiling"):
        return 0.8 <= lo <= 1.6
    if kind in ("ceiling", "tunnel", "table") and lo > 0.1:
        return 0.35 <= lo <= 0.95
    if kind in ("hurdle", "low_box"):
        return 0.08 <= hi <= 0.6
    if kind in ("gap", "narrow_gap", "doorway", "narrow_doorway", "cabinet_gap"):
        ys = sorted(b["center"][1] for b in m)
        inner = [b for b in m]
        w = (max(ys) - min(ys)) - sum(b["half"][1] for b in (inner[0], inner[-1]))
        return 0.30 <= w <= 1.2
    if kind in ("counter", "counter_pair"):
        return 0.6 <= hi <= 1.0
    return True


def one(w):
    from cat_mjlab.cos.default_motion import default_motion, pick_cycle, to_frame
    from cat_mjlab.cos.kinds import fits_for_event
    from cat_mjlab.cos.synth import Body, distractors, room_walls, verify
    a = ARGS
    rng = np.random.default_rng([int(hashlib.md5(w["id"].encode()).hexdigest()[:8], 16), a.seed])
    qw = np.asarray(np.load(w["qpos_file"], mmap_mode="r")[w["a"]:w["b"]], dtype=np.float64)
    T = len(qw)
    chord = qw[-1, :2] - qw[0, :2]; L = float(np.linalg.norm(chord))
    feas = FEAS.get(w["id"], {})
    info = dict(window=w["id"], T=T, chord=round(L, 3), tracked=feas.get("tracked"))
    if L < 0.5:
        info["skip"] = "no_travel"
        return info, [], []
    yaw = math.atan2(chord[1], chord[0])
    qc = to_frame(qw, qw[0, :2], yaw)
    cyc = pick_cycle(BANK, L / (T * DT), rng)
    Dq = default_motion(qc, cyc, phase0=float(rng.random()))
    M, D = Body(G, qc), Body(G, Dq)
    st = FF(qc)
    tr_c = None
    tz = Path(a.tracked_dir) / f"{w['id']}.npz" if a.tracked_dir else None
    if tz is not None and tz.exists():
        tr = np.load(tz)["tracked"].astype(np.float64)
        tr_c = to_frame(tr, qw[0, :2], yaw)
        # SONIC sees no global position, so its rollout drifts; re-anchor the root path and heading to the reference
        # and keep the tracked posture (height, roll, pitch, joints): tracked_clear then measures posture, not drift.
        from cat_mjlab.cos.default_motion import qmul, qyaw, yaw_of_q
        n = min(len(tr_c), T)
        tr_c = tr_c[:n].copy()
        tr_c[:, :2] = qc[:n, :2]
        tr_c[:, 3:7] = qmul(qyaw(yaw_of_q(qc[:n, 3:7]) - yaw_of_q(tr_c[:, 3:7])), tr_c[:, 3:7])
    attempts, per_event = [], []
    for e_i, (fam, f0, f1, mag) in enumerate(w["events"]):
        fa, fb = max(0, f0), min(T, f1)
        if fb - fa < 2:
            continue
        acc, stat = fits_for_event(dict(family=fam, fa=fa, fb=fb), M, D, st, rng, K=a.k, tries=a.tries)
        attempts.append(dict(window=w["id"], event=e_i, family=fam, source=w["source"], subset=w["subset"],
                             accepted=len(acc), kinds=sorted({x["kind"] for x in acc}), **stat))
        per_event.append((e_i, fam, fa, fb, acc))
    # scenes: single-event fits, then multi-obstacle composites (one fit per event, non-overlapping footprints)
    plans = [([(e_i, fam, fa, fb, o)], f"e{e_i}k{k}") for e_i, fam, fa, fb, acc in per_event for k, o in enumerate(acc)]
    ev_with = [x for x in per_event if x[4]]
    if len(ev_with) >= 2:
        for c in range(a.composites):
            pick = [(e_i, fam, fa, fb, acc[int(rng.integers(len(acc)))]) for e_i, fam, fa, fb, acc in ev_with]
            plans.append((pick, f"multi{c}"))
    xr = (float(min(qc[:, 0].min(), 0.)) - 1.0, float(qc[:, 0].max()) + 1.0)
    scenes = []
    for parts, tag in plans:
        main = [b for p in parts for b in p[4]["boxes"]]
        dis = distractors(M, D, main, rng, x_range=xr)
        walls = room_walls(M, D, rng, xr)
        boxes = main + dis + walls
        ver = verify(G, qc, Dq, boxes)
        fam0 = parts[0][1]
        label = "arms_in" if fam0 == "sideways" and ver["d_main_body"] >= 0 and len(parts) == 1 else fam0
        tclear = None
        if tr_c is not None:
            tclear = float(Body(G, tr_c).sep(boxes)[0].min())
        kinds = [p[4]["kind"] for p in parts]
        scenes.append(dict(scene=f"{w['id']}__{tag}", window=w["id"], clip=w["clip"], source=w["source"],
                           subset=w["subset"], licence=w["licence"], family=label if len(parts) == 1 else "multi",
                           families=[p[1] for p in parts], kinds=kinds, n_obstacles=len(parts),
                           event_frames=[[p[2], p[3]] for p in parts],
                           chord=dict(origin=qw[0, :2].tolist(), yaw=yaw, length=L), gait_src=cyc["src"],
                           quality=dict(m_clear=ver["m_clear"], d_pen=-ver["d_main"], d_pen_body=-ver["d_main_body"],
                                        slack_main=min(p[4]["m_clear_main"] for p in parts),
                                        realistic=all(realism(k, p[4]["boxes"]) for k, p in zip(kinds, parts))),
                           verified=ver["ok"], verified_pen=ver["ok_pen"], tracked=feas.get("tracked"),
                           tracked_clear=tclear, n_distractors=len(dis), boxes=boxes))
    if scenes:
        np.savez_compressed(Path(a.out) / "motions" / f"{w['id']}.npz", M=qc.astype(np.float32), D=Dq.astype(np.float32),
                            **({"tracked": tr_c.astype(np.float32)} if tr_c is not None else {}))
    return info, scenes, attempts


def cmd_synth(a):
    out = Path(a.out); (out / "motions").mkdir(parents=True, exist_ok=True)
    W = [json.loads(l) for l in open(Path(a.lib) / "windows.jsonl")][:a.limit]
    import multiprocessing as mp
    t = time.time()
    with mp.get_context("fork").Pool(a.jobs, initializer=init, initargs=(a,)) as pool, \
            open(out / "scenes.jsonl", "w") as fs, open(out / "attempts.jsonl", "w") as fa, open(out / "windows.jsonl", "w") as fw:
        for k, (info, scenes, attempts) in enumerate(pool.imap_unordered(one, W, chunksize=2)):
            fw.write(json.dumps(info, default=_j) + "\n")
            for s in scenes:
                fs.write(json.dumps(s, default=_j) + "\n")
            for x in attempts:
                fa.write(json.dumps(x, default=_j) + "\n")
            if k % 500 == 0:
                print(f"{k}/{len(W)} windows {time.time() - t:.0f}s", flush=True)
    print(f"SYNTH DONE {len(W)} windows {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("events", "synth"))
    ap.add_argument("--lib", required=True); ap.add_argument("--out"); ap.add_argument("--gait-bank")
    ap.add_argument("--feasibility"); ap.add_argument("--tracked-dir"); ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--tries", type=int, default=150)
    ap.add_argument("--composites", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    (cmd_events if a.cmd == "events" else cmd_synth)(a)


if __name__ == "__main__":
    main()
