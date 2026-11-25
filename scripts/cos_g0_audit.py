#!/usr/bin/env python3
"""G0 event audit (docs/cos_obstacles.md): run the deviation-event detectors on every clip of the G1 motion library.

  python scripts/cos_g0_audit.py --library .../library_v1 --out OUT.json [--features-cache F.npz] [--calibrate]
"""
from __future__ import annotations

import argparse, collections, json, re, time
from pathlib import Path

import numpy as np

from cat_mjlab.cos.events import CAL, CLEAN, FAMILIES, TH, clean, detect, DT
from cat_mjlab.cos.features import FrameFeatures


def base_take(clip_id):
    """Distinct-take key: PHUMA chunks of one sequence and OmniRetarget augmentations count as one take."""
    s = re.sub(r"_chunk_\d+$", "", clip_id)
    s = re.sub(r"_(rot|trans)_\d+$|_original$", "", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True, help="path prefix of library_v1.{npz,json}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--features-cache")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    meta = json.loads(Path(a.library + ".json").read_text())
    L = np.load(a.library + ".npz")
    clip = L["clip"]
    t = time.time()
    if a.features_cache and Path(a.features_cache).exists():
        F = dict(np.load(a.features_cache))
    else:
        F = FrameFeatures()(L["qpos"])
        if a.features_cache:
            np.savez(a.features_cache, **F)
    print(f"features {len(clip)} frames in {time.time() - t:.0f}s", flush=True)
    if a.calibrate:
        w = L["skills"][:, 0] > 0
        print({"top_body_up": float(np.median(F["top_body"][w])), "hand_z_walk": float(np.median(F["hand_z"][w])),
               "arm_lat_p5": float(np.percentile(F["elbow_lat"][w], 5)), "n_walk_frames": int(w.sum())})
        return
    starts = np.r_[0, np.where(np.diff(clip) != 0)[0] + 1]; ends = np.r_[starts[1:], len(clip)]
    events = []
    for s, e in zip(starts, ends):
        ci = int(clip[s]); c = meta["clips"][ci]
        Fc = {k: v[s:e] for k, v in F.items()}
        ev = detect(Fc)
        for fam, lst in ev.items():
            for a0, b0, mag in lst:
                events.append(dict(family=fam, clip=ci, clip_id=c["clip_id"], source=c["source"], category=c.get("category"),
                                   take=base_take(c["clip_id"]), obstacle_eligible=bool(c.get("obstacle_eligible", False)),
                                   frame0=int(s), t0=round(a0 * DT, 2), t1=round(b0 * DT, 2), dur=round((b0 - a0) * DT, 2),
                                   mag=round(mag, 3), clip_seconds=c["seconds"],
                                   clean=bool(clean(fam, a0, b0, Fc, c["clip_id"] + " " + c.get("category", ""), c["source"]))))
    # summary
    summ = {}
    for fam, tier in [(f, t) for t in ("raw", "clean") for f in FAMILIES]:
        E = [x for x in events if x["family"] == fam and (tier == "raw" or x["clean"])]
        by = collections.defaultdict(list)
        for x in E:
            by[x["source"]].append(x)
        summ[f"{tier}/{fam}"] = dict(events=len(E), takes=len({x["take"] for x in E}), clips=len({x["clip"] for x in E}),
                         seconds=round(sum(x["dur"] for x in E), 1),
                         dur_p50=round(float(np.median([x["dur"] for x in E])), 2) if E else None,
                         per_source={k: dict(events=len(v), takes=len({x["take"] for x in v}),
                                             seconds=round(sum(x["dur"] for x in v), 1)) for k, v in sorted(by.items())},
                         trainable=len(E) >= 20 and len({x["take"] for x in E}) >= 10)
    out = dict(thresholds={k: float(v) for k, v in TH.items()}, calibration=CAL, clean_rules={k: float(v) for k, v in CLEAN.items()}, n_clips=len(starts),
               n_frames=int(len(clip)), summary=summ, events=events)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for fam, s in summ.items():
        print(f"{fam:12s} events {s['events']:5d} takes {s['takes']:4d} sec {s['seconds']:7.1f} p50 {s['dur_p50']} "
              f"trainable={s['trainable']}  " + ", ".join(f"{k}:{v['events']}/{v['takes']}" for k, v in s["per_source"].items()))


if __name__ == "__main__":
    main()
