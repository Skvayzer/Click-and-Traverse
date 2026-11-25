#!/usr/bin/env python3
"""S2 oracle plans for every (scene, trial) start (docs/fullpath_plan.md): the start state is taken from a recorded episode
(starts are identical across speeds and policies for a scene and trial), the scene's boxes and field-grid bounds from the
recording and the bank manifest; margins are tried in order until a plan is found.

    fullpath_oracle.py --starts /tmp/cat-fullpath/runs/fix0_plan --library lib.npz --out /tmp/cat-fullpath/oracle --workers 28
Output per start: OUT/<scene>_<trial>.npz with qpos [n, 36] (50 Hz), modes, meta (ok, margin, min_sep, tried, ...).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFESTS = ("/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab/data/furniture/tablegap_v1_packed/manifest.json",
             "/tmp/cat-overnight/planning_v1_packed/manifest.json")
_S = {}


def setup(library):
    import torch
    torch.set_num_threads(2)
    from cat_mjlab.exact_geometry import ExactGeometry
    from cat_mjlab.fullpath import ClipLibrary
    _S["g"], _S["lib"] = ExactGeometry("cpu"), ClipLibrary.load(library)
    _S["grid"] = {}
    for m in MANIFESTS:
        for s in json.load(open(m))["scenes"]:
            _S["grid"][s["scene_id"]] = s


def one(job):
    rec, out, margins, budget = job
    from cat_mjlab.fullpath import plan_scene, MODE_NAMES
    if Path(out).exists():
        return json.loads(str(np.load(out)["meta"]))
    d = np.load(rec)
    m = json.loads(str(d["meta"]))
    boxes = {k: d[k] for k in ("box_center", "box_rot", "box_half")}
    s = _S["grid"][m["scene"]]
    o = np.array(s["origin"], float); e = o + (np.array(s["shape"], float) - 1) * float(s["dx"])
    gxy = np.array(m["goal"][:2]); u = gxy - np.array(m["start"][:2])
    radius = .3 if m["planning_benchmark"] else None
    tried = []
    for mg in margins:
        r = plan_scene(_S["lib"], _S["g"], boxes, d["qpos"][0], gxy, u, margin=mg, budget_s=budget, goal_radius=radius,
                       bounds=(o[0], e[0], o[1], e[1]))
        tried.append(dict(margin=mg, ok=r["ok"], seconds=r["seconds"], expansions=r["expansions"], progress=r["progress"]))
        if r["ok"]:
            break
    lib = _S["lib"]
    same = [lib.sources[c][0] == m["scene"] for c in r["clips"]]
    meta = dict(scene=m["scene"], trial=m["trial"], ok=r["ok"], margin=r["margin"] if r["ok"] else None, min_sep=r["min_sep"],
                tried=tried, route=r["route"], progress=r["progress"], seconds=round(sum(t["seconds"] for t in tried), 1),
                same_scene_clips=float(np.mean(same)) if same else None, duration_s=round(len(r["qpos"]) * .02, 2),
                mode_share=dict(zip(MODE_NAMES, np.asarray(r["modes"]).mean(0).round(3).tolist())) if len(r["modes"]) else None)
    np.savez_compressed(out, qpos=r["qpos"], modes=r["modes"], clips=r["clips"], meta=json.dumps(meta))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--starts", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--speed", default="0.15")
    p.add_argument("--margins", default="0.08,0.06,0.04,0.02")
    p.add_argument("--budget", type=float, default=150.)
    p.add_argument("--workers", type=int, default=28)
    p.add_argument("--reverse", action="store_true", help="process starts in reverse order (a second worker pool)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    margins = [float(x) for x in a.margins.split(",")]
    jobs = []
    for f in sorted(a.starts.glob(f"*_{a.speed}_*.npz")):
        scene, _, trial = f.stem.rsplit("_", 2)
        jobs.append((str(f), str(a.out / f"{scene}_{trial}.npz"), margins, a.budget))
    if a.reverse:
        jobs.reverse()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    with ProcessPoolExecutor(a.workers, initializer=setup, initargs=(str(a.library),)) as ex:
        metas = []
        for meta in ex.map(one, jobs):
            metas.append(meta)
            print(json.dumps({k: meta[k] for k in ("scene", "trial", "ok", "margin", "seconds")}), flush=True)
    ok = [m for m in metas if m["ok"]]
    print(json.dumps(dict(plans=len(metas), ok=len(ok), margins={str(mg): sum(m["margin"] == mg for m in ok) for mg in margins})))
    (a.out / "summary.json").write_text(json.dumps(metas))


if __name__ == "__main__":
    main()
