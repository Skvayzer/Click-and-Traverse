#!/usr/bin/env python3
"""Convert scripts/fullpath_rollout.py episodes into the scripts/record_plan_reflex.py recording layout (one world per
episode, stacked), so scripts/render_plan_reflex.py can render them (videos of docs/fullpath_plan.md).

    fullpath_to_rec.py EP1.npz [EP2.npz ...] --out rec.npz
Episodes must share dt; worlds are padded to the longest episode (the renderer stops at each world's end).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episodes", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    eps = [dict(np.load(f)) for f in a.episodes]
    metas = [json.loads(str(e["meta"])) for e in eps]
    T = max(len(e["qpos"]) - 1 for e in eps)
    n = len(eps)
    rec = dict(qpos=np.zeros((T + 1, n, 36), np.float32), plan=np.zeros((T + 1, n, 21, 36), np.float32),
               plan_t=np.zeros((T + 1, n), np.float32), released=np.zeros((T + 1, n, 2), bool), has_plan=np.zeros((T + 1, n), bool),
               hand_clear=np.ones((T + 1, n, 2), np.float32), fallback=np.zeros((T + 1, n), bool),
               body_clear=np.ones((T + 1, n), np.float32))
    end, outcome = [], []
    for w, (e, m) in enumerate(zip(eps, metas)):
        L = len(e["qpos"]) - 1                         # the last row is never filled (episode ended)
        q = e["qpos"][:L]
        rec["qpos"][:L, w] = q; rec["qpos"][L:, w] = q[-1]
        rec["plan_t"][:L, w] = e["plan_t"][:L]; rec["released"][:L, w] = e["released"][:L]
        rec["body_clear"][:L, w] = e["body_clear"][:L]
        rec["hand_clear"][:L, w] = e["body_clear"][:L, None]
        if "ev_t" in e:
            k = np.searchsorted(e["ev_t"], np.arange(L), side="left") - 1
            ok = k >= 0
            rec["plan"][:L, w][ok] = e["ev_chosen"][k[ok]]
            rec["has_plan"][:L, w] = ok & e["valid"][:L]
        end.append(L - 1); outcome.append(m["outcome"])
    meta = dict(checkpoint=metas[0]["checkpoint"], scenes=[m["scene"] for m in metas], mix=metas[0]["mix"],
                speed=metas[0]["speed"], band_m=metas[0]["band_m"], outcome=outcome, end=end, dt=metas[0]["dt"],
                bank=metas[0]["bank"], fallback=None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, meta=json.dumps(meta), **rec)
    print("converted", n, "episodes ->", a.out)


if __name__ == "__main__":
    main()
