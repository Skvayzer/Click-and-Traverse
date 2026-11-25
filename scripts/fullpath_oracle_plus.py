#!/usr/bin/env python3
"""S2 amendment (docs/fullpath_plan.md, 11:20): add the recorded clean traversals from the same start as oracle candidates.

    fullpath_oracle_plus.py --oracle /tmp/cat-fullpath/oracle --runs /tmp/cat-fullpath/runs --out /tmp/cat-fullpath/oracle_plus
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--floor", type=float, default=.02)
    a = p.parse_args()
    import torch
    torch.set_num_threads(8)
    from cat_mjlab.exact_geometry import ExactGeometry
    g = ExactGeometry("cpu")
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(a.oracle.glob("*_[01].npz")):
        scene, trial = f.stem.rsplit("_", 1)
        o = np.load(f); meta = json.loads(str(o["meta"]))
        best = dict(source="search", margin=meta["min_sep"] if meta["ok"] else -1., qpos=o["qpos"]) if meta["ok"] else None
        boxes = None
        for rec in sorted(a.runs.glob(f"*/{scene}_*_{trial}.npz")):
            d = np.load(rec); m = json.loads(str(d["meta"]))
            if m["outcome"]["result"] != "clean" or m["scene"] != scene:
                continue
            boxes = {k: d[k] for k in ("box_center", "box_rot", "box_half")}
            q = d["qpos"][:-1]
            sep = float(g.min_separation(q, boxes).min())
            if sep >= a.floor and (best is None or sep > best["margin"] + 1e-6):
                best = dict(source=f"recorded:{rec.parent.name}@{m['speed']:g}", margin=sep, qpos=q)
        out = dict(meta | dict(search_ok=meta["ok"], search_margin=meta["margin"]))
        if best is not None:
            out.update(ok=True, source=best["source"], min_sep=round(best["margin"], 4), duration_s=round(len(best["qpos"]) * .02, 2))
            np.savez_compressed(a.out / f.name, qpos=best["qpos"].astype(np.float32), meta=json.dumps(out))
        else:
            out.update(ok=False, source=None)
            np.savez_compressed(a.out / f.name, qpos=o["qpos"], meta=json.dumps(out))
        rows.append(out)
        print(json.dumps({k: out[k] for k in ("scene", "trial", "ok", "source", "min_sep", "search_ok")}), flush=True)
    (a.out / "summary.json").write_text(json.dumps(rows))
    import collections
    print(json.dumps(dict(n=len(rows), ok=sum(r["ok"] for r in rows),
                          sources=collections.Counter((r["source"] or "none").split("@")[0] for r in rows))))


if __name__ == "__main__":
    main()
