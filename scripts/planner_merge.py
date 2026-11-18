#!/usr/bin/env python3
"""Merge planner datasets collected from several policies into one root, by linking their shards (no copy).

    python scripts/planner_merge.py --out DATA_ALL DATA_V7 DATA_E2 DATA_T2

Shard k of source i becomes <out>/<split>/<i+1><k:02d>_<file>.npy (a symlink), so load_split reads one dataset. The
sources must share the bank and the scene split (same seed), which their info.json records; this script checks both.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("sources", type=Path, nargs="+")
    args = p.parse_args(argv)
    if len(args.sources) > 9:
        raise ValueError("at most 9 sources")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"{args.out} is not empty")
    infos = [json.loads((s / "info.json").read_text()) for s in args.sources]
    for key in ("bank", "train_scenes", "test_scenes", "grid", "horizon_frames", "frame_stride_steps"):
        if any(info[key] != infos[0][key] for info in infos[1:]):
            raise ValueError(f"sources differ in {key}: one dataset needs one bank and one scene split")
    ids = [info["source_id"] for info in infos]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate source ids {ids}: collect each policy with its own --source-id")
    linked = {}
    for i, src in enumerate(args.sources):
        for split in ("train", "test"):
            for f in sorted((src / split).glob("*_*.npy")):
                k, rest = f.name.split("_", 1)
                if int(k) > 99:
                    raise ValueError(f"{f}: more than 100 shards per source")
                target = args.out / split / f"{i + 1}{int(k):02d}_{rest}"
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(f.resolve(), target)
                linked[split] = linked.get(split, 0) + 1
    merged = {k: v for k, v in infos[0].items() if k not in ("checkpoint", "source_id", "examples", "seconds")}
    merged.update(schema="wbp-planner-data-v2-merged",
                  sources=[dict(path=str(s.resolve()), checkpoint=info["checkpoint"], source_id=info["source_id"],
                                examples=info["examples"]) for s, info in zip(args.sources, infos)],
                  examples={split: sum(info["examples"].get(split, 0) for info in infos) for split in ("train", "test")})
    (args.out / "info.json").write_text(json.dumps(merged, indent=1) + "\n")
    print("MERGE DONE", json.dumps(dict(out=str(args.out), files=linked, examples=merged["examples"], source_ids=ids)))


if __name__ == "__main__":
    main()
