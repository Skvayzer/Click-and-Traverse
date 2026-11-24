#!/usr/bin/env python3
"""Turn a plan-blind guide into a throw guide (A1): same episodes and initial states, plus the shared throw schedule."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_mjlab.threats import make_schedule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path, help="existing plan-blind guide at the same operator speed")
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=20261125)
    a = p.parse_args()
    with np.load(a.base, allow_pickle=False) as source:
        guide = dict(source)
    meta = json.loads(str(guide["meta"]))
    meta["variant"] = "throw"
    n = len(guide["scenes"])
    keep = dict(meta=json.dumps(meta, sort_keys=True), scenes=guide["scenes"], pose=guide["pose"], velocity=guide["velocity"])
    keep.update(make_schedule(n, a.seed))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **keep)
    print(json.dumps(dict(out=str(a.out), episodes=n, meta=meta, first_launch_mean=float(keep["throw_time"][:, 0].mean()))))


if __name__ == "__main__":
    main()
