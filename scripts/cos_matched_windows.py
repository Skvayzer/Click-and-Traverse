#!/usr/bin/env python3
"""Matched feasibility windows for the retargeting comparison: the same 6 s stretch of the same AMASS sequence in both
libraries (docs/cos_obstacles.md).

  python scripts/cos_matched_windows.py --a LIB_GMR --b LIB_EMBER --out-a A.jsonl --out-b B.jsonl [--per-subset 150]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cos_compare_retarget import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--out-a", required=True); ap.add_argument("--out-b", required=True)
    ap.add_argument("--per-subset", type=int, default=150); ap.add_argument("--seconds", type=float, default=6.)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    common = sorted(set(A) & set(B))
    rng = np.random.default_rng(1)
    n = int(a.seconds / 0.02)
    with open(a.out_a, "w") as fa, open(a.out_b, "w") as fb:
        for sub in sorted({k.split("/")[0] for k in common}):
            ks = [k for k in common if k.split("/")[0] == sub and min(A[k]["frames"], B[k]["frames"]) > n + 10]
            for k in [ks[i] for i in rng.permutation(len(ks))[:a.per_subset]]:
                t0 = int(rng.integers(0, min(A[k]["frames"], B[k]["frames"]) - n))
                for f, lib, c in ((fa, a.a, A[k]), (fb, a.b, B[k])):
                    f.write(json.dumps(dict(id=f"{sub}|{k}|{t0}", qpos_file=str(Path(lib) / f"qpos_{c['shard']:03d}.npy"),
                                            a=c["start"] + t0, b=c["start"] + t0 + n)) + "\n")


if __name__ == "__main__":
    main()
