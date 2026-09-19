#!/usr/bin/env python3
"""Generate a standalone matched contrastive specialist field bank."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_ppo.furniture.contrastive_bank import build_contrastive_bank

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    args = parser.parse_args()
    print(build_contrastive_bank(**vars(args), progress=lambda s: print(s, flush=True)))
