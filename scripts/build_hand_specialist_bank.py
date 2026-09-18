#!/usr/bin/env python3
"""Extract the existing hand-protection scenes into compact immutable banks."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cat_ppo.furniture.hand_specialist import build_hand_specialist_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-manifest", type=Path, required=True)
    parser.add_argument("--collision-bank", type=Path, required=True)
    parser.add_argument("--reset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_hand_specialist_bank(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
