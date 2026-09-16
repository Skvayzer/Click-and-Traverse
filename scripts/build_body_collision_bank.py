#!/usr/bin/env python3
"""Build an immutable shared body-collision bank; never initialize JAX/training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_ppo.furniture.body_collision_bank import build_body_collision_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-manifest", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, default=Path("docs/assets/collision-proxy-proposal-20260916/proposal.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-size", type=float, default=.25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download", action="store_true", help="Fetch missing released occupancy by its pinned LFS SHA256")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    result = build_body_collision_bank(args.field_manifest, args.proposal, args.output,
        cell_size=args.cell_size, workers=args.workers, download=args.download,
        inventory_only=args.inventory_only,
        progress=lambda record: print(json.dumps(record), flush=True))
    print(json.dumps({key: value for key, value in result.items() if key not in ("scenes", "arrays")}, indent=2))


if __name__ == "__main__":
    main()
