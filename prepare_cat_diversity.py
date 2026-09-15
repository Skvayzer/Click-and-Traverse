"""Build CAT's complete public procedural parameter sweep and irregular clutter rooms."""

import argparse
import json
import os
from pathlib import Path

# CPU workers each execute NumPy/SciPy; avoid accidental nested BLAS saturation.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from cat_ppo.furniture.expanded_fields import (
    DEFAULT_BASE_BANK, DEFAULT_DIFFICULTIES, DEFAULT_OUTPUT, DEFAULT_PROCEDURAL_SEEDS,
    generation_plan, prepare_expanded_fields,
)
from cat_ppo.furniture.generalist_fields import load_generalist_manifest
from cat_ppo.furniture.legacy_scenes import TYPICAL_SCENES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bank", type=Path, default=DEFAULT_BASE_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download", action="store_true", help="Download the 27 pinned typical-scene assets")
    parser.add_argument("--plan", action="store_true", help="Print requested coverage without writing assets")
    parser.add_argument("--validate", type=Path, help="Verify an existing completed manifest and every field hash")
    parser.add_argument("--difficulties", type=float, nargs="+", default=DEFAULT_DIFFICULTIES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_PROCEDURAL_SEEDS)
    parser.add_argument("--ground-counts", type=int, nargs="+", default=list(range(4)))
    parser.add_argument("--lateral-counts", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--overhead-counts", type=int, nargs="+", default=list(range(4)))
    parser.add_argument("--furniture-count", type=int, default=24)
    parser.add_argument("--generic-count", type=int, default=24)
    parser.add_argument("--furniture-first-seed", type=int, default=4001)
    parser.add_argument("--generic-first-seed", type=int, default=5001)
    parser.add_argument("--typical-scenes", nargs="*", choices=TYPICAL_SCENES, default=TYPICAL_SCENES)
    args = parser.parse_args(argv)
    if args.validate:
        manifest = load_generalist_manifest(args.validate)
        print(json.dumps(dict(manifest=str(args.validate), scenes=manifest["scene_count"],
                              fields_bytes=manifest["fields_bytes"], sha256=manifest["manifest_sha256"]), indent=2))
        return
    if min(args.furniture_count, args.generic_count) < 0:
        parser.error("Room counts must be nonnegative")
    options = dict(difficulties=args.difficulties, seeds=args.seeds,
        ground_counts=args.ground_counts, lateral_counts=args.lateral_counts, overhead_counts=args.overhead_counts,
        furniture_seeds=tuple(range(args.furniture_first_seed, args.furniture_first_seed + args.furniture_count)),
        generic_seeds=tuple(range(args.generic_first_seed, args.generic_first_seed + args.generic_count)),
        typical_scenes=args.typical_scenes)
    if args.plan:
        plan = generation_plan(**options)
        plan.pop("jobs")
        print(json.dumps(plan, indent=2))
        return
    def progress(done, total, scene_id):
        print(json.dumps(dict(completed=done, total=total, scene_id=scene_id)), flush=True)
    manifest_path = prepare_expanded_fields(args.base_bank, args.output, workers=args.workers,
                                            download=args.download, progress=progress, **options)
    manifest = load_generalist_manifest(manifest_path, verify_files=False)
    print(json.dumps(dict(manifest=str(manifest_path), scenes=manifest["scene_count"],
                          requested_generated=manifest["requested_generated_count"],
                          omitted_duplicates=manifest["duplicate_generated_count"],
                          retained_families=manifest["retained_family_counts"],
                          fields_bytes=manifest["fields_bytes"], sha256=manifest["manifest_sha256"]), indent=2))


if __name__ == "__main__":
    main()
