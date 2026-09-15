"""Prepare CAT's 37 configured scenes and two dense FMM rooms, verifying sources."""
from pathlib import Path
import argparse
import json

from cat_ppo.furniture.generalist_fields import load_generalist_manifest, prepare_generalist_fields


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-config", type=Path, default=Path("configs/cat_generalist_released.json"))
    parser.add_argument("--output", type=Path, default=Path("data/furniture/cat_generalist"))
    parser.add_argument("--download", action="store_true", help="Fetch pinned original public field arrays")
    parser.add_argument("--original-only", action="store_true")
    parser.add_argument("--reconstruct-missing-original", action="store_true",
                        help="Explicitly reconstruct unavailable D8G2L3O2S13 with the upstream generator; original byte identity is unknown")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate", type=Path, help="Validate an existing bank without regenerating it")
    args = parser.parse_args(argv)
    path = args.validate or prepare_generalist_fields(args.released_config, args.output,
        download=args.download, clutter=not args.original_only, seed=args.seed, workers=args.workers,
        reconstruct_missing=args.reconstruct_missing_original)
    manifest = load_generalist_manifest(path)
    print(json.dumps(dict(manifest=str(path), original_scenes=manifest["original_count"],
        scenes=manifest["scene_count"], field_bytes=manifest["fields_bytes"],
        byte_verified_originals=manifest["byte_verified_original_count"],
        reconstructed_originals=manifest["reconstructed_original_count"],
        sha256=manifest["manifest_sha256"]), indent=2))


if __name__ == "__main__":
    main()
