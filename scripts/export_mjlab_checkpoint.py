#!/usr/bin/env python3
"""Export model/Adam and shared sampler arrays in the OLD JAX environment.

Example:
  .venv/bin/python scripts/export_mjlab_checkpoint.py --runtime RUN/resume.msgpack \
      --output outputs/mjlab_migration/learner.npz --expected-source-sha256 HASH

Or expand the original released native model using the existing named-feature
mapping with --released-checkpoint PATH. The destination must be new. No source
checkpoint, optimizer, training state, W&B run or environment is modified.
The shared sampler and curriculum transfer only to the identical field bank;
physics, partial episodes and framework RNG are explicitly reset on migration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from cat_mjlab.checkpoint_arrays import expand_released_params, write_array_archive, extract_shared_sampling


def export_runtime(source, destination, expected_sha256=None):
    from flax import serialization
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Source must be a regular runtime checkpoint")
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Preserved source checksum differs")
    snapshot = serialization.msgpack_restore(payload)
    del payload
    if snapshot.get("schema") != "cat-ppo-runtime-v1":
        raise ValueError("Unrecognized runtime schema")
    if snapshot["contract"].get("normalize_observations") is not False:
        raise ValueError("The PyTorch migration requires normalization disabled")
    tree = snapshot["training_state"]
    if len(tree["paths"]) != len(tree["leaves"]) or len(set(tree["paths"])) != len(tree["paths"]):
        raise ValueError("Malformed runtime parameter paths")
    selected = [(path, array) for path, array in zip(tree["paths"], tree["leaves"])
                if path.startswith((".params.", ".optimizer_state"))]
    sampler = extract_shared_sampling(snapshot)
    bank_sha = snapshot["contract"].get("metadata", {}).get("bank_sha256")
    if not bank_sha:
        raise ValueError("Runtime lacks the field-bank SHA needed for sampler migration")
    arrays = [array for path, array in selected]
    sampling_indices = {}
    for key, value in sampler.items():
        sampling_indices[key] = len(arrays)
        arrays.append(value)
    metadata = dict(kind="runtime", source_path=str(source.resolve()), source_sha256=digest,
        source_schema=snapshot["schema"], step=int(snapshot["step"]), contract=snapshot["contract"],
        paths=[path for path, array in selected], training_array_count=len(selected),
        sampling_state=dict(bank_sha256=bank_sha, array_indices=sampling_indices),
        omitted="physics/episode state, JAX RNG, metrics windows, unused normalizer")
    return write_array_archive(destination, metadata, arrays)


def export_released(source, destination):
    from cat_ppo.furniture.learning import load_native
    from cat_ppo.furniture.control import wholebody_observation_contract
    source = Path(source)
    params, expansion = expand_released_params(load_native(source))
    paths, arrays = [], []
    for trunk, tree in (("policy", params[1]), ("value", params[2])):
        for name, layer in tree["params"].items():
            for key, value in layer.items():
                paths.append(f"{trunk}['params']['{name}']['{key}']")
                arrays.append(np.asarray(value))
    files = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(source.rglob("*")) if path.is_file()}
    metadata = dict(kind="released_expanded", source_path=str(source.resolve()), source_files_sha256=files,
        step=0, paths=paths, expansion=expansion, observation_contract=wholebody_observation_contract(),
        normalize_observations=False, omitted="unused normalizer; original release contains no Adam state")
    return write_array_archive(destination, metadata, arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--runtime", type=Path)
    sources.add_argument("--released-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    if args.expected_source_sha256 and not args.runtime:
        parser.error("--expected-source-sha256 applies to runtime files")
    result = (export_runtime(args.runtime, args.output, args.expected_source_sha256) if args.runtime
              else export_released(args.released_checkpoint, args.output))
    print(json.dumps(dict(output=str(args.output), bytes=args.output.stat().st_size,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), kind=result["kind"], step=result["step"],
        arrays=result["array_count"]), sort_keys=True))


if __name__ == "__main__":
    main()
