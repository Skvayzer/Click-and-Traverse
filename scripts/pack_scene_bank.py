#!/usr/bin/env python3
"""Convert a field bank to the packed encoding, writing a new bank beside it.

The source bank is never modified: existing checkpoints pin its manifest hash,
and rewriting it in place would invalidate both --resume and rollout recording.

Each scene keeps its own directory layout; only the field files change:
    sdf.npy  float32          -> sdf.npy   float16
    bf.npy   float32 (...,3)  -> bf.npy    int16 (...,2) + bf_valid.npy  bits
    gf.npy   float32 (...,3)  -> gf.npy    int16 (...,2) + gf_valid.npy  bits
    obs.npy  uint8            -> obs.npy   packed bits
Everything else in the scene directory is copied verbatim.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cat_mjlab.packing.field_packing import (  # noqa: E402
    SCHEMA, pack_direction, pack_occupancy, pack_scalar,
    unpack_direction, unpack_occupancy, unpack_scalar,
)
from cat_ppo.furniture.generalist_fields import (  # noqa: E402
    load_generalist_manifest, scene_directory,
)


def convert_scene(source, target, shape, *, verify):
    target.mkdir(parents=True, exist_ok=True)
    report = {}
    sdf = np.load(source / "sdf.npy")
    np.save(target / "sdf.npy", pack_scalar(sdf))
    if verify:
        report["sdf_max_error_m"] = float(np.abs(unpack_scalar(np.load(target / "sdf.npy")) - sdf).max())
    for name in ("bf", "gf"):
        field = np.load(source / f"{name}.npy")
        codes, valid = pack_direction(field)
        np.save(target / f"{name}.npy", codes)
        np.save(target / f"{name}_valid.npy", valid)
        if verify:
            back = unpack_direction(codes, valid, shape)
            norm = np.linalg.norm(field, axis=-1, keepdims=True)
            unit = np.where(norm > 0., field / np.maximum(norm, 1e-20), 0.)
            nonzero = norm[..., 0] > 0.
            cos = np.clip(np.sum(unit * back, axis=-1)[nonzero], -1., 1.)
            report[f"{name}_max_angle_deg"] = float(np.degrees(np.arccos(cos)).max())
            report[f"{name}_zeros_preserved"] = bool(np.all(back[~nonzero] == 0.))
    if (source / "obs.npy").exists():
        obs = np.load(source / "obs.npy")
        np.save(target / "obs.npy", pack_occupancy(obs))
        if verify:
            report["obs_identical"] = bool(np.array_equal(unpack_occupancy(np.load(target / "obs.npy"), obs.shape), obs))
    for extra in source.iterdir():
        if extra.suffix != ".npy":
            shutil.copy2(extra, target / extra.name)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New bank directory (must not exist)")
    p.add_argument("--limit", type=int, help="Convert only the first N scenes, for a trial run")
    p.add_argument("--verify", action="store_true", help="Decode every scene back and report worst error")
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output bank already exists; refusing to overwrite")

    manifest_path = args.manifest.resolve()
    manifest = load_generalist_manifest(manifest_path, verify_files=False)
    records = list(manifest["scenes"][: args.limit] if args.limit else manifest["scenes"])
    args.output.mkdir(parents=True)

    worst = {}
    raw = packed = 0
    for index, record in enumerate(records):
        source = scene_directory(manifest, manifest_path, record)
        target = (args.output / record["path"]).resolve()
        report = convert_scene(source, target, tuple(record["shape"]), verify=args.verify)
        for key, value in report.items():
            if isinstance(value, bool):
                worst[key] = worst.get(key, True) and value
            else:
                worst[key] = max(worst.get(key, 0.), value)
        # The record must describe the files that now exist, not the ones it came from.
        from cat_ppo.furniture.generalist_fields import _field_records
        record = dict(record, fields=_field_records(target))
        records[index] = record
        raw += sum(f.stat().st_size for f in source.iterdir() if f.suffix == ".npy")
        packed += sum(f.stat().st_size for f in target.iterdir() if f.suffix == ".npy")
        if (index + 1) % 50 == 0 or index + 1 == len(records):
            print(f"{index + 1}/{len(records)}  {raw / 1e9:.2f} GB -> {packed / 1e9:.2f} GB", flush=True)

    # The packed bank is self-contained: every field file now lives inside it, so the
    # inherited retention root must point here. Left as-is it would send most scenes
    # back to the original raw store and the packed copies would never be read.
    packed_manifest = dict(manifest, schema_packed=SCHEMA, packed_from=str(manifest_path),
                           scenes=records, scene_count=len(records))
    if "flat_balance" in packed_manifest:
        from cat_ppo.furniture.generalist_fields import sha256, _json_hash
        from cat_ppo.furniture.packed_balance_bank import SCHEMA as PACKED_SCHEMA
        packed_manifest["flat_balance"] = dict(
            packed_manifest["flat_balance"], schema=PACKED_SCHEMA,
            encoding=SCHEMA, retention_storage_root=str(args.output.resolve()),
            source=dict(manifest=str(manifest_path), sha256=sha256(manifest_path),
                        scene_records_sha256=_json_hash([{k: v for k, v in r.items() if k != "fields"}
                                                         for r in records])))
        packed_manifest.pop("manifest_sha256", None)
        packed_manifest["manifest_sha256"] = _json_hash(packed_manifest)
    (args.output / "manifest.json").write_text(json.dumps(packed_manifest, indent=1, sort_keys=True) + "\n")
    summary = dict(scenes=len(records), raw_bytes=raw, packed_bytes=packed,
                   ratio=raw / max(packed, 1), verification=worst)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
