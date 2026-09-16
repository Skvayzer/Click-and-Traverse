#!/usr/bin/env python3
"""Publish a corrected room bank while hardlinking unchanged CAT scene files.

The source bank is never rewritten. Completed rooms are resumable, and the new
manifest is published only after every field and source fingerprint validates.
Use an output directory on the same filesystem: this command intentionally does
not fall back to copying the multi-GB unchanged CAT collection.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from cat_ppo.furniture.expanded_fields import check_free_space_connectivity
from cat_ppo.furniture.generalist_fields import (
    EXPANDED_SCHEMA, FIELD_NAMES, _field_records, _json_hash,
    load_generalist_manifest, make_clutter_fields, sha256,
)


BUILDER_VERSION = "cat-room-navigation-bank-migration-v1"
ROOM_FAMILIES = {"furniture", "generic_clutter"}
CAT_FAMILIES = {"original_cat", "published_cat", "procedural_cat"}
REQUIRED_ROOM_METADATA = {
    "occupancy": "conservative-voxel-cell-OBB-intersection-v1",
    "room_navigation": "ordered-certified-route-v1",
    "route_used_for_guidance": True,
}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _code_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), root / "cat_ppo/furniture/generalist_fields.py",
             root / "cat_ppo/furniture/room_geometry.py"]
    # The runtime route controller is part of provenance when present. This is
    # optional only to keep the migration module importable during development.
    paths += sorted((root / "cat_ppo/furniture").glob("room_navigation*.py"))
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def _contained_directory(root, relative):
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Scene directory escapes its bank: {relative}")
    return path


def _hardlink_tree(source, destination):
    """Preserve exact files; never open a linked output file for writing."""
    count, saved_bytes = 0, 0
    for file in sorted(source.rglob("*")):
        if file.is_symlink():
            raise ValueError(f"Refusing a symlink in an immutable scene tree: {file}")
        if file.is_dir():
            continue
        if not file.is_file():
            raise ValueError(f"Scene contains a nonregular file: {file}")
        target = destination / file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or not os.path.samefile(file, target):
                raise FileExistsError(f"Existing destination is not its source hardlink: {target}")
        else:
            try:
                os.link(file, target)
            except OSError as error:
                raise OSError(
                    f"Could not hardlink {file} to {target}; choose a new output directory "
                    "on the source filesystem (copy fallback is intentionally disabled)") from error
        count += 1
        saved_bytes += file.stat().st_size
    return count, saved_bytes


def _require_corrected_room_source(source):
    for name, expected in REQUIRED_ROOM_METADATA.items():
        if source.get(name) != expected:
            raise ValueError(f"Room builder has not integrated the navigation fix: {name} must be {expected!r}")


def _validate_room_record(record, directory):
    _require_corrected_room_source(record["source"])
    if record["fields"] != _field_records(directory):
        raise ValueError(f"Rebuilt room fields differ from their hashes: {directory}")
    if sha256(directory / "scene.json") != record["scene_sha256"]:
        raise ValueError(f"Rebuilt room scene fingerprint differs: {directory}")
    if sha256(directory / "source.json") != record["source"]["metadata_sha256"]:
        raise ValueError(f"Rebuilt room source fingerprint differs: {directory}")
    if sha256(directory / "obs.npy") != record["source"]["occupancy_sha256"]:
        raise ValueError(f"Rebuilt room occupancy fingerprint differs: {directory}")


def _rebuild_room(payload):
    old, source_root, output_root, plan = payload
    source_directory = _contained_directory(Path(source_root), old["path"])
    directory = _contained_directory(Path(output_root), old["path"])
    cache_path = directory / "navigation_migration_record.json"
    cache_key = _json_hash(dict(plan_sha256=plan["plan_sha256"], original_record=old))
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("cache_key") != cache_key:
            raise ValueError(f"Cached room belongs to another migration: {directory}")
        _validate_room_record(cached["record"], directory)
        return cached["record"]
    scene_path = source_directory / "scene.json"
    if sha256(scene_path) != old["scene_sha256"]:
        raise ValueError(f"Source room changed during migration: {scene_path}")
    scene = json.loads(scene_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    generated = make_clutter_fields(scene, directory, dx=old["dx"])
    _require_corrected_room_source(generated["source"])
    for key in ("scene_id", "family", "shape", "origin", "dx", "start", "goal", "reset_xy_scale", "reset_yaw"):
        if generated[key] != old[key]:
            raise ValueError(f"Navigation migration unexpectedly changed room {key}: {old['scene_id']}")
    # Preserve task semantics, sampling weights and all unrelated per-scene keys.
    record = copy.deepcopy(old)
    record.update({key: generated[key] for key in ("fields", "source", "scene_sha256")})
    record.update({key: value for key, value in generated.items() if key not in old})
    rebuilt_scene = json.loads((directory / "scene.json").read_text())
    original_geometry = {key: value for key, value in scene.items() if key != "cat_field_generation"}
    new_geometry = {key: value for key, value in rebuilt_scene.items() if key != "cat_field_generation"}
    if new_geometry != original_geometry:
        raise ValueError(f"Navigation migration changed canonical room geometry: {old['scene_id']}")
    occupancy = np.load(directory / "obs.npy", allow_pickle=False)
    connectivity = check_free_space_connectivity(occupancy, np.asarray(record["origin"]), record["dx"],
                                                 record["start"], record["goal"])
    source = copy.deepcopy(old["source"])
    source.pop("metadata_sha256", None)
    source.update(generated["source"])
    source.update(occupancy_sha256=sha256(directory / "obs.npy"), connectivity=connectivity)
    source.setdefault("source_hashes", {}).update(plan["source_hashes"])
    source["navigation_migration"] = dict(
        builder=BUILDER_VERSION, plan_sha256=plan["plan_sha256"],
        source_bank_sha256=plan["source_manifest_sha256"],
        source_scene_sha256=old["scene_sha256"],
        source_field_sha256={name: old["fields"][name]["sha256"] for name in FIELD_NAMES},
        canonical_geometry_unchanged=True, task_semantics_unchanged=True)
    _write_json(directory / "source.json", source)
    record["source"] = dict(source, metadata_sha256=sha256(directory / "source.json"))
    _validate_room_record(record, directory)
    _write_json(cache_path, dict(cache_key=cache_key, record=record))
    return record


def rebuild_bank(source_manifest, output_dir, *, workers=1, progress=None):
    """Rebuild room fields only; keep the source bank intact and publish last."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    source_manifest = Path(source_manifest).resolve()
    source_root, output = source_manifest.parent, Path(output_dir).resolve()
    if output == source_root or output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("Use separate, non-nested source and destination bank directories")
    source_file_sha256 = sha256(source_manifest)
    original = load_generalist_manifest(source_manifest)
    if original["schema"] != EXPANDED_SCHEMA:
        raise ValueError("Navigation migration requires an expanded bank with explicit room/CAT semantics")
    if any(record["family"] not in ROOM_FAMILIES | CAT_FAMILIES for record in original["scenes"]):
        raise ValueError("Unknown scene family: refusing to infer whether it may be rebuilt")
    rooms = [record for record in original["scenes"] if record["family"] in ROOM_FAMILIES]
    if not rooms:
        raise ValueError("Source bank has no rooms to rebuild")
    plan = dict(builder=BUILDER_VERSION, source_manifest=str(source_manifest),
                source_manifest_file_sha256=source_file_sha256,
                source_manifest_sha256=original["manifest_sha256"], source_hashes=_code_hashes(),
                source_scene_count=original["scene_count"], room_scene_ids=[s["scene_id"] for s in rooms],
                unchanged_scene_count=original["scene_count"] - len(rooms),
                required_room_metadata=REQUIRED_ROOM_METADATA,
                unchanged_storage="hardlinks only; original CAT/procedural record bytes unchanged")
    plan["plan_sha256"] = _json_hash(plan)
    plan_path = output / "navigation_migration_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise FileExistsError("Destination contains a different migration plan; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Destination is not empty and has no matching migration plan")
        output.mkdir(parents=True, exist_ok=True)
        if source_root.stat().st_dev != output.stat().st_dev:
            raise ValueError("Source and destination must share a filesystem for hardlinks")
        _write_json(plan_path, plan)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = load_generalist_manifest(manifest_path)
        if existing.get("navigation_migration", {}).get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("Existing complete bank has different migration provenance")
        return manifest_path

    records = copy.deepcopy(original["scenes"])
    linked_count, linked_bytes = 0, 0
    unchanged_hashes = []
    for record in records:
        if record["family"] not in CAT_FAMILIES:
            continue
        source_directory = _contained_directory(source_root, record["path"])
        target_directory = _contained_directory(output, record["path"])
        count, size = _hardlink_tree(source_directory, target_directory)
        linked_count += count
        linked_bytes += size
        unchanged_hashes.append(dict(scene_id=record["scene_id"], family=record["family"],
            fields={name: record["fields"][name]["sha256"] for name in FIELD_NAMES}))
    payloads = [(record, str(source_root), str(output), plan) for record in rooms]
    rebuilt = {}
    if workers == 1:
        for index, payload in enumerate(payloads, 1):
            record = _rebuild_room(payload)
            rebuilt[record["scene_id"]] = record
            if progress:
                progress(index, len(rooms), record["scene_id"])
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_rebuild_room, payload) for payload in payloads]
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                rebuilt[record["scene_id"]] = record
                if progress:
                    progress(index, len(rooms), record["scene_id"])
    records = [rebuilt.get(record["scene_id"], record) for record in records]
    if source_file_sha256 != sha256(source_manifest) or plan["source_hashes"] != _code_hashes():
        raise ValueError("Source manifest or builder code changed while the replacement bank was being built")
    for old, new in zip(original["scenes"], records, strict=True):
        if old["family"] in CAT_FAMILIES and old != new:
            raise ValueError(f"An unchanged CAT record was modified: {old['scene_id']}")

    # Keep old generation coverage as a labelled historical artifact. Its room
    # field hashes describe the parent bank, so they must not masquerade as new.
    for filename, key in (("preparation_plan.json", "preparation_plan_sha256"),
                          ("generation_coverage.json", "generation_coverage_sha256")):
        source = source_root / filename
        if source.exists():
            target = output / "parent_generation" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not os.path.samefile(source, target):
                    raise FileExistsError(f"Historical provenance file differs: {target}")
            else:
                os.link(source, target)
    report = dict(builder=BUILDER_VERSION, plan_sha256=plan["plan_sha256"],
        source_manifest_sha256=original["manifest_sha256"],
        scene_count=len(records), rebuilt_room_count=len(rooms),
        unchanged_cat_scene_count=len(unchanged_hashes),
        unchanged_cat_scene_hashes=unchanged_hashes,
        hardlinked_file_count=linked_count, hardlinked_logical_bytes=linked_bytes,
        original_cat_and_procedural_hashes_identical=True,
        canonical_room_geometry_unchanged=True, scene_order_unchanged=True,
        retained_family_counts=dict(Counter(record["family"] for record in records)))
    _write_json(output / "navigation_migration_report.json", report)
    manifest = copy.deepcopy(original)
    manifest.pop("manifest_sha256")
    parent_generation = {key: manifest.pop(key) for key in
                         ("preparation_plan_sha256", "generation_coverage_sha256") if key in manifest}
    manifest.update(scenes=records,
        fields_bytes=sum(field["size_bytes"] for record in records for field in record["fields"].values()),
        navigation_migration=dict(
            builder=BUILDER_VERSION, plan_sha256=plan["plan_sha256"],
            report_sha256=sha256(output / "navigation_migration_report.json"),
            source_manifest_sha256=original["manifest_sha256"],
            parent_generation=parent_generation, original_cat_and_procedural_arrays_unchanged=True,
            room_geometry_unchanged=True, required_room_metadata=REQUIRED_ROOM_METADATA))
    manifest["manifest_sha256"] = _json_hash(manifest)
    temporary = output / ".manifest.validating.json"
    _write_json(temporary, manifest)
    # This rechecks every actual output array, including all linked originals,
    # against the original recorded SHA256 and all new room fingerprints.
    load_generalist_manifest(temporary)
    temporary.replace(manifest_path)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent CPU room builders; default 1")
    args = parser.parse_args()
    path = rebuild_bank(args.source_manifest, args.output_dir, workers=args.workers,
                        progress=lambda n, total, identity: print(f"[{n}/{total}] {identity}", flush=True))
    print(json.dumps(dict(manifest=str(path), manifest_sha256=load_generalist_manifest(path, verify_files=False)["manifest_sha256"])), flush=True)


if __name__ == "__main__":
    main()
