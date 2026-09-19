#!/usr/bin/env python3
"""Add matched table-height passages to an immutable cabinet contrastive bank."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cat_ppo.furniture.contrastive_bank import MARKER, TABLE_MARKER, ROLES, validate_contrastive_manifest
from cat_ppo.furniture.generalist_fields import (
    _field_records, _json_hash, load_generalist_manifest, make_clutter_fields, sha256,
)


def _copy_immutable(source, destination):
    """Hard-link local immutable arrays; fall back to a copy across filesystems."""
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"Existing immutable file differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build(base_manifest, output, *, seed=20260919, groups=16, split="train", progress=print):
    from cat_ppo.furniture.table_edge_passages import generate_table_edge_pair
    base_manifest, output = Path(base_manifest).resolve(), Path(output).resolve()
    if output == base_manifest.parent or output.is_relative_to(base_manifest.parent):
        raise ValueError("Write the extended bank outside its immutable source bank")
    if type(groups) is not int or groups < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Expected nonnegative seed and positive group count")
    base = load_generalist_manifest(base_manifest)
    if base["contrastive_specialist"]["schema"] != MARKER or base.get("split") != split:
        raise ValueError("Source must be a cabinet-only bank in the same dataset split")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "manifest.json"
    settings = dict(table_seed=seed, table_groups=groups, split=split,
                    base_manifest_sha256=sha256(base_manifest))
    if destination.exists():
        existing = load_generalist_manifest(destination)
        if existing["contrastive_specialist"].get("table_extension") != settings:
            raise ValueError("Existing immutable extended bank uses different arguments")
        return destination
    records = copy.deepcopy(base["scenes"])
    for record in records:
        source, target = base_manifest.parent / record["path"], output / record["path"]
        target.mkdir(parents=True, exist_ok=True)
        for name in ("sdf.npy", "bf.npy", "gf.npy", "obs.npy", "source.json", "scene.json"):
            _copy_immutable(source / name, target / name)
    for group_seed in range(seed, seed + groups):
        cached = sorted((output / "scenes").glob(f"table-contrast-{split}-{group_seed:06d}-*/scene-input.json"))
        if len(cached) == 2:
            scenes = [json.loads(path.read_text()) for path in cached]
            scenes.sort(key=lambda scene: ROLES.index(scene["hand_contrast"]["role"]))
        else:
            scenes = generate_table_edge_pair(group_seed, split=split)
        for scene in scenes:
            relative = Path("scenes") / scene["scene_id"]
            directory = output / relative
            directory.mkdir(parents=True, exist_ok=True)
            record_path = directory / "field-record.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if (record["fields"] != _field_records(directory)
                        or record["scene_sha256"] != sha256(directory / "scene.json")
                        or record["source"]["metadata_sha256"] != sha256(directory / "source.json")
                        or record["source"]["hand_contrast"] != scene["hand_contrast"]):
                    raise ValueError("Cached table scene bytes or metadata changed")
            else:
                record = make_clutter_fields(scene, directory, dx=.04)
                source = dict(record["source"], kind="contrastive-table-edge-passage", hand_contrast=scene["hand_contrast"])
                (directory / "source.json").write_text(json.dumps(source, indent=2) + "\n")
                source["metadata_sha256"] = sha256(directory / "source.json")
                record.update(path=str(relative), source=source, task_kind="room", reset_mode="room",
                    crossed_mode="goal_radius", episode_length=4000, sampling_group="generic_clutter")
                record_path.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            progress(f"Prepared table scene {len(records) - len(base['scenes'])}/{groups * 2}: {scene['scene_id']}")
    manifest = copy.deepcopy(base)
    manifest.pop("manifest_sha256")
    manifest.update(scenes=records, scene_count=len(records),
        fields_bytes=sum(f["size_bytes"] for scene in records for f in scene["fields"].values()))
    marker = manifest["contrastive_specialist"]
    marker.update(schema=TABLE_MARKER, group_count=base["contrastive_specialist"]["group_count"] + groups,
        geometry_group_counts=dict(cabinet=base["contrastive_specialist"]["group_count"], table_edges=groups),
        preserved_cabinet_scene_count=len(base["scenes"]),
        preserved_scene_records_sha256=_json_hash(base["scenes"]), table_extension=settings,
        split_unit="geometry family and matched seed group",
        purpose="Tall passages plus matched table-height hand-raising passages")
    validate_contrastive_manifest(manifest, path=destination, verify_files=True)
    manifest["manifest_sha256"] = _json_hash(manifest)
    temporary = output / "manifest.pending.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    load_generalist_manifest(temporary)
    temporary.replace(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "data/furniture/contrastive_hand_v2/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/furniture/contrastive_table_v4")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    args = parser.parse_args()
    print(build(**vars(args), progress=lambda value: print(value, flush=True)))
