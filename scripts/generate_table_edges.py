#!/usr/bin/env python3
"""Extend a bank with matched table-height passages (open / forward_protected pairs).

Walking-past-a-table-edge hand protection: the robot must raise its hands as it passes
a 70-72 cm table edge. Each group yields two matched scenes that differ only in gap
width. Same extension mechanism as generate_narrow_passages.py: the parent bank is
pinned and referenced in place, the new scenes are a hashed addition fragment.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cat_ppo.furniture.table_edge_passages import generate_table_edge_pair  # noqa: E402

EXTENDED_SCHEMA = "cat-extended-balance-v1"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--groups", type=int, default=32, help="Each group yields a matched open/protected pair")
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--split", default="train")
    p.add_argument("--dx", type=float, default=.04)
    p.add_argument("--base-manifest", type=Path,
                   default=ROOT / "data/furniture/procedural_rooms_v2_20260923/manifest.json",
                   help="Bank to extend; its scenes are referenced in place, never copied")
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output bank already exists; refusing to overwrite")
    from cat_ppo.furniture.generalist_fields import _json_hash, load_generalist_manifest, make_clutter_fields, sha256

    args.output.mkdir(parents=True)
    records = []
    for index in range(args.groups):
        group_seed = args.seed + index
        for scene in generate_table_edge_pair(group_seed, split=args.split):
            relative = Path("scenes") / scene["scene_id"]
            directory = args.output / relative
            directory.mkdir(parents=True, exist_ok=True)
            record = make_clutter_fields(scene, directory, dx=args.dx)
            # hand_contrast is kept: it is what makes the scene resolve inside this bank and carries
            # the zone metadata; the cabinet-only corridor gate skips geometry_family=table_edges.
            source = dict(record["source"], kind="table-edge-passage", hand_contrast=scene["hand_contrast"])
            (directory / "source.json").write_text(json.dumps(source, indent=2) + "\n")
            source["metadata_sha256"] = sha256(directory / "source.json")
            record.update(path=str(relative), source=source, task_kind="room", reset_mode="room",
                          crossed_mode="goal_radius", episode_length=4000, sampling_group="generic_clutter")
            (directory / "field-record.json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
        print(f"group {index + 1}/{args.groups}  scenes={len(records)}", flush=True)

    fragment = args.output / "addition.json"
    fragment.write_text(json.dumps(dict(kind="table_edge_passages", scene_count=len(records), scenes=records), indent=1) + "\n")
    base_path = args.base_manifest.resolve()
    base = json.loads(base_path.read_text())
    # Records carrying hand_contrast/flat_balance resolve against THIS bank's directory, so the
    # parent's such scenes must be reachable here: hard-link, never copy (see the narrow script).
    for record in base["scenes"]:
        source_meta = record.get("source", {})
        if not (source_meta.get("hand_contrast") or source_meta.get("flat_balance")):
            continue
        target = args.output / record["path"]
        if target.exists():
            continue
        source = (base_path.parent / record["path"]).resolve()
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            try:
                os.link(item, target / item.name)
            except OSError:
                shutil.copy2(item, target / item.name)
    combined = list(base["scenes"]) + records
    marker = dict(base["flat_balance"])
    marker.update(schema=EXTENDED_SCHEMA, parent=dict(manifest=str(base_path), sha256=sha256(base_path)),
                  additions=[dict(kind="table_edge_passages", manifest=str(fragment.resolve()), sha256=sha256(fragment), count=len(records))])
    manifest = dict(base, scene_count=len(combined), scenes=combined, flat_balance=marker,
                    fields_bytes=base.get("fields_bytes", 0) + sum(f["size_bytes"] for s in records for f in s["fields"].values()))
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _json_hash(manifest)
    temporary = args.output / "manifest.pending.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    load_generalist_manifest(temporary)
    temporary.replace(args.output / "manifest.json")
    print(json.dumps(dict(groups=args.groups, scenes=len(records), output=str(args.output)), indent=1))


if __name__ == "__main__":
    main()
