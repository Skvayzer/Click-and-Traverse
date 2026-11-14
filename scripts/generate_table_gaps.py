#!/usr/bin/env python3
"""Intermediate table-pair gaps for a width curriculum, as a packed extension of a packed bank.

The table-edge passages had only two widths: open (~1.5 m, nothing to protect) and forward_protected (0.43-0.44 m,
~4 cm body clearance with the hands raised, dynamic feasibility unvalidated). Nothing in between taught walking forward
between tables with the hands raised or tucked. This writes forward_protected scenes with gaps evenly spaced over
--gap-range (default 0.46-0.70 m: the hanging hands still hit the tables, a raised or tucked posture clears), certified
with the curriculum rule (cat_ppo/furniture/table_edge_passages.py). Fields are packed like generate_side_gaps.py
(float32 SDF, packed bf/gf, raw occupancy); records match generate_table_edges.py (room task, goal radius, kind
table-edge-passage, hand_contrast kept), so the table_* terms, raised resets and table metrics see them.

  generate_table_gaps.py --output data/furniture/tablegap_v1_packed --base-manifest data/furniture/sidegap_v2_packed/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXTENDED_SCHEMA = "cat-extended-balance-v1"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-manifest", type=Path, default=ROOT / "data/furniture/sidegap_v2_packed/manifest.json")
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--gap-range", type=float, nargs=2, default=(.46, .70))
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--split", default="train")
    p.add_argument("--dx", type=float, default=.04)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Output bank already exists; refusing to overwrite")
    from cat_ppo.furniture.generalist_fields import _field_records, _json_hash, load_generalist_manifest, make_clutter_fields, sha256
    from cat_ppo.furniture.table_edge_passages import generate_table_edge_pair
    from cat_mjlab.packing.field_packing import pack_direction

    args.output.mkdir(parents=True)
    records = []
    gaps = np.linspace(args.gap_range[0], args.gap_range[1], args.count)
    for index, gap in enumerate(gaps):
        try:
            pair = generate_table_edge_pair(args.seed + index, split=args.split, gap_range=(float(gap), float(gap)))
        except ValueError as error:
            print(f"skip gap {gap:.3f}: {error}", flush=True)
            continue
        scene = next(s for s in pair if s["hand_contrast"]["role"] == "forward_protected")
        relative = Path("scenes") / scene["scene_id"]
        directory = args.output / relative
        directory.mkdir(parents=True, exist_ok=True)
        record = make_clutter_fields(scene, directory, dx=args.dx)
        for name in ("bf", "gf"):
            codes, valid = pack_direction(np.load(directory / f"{name}.npy"))
            np.save(directory / f"{name}.npy", codes); np.save(directory / f"{name}_valid.npy", valid)
        if not np.array_equal(np.load(directory / "obs.npy").astype(bool), np.load(directory / "sdf.npy") < 0):
            raise ValueError("occupancy and SDF disagree")
        source = dict(record["source"], kind="table-edge-passage", hand_contrast=scene["hand_contrast"],
                      table_gap_curriculum=dict(gap_m=float(gap), range_m=[float(v) for v in args.gap_range]))
        (directory / "source.json").write_text(json.dumps(source, indent=2) + "\n")
        source["metadata_sha256"] = sha256(directory / "source.json")
        record.update(path=str(relative), source=source, task_kind="room", reset_mode="room", crossed_mode="goal_radius",
                      episode_length=4000, sampling_group="generic_clutter", fields=_field_records(directory))
        (directory / "field-record.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(f"{len(records)}/{args.count} gap {gap:.3f} body_min {scene['hand_contrast']['certificate']['body_min_separation_m']:.3f}", flush=True)

    fragment = args.output / "addition.json"
    fragment.write_text(json.dumps(dict(kind="table_gap_curriculum", scene_count=len(records), scenes=records), indent=1) + "\n")
    base_path = args.base_manifest.resolve()
    base = json.loads(base_path.read_text())
    # Parent scenes carrying hand_contrast / flat_balance resolve against THIS bank's directory: hard-link them here.
    for parent in base["scenes"]:
        meta = parent.get("source", {})
        if not (meta.get("hand_contrast") or meta.get("flat_balance")):
            continue
        target = args.output / parent["path"]
        if target.exists():
            continue
        origin_dir = (base_path.parent / parent["path"]).resolve()
        target.mkdir(parents=True, exist_ok=True)
        for item in origin_dir.iterdir():
            try:
                os.link(item, target / item.name)
            except OSError:
                shutil.copy2(item, target / item.name)
    combined = list(base["scenes"]) + records
    marker = dict(base["flat_balance"])
    marker.update(schema=EXTENDED_SCHEMA, parent=dict(manifest=str(base_path), sha256=sha256(base_path)),
                  additions=[dict(kind="table_gap_curriculum", manifest=str(fragment.resolve()), sha256=sha256(fragment), count=len(records))])
    manifest = dict(base, scene_count=len(combined), scenes=combined, flat_balance=marker,
                    fields_bytes=base.get("fields_bytes", 0) + sum(f["size_bytes"] for r in records for f in r["fields"].values()))
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _json_hash(manifest)
    temporary = args.output / "manifest.pending.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    load_generalist_manifest(temporary)
    temporary.replace(args.output / "manifest.json")
    print(json.dumps(dict(scenes=len(records), gaps_m=[round(float(gaps[0]), 3), round(float(gaps[-1]), 3)], output=str(args.output))))


if __name__ == "__main__":
    main()
