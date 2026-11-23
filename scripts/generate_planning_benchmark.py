#!/usr/bin/env python3
"""Append held-out U traps, turn-before-entry gaps and certified table passages."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cat_mjlab.packing.field_packing import pack_direction
from cat_ppo.furniture.generalist_fields import (
    _field_records, _json_hash, load_generalist_manifest, make_clutter_fields, scene_directory, sha256,
)
from cat_ppo.furniture.room_geometry import root_cylinder_segment_clearance
from cat_ppo.furniture.scenes import SCHEMA, _box, _digest
from cat_ppo.furniture.table_edge_passages import generate_table_edge_pair

FAMILIES = ("u-trap", "early-turn", "table-gap")
CASES_PER_FAMILY = 24


def wall_scene(family, index, seed):
    rng = np.random.default_rng(seed + index)
    length, width, height = 4.1, 2.8, 1.68
    mid = width / 2
    sign = 1 if index % 2 else -1
    boxes = []

    def wall(name, x0, x1, y0, y1):
        boxes.append(_box(name, [(x0 + x1) / 2, (y0 + y1) / 2, height / 2],
                          [(x1 - x0) / 2, (y1 - y0) / 2, height / 2], "wall"))

    wall("left_boundary", 0., length, -.08, 0.)
    wall("right_boundary", 0., length, width, width + .08)
    wall("back_boundary", -.08, 0., 0., width)
    wall("far_boundary", length, length + .08, 0., width)
    start, goal = [.48, mid, 0.], [3.62, mid]
    if family == "u-trap":
        mouth = rng.uniform(.94, 1.19)
        closed = rng.uniform(2.03, 2.37)
        half = rng.uniform(.47, .63)
        shift = sign * rng.uniform(.025, .10)
        low, high = mid + shift - half, mid + shift + half
        wall("near_cheek", mouth, closed, low - .09, low)
        wall("far_cheek", mouth, closed, high, high + .09)
        wall("dead_end", closed, closed + .10, low - .09, high + .09)
        outside = high + .40 if sign > 0 else low - .40
        route = [start[:2], [.64, outside], [closed + .43, outside], goal]
        details = {"mouth_x_m": mouth, "dead_end_x_m": closed, "pocket_width_m": 2 * half}
        radius = .25
    else:
        depth = rng.uniform(.71, .94)
        entry = rng.uniform(1.55, 1.78)
        gap = rng.uniform(.36, .43)
        offset = sign * rng.uniform(.47, .57)
        y = mid + offset
        low, high = y - gap / 2, y + gap / 2
        wall("lower_channel_wall", entry, entry + depth, 0., low)
        wall("upper_channel_wall", entry, entry + depth, high, width)
        # The approach cheek ends at the displaced lane, where a straight run-up is blocked.
        approach = rng.uniform(.52, .66)
        if sign > 0:
            wall("approach_cheek", entry - approach, entry, 0., low)
        else:
            wall("approach_cheek", entry - approach, entry, high, width)
        route = [start[:2], [.76, y], [entry + depth + .31, y], goal]
        details = {"gap_m": gap, "lane_offset_m": offset, "channel_depth_m": depth, "approach_cheek_m": approach}
        radius = .15

    points = np.asarray(route)
    clearance = float(root_cylinder_segment_clearance(points[:-1], points[1:], boxes, radius=radius).min())
    if clearance <= 0:
        raise ValueError(f"{family} {index} has no certified root route ({clearance:.3f} m)")
    dimensions = [length, width, height]
    digest = _digest(dict(boxes=boxes, room_dimensions=dimensions))
    return dict(
        schema=SCHEMA, scene_id=f"planning-{family}-{index:02d}-{digest[:10]}", family="generic_clutter",
        split="test", units="metres", coordinate_system="right-handed-z-up", boxes=boxes,
        room_dimensions=dimensions, geometry_hash=digest, start=start, goal=goal, route=route,
        time_budget=80., bottlenecks=[],
        counts=dict(tables=0, chairs=0, generic_objects=len(boxes), primitive_boxes=len(boxes), bottlenecks=0),
        feasibility=dict(method="swept root cylinder on authored route", root_route_validated=True,
                         root_min_clearance_m=clearance, root_radius_m=radius),
        planning_benchmark=dict(family=family, index=index, **details),
    )


def write_case(job):
    family, index, seed, output = job
    if family == "table-gap":
        gap = float(np.linspace(.46, .62, CASES_PER_FAMILY)[index])
        pair = generate_table_edge_pair(seed + 1000 + index, split="test", gap_range=(gap, gap))
        scene = next(s for s in pair if s["hand_contrast"]["role"] == "forward_protected")
        scene["scene_id"] = f"planning-table-gap-{index:02d}-{scene['geometry_hash'][:10]}"
        scene["planning_benchmark"] = dict(family=family, index=index, gap_m=gap)
    else:
        scene = wall_scene(family, index, seed)
    relative = Path("scenes") / scene["scene_id"]
    directory = output / relative
    record = make_clutter_fields(scene, directory)
    for name in ("bf", "gf"):
        codes, valid = pack_direction(np.load(directory / f"{name}.npy"))
        np.save(directory / f"{name}.npy", codes)
        np.save(directory / f"{name}_valid.npy", valid)
    if family != "table-gap":
        shift = np.array([*scene["start"][:2], 0.])
        spec = json.loads((directory / "scene.json").read_text())
        for box in spec["boxes"]:
            box["center"] = (np.asarray(box["center"]) - shift).tolist()
        spec["start"] = [0., 0., 0.]
        spec["goal"] = (np.asarray(spec["goal"]) - shift[:2]).tolist()
        spec["route"] = (np.asarray(spec["route"]) - shift[:2]).tolist()
        spec["runtime_shift_m"] = (-shift).tolist()
        (directory / "scene.json").write_text(json.dumps(spec, indent=2) + "\n")
        record.update(family="procedural_cat", task_kind="cat", reset_mode="cat", crossed_mode="x_plane",
                      sampling_group="procedural_cat", episode_length=1000, reset_xy_scale=[1., 1.],
                      start=[0., 0., .8], goal=(np.asarray(record["goal"]) - shift).tolist(),
                      origin=(np.asarray(record["origin"]) - shift).tolist())
    else:
        record.update(task_kind="room", reset_mode="room", crossed_mode="goal_radius", episode_length=4000,
                      sampling_group="generic_clutter")
    source = dict(record["source"], kind="planning-benchmark", planning_benchmark=scene["planning_benchmark"])
    if "hand_contrast" in scene:
        source["hand_contrast"] = scene["hand_contrast"]
    else:
        source["flat_balance"] = "planning-benchmark"
    (directory / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    source["metadata_sha256"] = sha256(directory / "source.json")
    record.update(path=str(relative), source=source, fields=_field_records(directory),
                  scene_sha256=sha256(directory / "scene.json"))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    base_path = args.base_manifest.resolve()
    output = args.output.resolve()
    base = load_generalist_manifest(base_path, verify_files=False)
    output.mkdir(parents=True)
    jobs = [(family, i, args.seed, output) for family in FAMILIES for i in range(CASES_PER_FAMILY)]
    records = []
    with ProcessPoolExecutor(args.workers) as workers:
        for record in workers.map(write_case, jobs):
            records.append(record)
            print(record["scene_id"], flush=True)
    for record in base["scenes"]:
        source = record["source"]
        if source.get("hand_contrast") or source.get("flat_balance"):
            parent = scene_directory(base, base_path, record)
            destination = output / record["path"]
            destination.mkdir(parents=True)
            for file in parent.iterdir():
                os.link(file, destination / file.name)
    fragment = output / "addition.json"
    fragment.write_text(json.dumps(dict(kind="planning-benchmark", scenes=records), indent=1) + "\n")
    marker = dict(base["flat_balance"], schema="cat-extended-balance-v1",
                  parent=dict(manifest=str(base_path), sha256=sha256(base_path)),
                  additions=[dict(kind="planning-benchmark", manifest=str(fragment),
                                  sha256=sha256(fragment), count=len(records))])
    manifest = dict(base, scenes=base["scenes"] + records, scene_count=base["scene_count"] + len(records),
                    flat_balance=marker, fields_bytes=base["fields_bytes"] +
                    sum(file["size_bytes"] for record in records for file in record["fields"].values()))
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = _json_hash(manifest)
    pending = output / "manifest.pending.json"
    pending.write_text(json.dumps(manifest, indent=1) + "\n")
    load_generalist_manifest(pending, verify_files=False)
    pending.replace(output / "manifest.json")
    (output / "scene_ids.txt").write_text("\n".join(r["scene_id"] for r in records) + "\n")


if __name__ == "__main__":
    main()
