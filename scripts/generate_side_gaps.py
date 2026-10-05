#!/usr/bin/env python3
"""Procedural sideways-gap scenes (CAT-type): walls whose only opening needs the robot to turn sideways.

Why: only 7 of ~2300 CAT training scenes need a sideways passage, two of them unpassable without touching
under our rule (published-side1: 0.24 m opening, 0 cm spare for the 0.236 m sideways body). The CAT expert
learned that side gaps are failures and refuses even a 0.44 m gap (10 cm spare per side).

Layout (CAT runtime frame: start at the origin facing +x, goal (2.0, 0) crossed as an x-plane, grid
x -0.6..2.6, y -1.1..1.1, z 0..1.52): one wall across the full width at x 0.8-1.3 (depth 0.04-0.25 m) with a
gap of 0.32-0.60 m (spare per side 4-18 cm when sideways) at a lateral offset; 30% a second staggered wall
0.5-0.7 m later; 15% a low bar in the gap (0.05-0.12 m); 10% a beam over the gap (bottom 1.05-1.20 m).
Fields: the usual CAT pipeline (occupancy -> SDF -> 3-D FMM toward the goal). Stored like the procedural CAT
scenes (float32 SDF, packed bf/gf, raw occupancy), family procedural_cat, source kind 'side-gap'.
Output: <output> extending the base bank; then build the collision and reset banks with --base-*.
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
LENGTH, WIDTH, HEIGHT = 3.0, 2.0, 1.52            # room frame: x 0..3, y 0..2; runtime = room - (0.5, 1.0)
START = np.array([.5, 1.])


def scene_boxes(rng):
    boxes, info = [], {}

    def add(name, category, lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        boxes.append(dict(name=name, category=category, center=[round(v, 4) for v in (lo + hi) / 2],
                          half_size=[round(v, 4) for v in (hi - lo) / 2], yaw=0.))

    def wall(name, x, depth, gap, offset):
        y0, y1 = WIDTH / 2 + offset - gap / 2, WIDTH / 2 + offset + gap / 2
        add(name + "_a", "wall_side", (x, -.15, 0.), (x + depth, y0, HEIGHT))
        add(name + "_b", "wall_side", (x, y1, 0.), (x + depth, WIDTH + .15, HEIGHT))
        return y0, y1

    x = START[0] + rng.uniform(.8, 1.3); depth = rng.uniform(.04, .25)
    gap = rng.uniform(.32, .60); offset = rng.uniform(-.35, .35)
    y0, y1 = wall("w1", x, depth, gap, offset)
    info.update(gap_m=round(gap, 3), wall_depth_m=round(depth, 3), offset_m=round(offset, 3), walls=1)
    r = rng.uniform()
    if r < .15:
        h = rng.uniform(.05, .12)
        add("bar", "hurdle", (x, y0, 0.), (x + min(depth, .08), y1, h)); info["bar_m"] = round(h, 3)
    elif r < .25:
        b = rng.uniform(1.05, 1.20)
        add("beam", "beam", (x - .05, y0, b), (x + depth + .05, y1, HEIGHT)); info["beam_bottom_m"] = round(b, 3)
    if rng.uniform() < .30 and x + depth + .5 < START[0] + 1.6:
        x2 = x + depth + rng.uniform(.5, .7); depth2 = rng.uniform(.04, .15)
        if x2 + depth2 < START[0] + 1.8:
            gap2 = rng.uniform(.34, .60); offset2 = float(np.clip(-offset + rng.uniform(-.15, .15), -.35, .35))
            wall("w2", x2, depth2, gap2, offset2); info.update(walls=2, gap2_m=round(gap2, 3), offset2_m=round(offset2, 3))
    return boxes, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=ROOT / "data/furniture/sidegap_v1_packed")
    p.add_argument("--base-manifest", type=Path, default=ROOT / "data/furniture/table_edges_v1_packed/manifest.json")
    p.add_argument("--count", type=int, default=160)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--dx", type=float, default=.04)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Output bank already exists; refusing to overwrite")
    from cat_ppo.furniture.generalist_fields import _field_records, _json_hash, load_generalist_manifest, make_clutter_fields, sha256
    from cat_ppo.furniture.scenes import SCHEMA, _digest
    from cat_mjlab.packing.field_packing import pack_direction

    args.output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    records = []
    shift = np.array([START[0], START[1], 0.])
    for n in range(args.count):
        boxes, info = scene_boxes(rng)
        dims = [LENGTH, WIDTH, HEIGHT]
        start, goal = [float(START[0]), float(START[1]), 0.], [float(START[0] + 2.), float(START[1])]
        scene_id = f"sidegap-{n:04d}-w{info['gap_m']:.2f}"
        scene = dict(schema=SCHEMA, scene_id=scene_id, family="furniture", split="train", units="metres",
                     coordinate_system="right-handed-z-up", boxes=boxes, room_dimensions=dims,
                     geometry_hash=_digest(dict(boxes=boxes, room_dimensions=dims)), start=start, goal=goal,
                     route=[start[:2], goal], bottlenecks=[], time_budget=20.,
                     counts=dict(tables=0, chairs=0, generic_objects=0, primitive_boxes=len(boxes), bottlenecks=0),
                     feasibility=dict(method="procedural side gap; CAT-type scene, no route certificate", root_route_validated=False))
        relative = Path("scenes") / scene_id
        directory = args.output / relative
        try:
            record = make_clutter_fields(scene, directory, dx=args.dx)
        except ValueError as error:                 # e.g. disconnected start/goal
            print("skip", scene_id, error, flush=True); shutil.rmtree(directory, ignore_errors=True); continue
        for name in ("bf", "gf"):
            codes, valid = pack_direction(np.load(directory / f"{name}.npy"))
            np.save(directory / f"{name}.npy", codes); np.save(directory / f"{name}_valid.npy", valid)
        if not np.array_equal(np.load(directory / "obs.npy").astype(bool), np.load(directory / "sdf.npy") < 0):
            raise ValueError("occupancy and SDF disagree")
        spec = json.loads((directory / "scene.json").read_text())
        for box in spec["boxes"]:
            box["center"] = [round(c - s, 6) for c, s in zip(box["center"], shift)]
        spec["start"] = [0., 0., 0.]; spec["goal"] = [2., 0.]; spec["route"] = [[0., 0.], [2., 0.]]
        spec["runtime_shift_m"] = (-shift).tolist(); spec["side_gap"] = info
        (directory / "scene.json").write_text(json.dumps(spec, indent=2) + "\n")
        source = dict(record["source"], kind="side-gap", flat_balance=True, side_gap=info,
                      occupancy_sha256=sha256(directory / "obs.npy"))
        (directory / "source.json").write_text(json.dumps(source, indent=2) + "\n")
        source["metadata_sha256"] = sha256(directory / "source.json")
        record.update(path=str(relative), family="procedural_cat", sampling_group="procedural_cat", task_kind="cat",
                      reset_mode="cat", crossed_mode="x_plane", episode_length=1000, sampling_weight=1.,
                      origin=(np.asarray(record["origin"]) - shift).tolist(), start=[0., 0., .8], goal=[2., 0., .75],
                      reset_xy_scale=[1., 1.], reset_yaw=0., source=source, scene_sha256=sha256(directory / "scene.json"),
                      fields=_field_records(directory))
        (directory / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        if len(records) % 20 == 0:
            print(f"{len(records)} scenes", flush=True)

    fragment = args.output / "addition.json"
    fragment.write_text(json.dumps(dict(kind="side_gaps", scene_count=len(records), scenes=records), indent=1) + "\n")
    base_path = args.base_manifest.resolve()
    base = json.loads(base_path.read_text())
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
                  additions=[dict(kind="side_gaps", manifest=str(fragment.resolve()), sha256=sha256(fragment), count=len(records))])
    manifest = dict(base, scene_count=len(combined), scenes=combined, flat_balance=marker,
                    fields_bytes=base.get("fields_bytes", 0) + sum(f["size_bytes"] for r in records for f in r["fields"].values()))
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _json_hash(manifest)
    temporary = args.output / "manifest.pending.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    load_generalist_manifest(temporary)
    temporary.replace(args.output / "manifest.json")
    gaps = [r["source"]["side_gap"]["gap_m"] for r in records]
    print(json.dumps(dict(scenes=len(records), gap_m=[round(min(gaps), 2), round(float(np.median(gaps)), 2), round(max(gaps), 2)],
                          two_walls=sum(r["source"]["side_gap"]["walls"] == 2 for r in records),
                          bars=sum("bar_m" in r["source"]["side_gap"] for r in records),
                          beams=sum("beam_bottom_m" in r["source"]["side_gap"] for r in records), output=str(args.output))))


if __name__ == "__main__":
    main()
