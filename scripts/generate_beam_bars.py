#!/usr/bin/env python3
"""Procedural beam and bar scenes (CAT-type) graded by difficulty, for the height curriculum (task 'height_curriculum').

Why: on the 25 original CAT courses with an overhead beam, the planner-reflex policy (COMBO2) refused 50-62% and passed
cleanly 0-6%; without plans the same network passed 8% and touched 82%; the low-bar courses were no better (0% clean).
No policy in the line had learned to duck under a beam or step over a bar, so the planner's training data held few
such passes either (15% of the windows in beam courses include a crouch). A curriculum of graded heights, as the table
width curriculum did for table gaps (forward passage 37% -> 95%), makes them learnable with strict contact termination
(the training simulator has no obstacle contact forces, so a soft touch would teach passing through a bar).

Layout (CAT runtime frame: start at the origin facing +x, goal (2.0, 0) crossed as an x-plane, grid x -0.6..2.6,
y -1.1..1.1, z 0..1.52), every obstacle across the full width so it cannot be walked around:
    beam      an overhead beam, its underside at BEAM_BOTTOM[level], 0.06-0.25 m deep, at x 0.9-1.3 m
    bar       a low bar BAR_TOP[level] high, 0.04-0.10 m deep, at x 0.8-1.2 m
    beam_bar  a bar and, 0.5-0.8 m later, a beam, both at the level's heights
Levels 0..7 run from easy (a beam at 1.40 m needs no crouch; a 3 cm bar) to the measured CAT range (beam undersides
0.84-1.24 m, median 1.12 m; bars 0.06-0.18 m).
Fields: the usual CAT pipeline (occupancy -> SDF -> 3-D FMM toward the goal), stored like the side-gap scenes (family
procedural_cat, source kind 'beam-bar' with the level and heights). Output: <output> extending the base bank; then
build the collision and reset banks with --base-* and the goal-seed cache with --reuse.
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
BEAM_BOTTOM = (1.40, 1.32, 1.25, 1.18, 1.11, 1.04, 0.97, 0.90)     # m, underside of the beam per level
BAR_TOP = (0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.16, 0.19)          # m, top of the bar per level
VARIANTS = (("beam", .4), ("bar", .4), ("beam_bar", .2))


def scene_boxes(rng, level, variant):
    boxes, info = [], dict(variant=variant, level=int(level))

    def add(name, category, x, depth, z0, z1):
        boxes.append(dict(name=name, category=category, center=[round(float(x + depth / 2), 4), round(WIDTH / 2, 4),
                                                                 round(float((z0 + z1) / 2), 4)],
                          half_size=[round(float(depth / 2), 4), round(WIDTH / 2 + .1, 4), round(float((z1 - z0) / 2), 4)],
                          yaw=0.))

    if variant in ("bar", "beam_bar"):
        x = START[0] + rng.uniform(.8, 1.2) - (.25 if variant == "beam_bar" else 0.); depth = rng.uniform(.04, .10)
        top = BAR_TOP[level]
        add("bar", "hurdle", x, depth, 0., top); info.update(bar_m=round(top, 3), bar_x_m=round(float(x - START[0]), 3),
                                                             bar_depth_m=round(float(depth), 3))
        if variant == "beam_bar":
            x = x + depth + rng.uniform(.5, .8)
    if variant in ("beam", "beam_bar"):
        if variant == "beam":
            x = START[0] + rng.uniform(.9, 1.3)
        depth = rng.uniform(.06, .25); bottom = BEAM_BOTTOM[level]
        add("beam", "beam", x, depth, bottom, HEIGHT)
        info.update(beam_bottom_m=round(bottom, 3), beam_x_m=round(float(x - START[0]), 3), beam_depth_m=round(float(depth), 3))
    return boxes, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-manifest", type=Path, required=True)
    p.add_argument("--per-level", type=int, default=75, help="scenes per level (8 levels)")
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--prefix", default="beambar")
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
    names, probs = zip(*VARIANTS)
    n = 0
    for level in range(len(BEAM_BOTTOM)):
        for _ in range(args.per_level):
            variant = str(rng.choice(names, p=probs))
            boxes, info = scene_boxes(rng, level, variant)
            dims = [LENGTH, WIDTH, HEIGHT]
            start, goal = [float(START[0]), float(START[1]), 0.], [float(START[0] + 2.), float(START[1])]
            scene_id = f"{args.prefix}-{n:04d}-{variant}-L{level}"; n += 1
            scene = dict(schema=SCHEMA, scene_id=scene_id, family="furniture", split="train", units="metres",
                         coordinate_system="right-handed-z-up", boxes=boxes, room_dimensions=dims,
                         geometry_hash=_digest(dict(boxes=boxes, room_dimensions=dims)), start=start, goal=goal,
                         route=[start[:2], goal], bottlenecks=[], time_budget=20.,
                         counts=dict(tables=0, chairs=0, generic_objects=0, primitive_boxes=len(boxes), bottlenecks=0),
                         feasibility=dict(method="procedural beam/bar; CAT-type scene, no route certificate", root_route_validated=False))
            relative = Path("scenes") / scene_id
            directory = args.output / relative
            try:
                record = make_clutter_fields(scene, directory, dx=args.dx)
            except ValueError as error:             # e.g. a bar that disconnects start and goal in the field solver
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
            spec["runtime_shift_m"] = (-shift).tolist(); spec["beam_bar"] = info
            (directory / "scene.json").write_text(json.dumps(spec, indent=2) + "\n")
            source = dict(record["source"], kind="beam-bar", flat_balance=True, beam_bar=info,
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
            if len(records) % 50 == 0:
                print(f"{len(records)} scenes", flush=True)

    fragment = args.output / "addition.json"
    fragment.write_text(json.dumps(dict(kind="beam_bars", scene_count=len(records), scenes=records), indent=1) + "\n")
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
    additions = list(marker.get("additions", [])) + [dict(kind="beam_bars", manifest=str(fragment.resolve()),
                                                          sha256=sha256(fragment), count=len(records))]
    marker.update(schema=EXTENDED_SCHEMA, parent=dict(manifest=str(base_path), sha256=sha256(base_path)), additions=additions)
    manifest = dict(base, scene_count=len(combined), scenes=combined, flat_balance=marker,
                    fields_bytes=base.get("fields_bytes", 0) + sum(f["size_bytes"] for r in records for f in r["fields"].values()))
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _json_hash(manifest)
    temporary = args.output / "manifest.pending.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    load_generalist_manifest(temporary)
    temporary.replace(args.output / "manifest.json")
    from collections import Counter
    print(json.dumps(dict(scenes=len(records), variants=Counter(r["source"]["beam_bar"]["variant"] for r in records),
                          levels=Counter(r["source"]["beam_bar"]["level"] for r in records), output=str(args.output))))


if __name__ == "__main__":
    main()
