#!/usr/bin/env python3
"""Build the keyboard-demo obstacle course: every CAT obstacle type in one long walled corridor.

Sections, 2 m apart along +x, sized from the published CAT scenes (measured from their SDFs):
low hurdle, high hurdle, high beam, low beam, side gap 0.44 m, side gap 0.34 m, double hurdle,
hurdle under a beam, side gap with a hurdle in it, and two tables with a 0.60 m gap.

The course is a CAT-type scene (task_kind "cat"), stored exactly like the procedural CAT scenes
(float32 SDF, packed bf/gf, raw occupancy): the CAT expert learned keyboard control in CAT scenes,
and room scenes would need a certified 0.23 m root route, i.e. no side gap under 0.46 m. Fields are
the usual CAT pipeline (occupancy -> SDF -> 3-D FMM guidance toward the course end); the runtime
coordinates put the start at the origin like every CAT scene. sampling_weight 0: never trained on.

Output: <bank>_packed, extending table_edges_v1_packed (parent scenes referenced, not copied).
Then build the collision and reset banks with --base-* (commands printed at the end).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXTENDED_SCHEMA = "cat-extended-balance-v1"
WIDTH, HEIGHT = 2.0, 1.6                     # corridor (CAT scenes: ~1.9 m wide, 1.5 m tall)
START_X = 1.5


def course_boxes():
    """(name, category, x, y, z centres and half sizes) in room coordinates; corridor y in [0, WIDTH]."""
    mid, boxes = WIDTH / 2, []

    def add(name, category, center, half):
        boxes.append(dict(name=name, category=category, center=[round(v, 4) for v in center],
                          half_size=[round(v, 4) for v in half], yaw=0.))

    def hurdle(name, x, height, depth=.10):
        add(name, "hurdle", (x, mid, height / 2), (depth / 2, mid, height / 2))

    def beam(name, x, bottom, depth=.16):
        add(name, "beam", (x, mid, (bottom + HEIGHT) / 2), (depth / 2, mid, (HEIGHT - bottom) / 2))

    def side_gap(name, x, gap, depth=.08):
        side = (WIDTH - gap) / 2
        add(name + "_left", "wall_side", (x, side / 2, HEIGHT / 2), (depth / 2, side / 2, HEIGHT / 2))
        add(name + "_right", "wall_side", (x, WIDTH - side / 2, HEIGHT / 2), (depth / 2, side / 2, HEIGHT / 2))

    def table(name, x, y0, y1, top=.75, depth=.8):
        add(name + "_top", "table_top", (x, (y0 + y1) / 2, top - .02), (depth / 2, (y1 - y0) / 2, .02))
        for i, (dx, y) in enumerate(((-1, y0), (-1, y1), (1, y0), (1, y1))):
            add(f"{name}_leg{i}", "table_leg", (x + dx * (depth / 2 - .04), y + (.04 if y == y0 else -.04), (top - .04) / 2),
                (.025, .025, (top - .04) / 2))

    # perimeter (as in the room generator: 6 cm half-thick walls centred on the boundary)
    length = 25.0
    add("wall_south", "wall", (length / 2, 0., HEIGHT / 2), (length / 2, .06, HEIGHT / 2))
    add("wall_north", "wall", (length / 2, WIDTH, HEIGHT / 2), (length / 2, .06, HEIGHT / 2))
    add("wall_west", "wall", (0., mid, HEIGHT / 2), (.06, mid, HEIGHT / 2))
    add("wall_east", "wall", (length, mid, HEIGHT / 2), (.06, mid, HEIGHT / 2))
    hurdle("s1_hurdle_low", 3.0, .10)
    hurdle("s2_hurdle_high", 5.0, .20, depth=.12)
    beam("s3_beam_high", 7.0, 1.16)
    beam("s4_beam_low", 9.0, 1.02)
    side_gap("s5_gap44", 11.0, .44)
    side_gap("s6_gap34", 13.0, .34)
    hurdle("s7_double_a", 14.7, .08, depth=.12)
    hurdle("s7_double_b", 15.3, .08, depth=.12)
    hurdle("s8_hurdle_under", 17.0, .05, depth=.08)
    beam("s8_beam_over", 17.0, 1.16)
    side_gap("s9_gap_hurdle", 19.0, .44)
    add("s9_gap_hurdle_bar", "hurdle", (19.0, mid, .05), (.04, .22, .05))
    table("s10_table_a", 21.2, .06, mid - .30)
    table("s10_table_b", 21.2, mid + .30, WIDTH - .06)
    return boxes, length


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=ROOT / "data/furniture/course_v1_packed")
    p.add_argument("--base-manifest", type=Path, default=ROOT / "data/furniture/table_edges_v1_packed/manifest.json")
    p.add_argument("--dx", type=float, default=.04)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Output bank already exists; refusing to overwrite")
    from cat_ppo.furniture.generalist_fields import _field_records, _json_hash, load_generalist_manifest, make_clutter_fields, sha256
    from cat_ppo.furniture.scenes import SCHEMA, _digest
    from cat_mjlab.packing.field_packing import pack_direction

    boxes, length = course_boxes()
    dims = [length, WIDTH, HEIGHT]
    start, goal = [START_X, WIDTH / 2, 0.], [length - 1.0, WIDTH / 2]
    scene_id = "course-mixed-obstacles-v1"
    scene = dict(schema=SCHEMA, scene_id=scene_id, family="furniture", split="demo", units="metres",
                 coordinate_system="right-handed-z-up", boxes=boxes, room_dimensions=dims,
                 geometry_hash=_digest(dict(boxes=boxes, room_dimensions=dims)),
                 start=start, goal=goal, route=[start[:2], goal], bottlenecks=[], time_budget=120.,
                 counts=dict(tables=2, chairs=0, generic_objects=0, primitive_boxes=len(boxes), bottlenecks=0),
                 feasibility=dict(method="hand-built keyboard-demo course; no route certificate (CAT-type scene)",
                                  root_route_validated=False))
    args.output.mkdir(parents=True)
    relative = Path("scenes") / scene_id
    directory = args.output / relative
    record = make_clutter_fields(scene, directory, dx=args.dx)
    # Pack directions like the procedural CAT scenes (sdf stays float32, occupancy stays raw uint8).
    for name in ("bf", "gf"):
        codes, valid = pack_direction(np.load(directory / f"{name}.npy"))
        np.save(directory / f"{name}.npy", codes); np.save(directory / f"{name}_valid.npy", valid)
    occupancy = np.load(directory / "obs.npy")
    if not np.array_equal(occupancy.astype(bool), np.load(directory / "sdf.npy") < 0):
        raise ValueError("occupancy and SDF disagree")
    # CAT runtime coordinates: start at the origin (the reset adds its offsets around it).
    shift = np.asarray([START_X, WIDTH / 2, 0.])
    spec = json.loads((directory / "scene.json").read_text())
    for box in spec["boxes"]:
        box["center"] = [round(c - s, 6) for c, s in zip(box["center"], shift)]
    spec["start"] = [0., 0., 0.]; spec["goal"] = [goal[0] - shift[0], goal[1] - shift[1]]
    spec["route"] = [spec["start"][:2], spec["goal"]]
    spec["runtime_shift_m"] = (-shift).tolist()
    (directory / "scene.json").write_text(json.dumps(spec, indent=2) + "\n")
    source = dict(record["source"], kind="keyboard-demo-course", flat_balance=True, physical_obstacles_in_training=False,
                  occupancy_sha256=sha256(directory / "obs.npy"),
                  note="hand-built demo course; CAT-type scene, never sampled for training")
    (directory / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    source["metadata_sha256"] = sha256(directory / "source.json")
    origin = (np.asarray(record["origin"]) - shift).tolist()
    record.update(path=str(relative), family="procedural_cat", sampling_group="procedural_cat", task_kind="cat",
                  reset_mode="cat", crossed_mode="x_plane", episode_length=1000, sampling_weight=0.,
                  origin=origin, start=[0., 0., .8], goal=[goal[0] - shift[0], 0., .75], reset_xy_scale=[1., 1.], reset_yaw=0.,
                  source=source, scene_sha256=sha256(directory / "scene.json"), fields=_field_records(directory))
    (directory / "record.json").write_text(json.dumps(record, indent=2) + "\n")

    fragment = args.output / "addition.json"
    fragment.write_text(json.dumps(dict(kind="demo_course", scene_count=1, scenes=[record]), indent=1) + "\n")
    base_path = args.base_manifest.resolve()
    base = json.loads(base_path.read_text())
    for parent in base["scenes"]:          # scenes resolving inside the bank directory: hard-link, never copy
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
    combined = list(base["scenes"]) + [record]
    marker = dict(base["flat_balance"])
    marker.update(schema=EXTENDED_SCHEMA, parent=dict(manifest=str(base_path), sha256=sha256(base_path)),
                  additions=[dict(kind="demo_course", manifest=str(fragment.resolve()), sha256=sha256(fragment), count=1)])
    manifest = dict(base, scene_count=len(combined), scenes=combined, flat_balance=marker,
                    fields_bytes=base.get("fields_bytes", 0) + sum(f["size_bytes"] for f in record["fields"].values()))
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _json_hash(manifest)
    temporary = args.output / "manifest.pending.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    load_generalist_manifest(temporary)
    temporary.replace(args.output / "manifest.json")
    prefix = str(args.output)[:-len("_packed")] if str(args.output).endswith("_packed") else str(args.output)
    print(json.dumps(dict(scene=scene_id, shape=record["shape"], boxes=len(boxes), output=str(args.output)), indent=1))
    print(f"next:\n  .venv-mjlab/bin/python scripts/build_body_collision_bank.py --field-manifest {args.output}/manifest.json "
          f"--base-collision-manifest data/furniture/table_edges_v1_collision/manifest.json --output {prefix}_collision\n"
          f"  .venv-mjlab/bin/python scripts/build_body_collision_resets.py --cpu-native --field-manifest {args.output}/manifest.json "
          f"--collision-bank {prefix}_collision/manifest.json --base-reset-manifest data/furniture/table_edges_v1_resets/manifest.json "
          f"--output {prefix}_resets")


if __name__ == "__main__":
    main()
