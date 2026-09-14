"""Generic clutter with the dense furniture layout's route topology.

Furniture is a demo domain, not a semantic input to the policy. This generator
replaces complete objects with crates, low blocks, partitions and shelves.
Fields and physical collision geometry are rebuilt from the resulting shapes.
"""
from __future__ import annotations

import argparse
import copy
import math
import random

from cat_ppo.furniture.scenes import (_box, _digest, _horizontal_distance, _root_obstacles, _route_clearance,
    generate_scene, validate_scene, write_scene_bundle)


def generate_clutter_scene(seed=0, split="train", family="mixed", difficulty="dense"):
    source = generate_scene(seed, split, family, difficulty)
    scene = copy.deepcopy(source)
    rng = random.Random(int(_digest(["generic-clutter-v1", seed, split, family, difficulty]), 16))
    objects = {}
    boxes = []
    for box in source["boxes"]:
        identity = box.get("furniture_id")
        if identity is None:
            boxes.append(copy.deepcopy(box))
        else:
            objects.setdefault(identity, []).append(box)
    kinds = []
    for index, (identity, parts) in enumerate(sorted(objects.items())):
        # Enclose each oriented primitive's true world footprint. Axis-aligned
        # inputs preserve their footprint; rotated inputs use a conservative AABB.
        bounds = []
        for part in parts:
            c, s = abs(math.cos(part["yaw"])), abs(math.sin(part["yaw"]))
            hx, hy = part["half_size"][:2]
            bounds.append((part["center"], [c * hx + s * hy, s * hx + c * hy]))
        lower = [min(center[i] - half[i] for center, half in bounds) for i in range(2)]
        upper = [max(center[i] + half[i] for center, half in bounds) for i in range(2)]
        x, y = [(a + b) / 2 for a, b in zip(lower, upper)]
        hx, hy = [(b - a) / 2 for a, b in zip(lower, upper)]
        kind = ("crate", "low_block", "partition", "shelf")[index % 4]
        heights = {"crate": (.72, .96), "low_block": (.48, .65),
                   "partition": (1.05, 1.30), "shelf": (.72, .88)}
        height = rng.uniform(*heights[kind])
        common = dict(object_id=f"clutter_{index:02d}", object_type=kind,
                      shape_identity=f"{split}/generic/{index:02d}")
        if kind == "shelf":
            for level, z in enumerate((height * .42, height)):
                boxes.append(_box(f"generic_{index}_shelf_{level}", [x, y, z - .025],
                                  [hx, hy, .025], "shelf_edge", **common))
            for side in (-1, 1):
                boxes.append(_box(f"generic_{index}_support_{side}",
                    [x + side * (hx - .025), y, height / 2], [.025, hy, height / 2], "support", **common))
        else:
            boxes.append(_box(f"generic_{index}_{kind}", [x, y, height / 2],
                              [hx, hy, height / 2], kind, **common))
        kinds.append(kind)
    scene["boxes"] = boxes
    scene["geometry_hash"] = _digest(dict(boxes=boxes, room_dimensions=scene["room_dimensions"]))
    scene["scene_id"] = "generic-" + source["scene_id"].rsplit("-", 1)[0] + "-" + scene["geometry_hash"][:12]
    scene["generator"] = dict(source["generator"], name="generic-clutter-v1",
        description="Crates, low blocks, partitions and shelves on an alternating aisle topology",
        source_furniture_geometry_hash=source["geometry_hash"], semantic_labels_observed_by_policy=False,
        shape_profiles=[], split_holdout={"kind": "inherited-disjoint-footprint-profiles-and-layout-seeds",
                                          "real_object_identity_holdout": False})
    scene["counts"] = dict(tables=0, chairs=0, generic_objects=len(objects),
        primitive_boxes=len(boxes), bottlenecks=len(scene["bottlenecks"]),
        **{("shelves" if kind == "shelf" else kind + "s"): kinds.count(kind) for kind in sorted(set(kinds))})
    root_boxes = _root_obstacles(boxes)
    for case in scene["start_goals"]:
        for gate in case["bottlenecks"]:
            gate["required_behavior"] = "coordinate body posture through generic clutter"
            gate["width_m"] = 2 * min(_horizontal_distance(gate["center"], box) for box in root_boxes)
            gate["width_definition"] = "twice-center-clearance-in-root-height-band"
        admission = _route_clearance(case["route"], boxes)
        if not admission["root_route_validated"]:
            raise ValueError("Generic replacement blocked the admitted root route")
    scene["bottlenecks"] = copy.deepcopy(scene["start_goals"][scene["goal_index"]]["bottlenecks"])
    scene["feasibility"] = _route_clearance(scene["route"], boxes)
    scene["reset_clearance"]["center_clearance_m"] = min(
        _horizontal_distance(scene["start"][:2], box) for box in root_boxes)
    validate_scene(scene)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--family", default="mixed")
    parser.add_argument("--difficulty", default="dense")
    parser.add_argument("--voxel-size", type=float, default=.10)
    args = parser.parse_args()
    scene = generate_clutter_scene(args.seed, args.split, args.family, args.difficulty)
    write_scene_bundle(scene, args.output, args.voxel_size)
    print(scene["scene_id"], scene["counts"])


if __name__ == "__main__":
    main()
