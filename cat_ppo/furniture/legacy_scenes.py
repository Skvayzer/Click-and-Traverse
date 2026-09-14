"""Adapt original CAT procedural occupancy into explicit physical cuboids.

The original occupied voxel union is preserved exactly at its source .04m
resolution, then translated into the positive-coordinate furniture scene schema.
Perimeter walls and physical whole-body collision are new. Fields are regenerated
with the upstream three-dimensional HumanoidPF algorithm on the selected grid;
this is an adapted task, not a bit-identical replay of released CAT training.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from cat_ppo.furniture.scenes import SCHEMA, SPLITS, _digest, _walls, validate_scene

UPSTREAM_COMMIT = "866ba392f1c1e84b92ad75fa66550f26e8af8e48"
GUIDANCE_METHOD = "cat-upstream-progressive-v1"
SOURCE_ROOT = Path(__file__).resolve().parents[2] / "procedural_obstacle_generation"
TYPICAL_SCENES = ("forward", "hurdle0", "hurdle1", "hurdle2", "hurdle3", "multi-hurdle0",
    "crouch0", "crouch1", "hurdle-crouch0", "hurdle-crouch1", "side0", "side1", "side2", "side3", "side4",
    "side-crouch0", "side-crouch1", "side-crouch2", "side-hurdle0", "side-hurdle1", "side-hurdle2",
    "side-hurdle3", "side-hurdle4", "side-hurdle-crouch0", "side-hurdle-crouch1", "side-hurdle-crouch2", "side-hurdle-crouch3")


def _upstream(name):
    # Original scripts use sibling absolute imports. Scope the search path change
    # to import only; never change cwd or rewrite algorithm source.
    sys.path.insert(0, str(SOURCE_ROOT))
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.remove(str(SOURCE_ROOT))
    if Path(module.__file__).resolve() != (SOURCE_ROOT / f"{name}.py").resolve():
        raise ImportError(f"Unexpected module shadows original CAT {name}")
    return module


def _source_hashes():
    names = ("typical_obstacle.py", "random_obstacle.py", "grid_config.py", "pf_grid_config.yaml", "pf_modular.py")
    return {name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest() for name in names}


def merge_occupied_voxels(occupancy):
    """Lossless, non-overlapping greedy cuboids [inclusive start, exclusive end]."""
    remaining = np.asarray(occupancy, dtype=bool).copy()
    if remaining.ndim != 3:
        raise ValueError("occupancy must be a 3-D xyz array")
    cuboids = []
    nx, ny, nz = remaining.shape
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if not remaining[x, y, z]:
                    continue
                x1, y1, z1 = x + 1, y + 1, z + 1
                while x1 < nx and remaining[x1, y, z]:
                    x1 += 1
                while y1 < ny and remaining[x:x1, y1, z].all():
                    y1 += 1
                while z1 < nz and remaining[x:x1, y:y1, z1].all():
                    z1 += 1
                remaining[x:x1, y:y1, z:z1] = False
                cuboids.append(((x, y, z), (x1, y1, z1)))
    return cuboids


def generate_legacy_scene(kind="typical", scene_type="forward", seed=0, split="train",
                          difficulty=.5, n_ground=1, n_lateral=1, n_overhead=1):
    """Call the original generators; no simulator, plot, checkpoint or GPU work."""
    if kind not in ("typical", "random") or split not in SPLITS:
        raise ValueError("kind must be typical/random and split train/validation/test")
    if type(seed) is not int or seed < 0 or not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError("seed must be nonnegative integer and difficulty in [0,1]")
    if any(type(n) is not int or n < 0 for n in (n_ground, n_lateral, n_overhead)):
        raise ValueError("obstacle counts must be nonnegative integers")
    grid = _upstream("grid_config").load_grid_config()
    source_seed = seed + {"train": 0, "validation": 100000000, "test": 200000000}[split]
    if kind == "typical":
        if scene_type not in TYPICAL_SCENES:
            raise ValueError(f"Unknown original CAT typical scene: {scene_type}")
        axes = [float(origin) + (np.arange(int(count)) + .5) * grid["voxel"]
                for origin, count in zip(grid["origin_w"], grid["shape"])]
        occupancy = _upstream("typical_obstacle").build_obstacles(scene_type, np.meshgrid(*axes, indexing="ij"))
    else:
        module = _upstream("random_obstacle")
        config = module.Cfg(seed=source_seed, difficulty=difficulty, n_rect_F=n_ground,
                            n_rect_L=n_lateral, n_rect_R=n_lateral, n_rect_C=n_overhead)
        occupancy, *axes = module.generate_and_save(config, save=False)
    occupancy = np.asarray(occupancy, dtype=bool)
    source_origin = np.asarray(grid["origin_w"], dtype=float)
    # Preserve z/floor, add space around the source xy bounds for physical walls.
    translation = np.asarray([.20 - source_origin[0], .20 - source_origin[1], 0.])
    effective_extent = np.asarray(occupancy.shape) * grid["voxel"]
    dimensions = [float(effective_extent[0] + .4), float(effective_extent[1] + .4), max(2.0, float(effective_extent[2]))]
    boxes = _walls(dimensions)
    cuboids = merge_occupied_voxels(occupancy)
    for index, (begin, end) in enumerate(cuboids):
        low = source_origin + np.asarray(begin) * grid["voxel"] + translation
        high = source_origin + np.asarray(end) * grid["voxel"] + translation
        boxes.append(dict(name=f"legacy_voxels_{index}", center=((low + high) / 2).tolist(),
                          half_size=((high - low) / 2).tolist(), yaw=0., category="legacy_obstacle",
                          source_voxel_begin=list(begin), source_voxel_end=list(end)))
    start3 = np.asarray(grid["start_w"], dtype=float) + translation
    goal3 = np.asarray(grid["goal_w"], dtype=float) + translation
    route = [start3[:2].tolist(), goal3[:2].tolist()]
    # The route provides progress accounting only. Native 3-D FMM guidance can
    # go around lateral obstacles; straight-line full-body feasibility is unknown.
    route_length = float(np.linalg.norm(goal3[:2] - start3[:2]))
    case = dict(start=[*route[0], 0.], goal=route[-1], route=route, route_length_m=route_length,
                time_budget=route_length / .3 + 12., bottlenecks=[])
    hashes = _source_hashes()
    legacy = dict(upstream_commit=UPSTREAM_COMMIT, source_hashes=hashes,
        kind=kind, scene_type=scene_type if kind == "typical" else "random", source_seed=source_seed,
        source_voxel_size_m=float(grid["voxel"]), source_shape=list(occupancy.shape),
        source_edge_origin=source_origin.tolist(), translation=translation.tolist(),
        occupied_voxels=int(occupancy.sum()), occupied_union_sha256=hashlib.sha256(occupancy.tobytes()).hexdigest(),
        cuboids=len(cuboids), merge="lossless-greedy-occupied-voxel-cuboids-v1",
        goal_height_m=float(goal3[2]), original_fields_bit_identical=False,
        physical_contacts_added=True, save_only_torch_import_is_lazy=True,
        generation_parameters=dict(difficulty=float(difficulty), n_ground=n_ground,
                                   n_lateral=n_lateral, n_overhead=n_overhead))
    geometry_hash = _digest(dict(boxes=boxes, room_dimensions=dimensions))
    scene = dict(schema=SCHEMA, scene_id=f"legacy-{kind}-{scene_type}-{split}-{seed}-{geometry_hash[:12]}",
                 seed=seed, split=split, family=f"legacy_{kind}", difficulty=float(difficulty),
                 boxes=boxes, room_dimensions=dimensions, geometry_hash=geometry_hash,
                 start_goals=[case], goal_index=0, **case,
                 legacy_cat=legacy,
                 generator=dict(name="original-cat-occupancy-to-physical-boxes-v1", guidance_method=GUIDANCE_METHOD,
                     geometry_source="original-CAT-occupied-voxel-union-plus-perimeter-walls",
                     split_holdout=dict(kind="disjoint-source-random-seeds" if kind == "random" else "deterministic-typical-template",
                                        typical_template_geometry_shared_across_splits=kind == "typical")),
                 counts=dict(tables=0, chairs=0, primitive_boxes=len(boxes), bottlenecks=0),
                 feasibility=dict(root_route_validated=False, full_body_validated=False, dynamic_feasibility_validated=False,
                     explanation="Original generator task; requires stepping/crouching/lateral navigation. No full-body certificate."),
                 reset_clearance=dict(root_cylinder_validated=False, full_body_validated=False,
                                      runtime_nominal_pose_collision_check_required=True))
    validate_scene(scene)
    return scene


def original_guidance_spec(scene):
    legacy = scene["legacy_cat"]
    return dict(method=GUIDANCE_METHOD, route=scene["route"], goal=scene["goal"],
                goal_height_m=legacy["goal_height_m"], source_sha256=legacy["source_hashes"]["pf_modular.py"],
                projection_voxels=5., speed_mps=.6, goal_seed_radius_m=.12)


def original_guidance_field(scene, sdf, bf, grid):
    """Original progressive 3-D FMM on the canonical physical obstacle union."""
    try:
        module = _upstream("pf_modular")
    except ModuleNotFoundError as error:
        raise ImportError("Original CAT 3-D guidance requires scikit-fmm; install requirements-furniture-cpu.txt") from error
    if _source_hashes()["pf_modular.py"] != scene["legacy_cat"]["source_hashes"]["pf_modular.py"]:
        raise ValueError("Original CAT field source changed since scene specification")
    config = module.PFConfig()
    config.voxel = float(grid["voxel_size"])
    axes = [origin + np.arange(size) * config.voxel for origin, size in zip(grid["sample_origin"], grid["shape"])]
    positions = np.meshgrid(*axes, indexing="ij")
    goal = np.asarray([*scene["goal"], scene["legacy_cat"]["goal_height_m"]])
    _, gf = module.make_guidance_field_progressive(config, positions, sdf <= 0, goal, bf, sdf)
    if not np.isfinite(gf).all():
        raise ValueError("Original CAT FMM produced nonfinite guidance")
    return gf.astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--kind", choices=("typical", "random"), default="typical")
    generate.add_argument("--scene", choices=TYPICAL_SCENES, default="forward")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--split", choices=SPLITS, default="train")
    generate.add_argument("--difficulty", type=float, default=.5)
    generate.add_argument("--n-ground", type=int, default=1)
    generate.add_argument("--n-lateral", type=int, default=1)
    generate.add_argument("--n-overhead", type=int, default=1)
    generate.add_argument("--voxel-size", type=float, default=.04)
    generate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    scene = generate_legacy_scene(args.kind, args.scene, args.seed, args.split, args.difficulty,
                                  args.n_ground, args.n_lateral, args.n_overhead)
    from cat_ppo.furniture.scenes import write_scene_bundle
    path = write_scene_bundle(scene, args.output, voxel_size=args.voxel_size)
    print(json.dumps(dict(bundle=str(path), scene_id=scene["scene_id"], legacy_cat=scene["legacy_cat"]), indent=2))


if __name__ == "__main__":
    main()
