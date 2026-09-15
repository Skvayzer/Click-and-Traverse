"""Reproducible CPU geometry check of compact CAT resets in the default rooms.

Queries the exact oriented-box SDF at the compiled robot's 13 field sites.
Hand/elbow sphere radii are subtracted. This checks sampled initial poses, not
mesh contacts, voxel interpolation, velocity evolution, or dynamic feasibility.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from cat_ppo.envs.g1 import constants
from cat_ppo.envs.g1.env_cat_wholebody import assemble_training_xml
from cat_ppo.furniture.generalist_fields import sha256
from cat_ppo.furniture.grippers import hand_sphere
from cat_ppo.furniture.random_rooms import (
    RESET_CENTER_CLEARANCE_M, RESET_HAND_ENVELOPE_RADIUS_M, generate_random_room,
)


def reset_cloud(model, samples, seed):
    names = ["head", "imu_in_pelvis", "imu_in_torso", *constants.FEET_SITES,
             *constants.HAND_SITES, *constants.KNEE_SITES, *constants.SHOULDER_SITES,
             "left_elbow_probe", "right_elbow_probe"]
    ids = [model.site(name).id for name in names]
    radii = np.array([0.] * 5 + [hand_sphere(side)["radius"] for side in ("left", "right")]
                     + [0.] * 4 + [.05, .05])
    lower, upper = model.jnt_range[1:].T
    center, half = (lower + upper) / 2, (upper - lower) * .95 / 2
    data, rng = mujoco.MjData(model), np.random.default_rng(seed)
    cloud, jitters = [], []
    for index in range(samples + 1):
        qpos = np.array(constants.DEFAULT_QPOS, dtype=float)
        qpos[2] = .8
        jitter = np.zeros(2)
        if index:
            jitter = rng.uniform(-.08, .08, 2)
            yaw = rng.uniform(-np.pi / 2, np.pi / 2)
            qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            qpos[7:] = np.clip(qpos[7:] * rng.uniform(.5, 1.5, 29), center - half, center + half)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        cloud.append(data.site_xpos[ids].copy())
        jitters.append(jitter)
    # Check a deterministic grid including the extreme joint multipliers in
    # addition to random resets. This records evidence, not an interval proof.
    arm_grid = {}
    for side in ("left", "right"):
        joints = [model.joint(f"{side}_{name}_joint") for name in ("shoulder_pitch", "shoulder_roll", "elbow")]
        addresses = np.array([int(joint.qposadr[0]) for joint in joints])
        grid_max = 0.
        for factors in product(np.linspace(.5, 1.5, 9), repeat=3):
            qpos = np.array(constants.DEFAULT_QPOS, dtype=float)
            qpos[2] = .8
            qpos[addresses] *= factors
            qpos[7:] = np.clip(qpos[7:], center - half, center + half)
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            radius = np.linalg.norm(data.site_xpos[model.site(f"{side}_palm").id, :2]) + hand_sphere(side)["radius"]
            grid_max = max(grid_max, float(radius))
        arm_grid[side] = grid_max
    return names, radii, np.array(cloud), np.array(jitters), arm_grid


def audit_scene(scene, cloud, jitters, radii, names):
    yaw = scene["start"][2]
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    points = cloud @ rotation.T
    # The environment applies XY jitter in world coordinates, then rotates only
    # the root orientation. Rotating the jitter with the robot would be wrong.
    points[:, :, :2] += np.asarray(scene["start"][:2]) + jitters[:, None, :]
    centers = np.array([box["center"] for box in scene["boxes"]])
    sizes = np.array([box["half_size"] for box in scene["boxes"]])
    angles = np.array([box["yaw"] for box in scene["boxes"]])
    c, s = np.cos(angles), np.sin(angles)
    delta = points[:, :, None, :] - centers
    local = delta.copy()
    local[..., 0] = delta[..., 0] * c + delta[..., 1] * s
    local[..., 1] = -delta[..., 0] * s + delta[..., 1] * c
    q = np.abs(local) - sizes
    distances = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)
    clearances = distances.min(axis=-1) - radii
    minimum = clearances.min(axis=-1)
    pose, site = np.unravel_index(clearances.argmin(), clearances.shape)
    return dict(kind=scene["family"], seed=scene["seed"], scene_id=scene["scene_id"],
                geometry_hash=scene["geometry_hash"], start=scene["start"],
                nominal_min_m=float(minimum[0]), sampled_min_m=float(minimum[1:].min()),
                colliding_samples=int((minimum[1:] < 0).sum()), samples=len(jitters) - 1,
                worst_site=names[site], worst_pose=int(pose),
                hand_collisions=int((clearances[1:, 5:7].min(axis=-1) < 0).sum()),
                nonhand_collisions=int((np.delete(clearances[1:], [5, 6], axis=1).min(axis=-1) < 0).sum()),
                root_center_clearance_m=scene["reset_clearance"]["center_clearance_m"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=90260916)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/cat-diversity-20260916/room-reset-audit.json")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    model = mujoco.MjModel.from_xml_string(assemble_training_xml())
    names, radii, cloud, jitters, arm_grid = reset_cloud(model, args.samples, args.seed)
    results = [audit_scene(generate_random_room(seed, kind=kind), cloud, jitters, radii, names)
               for kind, first in (("furniture", 4001), ("generic_clutter", 5001))
               for seed in range(first, first + 24)]
    summaries = {}
    for kind in ("furniture", "generic_clutter"):
        rows = [row for row in results if row["kind"] == kind]
        summaries[kind] = dict(rooms=len(rows), nominal_colliding_rooms=sum(row["nominal_min_m"] < 0 for row in rows),
            nominal_min_m=min(row["nominal_min_m"] for row in rows),
            sampled_colliding_rooms=sum(row["colliding_samples"] > 0 for row in rows),
            sampled_collisions=sum(row["colliding_samples"] for row in rows),
            sampled_total=sum(row["samples"] for row in rows), sampled_min_m=min(row["sampled_min_m"] for row in rows))
    report = dict(schema="cat-room-reset-audit-v1", seed=args.seed, samples_per_room=args.samples,
        reset_distribution=dict(xy_half_extent_m=.08, yaw_half_extent_rad=np.pi / 2,
                                joint_nominal_multiplier=[.5, 1.5], joint_soft_limit_factor=.95, root_height_m=.8),
        method="compiled MuJoCo forward kinematics; exact oriented-box SDF; 13 compact field sites",
        limitations=["Finite sample/grid audit, not a universal reset certificate", "No voxel-field interpolation",
                     "No robot mesh-contact or dynamic-feasibility test", "Route certificate remains root-cylinder only"],
        required_endpoint_center_clearance_m=RESET_CENTER_CLEARANCE_M,
        configured_reset_hand_envelope_radius_m=RESET_HAND_ENVELOPE_RADIUS_M,
        measured_arm_parameter_grid_radius_m=arm_grid, arm_grid_points_per_side=9 ** 3,
        source_sha256={name: sha256(ROOT / name) for name in (
            "cat_ppo/furniture/random_rooms.py", "cat_ppo/furniture/grippers.py",
            "cat_ppo/envs/g1/env_cat_wholebody.py", "scripts/audit_room_resets.py")},
        summaries=summaries, scenes=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), summaries=summaries, arm_grid=arm_grid), indent=2))
    if any(row["colliding_samples"] or row["nominal_min_m"] < 0 for row in results):
        raise SystemExit("Reset collisions found")


if __name__ == "__main__":
    main()
