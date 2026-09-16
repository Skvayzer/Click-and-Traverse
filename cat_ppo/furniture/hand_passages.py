"""Height-selective arm-protection passages with explicit static certificates.

Nominal *forward* traversal must intersect the approved hand envelopes. A
reachable raised-arm pose must clear both all 35 approved collision primitives
and CAT's actual conservative 4 cm SDF queries along the route. These are static
configuration checks, not dynamic demonstrations. Sideways alternatives are
audited and reported, never silently described as impossible.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import json
import math
from pathlib import Path
import random

import numpy as np

from cat_ppo.furniture.scenes import (
    SCHEMA, SPLITS, _box, _case, _digest, _route_clearance, _table, _walls,
    validate_scene,
)

KINDS = ("hand_table_aisle", "hand_shelf_passage")
DIFFICULTIES = ("easy", "medium", "hard")
GENERATOR = "height-selective-hand-passages-v1"
PROPOSAL = Path(__file__).resolve().parents[2] / "docs/assets/collision-proxy-proposal-20260916/proposal.json"


@lru_cache(maxsize=1)
def _robot():
    import mujoco
    from cat_ppo.envs.g1.env_cat_wholebody import assemble_training_xml
    from cat_ppo.furniture.body_collision_geometry import compile_proposal
    model = mujoco.MjModel.from_xml_string(assemble_training_xml())
    return model, compile_proposal(json.loads(PROPOSAL.read_text()), model)


def arm_pose(mode="nominal", fraction=1.):
    """Actual 29-joint G1/Dex3 qpos; arm offsets stay within ±0.8 rad."""
    import mujoco
    from cat_ppo.envs.g1.constants import DEFAULT_QPOS
    if mode not in ("nominal", "raised", "tucked") or not 0 <= fraction <= 1:
        raise ValueError("Invalid pose mode or interpolation fraction")
    model, _ = _robot()
    q = np.array(DEFAULT_QPOS, dtype=float)
    q[:2] = 0.
    if mode != "nominal":
        for side, sign in (("left", 1), ("right", -1)):
            changes = ({"shoulder_pitch": -.8, "shoulder_roll": -.15 * sign, "elbow": -.8}
                       if mode == "raised" else
                       {"shoulder_pitch": -.3, "shoulder_roll": -.25 * sign, "elbow": .7})
            for joint, delta in changes.items():
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}_joint")
                q[model.jnt_qposadr[joint_id]] += fraction * delta
    return q


def _pose_geometry(qpos):
    import mujoco
    from cat_ppo.furniture.body_collision_geometry import transform_primitives
    model, compiled = _robot()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return {key: np.asarray(value) for key, value in transform_primitives(
        data.xpos, data.xmat.reshape(-1, 3, 3), compiled["body_ids"],
        compiled["local_centers"], compiled["local_rotations"], compiled["local_endpoints"]).items()}


def _world_geometry(local, xyyaw):
    xyyaw = np.asarray(xyyaw, dtype=float).reshape(-1, 3)
    c, s = np.cos(xyyaw[:, 2]), np.sin(xyyaw[:, 2])
    rotation = np.zeros((len(c), 3, 3))
    rotation[:, 0, 0] = rotation[:, 1, 1] = c
    rotation[:, 0, 1], rotation[:, 1, 0], rotation[:, 2, 2] = -s, s, 1.
    position = np.c_[xyyaw[:, :2], np.zeros(len(c))]
    return dict(centers=np.einsum("pij,sj->psi", rotation, local["centers"]) + position[:, None],
                rotations=np.einsum("pij,sjk->psik", rotation, local["rotations"]),
                endpoints=np.einsum("pij,skj->pski", rotation, local["endpoints"]) + position[:, None, None])


def pose_separations(scene, xyyaw, *, mode="nominal", fraction=1.):
    """Per-pose full-body/hand minimum separation against canonical boxes."""
    from cat_ppo.furniture.body_collision_geometry import (
        box_box_separation, capsule_box_separation, sphere_box_separation,
    )
    _, compiled = _robot()
    world = _world_geometry(_pose_geometry(arm_pose(mode, fraction)), xyyaw)
    boxes = scene["boxes"]
    center = np.array([b["center"] for b in boxes])[None, None]
    half = np.array([b["half_size"] for b in boxes])[None, None]
    rotation = np.array([[[math.cos(b["yaw"]), -math.sin(b["yaw"]), 0],
                          [math.sin(b["yaw"]), math.cos(b["yaw"]), 0], [0, 0, 1]]
                         for b in boxes])[None, None]
    result = np.empty((len(world["centers"]), len(compiled["shape_names"])))
    for kind, ids in compiled["indices"].items():
        centers = world["centers"][:, ids][:, :, None]
        radii = compiled["radii"][ids][None, :, None]
        if kind == "sphere":
            values = sphere_box_separation(centers, radii, center, rotation, half)
        elif kind == "capsule":
            ends = world["endpoints"][:, ids]
            values = capsule_box_separation(ends[:, :, None, 0], ends[:, :, None, 1],
                                             radii, center, rotation, half)
        else:
            values = box_box_separation(centers, world["rotations"][:, ids][:, :, None],
                                        compiled["half_sizes"][ids][None, :, None], center, rotation, half)
        result[:, ids] = np.asarray(values).min(axis=-1)
    hands = [compiled["shape_names"].index(f"{side}_hand") for side in ("left", "right")]
    return dict(body=result.min(axis=-1), hands=result[:, hands],
                hand_centers=world["centers"][:, hands], hand_radii=compiled["radii"][hands])


def _route_poses(scene, spacing=.025):
    # These layouts use a constant torso heading along a gently offset route;
    # the certificate does not assume in-place body rotation at path corners.
    result = []
    for a, b in zip(scene["route"], scene["route"][1:]):
        count = max(1, math.ceil(math.dist(a, b) / spacing))
        for t in np.linspace(0., 1., count + 1):
            xy = (1 - t) * np.asarray(a) + t * np.asarray(b)
            result.append([*xy, scene["generator"]["passage_yaw_rad"]])
    return np.asarray(result)


def _sdf(scene, dx=.04):
    import skfmm
    from cat_ppo.furniture.room_geometry import conservative_rasterize_boxes
    shape = np.ceil((np.asarray(scene["room_dimensions"]) + [.2, .2, 0.]) / dx).astype(int)
    origin = np.array([-.1, -.1, 0.]) + .5 * dx
    occupancy = conservative_rasterize_boxes(scene["boxes"], shape, origin, dx)
    phi = np.where(occupancy, -1., 1.)
    return np.asarray(skfmm.distance(phi, dx=dx), dtype=np.float32), origin


def _hand_field_clearance(sdf, origin, sample, dx=.04):
    from cat_ppo.furniture.generalist_fields import sample_ragged_field
    values = np.asarray(sample_ragged_field(
        sdf.reshape(-1, 1), sample["hand_centers"].reshape(-1, 3), origin=origin,
        dx=dx, shape=np.array(sdf.shape), offset=0)).reshape(-1, 2)
    return values - sample["hand_radii"][None]


def certify_hand_passage(scene):
    """Reject weak forward-arm challenges or statically unsafe protected paths.

    The yaw audit checks entire constant-yaw centerline paths, including both
    sideways headings. It is explicitly not a complete SE(2) path search.
    """
    validate_scene(scene)
    poses = _route_poses(scene)
    sdf, origin = _sdf(scene)
    nominal = pose_separations(scene, poses)
    raised = pose_separations(scene, poses, mode="raised")
    interior_raised = pose_separations(scene, poses, mode="raised", fraction=.9)
    tucked = pose_separations(scene, poses, mode="tucked")
    raised_field = _hand_field_clearance(sdf, origin, raised)
    interior_raised_field = _hand_field_clearance(sdf, origin, interior_raised)
    tucked_field = _hand_field_clearance(sdf, origin, tucked)
    nominal_blocked = np.min(nominal["hands"], axis=-1) < -0.002
    transition_body, transition_field = [], []
    # Raise before entering and lower after exiting; static interpolation checks
    # include all 35 shapes and the actual runtime hand SDF at both endpoints.
    for fraction in np.linspace(0., 1., 17):
        sample = pose_separations(scene, poses[[0, -1]], mode="raised", fraction=float(fraction))
        transition_body.append(float(sample["body"].min()))
        transition_field.append(float(_hand_field_clearance(sdf, origin, sample).min()))
    yaw_paths = []
    for offset in np.linspace(-math.pi, math.pi, 25)[:-1]:
        alternative = poses.copy()
        alternative[:, 2] += offset
        sample = pose_separations(scene, alternative)
        clearance = float(sample["body"].min())
        field_clearance = float(_hand_field_clearance(sdf, origin, sample).min())
        yaw_paths.append(dict(yaw_offset_rad=float(offset), body_min_separation_m=clearance,
                              hand_field_min_clearance_m=field_clearance,
                              collision_free=clearance > 1e-5 and field_clearance > 1e-5))
    # Lateral offsets and yaw are also checked at the most challenging nominal
    # forward location. This local audit is not mislabeled as a path search.
    focus = poses[int(np.argmin(nominal["hands"].min(axis=-1)))].copy()
    normal = np.array([-math.sin(focus[2]), math.cos(focus[2])])
    local_audit_poses = []
    for lateral in np.linspace(-.15, .15, 13):
        for offset in np.linspace(-math.pi, math.pi, 25)[:-1]:
            local_audit_poses.append([*(focus[:2] + lateral * normal), focus[2] + offset])
    local_sample = pose_separations(scene, local_audit_poses)
    local_field = _hand_field_clearance(sdf, origin, local_sample).min(axis=-1)
    local_free = (local_sample["body"] > 1e-5) & (local_field > 1e-5)
    certificate = dict(
        method="approved-35-primitives-static-route-and-arm-transition-plus-exact-CAT-hand-field-v1",
        collision_proposal_sha256=__import__("hashlib").sha256(PROPOSAL.read_bytes()).hexdigest(),
        primitive_count=35, route_sample_spacing_m=.025, transition_samples_per_endpoint=17,
        voxel_size_m=.04, field_sampling="conservative-cell-OBB-occupancy; original-skfmm-SDF; exact-CAT-corner-order",
        nominal_forward_hand_collision_fraction=float(nominal_blocked.mean()),
        nominal_forward_hand_min_separation_m=float(nominal["hands"].min()),
        raised_route_body_min_separation_m=float(raised["body"].min()),
        raised_route_hand_field_min_clearance_m=float(raised_field.min()),
        raised_90percent_route_body_min_separation_m=float(interior_raised["body"].min()),
        raised_90percent_route_hand_field_min_clearance_m=float(interior_raised_field.min()),
        raised_qpos_relative_root=arm_pose("raised").tolist(),
        tucked_qpos_relative_root=arm_pose("tucked").tolist(),
        tucked_route_body_min_separation_m=float(tucked["body"].min()),
        tucked_route_hand_field_min_clearance_m=float(tucked_field.min()),
        tucked_route_static_safe=bool(tucked["body"].min() > .005 and tucked_field.min() > .005),
        raised_transition_body_min_separation_m=min(transition_body),
        raised_transition_hand_field_min_clearance_m=min(transition_field),
        nominal_constant_yaw_bypass_found=any(row["collision_free"] for row in yaw_paths),
        nominal_constant_yaw_path_audit=yaw_paths,
        nominal_lateral_yaw_local_audit=dict(lateral_range_m=[-.15, .15], lateral_step_m=.025,
            yaw_step_degrees=15, configuration_count=len(local_audit_poses),
            collision_free_configuration_count=int(local_free.sum()),
            focus_root_xyyaw=focus.tolist(), continuous_path_search=False),
        complete_se2_search=False, dynamic_feasibility_validated=False,
        arm_motion_physically_unique=False,
        explanation="Forward nominal arms collide; raising clears the sampled route and arm transitions. "
                    "Yaw audit includes sideways travel. Passing this check never proves raising is the only possible behavior.")
    if nominal_blocked.mean() < .12:
        raise ValueError("Nominal forward arms are not sufficiently challenged")
    if min(certificate["raised_route_body_min_separation_m"], min(transition_body)) <= .008:
        raise ValueError("Raised-arm route or transition intersects approved body volumes")
    if min(float(raised_field.min()), min(transition_field)) <= .025:
        raise ValueError(f"Raised-arm route or transition lacks 4cm-field hand clearance: route={float(raised_field.min()):.5f}, transition={min(transition_field):.5f}")
    if interior_raised["body"].min() <= .008 or interior_raised_field.min() <= .025:
        raise ValueError("Protected route must remain safe at 90% of the arm action limit")
    return certificate


def generate_hand_passage_scene(seed, split="train", kind="hand_table_aisle", difficulty="easy", certify=True):
    """Generate a deterministic height-selective, route-checked furniture scene."""
    if type(seed) is not int or seed < 0 or split not in SPLITS or kind not in KINDS or difficulty not in DIFFICULTIES:
        raise ValueError("Invalid hand passage seed/split/kind/difficulty")
    rng = random.Random(int(_digest([GENERATOR, seed, split, kind, difficulty]), 16))
    level = DIFFICULTIES.index(difficulty)
    dimensions = [4.8, 3.2, 1.8]
    yaw = rng.uniform(-.10, .10)
    center = np.array([2.4 + rng.uniform(-.05, .05), 1.6 + rng.uniform(-.07, .07)])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    gap = rng.uniform(*((.64, .67), (.60, .63), (.565, .595))[level])
    height = rng.uniform(.65, .70) if level < 2 else rng.uniform(.68, .735)
    parts = []
    if kind == "hand_table_aisle":
        for side in (-1, 1):
            parts.extend(_table(f"protective_table_{side:+d}", 0., side * (gap / 2 + .70),
                                2.1, 1.4, height, f"{split}/hand-table/{seed}"))
        local_route = [[-1.55, 0.], [1.55, 0.]]
        counts = dict(tables=2, chairs=0, generic_objects=0)
    else:
        offset = (0., .025, .04)[level]
        local_route = [[-1.55, 0.]]
        for index, x in enumerate((-.75, 0., .75)):
            local_y = (-1 if index % 2 else 1) * offset
            local_route.append([x, local_y])
            for side in (-1, 1):
                edge = local_y + side * gap / 2
                shelf_y = edge + side * .70
                identity = dict(furniture_id=f"shelf_{index}_{side:+d}", furniture_type="shelf")
                parts.append(_box(f"shelf_{index}_{side:+d}_edge", [x, shelf_y, height - .04],
                                  [.24, .70, .04], "shelf_edge", **identity))
                # Supports are deliberately outside the narrow arm challenge.
                parts.append(_box(f"shelf_{index}_{side:+d}_support", [x, edge + side * 1.1, height / 2],
                                  [.035, .035, height / 2], "shelf_support", **identity))
        local_route.append([1.55, 0.])
        counts = dict(tables=0, chairs=0, generic_objects=6)
    for box in parts:
        xy = rotation @ np.asarray(box["center"][:2]) + center
        box["center"][:2] = [round(float(v), 9) for v in xy]
        box["yaw"] = round(box["yaw"] + yaw, 9)
    boxes = _walls(dimensions) + parts
    route = [(rotation @ p + center).tolist() for p in local_route]
    case = _case(route, [])
    geometry_hash = _digest(dict(boxes=boxes, room_dimensions=dimensions))
    scene = dict(schema=SCHEMA, scene_id=f"{kind}-{difficulty}-{split}-{seed:06d}-{geometry_hash[:12]}",
                 seed=seed, split=split, family="furniture" if kind == KINDS[0] else "generic_clutter",
                 difficulty=difficulty, units="metres", coordinate_system="right-handed-z-up",
                 boxes=boxes, room_dimensions=dimensions, geometry_hash=geometry_hash,
                 start_goals=[case], goal_index=0, **copy.deepcopy(case))
    scene["counts"] = dict(**counts, primitive_boxes=len(boxes), bottlenecks=1 if kind == KINDS[0] else 3)
    scene["generator"] = dict(name=GENERATOR, passage_yaw_rad=yaw,
        geometry_source="canonical-oriented-boxes", floor_in_obstacle_field=False,
        semantic_labels_observed_by_policy=False, dynamic_feasibility_validated=False,
        purpose="Make nominal forward hands unsafe while a raised protective posture clears the route")
    scene["hand_protection"] = dict(kind=kind, difficulty=difficulty, level=level,
        passage_width_m=gap, furniture_top_height_m=height,
        nominal_forward_arms_must_change=True, arm_motion_physically_unique=False,
        task="Traverse safely; raising, tucking, or another safe maneuver is allowed")
    scene["source"] = dict(hand_protection=copy.deepcopy(scene["hand_protection"]))
    scene["feasibility"] = _route_clearance(route, boxes)
    scene["case_feasibility"] = [copy.deepcopy(scene["feasibility"])]
    if not scene["feasibility"]["root_route_validated"]:
        raise ValueError("Hand passage root route failed")
    validate_scene(scene)
    if certify:
        scene["hand_protection"]["certificate"] = certify_hand_passage(scene)
        scene["source"]["hand_protection"] = copy.deepcopy(scene["hand_protection"])
    return scene
