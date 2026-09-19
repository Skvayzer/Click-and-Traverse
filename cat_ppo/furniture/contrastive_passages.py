"""Matched hand-protection passages with explicit, static safety certificates.

Only corridor width changes within a group: open, forward with protected arms,
sideways-only for the checked postures, and protected/sideways/protected. The
certificates check the approved 35 body primitives and the actual conservative
4 cm hand field. They are sampled kinematic checks, never dynamic guarantees or
proofs that no other whole-body solution exists.
"""
from __future__ import annotations

import copy
import hashlib
import math
import random

import numpy as np

from cat_ppo.furniture.hand_passages import (
    PROPOSAL, _hand_field_clearance, _pose_geometry, _robot, _sdf, _world_geometry,
    arm_pose, pose_separations,
)
from cat_ppo.furniture.scenes import (
    SCHEMA, SPLITS, _box, _case, _digest, _route_clearance, _walls, validate_scene,
)

ROLES = ("open", "forward_protected", "narrow", "transition")
GENERATOR = "contrastive-hand-passages-v2"
MODULE_X = (-2.2, 0., 2.2)
ROUTE_LIMIT = 3.5
SAMPLE_SPACING = .05


def _hands(mode, fraction=1.):
    _, compiled = _robot()
    indices = [compiled["shape_names"].index(f"{side}_hand") for side in ("left", "right")]
    return _pose_geometry(arm_pose(mode, fraction))["centers"][indices], compiled["radii"][indices]


def _regions():
    # Interior of the existing action range; no demand to exceed a joint limit.
    centers = np.array([_hands("raised", .95)[0], _hands("tucked", .95)[0]])
    half = np.array([[.018, .010, .018], [.012, .008, .015]])[:, None, :]
    return centers - half, centers + half


def _world_xy(scene, local):
    yaw = scene["generator"]["passage_yaw_rad"]
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return np.asarray(local) @ rotation.T + scene["generator"]["passage_center_xy"]


def _poses(scene, xs, yaw_offset=0.):
    xy = _world_xy(scene, np.c_[np.asarray(xs), np.zeros(len(xs))])
    return np.c_[xy, np.full(len(xs), scene["generator"]["passage_yaw_rad"] + yaw_offset)]


def _navigation_report(route, boxes, radius):
    report = _route_clearance(route, boxes)
    report["root_radius_m"] = radius
    report["root_margin_lower_bound_m"] = report["center_clearance_lower_bound_m"] - radius
    report["root_route_validated"] = report["root_margin_lower_bound_m"] > 0.
    report["explanation"] = "Navigation-only radius; separate 35-primitive route and transition certificate required."
    return report


def generate_contrastive_group(seed, split="train", *, certify=True):
    """Return four deterministic, matched layouts in ``ROLES`` order.

    ``certify=False`` is for previews only. Training loaders must reject absent
    certificates, particularly for the smaller narrow-passage navigation radius.
    """
    if type(seed) is not int or seed < 0 or split not in SPLITS:
        raise ValueError("Expected a nonnegative integer seed and a known split")
    rng = random.Random(int(_digest([GENERATOR, split, seed]), 16))
    yaw = rng.uniform(-.10, .10)
    center = np.array([4.5 + rng.uniform(-.04, .04), 1.8 + rng.uniform(-.04, .04)])
    height, depth = rng.uniform(1.24, 1.28), rng.uniform(.55, .62)
    widths = dict(open=rng.uniform(1.50, 1.52), forward_protected=rng.uniform(.729, .741),
                  narrow=rng.uniform(.400, .410))
    group_id = f"contrast-{split}-{seed:06d}"
    region_min, region_max = _regions()
    scenes = []
    for role in ROLES:
        module_roles = ["forward_protected", "narrow", "forward_protected"] if role == "transition" else [role] * 3
        boxes = _walls([9., 3.6, 1.8])
        modules, zones = [], []
        for index, (x, module_role) in enumerate(zip(MODULE_X, module_roles)):
            width = widths[module_role]
            for side in (-1, 1):
                local = np.array([x, side * (width / 2 + depth / 2)])
                c, s = math.cos(yaw), math.sin(yaw)
                xy = np.array([[c, -s], [s, c]]) @ local + center
                boxes.append(_box(f"cabinet_{index}_{side:+d}", [*xy, height / 2],
                    [.325, depth / 2, height / 2], "cabinet", yaw=yaw,
                    furniture_id=f"cabinet_{index}_{side:+d}", furniture_type="cabinet"))
                # Seal the exterior lane: otherwise the policy could bypass
                # every contrastive gap around the ends of the cabinets.
                module_world_y = center[1] + s * x
                boundary_y = 3.6 if side > 0 else 0.
                edge = abs((boundary_y - module_world_y) / c) + .05
                inner = width / 2 + depth - .02
                local = np.array([x, side * (inner + edge) / 2])
                xy = np.array([[c, -s], [s, c]]) @ local + center
                boxes.append(_box(f"baffle_{index}_{side:+d}", [*xy, .9],
                    [.325, (edge - inner) / 2, .9], "baffle", yaw=yaw))
            protected = module_role == "forward_protected"
            # Ends do not overlap: 2.2 m module spacing leaves unshaped turn bays.
            zones.append(dict(start_m=x + ROUTE_LIMIT - .95, end_m=x + ROUTE_LIMIT + .70,
                fade_m=.15, forward_weight=float(module_role != "narrow"),
                hand_active=[int(protected), int(protected)],
                hand_regions_min=region_min.tolist(), hand_regions_max=region_max.tolist(),
                region_valid=[bool(protected), bool(protected)],
                hazard_start_m=x + ROUTE_LIMIT - .325, hazard_end_m=x + ROUTE_LIMIT + .325))
            modules.append(dict(center_local_x_m=x, length_m=.65, width_m=width, role=module_role))
        temporary = dict(generator=dict(passage_yaw_rad=yaw, passage_center_xy=center.tolist()))
        route = _world_xy(temporary, [[-ROUTE_LIMIT, 0.], [ROUTE_LIMIT, 0.]]).tolist()
        case = _case(route, [])
        geometry_hash = _digest(dict(boxes=boxes, room_dimensions=[9., 3.6, 1.8]))
        radius = .15 if role in ("narrow", "transition") else .23
        contrast = dict(schema="hand-contrast-v1", group_id=group_id, role=role,
            navigation_radius_m=radius, zones=zones, modules=modules,
            region_frame="xy:route tangent/left normal relative to root xy; z:absolute world metres",
            mode_names=["raised", "tucked"], cabinet_height_m=height,
            dynamic_feasibility_validated=False)
        scene = dict(schema=SCHEMA, scene_id=f"{group_id}-{role}-{geometry_hash[:12]}", seed=seed,
            split=split, family="generic_clutter", difficulty="contrastive", units="metres",
            coordinate_system="right-handed-z-up", room_dimensions=[9., 3.6, 1.8], boxes=boxes,
            geometry_hash=geometry_hash, start_goals=[case], goal_index=0, **copy.deepcopy(case))
        scene["generator"] = dict(name=GENERATOR, passage_yaw_rad=yaw, passage_center_xy=center.tolist(),
            geometry_source="canonical-oriented-boxes", floor_in_obstacle_field=False,
            semantic_labels_observed_by_policy=False, dynamic_feasibility_validated=False)
        scene["counts"] = dict(tables=0, chairs=0, generic_objects=12, primitive_boxes=len(boxes), bottlenecks=3)
        scene["hand_contrast"] = contrast
        scene["feasibility"] = _navigation_report(route, boxes, radius)
        scene["case_feasibility"] = [copy.deepcopy(scene["feasibility"])]
        validate_scene(scene)
        if certify:
            contrast["certificate"] = certify_contrastive_scene(scene)
        scene["source"] = dict(hand_contrast=copy.deepcopy(contrast))
        scenes.append(scene)
    return scenes


def module_free_intervals(scene, module):
    """Exact floor-level cross-section openings at a module's center plane.

    Intersect the infinite line normal to the route with every canonical OBB,
    then subtract the merged occupied intervals inside the perimeter walls.
    This catches exterior bypass lanes without a finite point sampling grid.
    """
    yaw = scene["generator"]["passage_yaw_rad"]
    origin = _world_xy(scene, [[module["center_local_x_m"], 0.]])[0]
    direction = np.array([-math.sin(yaw), math.cos(yaw)])
    domain = sorted([(0. - origin[1]) / direction[1], (3.6 - origin[1]) / direction[1]])
    occupied = []
    for box in scene["boxes"]:
        if not box["center"][2] - box["half_size"][2] <= .8 <= box["center"][2] + box["half_size"][2]:
            continue
        c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
        inverse = np.array([[c, s], [-s, c]])
        local = inverse @ (origin - box["center"][:2])
        vector = inverse @ direction
        low, high = domain
        for pos, vel, half in zip(local, vector, box["half_size"][:2]):
            if abs(vel) < 1e-10:
                if abs(pos) > half + 1e-9:
                    low, high = 1., -1.
                    break
            else:
                limits = sorted([(-half - pos) / vel, (half - pos) / vel])
                low, high = max(low, limits[0]), min(high, limits[1])
        if low <= high:
            occupied.append((low, high))
    merged = []
    for low, high in sorted(occupied):
        if merged and low <= merged[-1][1] + 1e-8:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    free, start = [], domain[0]
    for low, high in merged:
        if low - start > 1e-8:
            free.append([start, low])
        start = max(start, high)
    if domain[1] - start > 1e-8:
        free.append([start, domain[1]])
    return free


def _field_region_lower_bound(sdf, origin, centers, half, radii):
    """Lower bound for every interpolated query in swept hand-target boxes.

    CAT interpolation is a convex combination of the eight adjacent grid
    values (even with its historical corner ordering). Taking all such grid
    values bounds the complete region, not merely its eight corner samples.
    """
    values = []
    for p, h, radius in zip(centers.reshape(-1, 3), half.reshape(-1, 3), radii.reshape(-1)):
        lower = np.floor((p - h - origin) / .04).astype(int)
        upper = np.floor((p + h - origin) / .04).astype(int) + 1
        if np.any(lower < 0) or np.any(upper >= np.array(sdf.shape)):
            return -float("inf")
        values.append(float(sdf[tuple(slice(a, b + 1) for a, b in zip(lower, upper))].min()) - float(radius))
    return min(values)


def _region_certificate(scene, sdf, origin, poses, zone):
    """Check whole target hand spheres throughout each zone's swept boxes."""
    from cat_ppo.furniture.body_collision_geometry import sphere_box_separation
    yaw = scene["generator"]["passage_yaw_rad"]
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    box_centers = np.array([box["center"] for box in scene["boxes"]])
    box_half = np.array([box["half_size"] for box in scene["boxes"]])
    box_rotation = np.array([[[math.cos(b["yaw"]), -math.sin(b["yaw"]), 0.],
                             [math.sin(b["yaw"]), math.cos(b["yaw"]), 0.], [0., 0., 1.]] for b in scene["boxes"]])
    minimum, maximum = np.array(zone["hand_regions_min"]), np.array(zone["hand_regions_max"])
    radii = _hands("nominal")[1]
    results = []
    for lo, hi in zip(minimum, maximum):
        centers = ((lo + hi) / 2) @ rotation.T
        centers = centers[None] + np.c_[poses[:, :2], np.zeros(len(poses))][:, None]
        # Include half the route sampling interval so the bound covers the
        # entire straight sweep between adjacent sampled root positions.
        half_local = (hi - lo) / 2 + [SAMPLE_SPACING / 2, 0., 0.]
        half_world = np.abs(rotation)[None] @ half_local[..., None]
        half_world = np.broadcast_to(half_world[..., 0], centers.shape)
        separation = np.asarray(sphere_box_separation(centers[:, :, None], radii[None, :, None],
            box_centers[None, None], box_rotation[None, None], box_half[None, None]))
        analytic = float((separation.min(axis=-1) - np.linalg.norm(half_local, axis=-1)[None]).min())
        field = _field_region_lower_bound(sdf, origin, centers, half_world,
                                         np.broadcast_to(radii, centers.shape[:-1]))
        results.append(dict(analytic_swept_hand_min_clearance_m=analytic,
                            field_swept_hand_min_clearance_m=field,
                            valid=bool(analytic > .005 and field > .025)))
    return results


def _floor_referenced_pose(mode, variant):
    """Kinematic stepping/crouching samples with at least one foot on its floor.

    Sagittal joint deltas sum to zero on each leg, preserving the foot pitch.
    Root height is then corrected using the lower of the two foot centers.
    These are stance robustness samples, not balanced or dynamically certified
    motions, and do not lower the root while leaving the legs unchanged.
    """
    import mujoco
    q = arm_pose(mode, .95 if mode == "raised" else 1.)
    model, compiled = _robot()
    feet = [compiled["shape_names"].index(f"{side}_upper_foot") for side in ("left", "right")]
    reference_z = float(_pose_geometry(q)["centers"][feet, 2].min())
    if variant.startswith("crouch"):
        amount = .18 if variant == "crouch_shallow" else .36
        deltas = [(-amount / 2, amount, -amount / 2)] * 2
    else:
        deltas = [(-.10, .12, -.02), (.08, -.05, -.03)]
        if variant == "step_right":
            deltas.reverse()
    for side, values in zip(("left", "right"), deltas):
        for joint, delta in zip(("hip_pitch", "knee", "ankle_pitch"), values):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}_joint")
            q[model.jnt_qposadr[jid]] += delta
    q[2] += reference_z - float(_pose_geometry(q)["centers"][feet, 2].min())
    return q


def _configuration_separations(scene, poses, qpos):
    """Same approved volume kernels as pose_separations, for a custom leg pose."""
    from cat_ppo.furniture.body_collision_geometry import (
        box_box_separation, capsule_box_separation, sphere_box_separation,
    )
    _, compiled = _robot()
    world = _world_geometry(_pose_geometry(qpos), poses)
    centers = np.array([b["center"] for b in scene["boxes"]])[None, None]
    half = np.array([b["half_size"] for b in scene["boxes"]])[None, None]
    rotations = np.array([[[math.cos(b["yaw"]), -math.sin(b["yaw"]), 0.],
                           [math.sin(b["yaw"]), math.cos(b["yaw"]), 0.], [0., 0., 1.]]
                          for b in scene["boxes"]])[None, None]
    result = np.empty((len(poses), len(compiled["shape_names"])))
    for kind, ids in compiled["indices"].items():
        center = world["centers"][:, ids][:, :, None]
        radius = compiled["radii"][ids][None, :, None]
        if kind == "sphere":
            values = sphere_box_separation(center, radius, centers, rotations, half)
        elif kind == "capsule":
            ends = world["endpoints"][:, ids]
            values = capsule_box_separation(ends[:, :, None, 0], ends[:, :, None, 1], radius,
                                             centers, rotations, half)
        else:
            values = box_box_separation(center, world["rotations"][:, ids][:, :, None],
                compiled["half_sizes"][ids][None, :, None], centers, rotations, half)
        result[:, ids] = np.asarray(values).min(axis=-1)
    hands = [compiled["shape_names"].index(f"{side}_hand") for side in ("left", "right")]
    return dict(body=result.min(axis=-1), hands=result[:, hands],
                hand_centers=world["centers"][:, hands], hand_radii=compiled["radii"][hands])


def certify_contrastive_scene(scene):
    """Validate static route, arm/heading transitions, role and target regions.

    The certificate binds both canonical geometry and route. Navigation radius
    is a guidance approximation only; all safe-path checks use all 35 volumes.
    """
    validate_scene(scene)
    contrast = scene["hand_contrast"]
    role = contrast["role"]
    if role not in ROLES:
        raise ValueError("Unknown contrastive role")
    cross_sections = [module_free_intervals(scene, module) for module in contrast["modules"]]
    for module, openings in zip(contrast["modules"], cross_sections):
        expected = [[-module["width_m"] / 2, module["width_m"] / 2]]
        if len(openings) != 1 or not np.allclose(openings, expected, atol=1e-7, rtol=0.):
            raise ValueError(f"Exterior bypass or wrong module gap: {openings}")
    sdf, origin = _sdf(scene)
    xs = np.linspace(-ROUTE_LIMIT, ROUTE_LIMIT, int(2 * ROUTE_LIMIT / SAMPLE_SPACING) + 1)
    route_rows, transition_rows, role_rows, region_rows, stance_rows = [], [], [], [], []
    route_segments = []

    def record(target, positions, mode="nominal", fraction=1.):
        samples = pose_separations(scene, positions, mode=mode, fraction=fraction)
        target.append(dict(count=len(positions), body=float(samples["body"].min()),
                           field=float(_hand_field_clearance(sdf, origin, samples).min())))
        return samples

    if role == "transition":
        for mask, mode, yaw in ((xs < -1.1, "raised", 0.),
                               ((xs >= -1.1) & (xs <= 1.1), "nominal", math.pi / 2),
                               (xs > 1.1, "raised", 0.)):
            record(route_rows, _poses(scene, xs[mask], yaw), mode, .95 if mode == "raised" else 1.)
            route_segments.append((_poses(scene, xs[mask], yaw), mode))
        arm_locations = [-ROUTE_LIMIT, -1.1, 1.1, ROUTE_LIMIT]
        turn_locations = [-1.1, 1.1]
    elif role == "narrow":
        record(route_rows, _poses(scene, xs, math.pi / 2))
        route_segments.append((_poses(scene, xs, math.pi / 2), "nominal"))
        arm_locations, turn_locations = [], [-ROUTE_LIMIT, ROUTE_LIMIT]
    else:
        record(route_rows, _poses(scene, xs), "raised" if role == "forward_protected" else "nominal", .95)
        route_segments.append((_poses(scene, xs), "raised" if role == "forward_protected" else "nominal"))
        arm_locations, turn_locations = ([-ROUTE_LIMIT, ROUTE_LIMIT] if role == "forward_protected" else []), []
    for fraction in np.linspace(0., .95, 17):
        if arm_locations:
            record(transition_rows, _poses(scene, arm_locations), "raised", float(fraction))
    for yaw in np.linspace(0., math.pi / 2, 25):
        if turn_locations:
            record(transition_rows, _poses(scene, turn_locations, float(yaw)))
    if not transition_rows:
        record(transition_rows, _poses(scene, [-ROUTE_LIMIT, ROUTE_LIMIT]))

    for variant in ("crouch_shallow", "crouch_deep", "step_left", "step_right"):
        for positions, mode in route_segments:
            samples = _configuration_separations(scene, positions, _floor_referenced_pose(mode, variant))
            stance_rows.append(dict(variant=variant, count=len(positions), body=float(samples["body"].min()),
                field=float(_hand_field_clearance(sdf, origin, samples).min())))

    for module, zone in zip(contrast["modules"], contrast["zones"]):
        focus = _poses(scene, [module["center_local_x_m"]])
        nominal = pose_separations(scene, focus)
        raised = pose_separations(scene, focus, mode="raised", fraction=.95)
        tucked = pose_separations(scene, focus, mode="tucked", fraction=.95)
        sideways = pose_separations(scene, _poses(scene, [module["center_local_x_m"]], math.pi / 2))
        row = dict(role=module["role"], width_m=module["width_m"],
            nominal_forward_body_min_m=float(nominal["body"].min()),
            nominal_forward_hand_min_m=float(nominal["hands"].min()),
            raised_forward_body_min_m=float(raised["body"].min()),
            tucked_forward_body_min_m=float(tucked["body"].min()),
            nominal_sideways_body_min_m=float(sideways["body"].min()))
        if module["role"] == "open" and row["nominal_forward_body_min_m"] <= .06:
            raise ValueError("Open contrast does not comfortably admit nominal forward posture")
        if module["role"] == "forward_protected" and not (row["nominal_forward_hand_min_m"] < -.002 and row["raised_forward_body_min_m"] > .03):
            raise ValueError("Protected contrast must block nominal hands and admit raised forward posture")
        if module["role"] == "narrow" and not (max(row["raised_forward_body_min_m"], row["tucked_forward_body_min_m"] ,row["nominal_forward_body_min_m"]) < -.005 and row["nominal_sideways_body_min_m"] > .03):
            raise ValueError("Narrow contrast must block all tested forward postures and admit sideways posture")
        role_rows.append(row)
        if any(zone["hand_active"]):
            local_xs = np.linspace(zone["start_m"] - ROUTE_LIMIT, zone["end_m"] - ROUTE_LIMIT,
                                   math.ceil((zone["end_m"] - zone["start_m"]) / SAMPLE_SPACING) + 1)
            regions = _region_certificate(scene, sdf, origin, _poses(scene, local_xs), zone)
            zone["region_valid"] = [item["valid"] for item in regions]
            if not zone["region_valid"][0]:
                raise ValueError(f"Raised target region lacks conservative clearance: {regions}")
            region_rows.append(regions)

    all_rows = route_rows + transition_rows + stance_rows
    body_min, field_min = min(row["body"] for row in all_rows), min(row["field"] for row in all_rows)
    if body_min <= .008 or field_min <= .025:
        raise ValueError(f"Static route or transition unsafe: body={body_min}, hand field={field_min}")
    if role == "open" and field_min <= .30:
        raise ValueError("Open contrast must restore the native neutral-arm gate above 0.30 m hand clearance")
    radius = float(contrast["navigation_radius_m"])
    navigation = _navigation_report(scene["route"], scene["boxes"], radius)
    if not .14 <= radius <= .23 or not navigation["root_route_validated"]:
        raise ValueError("Unvalidated navigation radius")
    return dict(schema="hand-contrast-certificate-v1", primitive_count=35,
        body_proxy_sha256=hashlib.sha256(PROPOSAL.read_bytes()).hexdigest(),
        geometry_hash=scene["geometry_hash"], route_hash=_digest(scene["route"]),
        navigation_radius_m=radius, route_transition_validated=True,
        exterior_lanes_sealed=True, module_floor_cross_section_openings_m=cross_sections,
        route_sample_count=sum(row["count"] for row in route_rows),
        transition_sample_count=sum(row["count"] for row in transition_rows),
        body_min_separation_m=body_min, hand_field_min_clearance_m=field_min,
        route_sample_spacing_m=SAMPLE_SPACING, voxel_size_m=.04,
        module_audit=role_rows, target_region_audit=region_rows,
        stance_sample_count=sum(row["count"] for row in stance_rows), stance_audit=stance_rows,
        target_region_method="hand-sphere analytic Lipschitz bound and all adjacent SDF grid values over swept region boxes",
        motion_audit="sampled standing postures, two floor-referenced crouches, two kinematic stepping stances, full straight route, arm interpolation and nominal-arm turn bays",
        target_region_full_arm_ik_certified=False, sampled_kinematic_stances_validated=True,
        dynamic_feasibility_validated=False, complete_se2_search=False,
        explanation="Static checked posture family and sampled transitions only; safe hand-target regions do not certify every inverse-kinematic arm solution.")
