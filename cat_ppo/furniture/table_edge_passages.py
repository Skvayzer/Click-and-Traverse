"""Matched table-edge passages with whole-hand clearance above the tabletop.

The paired scenes preserve the layout and change the gap. Actual tabletop,
apron and leg boxes are used by both rendering and training collision fields.
Certificates are sampled kinematic checks, not guarantees of dynamic success.
"""
from __future__ import annotations

import copy
import hashlib
import math
import random

import numpy as np

from cat_ppo.furniture.contrastive_passages import (
    MODULE_X, ROUTE_LIMIT, SAMPLE_SPACING, _configuration_separations,
    _floor_referenced_pose, _hands, _navigation_report, _poses,
    _region_certificate, _world_xy, generate_contrastive_group,
)
from cat_ppo.furniture.hand_passages import PROPOSAL, _hand_field_clearance, _sdf, pose_separations
from cat_ppo.furniture.scenes import SPLITS, _box, _digest, validate_scene

GENERATOR = "contrastive-table-edge-passages-v1"
ROLES = ("open", "forward_protected")


def module_cross_section_openings(scene, module, height):
    """Exact OBB opening intervals at an explicitly selected horizontal height.

    Tables are intentionally open underneath. The tabletop band closes the
    outside detour to a standing robot; this is not a floor-level solid wall.
    """
    yaw = scene["generator"]["passage_yaw_rad"]
    origin = _world_xy(scene, [[module["center_local_x_m"], 0.]])[0]
    direction = np.array([-math.sin(yaw), math.cos(yaw)])
    domain = sorted([(0. - origin[1]) / direction[1], (scene["room_dimensions"][1] - origin[1]) / direction[1]])
    occupied = []
    for box in scene["boxes"]:
        if not box["center"][2] - box["half_size"][2] <= height <= box["center"][2] + box["half_size"][2]:
            continue
        c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
        inverse = np.array([[c, s], [-s, c]])
        local, vector = inverse @ (origin - box["center"][:2]), inverse @ direction
        low, high = domain
        for pos, vel, half in zip(local, vector, box["half_size"][:2]):
            if abs(vel) < 1e-10:
                if abs(pos) > half + 1e-9:
                    low, high = 1., -1.
                    break
            else:
                a, b = sorted([(-half - pos) / vel, (half - pos) / vel])
                low, high = max(low, a), min(high, b)
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


def generate_table_edge_pair(seed, split="train", *, certify=True):
    """Return a matched open / protected pair with table-height obstacles."""
    if type(seed) is not int or seed < 0 or split not in SPLITS:
        raise ValueError("Expected a nonnegative integer seed and a known split")
    rng = random.Random(int(_digest([GENERATOR, split, seed]), 16))
    height = rng.uniform(.700, .720)
    gap = rng.uniform(.430, .444)
    group = f"table-contrast-{split}-{seed:06d}"
    scenes = generate_contrastive_group(seed, split, certify=False)[:2]
    for scene in scenes:
        contrast = scene["hand_contrast"]
        role = contrast["role"]
        yaw = scene["generator"]["passage_yaw_rad"]
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        center = np.array(scene["generator"]["passage_center_xy"])
        original = {box["name"]: box for box in scene["boxes"] if box["category"] == "cabinet"}
        boxes = [box for box in scene["boxes"] if box["category"] not in ("cabinet", "baffle")]
        for i, module in enumerate(contrast["modules"]):
            x = module["center_local_x_m"]
            width = module["width_m"] if role == "open" else gap
            module.update(width_m=width, tabletop_top_z_m=height)
            for side in (-1, 1):
                source = original[f"cabinet_{i}_{side:+d}"]
                depth = 2 * source["half_size"][1]
                table = f"table_{i}_{side:+d}"
                local_y = side * (width / 2 + depth / 2)
                def add(name, local, half, category):
                    xy = rotation @ np.array(local[:2]) + center
                    boxes.append(_box(name, [*xy, local[2]], half, category, yaw=yaw,
                                      furniture_id=table, furniture_type="table"))
                add(f"{table}_top", [x, local_y, height - .030], [.325, depth / 2, .030], "table_top")
                # Four visible 5 cm legs and a shallow under-top support apron.
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        add(f"{table}_leg_{sx:+d}_{sy:+d}",
                            [x + sx * .275, local_y + sy * (depth / 2 - .050), (height - .06) / 2],
                            [.025, .025, (height - .06) / 2], "table_leg")
                for sy in (-1, 1):
                    add(f"{table}_apron_{sy:+d}", [x, local_y + sy * (depth / 2 - .04), height - .095],
                        [.275, .018, .035], "table_apron")
                module_world_y = center[1] + s * x
                boundary_y = scene["room_dimensions"][1] if side > 0 else 0.
                edge = abs((boundary_y - module_world_y) / c) + .05
                inner = width / 2 + depth - .02
                xy = rotation @ np.array([x, side * (inner + edge) / 2]) + center
                boxes.append(_box(f"baffle_{i}_{side:+d}", [*xy, .9], [.325, (edge - inner) / 2, .9], "baffle", yaw=yaw))
        scene["boxes"] = boxes
        scene["geometry_hash"] = _digest(dict(boxes=boxes, room_dimensions=scene["room_dimensions"]))
        scene["scene_id"] = f"{group}-{role}-{scene['geometry_hash'][:12]}"
        scene["generator"]["name"] = GENERATOR
        scene["counts"] = dict(tables=6, chairs=0, generic_objects=6, primitive_boxes=len(boxes), bottlenecks=3)
        contrast.pop("cabinet_height_m", None)
        contrast.update(group_id=group, geometry_family="table_edges", tabletop_top_z_m=height,
                        table_gap_m=gap if role == "forward_protected" else contrast["modules"][0]["width_m"],
                        target_posture="raised_above_tabletop", navigation_radius_m=.15 if role == "forward_protected" else .23)
        for zone in contrast["zones"]:
            zone["region_valid"] = [role == "forward_protected", False]
        scene["feasibility"] = _navigation_report(scene["route"], boxes, contrast["navigation_radius_m"])
        scene["case_feasibility"] = [copy.deepcopy(scene["feasibility"])]
        validate_scene(scene)
        if certify:
            contrast["certificate"] = certify_table_edge_scene(scene)
        scene["source"] = dict(hand_contrast=copy.deepcopy(contrast))
    return scenes


def certify_table_edge_scene(scene):
    """Certify a sampled protected route and complete swept hand-target region."""
    validate_scene(scene)
    contrast = scene["hand_contrast"]
    role = contrast["role"]
    if contrast.get("geometry_family") != "table_edges" or role not in ROLES:
        raise ValueError("Expected a matched table-edge scene")
    height = float(contrast["tabletop_top_z_m"])
    tops = [box for box in scene["boxes"] if box["category"] == "table_top"]
    legs = [box for box in scene["boxes"] if box["category"] == "table_leg"]
    if len(tops) != 6 or len(legs) != 24 or any(abs(box["center"][2] + box["half_size"][2] - height) > 1e-8 for box in tops):
        raise ValueError("Table geometry must contain six equally high tabletops with four legs each")
    check_z = height - .03
    cross_sections = [module_cross_section_openings(scene, module, check_z) for module in contrast["modules"]]
    for module, openings in zip(contrast["modules"], cross_sections):
        if len(openings) != 1 or not np.allclose(openings, [[-module["width_m"] / 2, module["width_m"] / 2]], atol=1e-7, rtol=0.):
            raise ValueError(f"Unsealed tabletop-level exterior lane: {openings}")
    sdf, origin = _sdf(scene)
    xs = np.linspace(-ROUTE_LIMIT, ROUTE_LIMIT, int(2 * ROUTE_LIMIT / SAMPLE_SPACING) + 1)
    positions = _poses(scene, xs)
    mode = "raised" if role == "forward_protected" else "nominal"
    route_rows, transition_rows, stance_rows, role_rows, region_rows = [], [], [], [], []

    def record(target, poses, pose_mode, fraction=1.):
        values = pose_separations(scene, poses, mode=pose_mode, fraction=fraction)
        target.append(dict(count=len(poses), body=float(values["body"].min()),
                           field=float(_hand_field_clearance(sdf, origin, values).min())))
        return values

    route = record(route_rows, positions, mode, .95)
    if role == "forward_protected":
        for fraction in np.linspace(0., .95, 17):
            record(transition_rows, _poses(scene, [-ROUTE_LIMIT, ROUTE_LIMIT]), "raised", float(fraction))
    else:
        record(transition_rows, _poses(scene, [-ROUTE_LIMIT, ROUTE_LIMIT]), "nominal")
    above_rows = []
    for variant in ("crouch_shallow", "crouch_deep", "step_left", "step_right"):
        values = _configuration_separations(scene, positions, _floor_referenced_pose(mode, variant))
        stance_rows.append(dict(variant=variant, count=len(positions), body=float(values["body"].min()),
                               field=float(_hand_field_clearance(sdf, origin, values).min())))
        if role == "forward_protected":
            above_rows.append(dict(variant=variant,
                above_top_clearance_m=(values["hand_centers"][:, :, 2] - values["hand_radii"][None] - height).min(axis=0).tolist()))
    for module, zone in zip(contrast["modules"], contrast["zones"]):
        focus = _poses(scene, [module["center_local_x_m"] - .20])
        nominal, raised = pose_separations(scene, focus), pose_separations(scene, focus, mode="raised", fraction=.95)
        tucked = pose_separations(scene, focus, mode="tucked", fraction=.95)
        row = dict(role=role, width_m=module["width_m"], nominal_forward_body_min_m=float(nominal["body"].min()),
                   nominal_forward_hand_min_m=float(nominal["hands"].min()), raised_forward_body_min_m=float(raised["body"].min()),
                   tucked_forward_body_min_m=float(tucked["body"].min()), tucked_forward_hand_min_m=float(tucked["hands"].min()),
                   above_top_clearance_m=(raised["hand_centers"][0, :, 2] - raised["hand_radii"] - height).tolist(),
                   raised_hand_sphere_lateral_overhang_m=(np.abs(_hands("raised", .95)[0][:, 1]) + _hands("raised", .95)[1] - module["width_m"] / 2).tolist())
        if role == "forward_protected" and not (row["nominal_forward_hand_min_m"] < -.002 and row["tucked_forward_hand_min_m"] < -.002 and row["raised_forward_body_min_m"] > .03 and min(row["above_top_clearance_m"]) > .08 and min(row["raised_hand_sphere_lateral_overhang_m"]) > .004):
            raise ValueError(f"Protected table must intersect nominal hands and clear raised hand spheres: {row}")
        if role == "open" and row["nominal_forward_body_min_m"] <= .06:
            raise ValueError("Open table pair does not admit nominal forward posture")
        role_rows.append(row)
        if any(zone["hand_active"]):
            local_xs = np.linspace(zone["start_m"] - ROUTE_LIMIT, zone["end_m"] - ROUTE_LIMIT,
                                   math.ceil((zone["end_m"] - zone["start_m"]) / SAMPLE_SPACING) + 1)
            regions = _region_certificate(scene, sdf, origin, _poses(scene, local_xs), zone)
            target_bottom = np.array(zone["hand_regions_min"])[0, :, 2] - _hands("raised")[1]
            regions[0]["above_top_clearance_m"] = (target_bottom - height).tolist()
            if not regions[0]["valid"] or min(regions[0]["above_top_clearance_m"]) <= .06:
                raise ValueError(f"Raised hand-target region does not clear tabletop: {regions}")
            # The tucked template remains present for a fixed metadata shape,
            # but cannot satisfy this task's required above-table hand posture.
            regions[1]["valid"] = False
            regions[1]["disabled_reason"] = "Tucked hands remain below the tabletop"
            zone["region_valid"] = [True, False]
            region_rows.append(regions)
    all_rows = route_rows + transition_rows + stance_rows
    body_min, field_min = min(r["body"] for r in all_rows), min(r["field"] for r in all_rows)
    if body_min <= .008 or field_min <= .025:
        raise ValueError(f"Unsafe table route/transition: body={body_min}, field={field_min}")
    if role == "open" and field_min <= .30:
        raise ValueError("Open pair must restore the native neutral-arm gate at >.30 m")
    if role == "forward_protected" and min(min(r["above_top_clearance_m"]) for r in above_rows) <= .05:
        raise ValueError("Sampled floor-referenced stances lower hands too close to tabletop")
    if not _navigation_report(scene["route"], scene["boxes"], contrast["navigation_radius_m"])["root_route_validated"]:
        raise ValueError("Unvalidated navigation radius")
    return dict(schema="hand-contrast-certificate-v1", primitive_count=35,
        body_proxy_sha256=hashlib.sha256(PROPOSAL.read_bytes()).hexdigest(),
        geometry_hash=scene["geometry_hash"], route_hash=_digest(scene["route"]),
        navigation_radius_m=contrast["navigation_radius_m"], route_transition_validated=True,
        exterior_lanes_sealed=True, exterior_lane_seal_height_m=check_z,
        exterior_lane_seal_scope="Tabletop-band cross-section, not floor-level: no standing exterior bypass; crawling is not exhaustively excluded",
        module_tabletop_cross_section_openings_m=cross_sections,
        tabletop_geometry_validated=True, tabletop_top_z_m=height,
        raised_hand_above_table_min_m=min(min(row["above_top_clearance_m"]) for row in role_rows),
        raised_hand_sphere_lateral_overhang_min_m=min(min(row["raised_hand_sphere_lateral_overhang_m"]) for row in role_rows),
        above_top_clearance_m=(route["hand_centers"][:, :, 2] - route["hand_radii"][None] - height).min(axis=0).tolist(),
        route_sample_count=sum(r["count"] for r in route_rows), transition_sample_count=sum(r["count"] for r in transition_rows),
        body_min_separation_m=body_min, hand_field_min_clearance_m=field_min,
        route_sample_spacing_m=SAMPLE_SPACING, voxel_size_m=.04,
        module_audit=role_rows, target_region_audit=region_rows, above_table_stance_audit=above_rows,
        stance_sample_count=sum(r["count"] for r in stance_rows), stance_audit=stance_rows,
        target_region_method="hand-sphere analytic Lipschitz bound and all adjacent SDF grid values over swept region boxes; lowest target sphere surface above tabletop",
        motion_audit="sampled standing posture, two floor-referenced crouches, two stepping stances, full straight route and arm interpolation at clear endpoints",
        target_region_full_arm_ik_certified=False, sampled_kinematic_stances_validated=True,
        dynamic_feasibility_validated=False, complete_se2_search=False,
        explanation="Static raised-hand route; open under-table spaces are physically modelled. No exhaustive crawl or dynamic feasibility guarantee.")
