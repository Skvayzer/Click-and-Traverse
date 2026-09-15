"""Irregular, seeded furniture rooms with an admitted root-cylinder route.

Objects are packed at continuous random XY positions and yaw angles. A route is
searched *after* packing; no grid of tables or pre-cleared corridor is imposed.
The route certificate covers a 23 cm root cylinder, not whole-body dynamics.
"""

from __future__ import annotations

import copy
import heapq
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from cat_ppo.furniture.scenes import (
    DIFFICULTIES, ROOT_CLEARANCE_RADIUS, ROOT_CLEARANCE_Z, SCHEMA, SPLITS,
    _box, _case, _chair, _digest, _horizontal_distance, _root_obstacles,
    _route_clearance, _table, _walls, validate_scene,
)


GENERATOR = "continuous-random-packed-rooms-v1"
KINDS = ("furniture", "generic_clutter")
GRAPH_SPACING_M = .08
# Each graph edge has a continuous clearance certificate from the distance
# field's 1-Lipschitz property, including diagonal edges.
GRAPH_CLEARANCE_M = ROOT_CLEARANCE_RADIUS + GRAPH_SPACING_M / math.sqrt(2) + .012
RESET_XY_JITTER_M = .08
# Randomized CAT joint resets can extend a hand sphere ~0.5033 m from the
# root, substantially beyond the route planner's 0.23 m root cylinder. Admit
# endpoints in a wider pocket while retaining the same packing and route graph.
# scripts/audit_room_resets.py checks the compiled robot, reset samples and arm
# parameter grid. This is an endpoint margin, not a whole-body route certificate.
RESET_HAND_ENVELOPE_RADIUS_M = .52
RESET_FIELD_MARGIN_M = .08
RESET_CENTER_CLEARANCE_M = math.ceil(100 * (
    RESET_HAND_ENVELOPE_RADIUS_M + math.sqrt(2) * RESET_XY_JITTER_M + RESET_FIELD_MARGIN_M)) / 100


def transform_object(parts: list[dict[str, Any]], x: float, y: float,
                     yaw: float) -> list[dict[str, Any]]:
    """Rigidly transform every primitive of a local-origin object, including legs."""
    result = copy.deepcopy(parts)
    c, s = math.cos(yaw), math.sin(yaw)
    for part in result:
        px, py = part["center"][:2]
        part["center"][:2] = [round(x + c * px - s * py, 9),
                              round(y + s * px + c * py, 9)]
        part["yaw"] = round(part["yaw"] + yaw, 9)
    return result


def _footprint(parts: list[dict[str, Any]]) -> tuple[float, float]:
    # Local-origin objects are axis aligned. This also encloses chair armrests.
    return tuple(max(abs(p["center"][axis]) + p["half_size"][axis]
                     for p in parts) for axis in (0, 1))


def _overlap(first, second, gap=.025):
    """Separating-axis test on conservative complete-object XY footprints."""
    ax, ay, ahx, ahy, aa = first
    bx, by, bhx, bhy, ba = second
    axes_a = ((math.cos(aa), math.sin(aa)), (-math.sin(aa), math.cos(aa)))
    axes_b = ((math.cos(ba), math.sin(ba)), (-math.sin(ba), math.cos(ba)))
    delta = (bx - ax, by - ay)
    for ux, uy in (*axes_a, *axes_b):
        distance = abs(delta[0] * ux + delta[1] * uy)
        ra = ahx * abs(axes_a[0][0] * ux + axes_a[0][1] * uy)
        ra += ahy * abs(axes_a[1][0] * ux + axes_a[1][1] * uy)
        rb = bhx * abs(axes_b[0][0] * ux + axes_b[0][1] * uy)
        rb += bhy * abs(axes_b[1][0] * ux + axes_b[1][1] * uy)
        if distance >= ra + rb + gap:
            return False
    return True


def _generic_parts(name, shape, width, depth, height, identity):
    common = dict(object_id=name, object_type=shape, shape_identity=identity)
    if shape != "shelf":
        return [_box(name, [0, 0, height / 2], [width / 2, depth / 2, height / 2],
                     shape, **common)]
    parts = [_box(name + f"_board_{index}", [0, 0, z],
                  [width / 2, depth / 2, .025], "shelf_edge", **common)
             for index, z in enumerate((height * .45, height - .025))]
    parts += [_box(name + f"_support_{side}",
                   [side * (width / 2 - .03), 0, height / 2],
                   [.03, depth / 2, height / 2], "support", **common)
              for side in (-1, 1)]
    return parts


def _pack_objects(rng, dimensions, kind, difficulty, split):
    if difficulty == "open_floor":
        return _walls(dimensions), [], dict(tables=0, chairs=0, generic_objects=0)
    dense = difficulty == "dense"
    large_count = rng.randint(8, 12) if dense else rng.randint(2, 4)
    small_count = rng.randint(25, 45) if dense else rng.randint(5, 9)
    specifications = []
    for index in range(large_count + small_count):
        large = index < large_count
        identity = f"{split}/{kind}/profile/{index:03d}"
        if kind == "furniture":
            if large:
                width, depth = rng.uniform(1.25, 1.95), rng.uniform(.65, .98)
                height = rng.uniform(.69, .87)
                name, shape = f"table_{index:02d}", "table"
                parts = _table(name, 0., 0., width, depth, height, identity)
            else:
                name, shape = f"chair_{index - large_count:02d}", "chair"
                width, height = rng.uniform(.38, .52), rng.uniform(.81, 1.10)
                parts = _chair(name, 0., 0., 0., width, height, rng.random() < .40, identity)
        else:
            name = f"clutter_{index:02d}"
            shape = ("crate", "low_block", "partition", "shelf")[index % 4]
            width = rng.uniform(1.1, 1.8) if large else rng.uniform(.38, .72)
            depth = rng.uniform(.55, .9) if large else rng.uniform(.30, .58)
            if shape == "partition":
                width, depth = rng.uniform(.9, 1.65), rng.uniform(.10, .22)
            height = rng.uniform(*{"crate": (.61, .99), "low_block": (.39, .65),
                                  "partition": (1.0, 1.55), "shelf": (.74, 1.18)}[shape])
            parts = _generic_parts(name, shape, width, depth, height, identity)
        specifications.append((parts, name, shape, identity))

    # Largest footprint first improves packing while every pose is still sampled
    # independently and continuously. No row, table-chair grouping or route mask.
    specifications.sort(key=lambda entry: math.prod(_footprint(entry[0])), reverse=True)
    boxes, accepted, profiles = _walls(dimensions), [], []
    for parts, name, shape, identity in specifications:
        hx, hy = _footprint(parts)
        for _ in range(1600):
            yaw = rng.uniform(-math.pi, math.pi)
            c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
            ex, ey = c * hx + s * hy, s * hx + c * hy
            x = rng.uniform(.105 + ex, dimensions[0] - .105 - ex)
            y = rng.uniform(.105 + ey, dimensions[1] - .105 - ey)
            footprint = (x, y, hx, hy, yaw)
            if any(_overlap(footprint, other) for other in accepted):
                continue
            accepted.append(footprint)
            boxes.extend(transform_object(parts, x, y, yaw))
            profiles.append(dict(identity=identity, object_id=name, kind=shape,
                                 center_xy_m=[x, y], yaw_rad=yaw,
                                 conservative_half_footprint_m=[hx, hy]))
            break
        else:
            return None
    counts = dict(tables=large_count if kind == "furniture" else 0,
                  chairs=small_count if kind == "furniture" else 0,
                  generic_objects=len(accepted) if kind == "generic_clutter" else 0)
    return boxes, profiles, counts


def _clearance_grid(boxes, dimensions):
    axes = [np.arange(.04, dimensions[i], GRAPH_SPACING_M) for i in (0, 1)]
    x, y = np.meshgrid(*axes, indexing="ij", sparse=True)
    clearance = np.full((len(axes[0]), len(axes[1])), np.inf)
    for box in _root_obstacles(boxes):
        c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
        dx, dy = x - box["center"][0], y - box["center"][1]
        qx = np.abs(c * dx + s * dy) - box["half_size"][0]
        qy = np.abs(-s * dx + c * dy) - box["half_size"][1]
        distance = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
        np.minimum(clearance, distance, out=clearance)
    return axes, clearance


def _search_path(start, goal, free, clearance):
    """A* with diagonal corner cutting disabled and a small clearance preference."""
    costs = {start: 0.}
    previous = {}
    queue = [(math.dist(start, goal), 0., start)]
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost != costs[current]:
            continue
        if current == goal:
            path = [current]
            while current in previous:
                current = previous[current]
                path.append(current)
            return list(reversed(path))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (-1, 1), (1, -1), (-1, -1)):
            nx, ny = current[0] + dx, current[1] + dy
            if not (0 <= nx < free.shape[0] and 0 <= ny < free.shape[1] and free[nx, ny]):
                continue
            if dx and dy and not (free[nx, current[1]] and free[current[0], ny]):
                continue
            step = math.hypot(dx, dy) * (1 + .025 / max(clearance[nx, ny] - ROOT_CLEARANCE_RADIUS, .01))
            candidate = cost + step
            if candidate >= costs.get((nx, ny), math.inf):
                continue
            costs[nx, ny] = candidate
            previous[nx, ny] = current
            heapq.heappush(queue, (candidate + math.dist((nx, ny), goal), candidate, (nx, ny)))
    return None


def _simplify_path(route, obstacles):
    # Keep turns, then greedily shortcut at most 1.2 m. Every replacement segment
    # is analytically checked at 4 cm; smoothing cannot cut through furniture.
    corners = [route[0]]
    for index in range(1, len(route) - 1):
        before = np.asarray(route[index]) - route[index - 1]
        after = np.asarray(route[index + 1]) - route[index]
        if abs(before[0] * after[1] - before[1] * after[0]) > 1e-8 or np.dot(before, after) <= 0:
            corners.append(route[index])
    corners.append(route[-1])
    result, index = [corners[0]], 0
    while index < len(corners) - 1:
        selected = index + 1
        for candidate in range(index + 2, min(index + 10, len(corners))):
            if math.dist(corners[index], corners[candidate]) > 1.2:
                break
            if _route_clearance([corners[index], corners[candidate]], obstacles)["root_margin_lower_bound_m"] >= .015:
                selected = candidate
        result.append(corners[selected])
        index = selected
    return result


def _admit_routes(rng, boxes, dimensions):
    from scipy import ndimage

    axes, clearance = _clearance_grid(boxes, dimensions)
    free = clearance >= GRAPH_CLEARANCE_M
    labels, nlabels = ndimage.label(free)
    if not nlabels:
        return None
    # Four-connected components deliberately exclude disconnected corner cells.
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    main_label = int(sizes.argmax())
    candidates = np.argwhere((labels == main_label) & (clearance >= RESET_CENTER_CLEARANCE_M))
    if len(candidates) < 10:
        return None
    xy = np.column_stack([axes[0][candidates[:, 0]], axes[1][candidates[:, 1]]])
    cases = []
    order = [0, 1]
    rng.shuffle(order)
    for axis in order:
        other = 1 - axis
        # Start and goal lie on opposing interior bands, never in a reserved aisle.
        interior = (xy[:, other] > .15 * dimensions[other]) & (xy[:, other] < .85 * dimensions[other])
        low = np.flatnonzero(interior & (xy[:, axis] < .25 * dimensions[axis]) & (xy[:, axis] > .10 * dimensions[axis]))
        high = np.flatnonzero(interior & (xy[:, axis] > .75 * dimensions[axis]) & (xy[:, axis] < .90 * dimensions[axis]))
        if not len(low) or not len(high):
            continue
        for _ in range(5):
            start_index = tuple(int(i) for i in candidates[rng.choice(low.tolist())])
            goal_index = tuple(int(i) for i in candidates[rng.choice(high.tolist())])
            path = _search_path(start_index, goal_index, free, clearance)
            if path is None:
                continue
            route = [[round(float(axes[0][i]), 8), round(float(axes[1][j]), 8)] for i, j in path]
            route = _simplify_path(route, boxes)
            admission = _route_clearance(route, boxes)
            if not admission["root_route_validated"]:
                continue
            cases.append(_case(route, []))
            cases.append(_case(list(reversed(route)), []))
            break
    if not cases:
        return None
    metrics = dict(graph_spacing_m=GRAPH_SPACING_M,
                   graph_center_clearance_m=GRAPH_CLEARANCE_M,
                   root_height_occupied_area_fraction=float(np.mean(clearance <= 0)),
                   root_inflated_blocked_area_fraction=float(np.mean(clearance < ROOT_CLEARANCE_RADIUS)),
                   traversable_component_count=int(nlabels),
                   largest_component_free_cell_fraction=float(sizes[main_label] / max(1, free.sum())),
                   route_stretch=cases[0]["route_length_m"] / math.dist(cases[0]["start"][:2], cases[0]["goal"]))
    return cases, metrics


def generate_random_room(seed: int, split: str = "train", kind: str = "furniture",
                         difficulty: str = "dense") -> dict[str, Any]:
    """Return a deterministic canonical scene, accepted only with a root route.

    Split names enter the RNG key, so identical numeric seeds in different splits
    produce different layouts. Evaluation seeds should also be held out by the
    dataset manifest. Admission conditions the uniform pose distribution: this
    is random *feasible* clutter, not unconstrained independent overlapping boxes.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if split not in SPLITS or kind not in KINDS or difficulty not in DIFFICULTIES:
        raise ValueError(f"Expected split in {SPLITS}, kind in {KINDS}, difficulty in {DIFFICULTIES}")
    rng = random.Random(int(_digest([GENERATOR, seed, split, kind, difficulty]), 16))
    for attempt in range(40):
        if difficulty == "dense":
            dimensions = [round(rng.uniform(8.6, 10.2), 3), round(rng.uniform(9., 11.), 3), 2.4]
        else:
            dimensions = [round(rng.uniform(5., 6.5), 3), round(rng.uniform(5., 6.5), 3), 2.4]
        packed = _pack_objects(rng, dimensions, kind, difficulty, split)
        if packed is None:
            continue
        boxes, profiles, counts = packed
        admitted = _admit_routes(rng, boxes, dimensions)
        if admitted is None:
            continue
        cases, metrics = admitted
        break
    else:
        raise RuntimeError(f"No connected randomly packed room for seed {seed} after 40 attempts")
    geometry_hash = _digest(dict(boxes=boxes, room_dimensions=dimensions))
    scene = dict(schema=SCHEMA, scene_id=f"random-{kind}-{difficulty}-{split}-{seed:06d}-{geometry_hash[:12]}",
                 seed=seed, split=split, family=kind, difficulty=difficulty,
                 units="metres", coordinate_system="right-handed-z-up", boxes=boxes,
                 room_dimensions=dimensions, geometry_hash=geometry_hash,
                 start_goals=cases, goal_index=0, **copy.deepcopy(cases[0]))
    scene["counts"] = dict(counts, primitive_boxes=len(boxes), bottlenecks=0)
    scene["generator"] = dict(name=GENERATOR,
        description="Continuous independent random object poses, non-overlapping packing, then root-route admission",
        geometry_source="canonical-oriented-boxes", floor_in_obstacle_field=False,
        shape_profiles=profiles, split_holdout=dict(kind="split-keyed-independent-layouts-and-explicit-manifest-seeds",
            real_object_identity_holdout=False),
        pose_distribution=dict(xy="uniform inside room conditional on non-overlap and route admission",
                               yaw="uniform [-pi, pi) per complete object"),
        route_topology="searched in final random geometry; no reserved corridor or grid",
        route_search="8-neighbor A* without diagonal corner cutting on conservatively inflated root occupancy",
        packing_attempt=attempt, full_body_route_certificate=False,
        semantic_labels_observed_by_policy=False,
        time_budget_method="route_length/0.40_m_per_s + 20_s; proposed benchmark budget")
    scene["layout_metrics"] = metrics
    scene["feasibility"] = _route_clearance(scene["route"], boxes)
    scene["case_feasibility"] = [_route_clearance(case["route"], boxes) for case in cases]
    center_clearance = min(_horizontal_distance(scene["start"][:2], box) for box in _root_obstacles(boxes))
    scene["reset_clearance"] = dict(root_radius_m=ROOT_CLEARANCE_RADIUS,
        root_z_interval_m=list(ROOT_CLEARANCE_Z), center_clearance_m=center_clearance,
        xy_jitter_half_extent_m=[RESET_XY_JITTER_M] * 2,
        worst_jitter_root_margin_lower_bound_m=center_clearance - ROOT_CLEARANCE_RADIUS - math.sqrt(2) * RESET_XY_JITTER_M,
        endpoint_admission="randomized-hand-envelope-margin-v2",
        required_center_clearance_m=RESET_CENTER_CLEARANCE_M,
        reset_hand_envelope_radius_m=RESET_HAND_ENVELOPE_RADIUS_M,
        field_discretization_margin_m=RESET_FIELD_MARGIN_M,
        root_cylinder_validated=True, full_body_validated=False,
        runtime_nominal_pose_collision_check_required=True)
    validate_scene(scene)
    return scene


def render_room_overview(scene: dict[str, Any], output: str | Path,
                         size: int = 1100) -> Path:
    """Write a compact top-view image of canonical rotated boxes and admitted A–B route."""
    from PIL import Image, ImageDraw, ImageFont

    validate_scene(scene)
    dimensions = scene["room_dimensions"]
    scale = (size - 100) / max(dimensions[:2])
    image = Image.new("RGB", (size, size + 70), "#f8fafc")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    title = f"{scene['counts']['tables']} tables + {scene['counts']['chairs']} chairs" if scene["family"] == "furniture" else f"{scene['counts']['generic_objects']} irregular clutter objects"
    draw.text((35, 20), title, fill="#182635", font=font)
    def pixel(point):
        return (50 + point[0] * scale, size + 35 - point[1] * scale)
    colors = {"wall": "#566270", "tabletop": "#cfad79", "table_leg": "#9e7f56",
              "chair_seat": "#7e9bad", "chair_back": "#597385", "chair_armrest": "#597385",
              "chair_leg": "#597385", "crate": "#bd9c7d", "low_block": "#889cab",
              "partition": "#69717e", "shelf_edge": "#bf9b72", "support": "#8f7252"}
    for box in sorted(scene["boxes"], key=lambda item: item["center"][2]):
        x, y = box["center"][:2]
        hx, hy = box["half_size"][:2]
        c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
        polygon = [pixel((x + c * sx * hx - s * sy * hy, y + s * sx * hx + c * sy * hy))
                   for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        draw.polygon(polygon, fill=colors.get(box["category"], "#8796a5"), outline="#566270")
    points = [pixel(point) for point in scene["route"]]
    draw.line(points, fill="#178962", width=5)
    for label, point in (("A", points[0]), ("B", points[-1])):
        draw.ellipse((point[0]-13, point[1]-13, point[0]+13, point[1]+13), fill="#178962")
        draw.text((point[0]+16, point[1]-13), label, fill="#075c40", font=font)
    draw.text((35, size+38), "Route checked for a 23 cm root radius; whole-body dynamics require evaluation.", fill="#384651", font=ImageFont.load_default(size=17))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output
