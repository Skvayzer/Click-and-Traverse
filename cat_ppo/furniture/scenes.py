"""Deterministic primitive furniture rooms and matching CAT potential fields.

``scene.json`` is the canonical geometry, in metres with z up.  The collision
adapter and the numeric fields consume the same oriented boxes.  The floor is
separate: it is deliberately absent from the obstacle SDF.  This generator
checks a small root cylinder along a supplied route; it does not certify a
robot's full body, dynamic feasibility, or finger safety.

Furniture guidance is a route lookahead with boundary projection. Adapted
legacy CAT scenes use the upstream three-dimensional fast-marching guidance.
Every bundle records that distinction. Furniture fields require only NumPy;
legacy fields additionally require scikit-fmm. Layout generation is stdlib-only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Iterable


SCHEMA = "cat-furniture-scene-v1"
FAMILIES = ("table", "chair", "slalom", "asymmetric", "overhead", "turn")
SPLITS = ("train", "validation", "test")
DIFFICULTIES = ("open_floor", "pilot", "dense")
GUIDANCE_METHOD = "route-lookahead-with-boundary-projection-v1"
ROOT_CLEARANCE_RADIUS = 0.23
ROOT_CLEARANCE_Z = (0.35, 1.05)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box(name: str, center: Iterable[float], half_size: Iterable[float],
         category: str, yaw: float = 0.0, **extra: Any) -> dict[str, Any]:
    return dict(name=name, center=[round(float(v), 9) for v in center],
                half_size=[round(float(v), 9) for v in half_size],
                yaw=round(float(yaw), 9), category=category, **extra)


def _walls(dimensions: list[float]) -> list[dict[str, Any]]:
    x, y, z = dimensions
    return [
        _box("wall_west", [0, y / 2, z / 2], [.06, y / 2, z / 2], "wall"),
        _box("wall_east", [x, y / 2, z / 2], [.06, y / 2, z / 2], "wall"),
        _box("wall_south", [x / 2, 0, z / 2], [x / 2, .06, z / 2], "wall"),
        _box("wall_north", [x / 2, y, z / 2], [x / 2, .06, z / 2], "wall"),
    ]


def _table(name: str, x: float, y: float, width: float, depth: float,
           height: float, identity: str) -> list[dict[str, Any]]:
    common = dict(furniture_id=name, furniture_type="table", shape_identity=identity)
    thickness, leg = .055, .045
    result = [_box(name + "_top", [x, y, height - thickness / 2],
                   [width / 2, depth / 2, thickness / 2], "tabletop", **common)]
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        result.append(_box(name + f"_leg_{i}",
                           [x + sx * (width / 2 - .095),
                            y + sy * (depth / 2 - .085), (height - thickness) / 2],
                           [leg, leg, (height - thickness) / 2], "table_leg", **common))
    return result


def _chair(name: str, x: float, y: float, yaw: float, width: float,
           back_height: float, armrests: bool, identity: str) -> list[dict[str, Any]]:
    common = dict(furniture_id=name, furniture_type="chair", shape_identity=identity)
    c, s = math.cos(yaw), math.sin(yaw)

    def part(suffix: str, center: list[float], half: list[float], category: str):
        px, py, pz = center
        return _box(name + suffix, [x + c * px - s * py, y + s * px + c * py, pz],
                    half, category, yaw=yaw, **common)

    result = [part("_seat", [0, 0, .45], [width / 2, .20, .028], "chair_seat"),
              part("_back", [0, -.20, (.48 + back_height) / 2],
                   [width / 2, .025, (back_height - .48) / 2], "chair_back")]
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        result.append(part(f"_leg_{i}", [sx * (width / 2 - .045), sy * .155, .211],
                           [.023, .023, .211], "chair_leg"))
    if armrests:
        for side in (-1, 1):
            result.append(part(f"_arm_{side:+d}", [side * (width / 2 + .015), 0, .66],
                               [.025, .20, .025], "chair_armrest"))
    return result


def _route_lengths(route: list[list[float]]) -> tuple[list[float], float]:
    cumulative = [0.0]
    for first, second in zip(route, route[1:]):
        length = math.dist(first, second)
        if length <= 1e-9:
            raise ValueError("Route segments must have positive length")
        cumulative.append(cumulative[-1] + length)
    return cumulative, cumulative[-1]


def _project_route(point: Iterable[float], route: list[list[float]]) -> tuple[float, float]:
    """Return distance to route and progress along it (metres)."""
    x, y = point
    cumulative, _ = _route_lengths(route)
    best = (float("inf"), 0.0)
    for i, (first, second) in enumerate(zip(route, route[1:])):
        dx, dy = second[0] - first[0], second[1] - first[1]
        length_sq = dx * dx + dy * dy
        t = min(1.0, max(0.0, ((x - first[0]) * dx + (y - first[1]) * dy) / length_sq))
        distance = math.hypot(x - first[0] - t * dx, y - first[1] - t * dy)
        candidate = (distance, cumulative[i] + t * math.sqrt(length_sq))
        if candidate[0] < best[0]:
            best = candidate
    return best


def _horizontal_distance(point: Iterable[float], box: dict[str, Any]) -> float:
    x, y = point
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    dx, dy = x - box["center"][0], y - box["center"][1]
    qx = abs(c * dx + s * dy) - box["half_size"][0]
    qy = abs(-s * dx + c * dy) - box["half_size"][1]
    return math.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0)


def _root_obstacles(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    low, high = ROOT_CLEARANCE_Z
    return [box for box in boxes
            if box["center"][2] + box["half_size"][2] >= low
            and box["center"][2] - box["half_size"][2] <= high]


def _route_clearance(route: list[list[float]], boxes: list[dict[str, Any]]) -> dict[str, Any]:
    """A conservative Lipschitz lower bound between route samples, not robot IK."""
    obstacles = _root_obstacles(boxes)
    spacing = .04
    minimum = float("inf")
    max_spacing = 0.0
    for first, second in zip(route, route[1:]):
        length = math.dist(first, second)
        count = max(1, math.ceil(length / spacing))
        max_spacing = max(max_spacing, length / count)
        for k in range(count + 1):
            point = [first[j] + (second[j] - first[j]) * k / count for j in range(2)]
            minimum = min(minimum, min(_horizontal_distance(point, box) for box in obstacles))
    lower_bound = minimum - max_spacing / 2
    return dict(method="analytic-obb-root-cylinder-route-samples-with-lipschitz-bound",
                root_radius_m=ROOT_CLEARANCE_RADIUS, root_z_interval_m=list(ROOT_CLEARANCE_Z),
                sample_spacing_max_m=max_spacing, sampled_center_clearance_min_m=minimum,
                center_clearance_lower_bound_m=lower_bound,
                root_margin_lower_bound_m=lower_bound - ROOT_CLEARANCE_RADIUS,
                root_route_validated=lower_bound >= ROOT_CLEARANCE_RADIUS,
                full_body_validated=False, dynamic_feasibility_validated=False,
                explanation="Root cylinder only; limb clearance, crouching and tracking remain unvalidated.")


def _case(route: list[list[float]], bottlenecks: list[dict[str, Any]]) -> dict[str, Any]:
    route = [[round(float(v), 8) for v in p] for p in route]
    _, length = _route_lengths(route)
    gates = copy.deepcopy(bottlenecks)
    for gate in gates:
        _, progress = _project_route(gate["center"], route)
        gate["route_distance_m"] = round(progress, 8)
        gate["route_progress_fraction"] = round(progress / length, 8)
    gates.sort(key=lambda gate: gate["route_distance_m"])
    yaw = math.atan2(route[1][1] - route[0][1], route[1][0] - route[0][0])
    return dict(start=route[0] + [yaw], goal=list(route[-1]), route=route, bottlenecks=gates,
                route_length_m=length, time_budget=round(length / .40 + 20.0, 3))


def generate_scene(seed: int = 0, split: str = "train", family: str = "mixed",
                   difficulty: str = "dense") -> dict[str, Any]:
    """Build a deterministic layout with split-disjoint primitive shape profiles.

    Dense rooms contain nine tables and 36 chairs.  Split holdout concerns these
    synthetic shape profiles and layout seeds, not real furniture categories.
    Three start/goal cases share the exact same geometry.
    """
    difficulty = difficulty.replace("-", "_")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if family not in (*FAMILIES, "mixed"):
        raise ValueError(f"family must be one of {(*FAMILIES, 'mixed')}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = random.Random(int(_digest([split, family, difficulty, seed]), 16))
    width_ranges = {"train": (2.00, 2.04), "validation": (2.075, 2.105), "test": (2.14, 2.18)}
    chair_width_ranges = {"train": (.400, .410), "validation": (.425, .435), "test": (.450, .460)}
    width_range = width_ranges[split]
    chair_range = chair_width_ranges[split]
    dimensions = [9.0, 9.4, 2.4] if difficulty == "dense" else [4.0, 3.0, 2.4]
    boxes = _walls(dimensions)
    bottlenecks: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []

    if difficulty == "dense":
        first_row = rng.uniform(2.02, 2.18)
        spacing_y = rng.uniform(2.55, 2.70)
        rows = [first_row + row * spacing_y for row in range(3)]
        first_column = rng.uniform(1.57, 1.73)
        spacing_x = rng.uniform(2.25, 2.35)
        dimensions[1] = rows[-1] + 2.20
        boxes = _walls(dimensions)
        aisle_centers = [(rows[i] + rows[i + 1]) / 2 for i in range(2)]
        gate_x = [first_column + .55 + spacing_x * col for col in range(3)]
        table_centers: list[tuple[float, float]] = []
        chair_centers: list[tuple[float, float]] = []
        for row, y in enumerate(rows):
            x_values = [first_column + spacing_x * col + (1.10 if row == 1 else 0) for col in range(3)]
            row_widths = []
            for col, x in enumerate(x_values):
                index = row * 3 + col
                width = rng.uniform(*width_range)
                row_widths.append(width)
                depth = rng.uniform(.82, .88)
                height = rng.uniform(.71, .83) if family in ("table", "mixed") else rng.uniform(.74, .78)
                table_id = f"table_{index:02d}"
                identity = f"{split}/table-profile/{index:02d}"
                boxes.extend(_table(table_id, x, y, width, depth, height, identity))
                table_centers.append((x, y))
                profiles.append(dict(identity=identity, width_m=width, depth_m=depth, height_m=height))
                for side in (-1, 1):
                    for position in (-1, 1):
                        chair_index = len(chair_centers)
                        pull = rng.uniform(-.008, .008)
                        if family in ("slalom", "mixed"):
                            pull += .020 * side * (1 if (col + row) % 2 else -1)
                        elif family == "asymmetric":
                            pull += .025 if side == 1 else -.015
                        chair_x, chair_y = x + position * .55, y + side * (.75 + pull)
                        chair_centers.append((chair_x, chair_y))
                        armrests = family in ("chair", "mixed") and chair_index % 3 == 0
                        boxes.extend(_chair(f"chair_{chair_index:02d}", chair_x, chair_y,
                                            math.pi if side == 1 else 0.0,
                                            rng.uniform(*chair_range), rng.uniform(.88, .98),
                                            armrests, f"{split}/chair-profile/{chair_index:02d}"))
            if row % 2 == 0:
                edge = x_values[0] - row_widths[0] / 2
                boxes.append(_box(f"row_{row}_closed_end", [edge / 2, y, 1.2],
                                  [edge / 2, .42, 1.2], "wall"))
            else:
                edge = x_values[-1] + row_widths[-1] / 2
                boxes.append(_box(f"row_{row}_closed_end", [(edge + 9) / 2, y, 1.2],
                                  [(9 - edge) / 2, .42, 1.2], "wall"))
        if family in ("overhead", "mixed"):
            for gate_index, y in enumerate(aisle_centers):
                x = gate_x[1]
                underside = rng.uniform(1.20, 1.25)
                boxes.append(_box(f"overhead_{gate_index}_beam", [x, y, underside + .065],
                                  [.22, spacing_y / 2 + .18, .065], "overhead"))
                for side in (-1, 1):
                    boxes.append(_box(f"overhead_{gate_index}_post_{side:+d}",
                                      [x, y + side * spacing_y / 2, underside / 2],
                                      [.045, .045, underside / 2], "overhead_support"))
        if family == "turn":
            for row in (0, 2):
                boxes.append(_box(f"turn_end_{row}", [7.86, rows[row], .43],
                                  [.265, .34, .43], "turn_obstacle"))
        start_y, final_y = rows[0] - 1.55, rows[-1] + 1.65
        route = [[.60, start_y], [8.42, start_y], [8.42, aisle_centers[0]], [.60, aisle_centers[0]],
                 [.60, aisle_centers[1]], [8.42, aisle_centers[1]], [8.42, final_y], [.60, final_y]]
        root_boxes = _root_obstacles(boxes)
        for aisle, y in enumerate(aisle_centers):
            for column, x in enumerate(gate_x):
                local_clearance = min(_horizontal_distance([x, y], box) for box in root_boxes)
                bottlenecks.append(dict(id=f"aisle_{aisle}_gate_{column}", center=[x, y],
                                        width_m=2 * local_clearance,
                                        width_definition="twice-center-clearance-in-root-height-band",
                                        required_behavior="coordinated limb clearance; overhead clearance where present",
                                        threatened_side="both", full_body_feasible=None))
        layout_description = "Three furniture rows with alternating closed ends and six opposed-chair gates."
    else:
        route = [[.55, 1.50], [3.45, 1.50]]
        if difficulty == "pilot":
            width = rng.uniform(*width_range) * .64
            boxes.extend(_table("table_00", 1.75, .61, width, 1.02, .76,
                                f"{split}/table-profile/pilot"))
            profiles.append(dict(identity=f"{split}/table-profile/pilot", width_m=width))
            if family != "table":
                boxes.extend(_chair("chair_00", 2.60, 2.13, 0.0, rng.uniform(*chair_range),
                                    .94, family in ("chair", "mixed"),
                                    f"{split}/chair-profile/pilot"))
            if family == "overhead":
                boxes.append(_box("pilot_overhead", [2, 1.5, 1.31], [.18, 1.45, .06], "overhead"))
            bottlenecks.append(dict(id="pilot_encounter", center=[1.75, 1.5], width_m=.76,
                                    width_definition="twice-table-edge-center-clearance",
                                    required_behavior="raise or tuck hand at tabletop", threatened_side="right",
                                    full_body_feasible=None))
        layout_description = ("Isolated diagnostic encounter; not a dense-room benchmark."
                              if difficulty == "pilot" else "Open-floor control, with perimeter walls only.")

    variant_route = copy.deepcopy(route)
    variant_route[0][0] += .22
    variant_route[-1][0] += .22 if difficulty == "dense" else -.22
    cases = [_case(route, bottlenecks), _case(variant_route, bottlenecks),
             _case(list(reversed(route)), bottlenecks)]
    feasibility = _route_clearance(route, boxes)
    if not feasibility["root_route_validated"]:
        raise ValueError(f"Constructed route failed root-cylinder clearance: {feasibility}")
    geometry_hash = _digest(dict(boxes=boxes, room_dimensions=dimensions))
    furniture = {box["furniture_id"]: box["furniture_type"] for box in boxes if "furniture_id" in box}
    scene = dict(schema=SCHEMA, scene_id=f"{difficulty}-{family}-{split}-{seed:06d}-{geometry_hash[:12]}",
                 seed=seed, split=split, family=family, difficulty=difficulty,
                 units="metres", coordinate_system="right-handed-z-up", boxes=boxes,
                 room_dimensions=dimensions, geometry_hash=geometry_hash, start_goals=cases,
                 goal_index=0, **copy.deepcopy(cases[0]))
    scene["counts"] = dict(tables=sum(kind == "table" for kind in furniture.values()),
                           chairs=sum(kind == "chair" for kind in furniture.values()),
                           primitive_boxes=len(boxes), bottlenecks=len(bottlenecks))
    scene["generator"] = dict(name="dense-primitive-furniture-v1", description=layout_description,
                              geometry_source="canonical-oriented-boxes", floor_in_obstacle_field=False,
                              shape_profiles=profiles,
                              split_holdout=dict(kind="disjoint-synthetic-dimension-profiles",
                                                 table_width_range_m=list(width_range),
                                                 chair_width_range_m=list(chair_range),
                                                 real_object_identity_holdout=False),
                              route_topology="alternating-closed-row-ends" if difficulty == "dense" else "straight",
                              full_body_route_certificate=False,
                              time_budget_method="route_length/0.40_m_per_s + 20_s; proposed benchmark budget")
    scene["feasibility"] = feasibility
    scene["reset_clearance"] = dict(root_radius_m=ROOT_CLEARANCE_RADIUS,
                                    root_z_interval_m=list(ROOT_CLEARANCE_Z),
                                    center_clearance_m=min(_horizontal_distance(scene["start"][:2], box)
                                                           for box in _root_obstacles(boxes)),
                                    root_cylinder_validated=True, full_body_validated=False,
                                    runtime_nominal_pose_collision_check_required=True)
    return scene


def validate_scene(scene: dict[str, Any]) -> None:
    """Reject malformed geometry before it is used for simulation or fields."""
    if scene.get("schema") != SCHEMA:
        raise ValueError(f"Expected scene schema {SCHEMA}")
    dimensions = scene.get("room_dimensions", [])
    if len(dimensions) != 3 or not all(math.isfinite(v) and v > 0 for v in dimensions):
        raise ValueError("room_dimensions must contain three positive finite lengths")
    names: set[str] = set()
    if not scene.get("boxes"):
        raise ValueError("A scene must contain its perimeter collision boxes")
    for box in scene["boxes"]:
        if not isinstance(box.get("name"), str) or box["name"] in names:
            raise ValueError("Primitive names must be strings and unique")
        names.add(box["name"])
        if len(box.get("center", [])) != 3 or not all(math.isfinite(v) for v in box["center"]):
            raise ValueError("Box centers must contain three finite coordinates")
        if len(box.get("half_size", [])) != 3 or not all(math.isfinite(v) and v > 0 for v in box["half_size"]):
            raise ValueError("Box half sizes must contain three positive finite lengths")
        if not math.isfinite(box.get("yaw", float("nan"))):
            raise ValueError("Box yaw must be finite")
    route = scene.get("route", [])
    if len(route) < 2 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in route):
        raise ValueError("Route must contain at least two finite xy points")
    _route_lengths(route)
    start, goal = scene.get("start", []), scene.get("goal", [])
    if len(start) != 3 or len(goal) != 2 or not all(math.isfinite(v) for v in (*start, *goal)):
        raise ValueError("start must be finite [x,y,yaw], goal finite [x,y]")
    if math.dist(start[:2], route[0]) > 1e-7 or math.dist(goal, route[-1]) > 1e-7:
        raise ValueError("Route endpoints must equal selected start and goal")
    if not math.isfinite(scene.get("time_budget", float("nan"))) or scene["time_budget"] <= 0:
        raise ValueError("time_budget must be positive and finite")
    if "start_goals" in scene:
        cases, selected_index = scene["start_goals"], scene.get("goal_index", 0)
        if type(selected_index) is not int or not 0 <= selected_index < len(cases):
            raise ValueError("Selected goal_index is outside start_goals")
        for key in ("start", "goal", "route", "time_budget", "bottlenecks"):
            if scene.get(key) != cases[selected_index].get(key):
                raise ValueError(f"Selected {key} differs from start_goals[goal_index]")
    actual_hash = _digest(dict(boxes=scene["boxes"], room_dimensions=dimensions))
    if scene.get("geometry_hash") != actual_hash:
        raise ValueError("Canonical geometry_hash does not match boxes and room dimensions")
    if "grid" in scene:
        grid = scene["grid"]
        dx = grid.get("voxel_size", float("nan"))
        if not math.isfinite(dx) or dx <= 0 or grid.get("axis_order") != "xyz":
            raise ValueError("Grid must have positive finite voxel_size and xyz axis_order")
        shape = grid.get("shape", [])
        if len(shape) != 3 or not all(type(v) is int and v >= 3 for v in shape):
            raise ValueError("Grid shape must contain three integers of at least three")
        edge, sample = grid.get("edge_origin", []), grid.get("sample_origin", [])
        if len(edge) != 3 or len(sample) != 3 or not all(math.isfinite(v) for v in (*edge, *sample)):
            raise ValueError("Grid origins must contain finite xyz coordinates")
        if any(abs(s - e - .5 * dx) > 1e-9 for e, s in zip(edge, sample)):
            raise ValueError("Grid sample_origin must be half a voxel above edge_origin")


def signed_distance(points: Any, boxes: list[dict[str, Any]]) -> Any:
    """Analytic union of oriented-box SDFs; positive in free space, in metres.

    Overlap interiors use min(primitive SDF), which is conservative rather than
    exact distance to the union boundary.  Outside the union distance is exact.
    """
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("Points must have trailing xyz dimension")
    result = np.full(points.shape[:-1], np.inf, dtype=np.float64)
    for box in boxes:
        displacement = points - np.asarray(box["center"])
        c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
        local = np.stack((c * displacement[..., 0] + s * displacement[..., 1],
                          -s * displacement[..., 0] + c * displacement[..., 1],
                          displacement[..., 2]), axis=-1)
        q = np.abs(local) - np.asarray(box["half_size"])
        distance = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, axis=-1), 0)
        np.minimum(result, distance, out=result)
    return result


def route_guidance(points: Any, route: list[list[float]], *, lookahead: float = .40,
                   speed: float = .60) -> Any:
    """Goal-bound xy velocity with nearest-segment projection and no extrapolation."""
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("Points must have trailing xyz dimension")
    if not math.isfinite(lookahead) or lookahead <= 0 or not math.isfinite(speed) or speed <= 0:
        raise ValueError("lookahead and speed must be positive and finite")
    cumulative, length = _route_lengths(route)
    flat = points.reshape(-1, 3)
    xy = flat[:, :2]
    best_distance = np.full(len(flat), np.inf)
    progress = np.zeros(len(flat))
    for i, (first, second) in enumerate(zip(route, route[1:])):
        first, second = np.asarray(first), np.asarray(second)
        delta = second - first
        segment_length = cumulative[i + 1] - cumulative[i]
        displacement = xy - first
        dot = displacement[:, 0] * delta[0] + displacement[:, 1] * delta[1]
        t = np.clip(dot / (segment_length ** 2), 0, 1)
        projection = first + t[:, None] * delta
        distance = np.sum((xy - projection) ** 2, axis=-1)
        replace = distance < best_distance
        progress[replace] = cumulative[i] + t[replace] * segment_length
        best_distance[replace] = distance[replace]
    target_progress = np.minimum(progress + lookahead, length)
    segment = np.minimum(np.searchsorted(cumulative, target_progress, side="right") - 1, len(route) - 2)
    vertices = np.asarray(route)
    cumulative = np.asarray(cumulative)
    fraction = ((target_progress - cumulative[segment]) /
                (cumulative[segment + 1] - cumulative[segment]))
    target = vertices[segment] + fraction[:, None] * (vertices[segment + 1] - vertices[segment])
    direction = target - xy
    norm = np.linalg.norm(direction, axis=-1)
    taper = np.minimum(np.linalg.norm(xy - vertices[-1], axis=-1) / .30, 1.0)
    taper = taper * taper * (3 - 2 * taper)
    result = np.zeros_like(flat)
    result[:, :2] = direction / np.maximum(norm[:, None], 1e-12) * (speed * taper)[:, None]
    return result.reshape(points.shape)


def write_scene_bundle(scene: dict[str, Any], output_dir: str | Path,
                       voxel_size: float = .10, *, goal_index: int = 0,
                       chunk_size: int = 32768) -> Path:
    """Atomically publish scene.json and matching xyz-order float32 field arrays.

    Existing destinations are never overwritten.  The selected start/goal has
    its own guidance hash; geometry_hash is shared across all three variants.
    Field generation is chunked to avoid a points-by-primitives allocation.
    """
    import numpy as np

    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    selected = copy.deepcopy(scene)
    cases = selected.get("start_goals", [scene])
    if isinstance(goal_index, bool) or not isinstance(goal_index, int) or not 0 <= goal_index < len(cases):
        raise ValueError("goal_index is outside start_goals")
    for key in ("start", "goal", "route", "time_budget", "bottlenecks", "route_length_m"):
        if key in cases[goal_index]:
            selected[key] = copy.deepcopy(cases[goal_index][key])
    selected["goal_index"] = goal_index
    validate_scene(selected)
    selected["reset_clearance"] = dict(selected.get("reset_clearance", {}),
        center_clearance_m=min(_horizontal_distance(selected["start"][:2], box)
                               for box in _root_obstacles(selected["boxes"])))
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Scene destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    dimensions = np.asarray(selected["room_dimensions"], dtype=float)
    edge_origin = np.array([-.10, -.10, 0.0])
    legacy_guidance = selected.get("generator", {}).get("guidance_method") == "cat-upstream-progressive-v1"
    if legacy_guidance:
        legacy = selected["legacy_cat"]
        source_edge = np.asarray(legacy["source_edge_origin"]) + legacy["translation"]
        # At the original .04m resolution, reproduce its translated cell centers
        # exactly instead of sampling on voxel boundaries after translation.
        edge_origin[:2] = source_edge[:2] - np.ceil((source_edge[:2] + .10) / voxel_size) * voxel_size
    extent = dimensions - edge_origin + np.array([.10, .10, 0.0])
    shape = np.maximum(3, np.ceil(extent / voxel_size).astype(int))
    total = int(np.prod(shape))
    if total > 10_000_000:
        raise ValueError("Requested field exceeds 10 million voxels; increase voxel_size")
    sample_origin = edge_origin + .5 * voxel_size
    selected["grid"] = dict(voxel_size=float(voxel_size), shape=shape.tolist(),
                            edge_origin=edge_origin.tolist(), sample_origin=sample_origin.tolist(),
                            axis_order="xyz", sample_location="cell_center", dtype="float32",
                            effective_edge_extent=(shape * voxel_size).tolist(),
                            resolution_error_bound_m=math.sqrt(3) * voxel_size / 2)
    selected["field_provenance"] = dict(
        sdf="min-analytic-oriented-box-sdf; exact outside union, conservative overlap interiors",
        boundary="finite-difference-gradient-of-sdf", guidance=GUIDANCE_METHOD,
        guidance_source="explicit-generator-route", goal_index=goal_index,
        geometry_hash=selected["geometry_hash"],
        grid_hash=_digest(selected["grid"]),
        guidance_hash=_digest(dict(route=selected["route"], goal=selected["goal"],
                                   method=GUIDANCE_METHOD, lookahead_m=.40, speed_mps=.60)),
        guidance_lookahead_m=.40, guidance_speed_mps=.60, guidance_goal_taper_radius_m=.30,
        boundary_projection_radius_m=.20, floor_in_obstacle_sdf=False,
        distances_are_voxel_center_samples=True, thin_primitives_may_lack_negative_voxel_samples=True,
        occupancy_or_full_body_feasibility_certificate=False)
    if legacy_guidance:
        from cat_ppo.furniture.legacy_scenes import original_guidance_spec
        spec = original_guidance_spec(selected)
        selected["field_provenance"].update(
            guidance=spec["method"], guidance_source="original-CAT-progressive-3D-FMM-on-adapted-physical-occupancy",
            guidance_hash=_digest(spec), original_cat_guidance=spec,
            guidance_lookahead_m=None, boundary_projection_radius_m=5 * voxel_size,
            original_fields_bit_identical=False)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        sdf = np.empty(tuple(shape), dtype=np.float32)
        sdf_flat = sdf.reshape(-1)
        for offset in range(0, total, chunk_size):
            indices = np.arange(offset, min(offset + chunk_size, total))
            ijk = np.stack(np.unravel_index(indices, tuple(shape)), axis=-1)
            points = sample_origin + ijk * voxel_size
            sdf_flat[offset:offset + len(indices)] = signed_distance(points, selected["boxes"])
        bf = np.stack(np.gradient(sdf, voxel_size, edge_order=2), axis=-1).astype(np.float32)
        if legacy_guidance:
            from cat_ppo.furniture.legacy_scenes import original_guidance_field
            gf = original_guidance_field(selected, sdf, bf, selected["grid"])
        else:
            gf = np.empty(tuple(shape) + (3,), dtype=np.float32)
            bf_flat, gf_flat = bf.reshape(-1, 3), gf.reshape(-1, 3)
            for offset in range(0, total, chunk_size):
                indices = np.arange(offset, min(offset + chunk_size, total))
                ijk = np.stack(np.unravel_index(indices, tuple(shape)), axis=-1)
                points = sample_origin + ijk * voxel_size
                g = route_guidance(points, selected["route"])
                b = bf_flat[offset:offset + len(indices)].astype(float)
                b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
                d = sdf_flat[offset:offset + len(indices)]
                weight = np.clip(1.0 - np.maximum(d, 0) / .20, 0, 1)
                weight = weight * weight * (3 - 2 * weight)
                g -= weight[:, None] * np.sum(g * b, axis=-1, keepdims=True) * b
                # Interior vectors point out of obstacles; free-space goal vectors
                # keep their taper and cannot gain speed through projection.
                g[d < 0] = .60 * b[d < 0]
                gf_flat[offset:offset + len(indices)] = g
        fields = {}
        for name, array in (("sdf", sdf), ("bf", bf), ("gf", gf)):
            file = staging / f"{name}.npy"
            np.save(file, array, allow_pickle=False)
            fields[name] = dict(file=file.name, shape=list(array.shape),
                                dtype=str(array.dtype), size_bytes=file.stat().st_size,
                                sha256=_file_digest(file))
        selected["fields"] = fields
        (staging / "scene.json").write_text(json.dumps(selected, indent=2, sort_keys=True, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError(f"Scene destination appeared during generation: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def load_scene(path: str | Path) -> dict[str, Any]:
    """Validate canonical geometry and, when present, every field's bytes/shape.

    Standalone layout JSON without field metadata is accepted. Bundles declaring
    fields must include all three immutable field files with matching hashes.
    """
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "scene.json"
    scene = json.loads(path.read_text())
    validate_scene(scene)
    if "fields" in scene:
        import numpy as np

        if "grid" not in scene or not {"sdf", "bf", "gf"}.issubset(scene["fields"]):
            raise ValueError("Field bundle must declare grid and sdf/bf/gf metadata")
        grid_shape = tuple(scene["grid"]["shape"])
        for name in ("sdf", "bf", "gf"):
            record = scene["fields"][name]
            if record.get("file") != f"{name}.npy":
                raise ValueError(f"Unexpected canonical field filename for {name}")
            file = path.parent / record["file"]
            if not file.is_file():
                raise ValueError(f"Missing declared field: {file}")
            if "size_bytes" in record and file.stat().st_size != record["size_bytes"]:
                raise ValueError(f"Field size mismatch: {name}")
            if _file_digest(file) != record.get("sha256"):
                raise ValueError(f"Field sha256 mismatch: {name}")
            array = np.load(file, mmap_mode="r", allow_pickle=False)
            expected_shape = grid_shape if name == "sdf" else grid_shape + (3,)
            if array.shape != expected_shape or list(array.shape) != record.get("shape"):
                raise ValueError(f"Field grid/metadata shape mismatch: {name}")
            if array.dtype != np.float32 or record.get("dtype", "float32") != "float32":
                raise ValueError(f"Field dtype must be float32: {name}")
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite field samples: {name}")
        provenance = scene.get("field_provenance", {})
        if provenance.get("geometry_hash") != scene["geometry_hash"]:
            raise ValueError("Field provenance geometry_hash differs from canonical geometry")
        if provenance.get("grid_hash") != _digest(scene["grid"]):
            raise ValueError("Field provenance grid_hash differs from declared sample grid")
        method = GUIDANCE_METHOD
        guidance_spec = dict(route=scene["route"], goal=scene["goal"], method=method, lookahead_m=.40, speed_mps=.60)
        if scene.get("generator", {}).get("guidance_method") == "cat-upstream-progressive-v1":
            from cat_ppo.furniture.legacy_scenes import original_guidance_spec
            guidance_spec = original_guidance_spec(scene)
            method = guidance_spec["method"]
            if provenance.get("original_cat_guidance") != guidance_spec:
                raise ValueError("Original CAT field guidance specification differs from declared scene")
        expected_guidance = _digest(guidance_spec)
        if (provenance.get("guidance") != method
                or provenance.get("guidance_hash") != expected_guidance
                or provenance.get("goal_index") != scene.get("goal_index", 0)):
            raise ValueError("Field guidance provenance differs from selected route/goal")
    return scene


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="write a deterministic room and matching field bundle")
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--split", choices=SPLITS, default="train")
    generate.add_argument("--family", choices=(*FAMILIES, "mixed"), default="mixed")
    generate.add_argument("--difficulty", choices=(*DIFFICULTIES, "open-floor"), default="dense")
    generate.add_argument("--voxel-size", type=float, default=.10)
    generate.add_argument("--goal-index", type=int, choices=range(3), default=0)
    args = parser.parse_args(argv)
    scene = generate_scene(seed=args.seed, split=args.split, family=args.family, difficulty=args.difficulty)
    path = write_scene_bundle(scene, args.output, voxel_size=args.voxel_size, goal_index=args.goal_index)
    print(json.dumps(dict(bundle=str(path), scene_id=scene["scene_id"], counts=scene["counts"],
                          goal_index=args.goal_index, full_body_validated=False), indent=2))


if __name__ == "__main__":
    main()
