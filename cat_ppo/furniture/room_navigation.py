"""Ordered room-route following without adding policy observations.

Legacy room routes admit an upright root cylinder (radius 0.23 m, world height
0.35--1.05 m), not the articulated whole body. Certified contrastive passages
may use a smaller navigation radius alongside mandatory full-body collision
checks. Hand/elbow fields and native
CAT task behavior remain the caller's responsibility.  All runtime functions
are pure JAX: route progress belongs in episode info, not Python attributes.
"""
from __future__ import annotations

import math

import jax.numpy as jp
import numpy as np


ROOT_RADIUS = .23
ROOT_HEIGHT = (.35, 1.05)
_EPS = 1e-7
_VISIBILITY_MARGIN = 2e-6


def scene_navigation_radius(scene):
    """Return the legacy radius or a geometry-bound contrastive certificate.

    This validates a sampled configuration certificate, not a proof of dynamic
    traversability. Runtime users must additionally match its body-proxy hash
    to an enabled collision bank; the reduced cylinder is only route guidance.
    """
    contrast = None if scene is None else scene.get("hand_contrast")
    if contrast is None:
        return ROOT_RADIUS
    if not isinstance(contrast, dict) or contrast.get("schema") != "hand-contrast-v1":
        raise ValueError("Unsupported hand-contrast navigation schema")
    radius = contrast.get("navigation_radius_m")
    if (isinstance(radius, bool) or not isinstance(radius, (int, float))
            or not math.isfinite(radius) or not .14 <= radius <= ROOT_RADIUS):
        raise ValueError("Hand-contrast navigation radius must be within [0.14, 0.23] metres")
    certificate = contrast.get("certificate", {})
    if (not isinstance(certificate, dict)
            or certificate.get("schema") != "hand-contrast-certificate-v1"
            or certificate.get("route_transition_validated") is not True
            or certificate.get("navigation_radius_m") != radius
            or certificate.get("primitive_count") != 35):
        raise ValueError("Hand-contrast navigation needs a matching 35-primitive transition certificate")
    from cat_ppo.furniture.scenes import _digest
    geometry_hash = _digest(dict(boxes=scene["boxes"], room_dimensions=scene["room_dimensions"]))
    if (certificate.get("geometry_hash") != geometry_hash
            or certificate.get("route_hash") != _digest(scene["route"])):
        raise ValueError("Hand-contrast certificate does not match scene geometry/route")
    fingerprint = certificate.get("body_proxy_sha256", "")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)):
        raise ValueError("Hand-contrast certificate needs a body-proxy SHA256")
    for key in ("route_sample_count", "transition_sample_count"):
        if type(certificate.get(key)) is not int or certificate[key] < 1:
            raise ValueError("Hand-contrast certificate requires sampled route and posture transitions")
    for key in ("body_min_separation_m", "hand_field_min_clearance_m"):
        value = certificate.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0.):
            raise ValueError("Hand-contrast route and transitions must have positive certified clearance")
    return float(radius)


def pack_room_scenes(scenes):
    """Pack canonical scenes; use ``None`` for unchanged original CAT tasks.

    Returned NumPy arrays have leading scene dimension and may be converted to
    JAX arrays once. Routes are padded by repeating their final point. Each OBB
    row is ``(cx, cy, hx, hy, cos(yaw), sin(yaw))``; only boxes intersecting the
    fixed root-height interval are retained. Counts mask padding exactly.
    navigation_radius carries the validated scene-specific route cylinder.
    These small immutable metadata arrays do not belong in policy observations.
    """
    scenes = list(scenes)
    if not scenes:
        raise ValueError("At least one scene slot is required")
    routes, all_obstacles = [], []
    for scene in scenes:
        if scene is None:
            routes.append(np.empty((0, 2), dtype=np.float32))
            all_obstacles.append(np.empty((0, 6), dtype=np.float32))
            continue
        route = np.asarray(scene["route"], dtype=np.float64)
        if (route.ndim != 2 or route.shape[1] != 2 or len(route) < 2
                or not np.isfinite(route).all()
                or np.any(np.linalg.norm(np.diff(route, axis=0), axis=-1) <= 1e-6)):
            raise ValueError("Room route needs at least two distinct finite XY waypoints")
        if "start" in scene and not np.allclose(route[0], scene["start"][:2], atol=1e-6, rtol=0):
            raise ValueError("Room route does not begin at its declared start")
        if "goal" in scene and not np.allclose(route[-1], scene["goal"][:2], atol=1e-6, rtol=0):
            raise ValueError("Room route does not end at its declared goal")
        obstacles = []
        for box in scene["boxes"]:
            if box.get("category") == "floor":
                continue
            center = np.asarray(box["center"], dtype=np.float64)
            half = np.asarray(box["half_size"], dtype=np.float64)
            yaw = float(box.get("yaw", 0.))
            if (center.shape != (3,) or half.shape != (3,) or np.any(half < 0)
                    or not np.isfinite(center).all() or not np.isfinite(half).all()
                    or not math.isfinite(yaw)):
                raise ValueError("Room boxes need finite XYZ center/half sizes/yaw")
            if center[2] + half[2] >= ROOT_HEIGHT[0] and center[2] - half[2] <= ROOT_HEIGHT[1]:
                obstacles.append([*center[:2], *half[:2], math.cos(yaw), math.sin(yaw)])
        routes.append(route.astype(np.float32))
        all_obstacles.append(np.asarray(obstacles, dtype=np.float32).reshape(-1, 6))
    k = max(2, max(map(len, routes)))
    m = max(1, max(map(len, all_obstacles)))
    packed_route = np.zeros((len(scenes), k, 2), dtype=np.float32)
    packed_obstacles = np.zeros((len(scenes), m, 6), dtype=np.float32)
    packed_obstacles[..., 4] = 1.
    for index, (route, obstacles) in enumerate(zip(routes, all_obstacles)):
        if len(route):
            packed_route[index] = route[-1]
            packed_route[index, :len(route)] = route
        packed_obstacles[index, :len(obstacles)] = obstacles
    return dict(enabled=np.asarray([scene is not None for scene in scenes]),
                route=packed_route, route_count=np.asarray(list(map(len, routes)), dtype=np.int32),
                obstacles=packed_obstacles,
                obstacle_count=np.asarray(list(map(len, all_obstacles)), dtype=np.int32),
                navigation_radius=np.asarray([scene_navigation_radius(scene) for scene in scenes],
                                              dtype=np.float32))


def _local(points, obstacles):
    displacement = jp.asarray(points, dtype=jp.float32)[..., None, :] - obstacles[:, :2]
    x, y = displacement[..., 0], displacement[..., 1]
    c, s = obstacles[:, 4], obstacles[:, 5]
    return jp.stack((c * x + s * y, -s * x + c * y), axis=-1)


def root_clearance(points, obstacles, obstacle_count, *, radius=ROOT_RADIUS):
    """Signed exact OBB distance minus cylinder radius; supports (..., 2)."""
    obstacles = jp.asarray(obstacles, dtype=jp.float32)
    q = jp.abs(_local(points, obstacles)) - obstacles[:, 2:4]
    distance = jp.linalg.norm(jp.maximum(q, 0.), axis=-1) + jp.minimum(jp.max(q, axis=-1), 0.)
    valid = jp.arange(obstacles.shape[0]) < obstacle_count
    return jp.min(jp.where(valid, distance - radius, jp.inf), axis=-1)


def swept_root_clearance(first, last, obstacles, obstacle_count, *, radius=ROOT_RADIUS):
    """Exact continuous segment/OBB distance minus radius, in float32.

    Endpoint-only sampling misses thin obstacles. A slab intersection catches
    crossings, including parallel and zero-length segments. For disjoint sets,
    minima are segment endpoints to rectangle or rectangle corners to segment.
    Supports broadcastable (..., 2) endpoints and returns (...,) clearances.
    Intersection returns -radius; negative values are not penetration depths.
    """
    obstacles = jp.asarray(obstacles, dtype=jp.float32)
    first, last = jp.broadcast_arrays(jp.asarray(first, dtype=jp.float32),
                                     jp.asarray(last, dtype=jp.float32))
    a, b = _local(first, obstacles), _local(last, obstacles)
    half = obstacles[:, 2:4]
    delta = b - a
    moving = delta != 0.
    divisor = jp.where(moving, delta, 1.)
    low, high = (-half - a) / divisor, (half - a) / divisor
    enter = jp.maximum(jp.max(jp.where(moving, jp.minimum(low, high), -jp.inf), axis=-1), 0.)
    leave = jp.minimum(jp.min(jp.where(moving, jp.maximum(low, high), jp.inf), axis=-1), 1.)
    intersects = jp.all(moving | (jp.abs(a) <= half), axis=-1) & (enter <= leave)
    distance = jp.minimum(jp.linalg.norm(jp.maximum(jp.abs(a) - half, 0.), axis=-1),
                          jp.linalg.norm(jp.maximum(jp.abs(b) - half, 0.), axis=-1))
    denominator = jp.maximum(jp.sum(delta * delta, axis=-1), 1e-20)
    for sx, sy in ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)):
        offset = half * jp.asarray([sx, sy]) - a
        fraction = jp.clip(jp.sum(offset * delta, axis=-1) / denominator, 0., 1.)
        residual = offset - fraction[..., None] * delta
        distance = jp.minimum(distance, jp.linalg.norm(residual, axis=-1))
    distance = jp.where(intersects, 0., distance)
    valid = jp.arange(obstacles.shape[0]) < obstacle_count
    return jp.min(jp.where(valid, distance - radius, jp.inf), axis=-1)


def _segment_target(root, route, segment, lookahead):
    a, b = route[segment], route[segment + 1]
    delta = b - a
    length = jp.maximum(jp.linalg.norm(delta), _EPS)
    progress = jp.clip(jp.sum((root - a) * delta) / length, 0., length)
    projection = a + progress * delta / length
    target = a + jp.minimum(progress + lookahead, length) * delta / length
    return target, projection, b


def route_context(root_xy, previous_xy, segment, sticky_violation, route, route_count,
                  obstacles, obstacle_count, *, lookahead=.4, waypoint_radius=.15,
                  radius=ROOT_RADIUS, goal_radius=.20, speed=.6):
    """Advance one ordered segment and choose a continuously visible target.

    Call once for the actual root after each physical step (and at reset with
    previous_xy=root_xy, segment=0, sticky_violation=False). Save returned
    segment/violation and current root in episode info. Both true and delayed
    perception should use this SAME progress state rather than advance twice.

    A waypoint may be rounded only from its vicinity and if the straight swept
    cylinder to the next segment's carrot is clear. No global nearest-segment
    search is used. Off-route recovery goes to the current segment's projection
    only when visible; otherwise the command is zero and blocked=True.

    target and direction are XY; guidance is XYZ with z=0. tangent is the current
    ordered route segment, independently of the robot's heading or recovery
    direction; progress_m is its clamped projection plus prior segment lengths.
    route_complete is
    final ordered segment + endpoint proximity + no previous/current collision.
    This geometric controller does not certify dynamic policy tracking.
    """
    root = jp.asarray(root_xy, dtype=jp.float32)
    previous = jp.asarray(previous_xy, dtype=jp.float32)
    route = jp.asarray(route, dtype=jp.float32)
    last_segment = jp.maximum(route_count - 2, 0)
    segment = jp.clip(jp.asarray(segment, dtype=jp.int32), 0, last_segment)
    next_segment = jp.minimum(segment + 1, last_segment)
    _, _, corner = _segment_target(root, route, segment, lookahead)
    next_target, _, _ = _segment_target(root, route, next_segment, lookahead)
    next_clearance = swept_root_clearance(root, next_target, obstacles, obstacle_count, radius=radius)
    advance = ((segment < last_segment) & (jp.linalg.norm(root - corner) <= waypoint_radius)
               & (next_clearance >= _VISIBILITY_MARGIN))
    segment = jp.where(advance, next_segment, segment)
    carrot, projection, _ = _segment_target(root, route, segment, lookahead)
    segment_delta = route[1:] - route[:-1]
    segment_lengths = jp.linalg.norm(segment_delta, axis=-1)
    tangent = segment_delta[segment] / jp.maximum(segment_lengths[segment], _EPS)
    progress_m = (jp.sum(jp.where(jp.arange(len(segment_lengths)) < segment,
                                  segment_lengths, 0.))
                  + jp.linalg.norm(projection - route[segment]))
    targets = jp.stack((carrot, projection))
    target_clearances = swept_root_clearance(root, targets, obstacles, obstacle_count, radius=radius)
    carrot_visible = target_clearances[0] >= _VISIBILITY_MARGIN
    projection_visible = target_clearances[1] >= _VISIBILITY_MARGIN
    target = jp.where(carrot_visible, carrot, jp.where(projection_visible, projection, root))
    route_valid = route_count >= 2
    finite = jp.all(jp.isfinite(root)) & jp.all(jp.isfinite(previous))
    root_margin = root_clearance(root, obstacles, obstacle_count, radius=radius)
    sweep_margin = swept_root_clearance(previous, root, obstacles, obstacle_count, radius=radius)
    swept_violation = (sweep_margin < -_VISIBILITY_MARGIN) | ~finite
    violation = jp.asarray(sticky_violation, dtype=bool) | swept_violation
    goal = route[jp.maximum(route_count - 1, 0)]
    goal_distance = jp.linalg.norm(root - goal)
    route_complete = (route_valid & (segment == last_segment) & (goal_distance <= goal_radius)
                      & ~violation)
    displacement = target - root
    target_distance = jp.linalg.norm(displacement)
    blocked = ~route_valid | ~finite | ((~carrot_visible) & ((~projection_visible) | (target_distance <= _EPS)))
    active = ~blocked & ~route_complete & ~violation
    direction = jp.where(active, displacement / jp.maximum(target_distance, _EPS), jp.zeros(2))
    # Only the final segment tapers to the final goal. Intermediate waypoints
    # must never stall merely because a nearby route endpoint is across a wall.
    taper = jp.where(segment == last_segment, jp.clip(goal_distance / .30, 0., 1.), 1.)
    guidance = jp.concatenate((direction * (speed * taper), jp.zeros(1, dtype=jp.float32)))
    return dict(segment=segment, target=target, direction=direction, guidance=guidance,
                progress_m=progress_m, tangent=tangent,
                root_clearance=root_margin, swept_clearance=sweep_margin,
                swept_violation=swept_violation, violation=violation,
                route_complete=route_complete, blocked=blocked,
                cross_track=jp.linalg.norm(root - projection))
