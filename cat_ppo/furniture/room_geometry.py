"""Conservative room voxels and analytic clearance against canonical boxes.

These helpers are deliberately separate from released/procedural CAT assets.
Room primitives have yaw only, so rectangle separating axes plus a Z interval
give an exact box/cell intersection test, including touching boundaries.
"""

from __future__ import annotations

import math

import numpy as np


def _box_parameters(box):
    center = np.asarray(box["center"], dtype=np.float64)
    half = np.asarray(box["half_size"], dtype=np.float64)
    yaw = float(box.get("yaw", 0.))
    if (center.shape != (3,) or half.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(half).all()
            or np.any(half < 0) or not math.isfinite(yaw)):
        raise ValueError("Box needs finite XYZ center, nonnegative half sizes and yaw")
    return center, half, math.cos(yaw), math.sin(yaw)


def conservative_rasterize_boxes(boxes, shape, sample_origin, dx):
    """Mark cells intersecting yaw-oriented boxes, without adding a floor.

    ``sample_origin + index * dx`` is a cell CENTER, matching CAT's runtime
    field samples. Each cell extends ``dx/2`` on every side. Four rectangle
    separating axes (world X/Y and box X/Y), followed by the Z slab, avoid both
    dropped sub-voxel legs and the excessive inflation of an AABB-only fill.
    Explicit ``category='floor'`` boxes are excluded from obstacle occupancy.

    Only a box's local index window is allocated. Boundary contact is occupied;
    a scale-dependent roundoff tolerance is the only geometric overcoverage.
    The occupied-cell union lies at most one voxel diagonal from true geometry.
    """
    raw_shape = np.asarray(shape)
    origin = np.asarray(sample_origin, dtype=np.float64)
    if (raw_shape.shape != (3,) or not np.isfinite(raw_shape).all()
            or np.any(raw_shape <= 0) or np.any(raw_shape != np.floor(raw_shape))):
        raise ValueError("shape must contain three positive integer cell counts")
    shape = raw_shape.astype(np.int64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("sample_origin must contain three finite cell-center coordinates")
    if not math.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive")
    occupancy = np.zeros(tuple(shape), dtype=bool)
    cell_half = .5 * dx
    for box in boxes:
        if box.get("category") == "floor":
            continue
        center, half, c, s = _box_parameters(box)
        ac, ass = abs(c), abs(s)
        extent = np.array([ac * half[0] + ass * half[1],
                           ass * half[0] + ac * half[1], half[2]])
        tolerance = 64 * np.finfo(np.float64).eps * max(
            1., float(np.max(np.abs(center))), float(np.max(half)), dx)
        # Broad phase is exact for the world X/Y/Z separating axes. Clip before
        # integer conversion so distant (but finite) boxes cannot overflow it.
        lower = np.ceil((center - extent - cell_half - origin - tolerance) / dx)
        upper = np.floor((center + extent + cell_half - origin + tolerance) / dx) + 1
        begin = np.clip(lower, 0, shape).astype(np.int64)
        end = np.clip(upper, 0, shape).astype(np.int64)
        if np.any(end <= begin):
            continue
        x = origin[0] + np.arange(begin[0], end[0])[:, None] * dx - center[0]
        y = origin[1] + np.arange(begin[1], end[1])[None, :] * dx - center[1]
        projected_cell_half = cell_half * (ac + ass)
        xy_intersects = ((np.abs(c * x + s * y) <= half[0] + projected_cell_half + tolerance)
                         & (np.abs(-s * x + c * y) <= half[1] + projected_cell_half + tolerance)
                         & (np.abs(x) <= extent[0] + cell_half + tolerance)
                         & (np.abs(y) <= extent[1] + cell_half + tolerance))
        z = origin[2] + np.arange(begin[2], end[2]) * dx - center[2]
        z_intersects = np.abs(z) <= half[2] + cell_half + tolerance
        target = tuple(slice(a, b) for a, b in zip(begin, end))
        occupancy[target] |= xy_intersects[..., None] & z_intersects
    return occupancy


def _root_boxes(boxes, radius, z_interval):
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("Root radius must be finite and nonnegative")
    z_interval = np.asarray(z_interval, dtype=np.float64)
    if (z_interval.shape != (2,) or not np.isfinite(z_interval).all()
            or z_interval[1] < z_interval[0]):
        raise ValueError("Root height interval must contain finite ordered endpoints")
    for box in boxes:
        if box.get("category") == "floor":
            continue
        center, half, c, s = _box_parameters(box)
        if center[2] + half[2] >= z_interval[0] and center[2] - half[2] <= z_interval[1]:
            yield center, half, c, s


def _xy(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 2 or not np.isfinite(points).all():
        raise ValueError("Points must have shape (..., 2) and finite XY coordinates")
    return points


def _local_xy(points, center, c, s):
    displacement = points - center[:2]
    return np.stack((c * displacement[..., 0] + s * displacement[..., 1],
                     -s * displacement[..., 0] + c * displacement[..., 1]), axis=-1)


def root_cylinder_clearance(points, boxes, *, radius=.23, z_interval=(.35, 1.05)):
    """Signed XY obstacle clearance minus the root-cylinder radius, in metres.

    Evaluates canonical box geometry analytically, independently of voxel SDFs.
    Positive values are clear; zero is touching. This certifies the fixed-height
    cylinder only, not arms, feet, whole-body dynamics or a different root height.
    No overlapping obstacle heights gives ``+inf``. Supports ``(..., 2)`` points.
    """
    points = _xy(points)
    margin = np.full(points.shape[:-1], np.inf, dtype=np.float64)
    for center, half, c, s in _root_boxes(boxes, radius, z_interval):
        q = np.abs(_local_xy(points, center, c, s)) - half[:2]
        outside = np.maximum(q, 0.)
        signed = np.hypot(outside[..., 0], outside[..., 1]) + np.minimum(np.max(q, axis=-1), 0.)
        np.minimum(margin, signed - radius, out=margin)
    return margin


def root_cylinder_segment_clearance(starts, ends, boxes, *, radius=.23, z_interval=(.35, 1.05)):
    """Exact continuous segment-to-box XY distance minus cylinder radius.

    Supports broadcastable ``(..., 2)`` endpoints. A positive result proves the
    entire straight segment clears every obstacle in the fixed root-height
    band; this is not an endpoint or sampled-distance test. Negative values flag
    collision, but are not penetration depths: center-line intersection returns
    ``-radius``. A zero-length segment is supported. Boundary contact gives zero.
    """
    starts, ends = np.broadcast_arrays(_xy(starts), _xy(ends))
    margin = np.full(starts.shape[:-1], np.inf, dtype=np.float64)
    for center, half, c, s in _root_boxes(boxes, radius, z_interval):
        first = _local_xy(starts, center, c, s)
        last = _local_xy(ends, center, c, s)
        delta = last - first
        # Liang-Barsky slab intersection including parallel and zero-length
        # segments. Safe denominators keep masked branches finite.
        enter = np.zeros(first.shape[:-1])
        leave = np.ones(first.shape[:-1])
        intersects = np.ones(first.shape[:-1], dtype=bool)
        for axis in (0, 1):
            moving = delta[..., axis] != 0
            divisor = np.where(moving, delta[..., axis], 1.)
            low = (-half[axis] - first[..., axis]) / divisor
            high = (half[axis] - first[..., axis]) / divisor
            enter = np.maximum(enter, np.where(moving, np.minimum(low, high), -np.inf))
            leave = np.minimum(leave, np.where(moving, np.maximum(low, high), np.inf))
            intersects &= moving | (np.abs(first[..., axis]) <= half[axis])
        intersects &= enter <= leave
        # If disjoint, a minimizing pair consists of either a segment endpoint
        # and its rectangle projection, or a rectangle corner and its segment
        # projection. These cases cover parallel edge minima too.
        endpoint_offsets = np.maximum(np.abs(np.stack((first, last), axis=-2)) - half[:2], 0.)
        distance = np.min(np.hypot(endpoint_offsets[..., 0], endpoint_offsets[..., 1]), axis=-1)
        length_squared = np.sum(delta * delta, axis=-1)
        denominator = np.where(length_squared > 0, length_squared, 1.)
        for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            corner_delta = np.array([sx * half[0], sy * half[1]]) - first
            fraction = np.clip(np.sum(corner_delta * delta, axis=-1) / denominator, 0., 1.)
            residual = corner_delta - fraction[..., None] * delta
            distance = np.minimum(distance, np.hypot(residual[..., 0], residual[..., 1]))
        distance = np.where(intersects, 0., distance)
        np.minimum(margin, distance - radius, out=margin)
    return margin
