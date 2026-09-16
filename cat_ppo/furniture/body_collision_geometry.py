"""Batched JAX volume tests for the approved robot collision primitives.

These kernels consume a small, already selected set of obstacle boxes. They
neither construct all robot/scene pairs nor change policy observations. All
rotations map local coordinates to world coordinates; leading array dimensions
broadcast, so ``[shape, candidate, ...]`` and vmapped environment batches work.

Nonpositive separation means touching/overlapping. Box SAT separation is a
separating-axis metric, not Euclidean distance or an exact penetration depth.
Capsule separation is exact segment-to-solid-box distance minus capsule radius;
when the segment enters the box it returns minus the radius, not penetration
depth. Tests are instantaneous volume tests, not swept-motion certification.
"""
from __future__ import annotations

import jax.numpy as jp
import numpy as np


BOX, CAPSULE, SPHERE = 0, 1, 2


def _local(point, center, rotation):
    return jp.einsum("...ji,...j->...i", rotation, point - center)


def sphere_box_separation(center, radius, box_center, box_rotation, box_half_size):
    """Exact signed box distance at sphere center, minus sphere radius."""
    center, radius = jp.asarray(center), jp.asarray(radius)
    local = _local(center, jp.asarray(box_center), jp.asarray(box_rotation))
    offset = jp.abs(local) - jp.asarray(box_half_size)
    signed_distance = (jp.linalg.norm(jp.maximum(offset, 0.), axis=-1)
                       + jp.minimum(jp.max(offset, axis=-1), 0.))
    return signed_distance - radius


def segment_box_squared_distance_local(start, end, half_size):
    """Exact squared distance from a segment to a centered axis-aligned box.

    The squared distance along a segment is piecewise quadratic. Its only
    breakpoints are the six box face planes. Sorting those plus the endpoints
    yields seven intervals; minimizing their quadratics finds the global
    minimum without sampling or an iterative optimizer. Parallel and zero-
    length segments are supported. This is three-dimensional, including cases
    where closest points lie inside an edge or face rather than at corners.
    """
    start, end, half_size = jp.broadcast_arrays(
        jp.asarray(start), jp.asarray(end), jp.asarray(half_size))
    delta = end - start
    moving = delta != 0.
    divisor = jp.where(moving, delta, 1.)
    crossings = jp.concatenate(((-half_size - start) / divisor,
                                (half_size - start) / divisor), axis=-1)
    crossings = jp.where(jp.concatenate((moving, moving), axis=-1), crossings, 0.)
    crossings = jp.clip(crossings, 0., 1.)
    zero = jp.zeros_like(start[..., :1])
    boundaries = jp.sort(jp.concatenate((zero, zero + 1., crossings), axis=-1), axis=-1)
    lower, upper = boundaries[..., :-1], boundaries[..., 1:]
    middle = .5 * (lower + upper)
    midpoint = start[..., None, :] + middle[..., :, None] * delta[..., None, :]
    active = jp.abs(midpoint) > half_size[..., None, :]
    side = jp.where(midpoint >= 0., 1., -1.)
    offset = start[..., None, :] - side * half_size[..., None, :]
    quadratic = jp.sum(jp.where(active, delta[..., None, :] ** 2, 0.), axis=-1)
    linear = jp.sum(jp.where(active, delta[..., None, :] * offset, 0.), axis=-1)
    optimum = jp.where(quadratic > 0., -linear / jp.where(quadratic > 0., quadratic, 1.), middle)
    optimum = jp.clip(optimum, lower, upper)
    position = start[..., None, :] + optimum[..., :, None] * delta[..., None, :]
    distance = jp.maximum(jp.abs(position) - half_size[..., None, :], 0.)
    return jp.min(jp.sum(distance ** 2, axis=-1), axis=-1)


def capsule_box_separation(start, end, radius, box_center, box_rotation, box_half_size):
    """Exact segment-to-solid-OBB distance minus capsule radius."""
    box_center, box_rotation = jp.asarray(box_center), jp.asarray(box_rotation)
    local_start = _local(jp.asarray(start), box_center, box_rotation)
    local_end = _local(jp.asarray(end), box_center, box_rotation)
    squared = segment_box_squared_distance_local(local_start, local_end, box_half_size)
    return jp.sqrt(jp.maximum(squared, 0.)) - jp.asarray(radius)


def box_box_separation(center, rotation, half_size, box_center, box_rotation,
                       box_half_size, *, parallel_axis_epsilon=1e-7):
    """OBB/OBB separating-axis test using all 3+3+9 candidate axes.

    Face axes and all edge cross-product axes are tested. The maximum normalized
    gap is positive for separated boxes and nonpositive for overlap. Exactly
    parallel cross axes are redundant. Cross axes shorter than the tiny stated
    threshold are conservatively ignored to avoid division by roundoff-sized
    lengths; this can add nearly-parallel false contacts, never remove a
    contact by itself. Shapes must have orthonormal rotation matrices and
    nonnegative half sizes. No mesh/SDF or convex-hull approximation is involved.
    """
    center, rotation, half_size = jp.asarray(center), jp.asarray(rotation), jp.asarray(half_size)
    box_center = jp.asarray(box_center)
    box_rotation, box_half_size = jp.asarray(box_rotation), jp.asarray(box_half_size)
    relative = jp.einsum("...ji,...jk->...ik", rotation, box_rotation)
    translation = _local(box_center, center, rotation)
    absolute = jp.abs(relative)
    first_axes = (jp.abs(translation) - half_size
                  - jp.einsum("...ij,...j->...i", absolute, box_half_size))
    second_axes = (jp.abs(jp.einsum("...ij,...i->...j", relative, translation))
                   - box_half_size - jp.einsum("...ij,...i->...j", absolute, half_size))

    one, two = jp.array([1, 2, 0]), jp.array([2, 0, 1])
    row_one = jp.take(relative, one, axis=-2)
    row_two = jp.take(relative, two, axis=-2)
    projected = jp.abs(jp.take(translation, two, axis=-1)[..., :, None] * row_one
                       - jp.take(translation, one, axis=-1)[..., :, None] * row_two)
    first_radius = (jp.take(half_size, one, axis=-1)[..., :, None] * jp.abs(row_two)
                    + jp.take(half_size, two, axis=-1)[..., :, None] * jp.abs(row_one))
    second_radius = (jp.take(box_half_size, one, axis=-1)[..., None, :]
                     * jp.take(absolute, two, axis=-1)
                     + jp.take(box_half_size, two, axis=-1)[..., None, :]
                     * jp.take(absolute, one, axis=-1))
    # Computing the cross length from its two actual components is more stable
    # near parallel axes than sqrt(1 - dot(axis_a, axis_b)**2).
    length = jp.sqrt(row_one ** 2 + row_two ** 2)
    usable = length > parallel_axis_epsilon
    cross_axes = jp.where(usable, (projected - first_radius - second_radius)
                          / jp.where(usable, length, 1.), -jp.inf)
    return jp.maximum(jp.maximum(jp.max(first_axes, axis=-1), jp.max(second_axes, axis=-1)),
                      jp.max(cross_axes, axis=(-2, -1)))


def transform_primitives(body_positions, body_rotations, body_ids, local_centers,
                         local_rotations, local_endpoints):
    """Transform shape constants through current owning-body poses.

    Body poses have shapes ``[..., body, 3]`` and ``[..., body, 3, 3]``.
    Constants have a single shape axis. Returns centers, rotations and capsule
    endpoints with the same leading environment dimensions. Fixed descendants
    must already have been expressed in the owning body frame offline.
    """
    positions = jp.take(jp.asarray(body_positions), jp.asarray(body_ids), axis=-2)
    rotations = jp.take(jp.asarray(body_rotations), jp.asarray(body_ids), axis=-3)
    return {
        "centers": positions + jp.einsum("...sij,sj->...si", rotations, jp.asarray(local_centers)),
        "rotations": jp.einsum("...sij,sjk->...sik", rotations, jp.asarray(local_rotations)),
        "endpoints": positions[..., :, None, :]
                     + jp.einsum("...sij,skj->...ski", rotations, jp.asarray(local_endpoints)),
    }


def box_aabb(center, rotation, half_size):
    extent = jp.einsum("...ij,...j->...i", jp.abs(jp.asarray(rotation)), jp.asarray(half_size))
    return jp.asarray(center) - extent, jp.asarray(center) + extent


def sphere_aabb(center, radius):
    radius = jp.asarray(radius)[..., None]
    return jp.asarray(center) - radius, jp.asarray(center) + radius


def capsule_aabb(start, end, radius):
    radius = jp.asarray(radius)[..., None]
    return jp.minimum(start, end) - radius, jp.maximum(start, end) + radius


def reduce_candidates(separation, candidate_mask, *, contact_tolerance=1e-6):
    """Return ``(any_collision, minimum_separation)`` per queried primitive.

    The final dimension is the candidate axis; all-invalid candidates give
    False and +inf. The tolerance is in metres and only handles numeric contact
    ambiguity. Padded candidates must contain finite geometry even when masked.
    """
    separation = jp.where(jp.asarray(candidate_mask), jp.asarray(separation), jp.inf)
    minimum = jp.min(separation, axis=-1)
    return minimum <= contact_tolerance, minimum


def compile_proposal(proposal, model):
    """Compile the approved review JSON to small NumPy constants grouped by type.

    Resolves body IDs from names against the actual simulator model, instead of
    trusting numeric IDs saved with a different MJCF. Does not modify the model
    or proposal. Runtime integration owns approval/version checks, obstacle
    semantics, training behavior and GPU capacity measurement.
    """
    import mujoco

    shapes = proposal.get("shapes", [])
    if not shapes:
        raise ValueError("Collision proposal has no shapes")
    kinds = {"box": BOX, "capsule": CAPSULE, "sphere": SPHERE}
    bodies, centers, rotations, endpoints, halves, radii, kind_ids = [], [], [], [], [], [], []
    names = [shape["id"] for shape in shapes]
    if len(set(names)) != len(names):
        raise ValueError("Collision proposal shape IDs must be unique")
    for shape in shapes:
        kind = shape.get("kind")
        if kind not in kinds:
            raise ValueError(f"Unsupported collision primitive {kind!r}")
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, shape["body_name"])
        if body <= 0:
            raise ValueError(f"Robot collision body is missing: {shape['body_name']}")
        center = np.asarray(shape["center"], dtype=np.float64)
        quat = np.asarray(shape["quat"], dtype=np.float64)
        if (center.shape != (3,) or not np.isfinite(center).all() or quat.shape != (4,)
                or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12):
            raise ValueError("Collision primitive has invalid pose")
        quat = quat / np.linalg.norm(quat)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        half = np.asarray(shape.get("half_size", [0., 0., 0.]), dtype=np.float64)
        radius = float(shape.get("radius", 0.))
        endpoint = np.asarray(shape.get("endpoints", [center, center]), dtype=np.float64)
        if (half.shape != (3,) or not np.isfinite(half).all() or endpoint.shape != (2, 3)
                or not np.isfinite(endpoint).all() or not np.isfinite(radius)
                or (kind == "box" and np.any(half <= 0.))
                or (kind != "box" and radius <= 0.)):
            raise ValueError("Collision primitive has invalid size/endpoints")
        bodies.append(body)
        centers.append(center)
        rotations.append(rotation.reshape(3, 3))
        endpoints.append(endpoint)
        halves.append(half)
        radii.append(radius)
        kind_ids.append(kinds[kind])
    kind_ids = np.asarray(kind_ids, dtype=np.int32)
    return {"body_ids": np.asarray(bodies, dtype=np.int32),
            "local_centers": np.asarray(centers, dtype=np.float32),
            "local_rotations": np.asarray(rotations, dtype=np.float32),
            "local_endpoints": np.asarray(endpoints, dtype=np.float32),
            "half_sizes": np.asarray(halves, dtype=np.float32),
            "radii": np.asarray(radii, dtype=np.float32), "kind_ids": kind_ids,
            "indices": {name: np.flatnonzero(kind_ids == value).astype(np.int32)
                        for name, value in kinds.items()},
            "shape_names": tuple(names), "groups": tuple(shape["group"] for shape in shapes)}
