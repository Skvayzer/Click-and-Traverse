"""JAX field queries with explicit unknowns; controlled map corruption, not a sensor stack."""
from itertools import product
import jax.numpy as jp


def sample_grid(field, positions, origin, voxel_size):
    """Trilinear xyz sampling. Return values and known mask, never silent free space.

    Unlike the released CAT sampler, corner and flattened weight order agree.
    Last-cell interpolation is valid; out-of-map positions are flagged unknown.
    """
    index = (positions - origin) / voxel_size
    maximum = jp.asarray(field.shape[:3]) - 1
    known = jp.all((index >= 0) & (index <= maximum), axis=-1) & jp.all(jp.isfinite(index), axis=-1)
    clipped = jp.clip(jp.nan_to_num(index), 0, maximum)
    base = jp.minimum(jp.floor(clipped).astype(jp.int32), maximum - 1)
    fraction = clipped - base
    offsets = jp.asarray(list(product((0, 1), repeat=3)), dtype=jp.int32)
    corners = base[:, None, :] + offsets[None, :, :]
    weights = jp.prod(jp.where(offsets[None, :, :] == 1, fraction[:, None, :], 1 - fraction[:, None, :]), axis=-1)
    values = field[corners[..., 0], corners[..., 1], corners[..., 2]]
    if field.ndim == 3:
        values = values[..., None]
    result = jp.sum(weights[..., None] * values, axis=1)
    return result, known


def probe_features(sdf, bf, positions, velocity, radii, origin, dx, *, age=0.0,
                   position_offset=None, unknown=None, uncertainty=0.0,
                   prediction_enabled=True):
    """Constant-velocity .2/.4s clearance forecast with masks for every horizon.

    This is a geometry proxy, not forward dynamics or a collision guarantee.
    Unknown features use a conservative negative clearance and zero direction.
    """
    if position_offset is not None:
        positions = positions + position_offset
    clearances, masks = [], []
    for horizon in (0.0, 0.2, 0.4):
        horizon = horizon if prediction_enabled else 0.0
        distance, known = sample_grid(sdf, positions + horizon * velocity, origin, dx)
        clearances.append(distance[:, 0] - radii)
        masks.append(known)
    known = jp.all(jp.stack(masks, axis=-1), axis=-1)
    direction, direction_known = sample_grid(bf, positions, origin, dx)
    known &= direction_known
    if unknown is not None:
        known &= ~unknown
    direction /= jp.maximum(jp.linalg.norm(direction, axis=-1, keepdims=True), 1e-6)
    clearance = jp.stack(clearances, axis=-1)
    clearance = jp.where(known[:, None], jp.clip(clearance, -1.0, 2.0), -jp.ones_like(clearance))
    direction = jp.where(known[:, None], direction, 0.0)
    return jp.concatenate([clearance, direction,
        jp.broadcast_to(jp.asarray(age), (len(positions), 1)),
        (~known).astype(jp.float32)[:, None],
        jp.broadcast_to(jp.asarray(uncertainty), (len(positions), 1))], axis=-1)


def apply_uncertainty_margin(features, *, enabled=True, fixed_margin=0.01,
                             std_multiplier=2.0, relative_motion_bound=1.0,
                             resolution_error_bound=0.0):
    """Proposed robust margin, not a calibrated probability or safety guarantee.

    Capture age is real elapsed simulation time. Translation noise standard
    deviation and an explicit relative-motion bound determine extra clearance.
    """
    margin = fixed_margin + resolution_error_bound + std_multiplier * features[:, 8] + relative_motion_bound * features[:, 6]
    margin = jp.where(enabled, margin, 0.0)
    return features.at[:, :3].add(-margin[:, None])


def hand_protection_cost(features, velocities, *, clearance_margin=.12, urgency_gain=1.0):
    """Hand-envelope proximity cost amplified when moving toward a surface.

    The boundary gradient points into free space. Closing speed is its negative
    projection; moving away cannot increase urgency. Predicted clearances are
    already disabled by the caller's prediction ablation when requested.
    This shaping term is a simulation objective, not an impact-damage model.
    """
    distance = jp.min(features[:, :3], axis=-1)
    deficit = jp.maximum(clearance_margin - distance, 0.0)
    closing_speed = jp.maximum(-jp.sum(velocities * features[:, 3:6], axis=-1), 0.0)
    urgency = jp.clip(closing_speed / jp.maximum(features[:, 0], .02), 0.0, 10.0)
    return jp.mean(deficit ** 2 * (1.0 + urgency_gain * urgency))
