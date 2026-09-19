"""Training-only first-outcome posture qualification; no evaluation rollouts."""
from __future__ import annotations

import jax.numpy as jp
import numpy as np

COUNTS = "pf_contrast_outcome_counts"
STEPS = "wholebody_contrast_zone_steps"
GROUPS = ("forward_protected", "narrow_passage", "posture_transition")


def initial_contrast_outcomes(batch_shape):
    return {COUNTS: jp.zeros(tuple(batch_shape) + (3, 2), jp.int32),
            STEPS: jp.zeros(tuple(batch_shape) + (6, 3), jp.int32)}


def update_contrast_outcomes(info, resolved, successful):
    context, telemetry = info["hand_contrast"], info["wholebody_telemetry"]
    core = context["core_active"]
    zone = jp.arange(6) == context["zone_index"][..., None]
    core_zone = core[..., None] & zone
    values = jp.stack((core_zone,
        core_zone & (telemetry["hand_contrast_heading_good"][..., None] > .5),
        core_zone & (telemetry["hand_contrast_hand_good"][..., None] > .5)), axis=-1)
    # Navigation's latch is already updated; successful/resolved masks refer to
    # the old latch. Freeze once an outcome was counted in a previous step.
    active = ~info["wholebody_navigation_outcome_counted"] | resolved
    info[STEPS] = steps = info[STEPS] + (values & active[..., None, None]).astype(jp.int32)
    seen = steps[..., 0] > 0
    heading = seen & (steps[..., 1] >= .9 * steps[..., 0])
    hands = seen & (steps[..., 2] >= .9 * steps[..., 0])
    qualified = (jp.all(~context["required_forward_zones"] | heading, axis=-1)
                 & jp.all(~context["required_hand_zones"] | hands, axis=-1))
    role = context["role"]
    valid = (role >= 1) & (role <= 3)
    groups = jp.clip(role - 1, 0, 2)
    finished = resolved & valid
    passed = successful & valid & qualified
    increments = jp.stack((jp.bincount(groups, weights=finished.astype(jp.int32), length=3),
                           jp.bincount(groups, weights=passed.astype(jp.int32), length=3)), axis=-1)
    previous = info[COUNTS].reshape(-1, 3, 2)[0]
    info[COUNTS] = jp.broadcast_to(previous + increments, info[COUNTS].shape)


def contrast_count_snapshot(info):
    counts = info[COUNTS]
    if counts.ndim == 2:
        return counts
    if counts.ndim == 3:
        return counts[0]
    if counts.ndim == 4:
        return jp.sum(counts[:, 0], axis=0)
    raise ValueError("Unexpected contrastive outcome counter shape")


def contrast_rollout_metrics(previous, current):
    previous, current = np.asarray(previous), np.asarray(current)
    if previous.shape != (3, 2) or current.shape != (3, 2):
        raise ValueError("Expected 3x2 contrastive snapshots")
    counts = current.astype(np.int64) - previous.astype(np.int64)
    if np.any(counts < 0) or np.any(counts[:, 1] > counts[:, 0]):
        raise ValueError("Invalid contrastive outcome increments")
    metrics = {}
    for name, (resolved, success) in zip(GROUPS, counts):
        metrics[f"success/{name}_resolved_count"] = int(resolved)
        metrics[f"success/{name}_success_count"] = int(success)
        if resolved:
            metrics[f"success/{name}_success_rate"] = float(success / resolved)
    return metrics
