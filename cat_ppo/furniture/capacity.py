"""Conservative empirical sizing for this fixed G1/Dex3 PPO configuration.

This is a planning estimate, not a GPU memory guarantee. The envelope exceeds
the observed full-PPO peaks at 5, 6, 8, 17 and 42 primitive boxes by at least
13%. It includes the robot, physics state, rollout and PPO working buffers;
it must not be confused with standalone MJX step memory or allocator pool size.
Unseen topologies still need observed allocation and bounded OOM recovery.
"""
from __future__ import annotations

import math


def estimated_memory_cap(scene, num_envs, budget_gib, min_envs=128):
    if type(num_envs) is not int or num_envs < 4 or num_envs % 4:
        raise ValueError("Parallel environment count must be a positive multiple of four")
    if type(min_envs) is not int or min_envs < 4 or min_envs % 4:
        raise ValueError("Minimum environment count must be a positive multiple of four")
    boxes = scene.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("Capacity estimate requires a materialized scene with primitive boxes")
    if budget_gib is not None and (isinstance(budget_gib, bool) or
            not isinstance(budget_gib, (int, float)) or not math.isfinite(budget_gib) or budget_gib <= 0):
        raise ValueError("Estimated memory budget must be finite and positive")
    per_env_bytes = (1 + .25 * len(boxes)) * 2**20
    count = num_envs
    if budget_gib is not None:
        while count * per_env_bytes > budget_gib * 2**30:
            reduced = count // 2
            if count % 2 or reduced < min_envs or reduced % 4:
                raise ValueError("Scene exceeds estimated memory budget even at the allowed batch floor")
            count = reduced
    return dict(schema="cat-empirical-capacity-v1", configured_num_envs=num_envs,
        num_envs=count, primitive_boxes=len(boxes), estimated_memory_budget_gib=budget_gib,
        estimated_peak_bytes=int(count * per_env_bytes),
        estimate="num_envs * (1 + 0.25 * primitive_boxes) MiB; empirical full-PPO envelope",
        guarantee=False)
