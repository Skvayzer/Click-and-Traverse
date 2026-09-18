"""Small, opt-in success curriculum inside CAT's existing scene families.

The four family masses remain unchanged. Each room family reserves half its
mass for ordinary irregular rooms and half for eligible hand-protection tasks.
Only real, fault-free endpoint successes unlock more difficult hand tasks;
CAT's historical timeout-survival statistic remains unchanged for old scenes.
Scene levels are immutable O(number of scenes) constants, never batch state.
"""
from __future__ import annotations

import jax.numpy as jp
import numpy as np

HAND_FRACTION = .5
MIN_COMPLETED = 64
SUCCESS_THRESHOLD = .6
KINDS = {"hand_table_aisle": "furniture", "hand_shelf_passage": "generic_clutter"}
DIFFICULTIES = ("easy", "medium", "hard")
STATE_KEYS = ("pf_hand_curriculum_stage", "pf_hand_curriculum_completed", "pf_hand_curriculum_goals")
FAILURE_KEYS = ("fall", "obstacle", "self_contact", "numerical", "body_collision",
                "hand_violation", "elbow_violation", "outside_bounds")
NAVIGATION_COUNTS_KEY = "pf_navigation_outcome_counts"
NAVIGATION_COUNTED_KEY = "wholebody_navigation_outcome_counted"
NAVIGATION_GROUPS = ("cat", "ordinary_clutter", "hand_protection", "all")


def curriculum_levels(manifest, *, enabled):
    """Return levels (-1 for old scenes), or None unless both opt-ins are set.

Fail closed for an enabled malformed curriculum: an absent easy subgroup would
otherwise silently redistribute room mass or make the initial sampler empty.
"""
    specialist = "specialist" in manifest
    if specialist:
        from cat_ppo.furniture.hand_specialist import validate_specialist_manifest
        validate_specialist_manifest(manifest)
        if not enabled:
            raise ValueError("Hand-protection specialist requires the hand curriculum to be enabled")
    if not enabled or not manifest.get("hand_protection_curriculum"):
        return None
    if manifest.get("schema") != "cat-generalist-field-bank-v2":
        raise ValueError("Hand curriculum requires the expanded four-family bank")
    levels = np.full(len(manifest["scenes"]), -1, dtype=np.int32)
    for index, scene in enumerate(manifest["scenes"]):
        record = scene.get("source", {}).get("hand_protection")
        if record is None:
            continue
        kind, level = record.get("kind"), record.get("level")
        if (kind not in KINDS or type(level) is not int or level not in range(3)
                or record.get("difficulty") != DIFFICULTIES[level]
                or scene.get("sampling_group") != KINDS[kind]):
            raise ValueError("Invalid hand-protection scene kind, level, difficulty or family")
        levels[index] = level
    groups = np.asarray([scene.get("sampling_group") for scene in manifest["scenes"]])
    for group in ("furniture", "generic_clutter"):
        selected = levels[groups == group]
        if selected.size and ((not specialist and not np.any(selected == -1)) or not np.any(selected == 0)):
            raise ValueError("Each represented room family needs ordinary rooms and easy hand tasks")
    if not np.any(levels >= 0):
        raise ValueError("An enabled hand curriculum must contain hand-protection tasks")
    return levels


def initial_curriculum_state():
    """Unbatched global counters; the existing Vmap wrapper broadcasts them."""
    return dict(zip(STATE_KEYS, (jp.int32(0), jp.zeros(3, jp.int32), jp.zeros(3, jp.int32))))


def navigation_scene_groups(manifest):
    """Static scene lookup for mutually exclusive online navigation populations."""
    groups = []
    for scene in manifest["scenes"]:
        room = scene["family"] in ("furniture", "generic_clutter")
        hand = bool(scene.get("source", {}).get("hand_protection"))
        if hand and not room:
            raise ValueError("Hand navigation scenes must belong to a clutter family")
        groups.append(2 if hand else 1 if room else 0)
    if not groups:
        raise ValueError("Online navigation outcomes require a nonempty scene bank")
    return np.asarray(groups, dtype=np.int32)


def initial_navigation_outcomes(batch_shape):
    """Eight replicated global integers and one boolean per physical episode.

    These are bookkeeping only: no policy observations, reward changes, or new
    reset conditions. The pf_ counters survive scene resets; the wholebody_
    episode latch follows the existing extension-state reset mechanism.
    """
    return {NAVIGATION_COUNTS_KEY: jp.zeros(tuple(batch_shape) + (4, 2), jp.int32),
            NAVIGATION_COUNTED_KEY: jp.zeros(tuple(batch_shape), jp.bool_)}


def update_navigation_outcomes(info, done, scene_groups):
    """Resolve each physical episode once at first clean goal or first failure.

    Native failures win over simultaneous goals. A clean goal at the horizon
    still succeeds. After first arrival, later collisions remain ordinary
    environment failures but cannot rewrite this navigation outcome. Goal
    arrival does not terminate the environment or modify its reward.

    Called before SamplePFWrapper replaces the scene/reset information. Returns
    the two event masks for optional first-outcome curriculum progression.
    """
    done = jp.asarray(done, jp.bool_)
    counted = info[NAVIGATION_COUNTED_KEY]
    truncation = jp.asarray(info.get("truncation", jp.zeros_like(done)), jp.bool_)
    failure = done & ~truncation
    episode, metrics = info.get("wholebody_episode", {}), info.get("episode_metrics", {})
    for key in FAILURE_KEYS:
        value = metrics.get("wb_" + key, episode.get(key))
        if value is not None:
            failure = failure | jp.asarray(value, jp.bool_)
    failure = failure | info.get("wholebody_faults", {}).get("any", jp.zeros_like(done))
    success = clean_goals(info) & ~failure
    resolved = ~counted & (success | failure | done)
    successful = resolved & success
    groups = jp.asarray(scene_groups, jp.int32)[info["pf_id"]]
    resolved_counts = jp.bincount(groups, weights=resolved.astype(jp.int32), length=3)
    success_counts = jp.bincount(groups, weights=successful.astype(jp.int32), length=3)
    increments = jp.stack((resolved_counts, success_counts), axis=-1)
    increments = jp.concatenate((increments, jp.sum(increments, axis=0, keepdims=True)), axis=0)
    previous = info[NAVIGATION_COUNTS_KEY].reshape(-1, 4, 2)[0]
    info[NAVIGATION_COUNTS_KEY] = jp.broadcast_to(previous + increments, info[NAVIGATION_COUNTS_KEY].shape)
    info[NAVIGATION_COUNTED_KEY] = counted | resolved
    return resolved, successful


def navigation_count_snapshot(info):
    """Return eight device scalars without copying the replicated environment batch.

    Accept a raw (4,2), a batch (B,4,2), or a device batch (D,B,4,2). Global
    counters are replicated within each device's environment batch, not across
    devices: sum one copy per device. The caller may np.asarray this tiny result.
    """
    counts = info[NAVIGATION_COUNTS_KEY]
    if counts.ndim == 2:
        return counts
    if counts.ndim == 3:
        return counts[0]
    if counts.ndim == 4:
        return jp.sum(counts[:, 0], axis=0)
    raise ValueError("Unexpected online navigation counter shape")


def navigation_rollout_metrics(previous, current):
    """Host-side rates from outcomes resolved during exactly one learner update.

    Keep both snapshots in the caller's update loop; exact runtime resume already
    restores the counters in env_state. No extra metric windows or historical
    averages are used. Zero-outcome groups emit counts but omit undefined rates.
    """
    previous, current = np.asarray(previous), np.asarray(current)
    if (previous.shape != (4, 2) or current.shape != (4, 2)
            or not np.issubdtype(previous.dtype, np.integer)
            or not np.issubdtype(current.dtype, np.integer)):
        raise ValueError("Online navigation snapshots must be 4x2 integer counts")
    for snapshot in (previous, current):
        if (np.any(snapshot < 0) or np.any(snapshot[:, 1] > snapshot[:, 0])
                or not np.array_equal(snapshot[3], snapshot[:3].sum(axis=0))):
            raise ValueError("Online navigation snapshot has inconsistent populations")
    counts = current.astype(np.int64) - previous.astype(np.int64)
    if np.any(counts < 0) or np.any(counts[:, 1] > counts[:, 0]):
        raise ValueError("Online navigation counters regressed within a learner update")
    result = {}
    for group, (resolved, success) in zip(NAVIGATION_GROUPS, counts):
        prefix = "training/" + ("" if group == "all" else group + "_")
        result[prefix + "resolved_count"] = int(resolved)
        result[prefix + "goal_success_count"] = int(success)
        if resolved:
            result[prefix + "goal_success_rate"] = float(success / resolved)
    return result


def hand_scene_logits(weights, group_ids, group_masses, levels, stage):
    """Inverse-failure weights normalized separately in old/hand subgroups.

Groups 0 and 1 retain their complete original masses. In room groups 2 and 3,
eligible hand levels share 50% and ordinary rooms share 50%. Locked levels have
exactly zero probability (-inf logits), including during initial reset.
"""
    is_hand = levels >= 0
    eligible = ~is_hand | (levels <= stage)
    subgroup = group_ids * 2 + is_hand.astype(jp.int32)
    active_weights = jp.where(eligible, weights, 0.)
    totals = jp.zeros(8, weights.dtype).at[subgroup].add(active_weights)
    subgroup_mass = group_masses[group_ids] * jp.where(group_ids >= 2, HAND_FRACTION, 1.)
    probabilities = active_weights / jp.maximum(totals[subgroup], 1e-20) * subgroup_mass
    return jp.where(probabilities > 0., jp.log(jp.maximum(probabilities, 1e-30)), -jp.inf)


def clean_goals(info):
    """Sticky endpoint flags with all accumulated faults excluded, never truncation.

Episode metrics are captured before autoreset and survive wrapper changes. The
raw sticky flags are equivalent and support callers outside EpisodeWrapper.
"""
    metrics = info.get("episode_metrics", {})
    episode = info.get("wholebody_episode", {})
    if "wb_goal_reached" in metrics:
        goal = metrics["wb_goal_reached"].astype(jp.bool_)
        for key in FAILURE_KEYS:
            if "wb_" + key in metrics:
                goal = goal & ~metrics["wb_" + key].astype(jp.bool_)
    else:
        goal = episode["goal_reached"].astype(jp.bool_)
        for key in FAILURE_KEYS:
            if key in episode:
                goal = goal & ~episode[key].astype(jp.bool_)
    return goal


def advance_curriculum(stage, completed, goals, scene_levels, done, clean_goal):
    """Update three tiny counters and unlock at most one level per transition.

The caller supplies physical done masks in legacy mode, or first-outcome masks
in the opt-in mode. Each attempt is counted once, rather than once per goal step.
Only the current level qualifies for an unlock; easy successes cannot unlock
hard scenes after medium starts. Previously unlocked levels remain available.
"""
    valid = (scene_levels >= 0) & done.astype(jp.bool_)
    indices = jp.maximum(scene_levels, 0)
    completed = completed.at[indices].add(valid.astype(jp.int32))
    goals = goals.at[indices].add((valid & clean_goal).astype(jp.int32))
    rate = goals[stage] / jp.maximum(completed[stage], 1)
    unlock = (stage < 2) & (completed[stage] >= MIN_COMPLETED) & (rate >= SUCCESS_THRESHOLD)
    return stage + unlock.astype(jp.int32), completed, goals


def curriculum_metrics(info):
    """Scalar JAX metrics for the learner's existing single W&B run.

    Accepts unbatched state, a training batch, or a device-by-batch state.
"""
    if STATE_KEYS[0] not in info:
        return {}
    stage, completed, goals = (info[key] for key in STATE_KEYS)
    stage = stage.reshape(-1)[0]
    completed, goals = completed.reshape(-1, 3)[0], goals.reshape(-1, 3)[0]
    metrics = {"hand_curriculum/stage": stage,
               "hand_curriculum/active_level_completed": completed[stage],
               "hand_curriculum/active_level_clean_goal_rate": goals[stage] / jp.maximum(completed[stage], 1)}
    for level, name in enumerate(DIFFICULTIES):
        metrics["hand_curriculum/" + name + "_completed"] = completed[level]
        metrics["hand_curriculum/" + name + "_clean_goal_rate"] = goals[level] / jp.maximum(completed[level], 1)
    return metrics
