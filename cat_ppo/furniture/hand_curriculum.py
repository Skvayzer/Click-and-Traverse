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


def curriculum_levels(manifest, *, enabled):
    """Return levels (-1 for old scenes), or None unless both opt-ins are set.

Fail closed for an enabled malformed curriculum: an absent easy subgroup would
otherwise silently redistribute room mass or make the initial sampler empty.
"""
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
        if selected.size and (not np.any(selected == -1) or not np.any(selected == 0)):
            raise ValueError("Each represented room family needs ordinary rooms and easy hand tasks")
    if not np.any(levels >= 0):
        raise ValueError("An enabled hand curriculum must contain hand-protection tasks")
    return levels


def initial_curriculum_state():
    """Unbatched global counters; the existing Vmap wrapper broadcasts them."""
    return dict(zip(STATE_KEYS, (jp.int32(0), jp.zeros(3, jp.int32), jp.zeros(3, jp.int32))))


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

Counters use complete episodes, rather than an EMA or per-step goal events.
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
