"""Hand-task curriculum probabilities, clean goals and persistent state."""
from copy import deepcopy
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest

from cat_ppo.furniture.hand_curriculum import (
    STATE_KEYS, advance_curriculum, clean_goals, curriculum_levels,
    curriculum_metrics, hand_scene_logits, initial_curriculum_state,
)
from cat_ppo.learning.train.pf_utils import SamplePFWrapper

GROUPS = jp.array([0, 1, 2, 2, 2, 2, 3, 3, 3, 3])
LEVELS = jp.array([-1, -1, -1, 0, 1, 2, -1, 0, 1, 2])
MASSES = jp.array([.2, .4, .25, .15])


def _manifest():
    scenes = []
    for group, level in zip(np.asarray(GROUPS), np.asarray(LEVELS)):
        scene = {"sampling_group": ("original_cat", "procedural_cat", "furniture", "generic_clutter")[group],
                 "source": {}}
        if level >= 0:
            scene["source"]["hand_protection"] = {
                "kind": "hand_table_aisle" if group == 2 else "hand_shelf_passage",
                "difficulty": ("easy", "medium", "hard")[level], "level": int(level)}
        scenes.append(scene)
    return {"schema": "cat-generalist-field-bank-v2", "hand_protection_curriculum": True,
            "scenes": scenes}


def _info(scene_ids, *, stage=0):
    batch = len(scene_ids)
    tile = lambda value: jp.broadcast_to(value, (batch,) + value.shape)
    info = {
        "pf_id": jp.asarray(scene_ids), "truncation": jp.ones(batch),
        "pf_success_ema": jp.zeros((batch, len(LEVELS))),
        "pf_episode_ema": jp.zeros((batch, len(LEVELS))),
        "pf_sampling_logits": tile(hand_scene_logits(jp.ones(len(LEVELS)), GROUPS, MASSES, LEVELS, jp.int32(stage))),
        "pf_sampling_ema_decay": jp.ones(batch) * .95, "pf_sampling_alpha": jp.ones(batch),
        "pf_sampling_group_ids": tile(GROUPS), "pf_sampling_group_masses": tile(MASSES),
        "wholebody_episode": {"goal_reached": jp.zeros(batch, bool), "body_collision": jp.zeros(batch, bool)},
    }
    info.update({key: tile(value) for key, value in initial_curriculum_state().items()})
    info[STATE_KEYS[0]] = jp.full(batch, stage, jp.int32)
    return info


def _update(info, done):
    state, logits = SamplePFWrapper._update_pf_sampling_info(
        SimpleNamespace(info=info), done, LEVELS)
    return state.info, logits


@pytest.mark.parametrize("stage", range(3))
def test_family_and_old_room_mass_survive_different_difficulties_and_weights(stage):
    # Different counts and adaptive weights must not remove ordinary rooms or
    # change CAT's 60% total. Locked levels must stay exactly unsampleable.
    groups = jp.concatenate((GROUPS, jp.array([0] * 13 + [2] * 4 + [3] * 2)))
    levels = jp.concatenate((LEVELS, jp.array([-1] * 13 + [0, 0, 1, 2, 1, 2])))
    weights = jp.linspace(.001, 1., len(groups))
    probabilities = np.asarray(jax.nn.softmax(jax.jit(hand_scene_logits)(weights, groups, MASSES, levels, jp.int32(stage))))
    np.testing.assert_allclose(np.bincount(np.asarray(groups), weights=probabilities), MASSES, atol=1e-7)
    for group, expected in ((2, .125), (3, .075)):
        selected = np.asarray(groups) == group
        np.testing.assert_allclose(probabilities[selected & (np.asarray(levels) < 0)].sum(), expected, atol=1e-7)
        np.testing.assert_allclose(probabilities[selected & (np.asarray(levels) >= 0)].sum(), expected, atol=1e-7)
    np.testing.assert_array_equal(probabilities[np.asarray(levels) > stage], 0.)
    assert np.all(probabilities[np.asarray(levels) < 0] > 0.)


def test_requires_both_opt_ins_and_validated_family_level_metadata():
    manifest = _manifest()
    np.testing.assert_array_equal(curriculum_levels(manifest, enabled=True), LEVELS)
    assert curriculum_levels(manifest, enabled=False) is None
    manifest.pop("hand_protection_curriculum")
    assert curriculum_levels(manifest, enabled=True) is None
    invalid = _manifest()
    invalid["scenes"][3]["source"]["hand_protection"]["difficulty"] = "hard"
    with pytest.raises(ValueError, match="Invalid hand-protection"):
        curriculum_levels(invalid, enabled=True)
    invalid = _manifest()
    invalid["scenes"] = [scene for index, scene in enumerate(invalid["scenes"]) if index != 2]
    with pytest.raises(ValueError, match="ordinary rooms"):
        curriculum_levels(invalid, enabled=True)


def test_unlock_requires_64_completed_clean_goals_not_timeouts_or_step_events():
    info = _info([3] * 64)
    # Truncation is one for every episode. Timeout survival is not a hand goal.
    failed, logits = jax.jit(_update)(deepcopy(info), jp.ones(64))
    np.testing.assert_array_equal(failed[STATE_KEYS[0]], 0)
    np.testing.assert_array_equal(failed[STATE_KEYS[1]][0], [64, 0, 0])
    np.testing.assert_array_equal(failed[STATE_KEYS[2]][0], 0)
    np.testing.assert_array_equal(failed["pf_success_ema"], 0)
    assert np.isneginf(np.asarray(logits)[4])
    info["wholebody_episode"]["goal_reached"] = jp.ones(64, bool)
    unfinished, _ = jax.jit(_update)(deepcopy(info), jp.zeros(64))
    np.testing.assert_array_equal(unfinished[STATE_KEYS[1]], 0)
    np.testing.assert_array_equal(unfinished[STATE_KEYS[0]], 0)
    partial, _ = jax.jit(_update)(deepcopy(info), jp.array([1.] * 63 + [0.]))
    np.testing.assert_array_equal(partial[STATE_KEYS[0]], 0)
    unlocked, logits = jax.jit(_update)(info, jp.ones(64))
    np.testing.assert_array_equal(unlocked[STATE_KEYS[0]], 1)
    assert np.isfinite(np.asarray(logits)[4]) and np.isneginf(np.asarray(logits)[5])
    np.testing.assert_array_equal(unlocked[STATE_KEYS[2]][0], [64, 0, 0])


def test_collided_goals_do_not_unlock_and_old_scene_survival_is_unchanged():
    info = _info([3] * 64 + [0, 2])
    info["wholebody_episode"]["goal_reached"] = jp.ones(66, bool)
    info["wholebody_episode"]["body_collision"] = jp.array([True] * 26 + [False] * 40)
    updated, _ = jax.jit(_update)(info, jp.ones(66))
    np.testing.assert_array_equal(updated[STATE_KEYS[0]], 0)  # 38/64 < .6
    np.testing.assert_array_equal(updated[STATE_KEYS[2]][0], [38, 0, 0])
    np.testing.assert_array_equal(updated["pf_success_ema"][0, jp.array([0, 2, 3])], [1, 1, 38])
    snapshot = {"episode_metrics": {"wb_goal_reached": jp.array([1., 1.]),
                                    "wb_outside_bounds": jp.array([0., 1.])}}
    np.testing.assert_array_equal(clean_goals(snapshot), [True, False])


def test_stage_is_monotonic_and_easy_goals_cannot_unlock_hard():
    advance = jax.jit(advance_curriculum)
    stage, completed, goals = jp.int32(1), jp.array([64, 0, 0]), jp.array([64, 0, 0])
    stage, completed, goals = advance(stage, completed, goals, jp.zeros(1000, jp.int32),
                                     jp.ones(1000, bool), jp.ones(1000, bool))
    assert int(stage) == 1
    stage, completed, goals = advance(stage, completed, goals, jp.ones(64, jp.int32),
                                     jp.ones(64, bool), jp.array([True] * 39 + [False] * 25))
    assert int(stage) == 2
    stage, completed, goals = advance(stage, completed, goals, jp.full(128, 2, jp.int32),
                                     jp.ones(128, bool), jp.zeros(128, bool))
    assert int(stage) == 2
    metrics = curriculum_metrics(dict(zip(STATE_KEYS, (stage, completed, goals))))
    assert int(metrics["hand_curriculum/stage"]) == 2
    assert float(metrics["hand_curriculum/active_level_clean_goal_rate"]) == 0.


def test_disabled_update_remains_byte_equal_to_explicit_no_curriculum():
    info = _info([0, 1, 2, 6])
    for key in STATE_KEYS:
        del info[key]
    a, la = SamplePFWrapper._update_pf_sampling_info(SimpleNamespace(info=deepcopy(info)), jp.ones(4))
    b, lb = SamplePFWrapper._update_pf_sampling_info(SimpleNamespace(info=deepcopy(info)), jp.ones(4), None)
    for x, y in zip(jax.tree.leaves(a.info), jax.tree.leaves(b.info)):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(la, lb)
