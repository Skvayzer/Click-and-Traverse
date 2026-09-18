"""Online traversal outcomes resolve once without changing physical episodes."""
from copy import deepcopy

from brax.envs.wrappers import training
from flax import serialization
import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco_playground import wrapper

from cat_ppo.furniture.generalist_training import SceneEpisodeWrapper
from cat_ppo.furniture.hand_curriculum import (
    NAVIGATION_COUNTS_KEY, NAVIGATION_COUNTED_KEY, STATE_KEYS,
    initial_navigation_outcomes, navigation_count_snapshot, navigation_rollout_metrics,
    navigation_scene_groups, update_navigation_outcomes,
)
from cat_ppo.learning.train.pf_utils import SamplePFWrapper
from test_cat_collision_transition import _CollisionTask, _assert_tree_equal
from test_hand_curriculum import _info, LEVELS


def _update(info, done, groups):
    resolved, success = update_navigation_outcomes(info, done, groups)
    return info, resolved, success


def _outcomes(scene_ids):
    size = len(scene_ids)
    return {"pf_id": jp.asarray(scene_ids), "truncation": jp.zeros(size),
            "wholebody_episode": {"goal_reached": jp.zeros(size, bool),
                                  "body_collision": jp.zeros(size, bool),
                                  "outside_bounds": jp.zeros(size, bool)},
            **initial_navigation_outcomes((size,))}


def test_scene_populations_are_mutually_exclusive_and_include_all_original_types():
    manifest = {"scenes": [dict(family=family) for family in
                ("original_cat", "published_cat", "procedural_cat", "furniture", "generic_clutter")]}
    manifest["scenes"].append(dict(family="furniture", source=dict(hand_protection={"level": 0})))
    np.testing.assert_array_equal(navigation_scene_groups(manifest), [0, 0, 0, 1, 1, 2])
    with pytest.raises(ValueError, match="nonempty"):
        navigation_scene_groups({"scenes": []})
    manifest["scenes"][-1]["family"] = "original_cat"
    with pytest.raises(ValueError, match="clutter family"):
        navigation_scene_groups(manifest)


def test_first_goal_failure_timeout_and_family_counts_have_correct_precedence():
    info = _outcomes([0, 1, 2, 3, 3, 3, 2, 0])
    info["wholebody_episode"]["goal_reached"] = jp.array([1, 0, 0, 1, 1, 0, 1, 0], bool)
    info["wholebody_episode"]["body_collision"] = jp.array([0, 1, 0, 0, 1, 0, 0, 0], bool)
    info["wholebody_episode"]["outside_bounds"] = jp.array([0, 0, 0, 0, 0, 0, 0, 1], bool)
    info["truncation"] = jp.array([0, 0, 1, 0, 0, 0, 1, 0])
    done = jp.array([0, 1, 1, 0, 1, 0, 1, 0])
    groups = jp.array([0, 0, 1, 2])
    updated, resolved, successful = jax.jit(_update)(info, done, groups)
    np.testing.assert_array_equal(resolved, [1, 1, 1, 1, 1, 0, 1, 1])
    np.testing.assert_array_equal(successful, [1, 0, 0, 1, 0, 0, 1, 0])
    expected = np.array([[3, 1], [2, 1], [2, 1], [7, 3]])
    np.testing.assert_array_equal(navigation_count_snapshot(updated), expected)
    # Once counted, later collisions cannot subtract successes or add attempts.
    updated["wholebody_episode"]["body_collision"] = jp.ones(8, bool)
    updated["truncation"] = jp.zeros(8)
    again, resolved, successful = jax.jit(_update)(updated, jp.ones(8), groups)
    np.testing.assert_array_equal(resolved, [0, 0, 0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(successful, 0)
    np.testing.assert_array_equal(navigation_count_snapshot(again), [[3, 1], [2, 1], [3, 1], [8, 3]])


def test_terminal_metric_snapshot_survives_raw_autoreset_and_failure_wins_goal():
    info = _outcomes([0, 0])
    # Raw flags may already have been reset; SceneEpisodeWrapper's captured
    # metrics describe the transition whose navigation outcome is being scored.
    info["episode_metrics"] = {"wb_goal_reached": jp.ones(2), "wb_body_collision": jp.array([1., 0.])}
    info["truncation"] = jp.array([0., 1.])
    updated, _, successful = _update(info, jp.ones(2), jp.array([0]))
    np.testing.assert_array_equal(successful, [False, True])
    np.testing.assert_array_equal(navigation_count_snapshot(updated)[0], [2, 1])


def test_first_outcome_curriculum_unlocks_before_horizon_and_does_not_double_count():
    info = _info([3] * 64)
    info.update(initial_navigation_outcomes((64,)))
    info["truncation"] = jp.zeros(64)
    info["wholebody_episode"]["goal_reached"] = jp.ones(64, bool)
    groups = jp.where(LEVELS >= 0, 2, 1)
    def update(value, done):
        from types import SimpleNamespace
        state, _ = SamplePFWrapper._update_pf_sampling_info(SimpleNamespace(info=value), done, LEVELS, groups)
        return state.info
    first = jax.jit(update)(info, jp.zeros(64))
    np.testing.assert_array_equal(first[STATE_KEYS[0]], 1)
    np.testing.assert_array_equal(first[STATE_KEYS[1]][0], [64, 0, 0])
    np.testing.assert_array_equal(first[STATE_KEYS[2]][0], [64, 0, 0])
    np.testing.assert_array_equal(first[NAVIGATION_COUNTS_KEY][0, 2], [64, 64])
    # Existing physical-episode sampling statistics still await terminal events.
    np.testing.assert_array_equal(first["pf_episode_ema"], 0)
    second = jax.jit(update)(first, jp.zeros(64))
    np.testing.assert_array_equal(second[STATE_KEYS[1]], first[STATE_KEYS[1]])
    second["wholebody_episode"]["body_collision"] = jp.ones(64, bool)
    third = jax.jit(update)(second, jp.ones(64))
    np.testing.assert_array_equal(third[STATE_KEYS[2]], first[STATE_KEYS[2]])
    np.testing.assert_array_equal(third[NAVIGATION_COUNTS_KEY], first[NAVIGATION_COUNTS_KEY])


def test_snapshot_reads_one_replica_per_device_and_rollout_rates_use_deltas():
    previous = jp.array([[10, 5], [7, 1], [2, 0], [19, 6]], jp.int32)
    increments = jp.array([[4, 3], [0, 0], [8, 2], [12, 5]], jp.int32)
    current = previous + increments
    for shape in ((4, 2), (16, 4, 2), (1, 16, 4, 2)):
        info = {NAVIGATION_COUNTS_KEY: jp.broadcast_to(current, shape)}
        np.testing.assert_array_equal(navigation_count_snapshot(info), current)
    devices = jp.stack([jp.broadcast_to(previous, (16, 4, 2)), jp.broadcast_to(current, (16, 4, 2))])
    np.testing.assert_array_equal(navigation_count_snapshot({NAVIGATION_COUNTS_KEY: devices}), previous + current)
    metrics = navigation_rollout_metrics(previous, current)
    assert metrics["training/goal_success_rate"] == 5 / 12
    assert metrics["training/resolved_count"] == 12 and metrics["training/goal_success_count"] == 5
    assert metrics["training/cat_goal_success_rate"] == .75
    assert metrics["training/hand_protection_goal_success_rate"] == .25
    assert metrics["training/ordinary_clutter_resolved_count"] == 0
    assert "training/ordinary_clutter_goal_success_rate" not in metrics
    empty = navigation_rollout_metrics(current, current)
    assert len(empty) == 8 and not any("rate" in key for key in empty)
    assert all(value == 0 for value in empty.values())


@pytest.mark.parametrize("bad", ["float", "shape", "negative", "success_gt_resolved", "total", "rewind", "success_delta"])
def test_bad_rollout_populations_fail_instead_of_reporting_misleading_rates(bad):
    previous = np.array([[2, 1], [0, 0], [0, 0], [2, 1]], np.int32)
    current = previous.copy()
    if bad == "float":
        current = current.astype(float)
    elif bad == "shape":
        current = current[:3]
    elif bad == "negative":
        current[0] = current[3] = [-1, 0]
    elif bad == "success_gt_resolved":
        current[0] = current[3] = [2, 3]
    elif bad == "total":
        current[3, 0] += 1
    elif bad == "rewind":
        current[:] = 0
    elif bad == "success_delta":
        current[0] = current[3] = [2, 2]
    with pytest.raises(ValueError):
        navigation_rollout_metrics(previous, current)


class _NavigationTask(_CollisionTask):
    field_bank_manifest = {"scenes": [dict(family="original_cat"), dict(family="procedural_cat"),
        dict(family="furniture"), dict(family="generic_clutter", source=dict(hand_protection={"level": 0}))]}

    def __init__(self, enabled):
        super().__init__(True)
        self._config.wholebody_first_outcome_metrics = enabled

    def reset(self, key):
        state = super().reset(key)
        state.info["wholebody_episode"]["goal_reached"] = jp.array(False)
        return state

    def step(self, state, action):
        result = super().step(state, action)
        result.info["wholebody_episode"]["goal_reached"] = (
            state.info["wholebody_episode"]["goal_reached"] | (action[0] < 0) | (action[0] > 1))
        return result


def _wrapped_online(enabled=True, horizon=4):
    task = _NavigationTask(enabled)
    return SamplePFWrapper(wrapper.BraxAutoResetWrapper(
        SceneEpisodeWrapper(training.VmapWrapper(task), horizon, 1, jp.full(4, horizon))))


def test_physical_episode_continues_after_arrival_then_collision_resets_only_latch():
    env = _wrapped_online()
    state = env.reset(jp.array([[2, 7], [3, 7]], jp.uint32))
    advance = jax.jit(env.step)
    first = advance(state, -jp.ones((2, 1)))
    np.testing.assert_array_equal(first.done, 0)
    np.testing.assert_array_equal(first.reward, 1.)
    np.testing.assert_array_equal(first.info[NAVIGATION_COUNTED_KEY], True)
    np.testing.assert_array_equal(navigation_count_snapshot(first.info), [[0, 0], [1, 1], [1, 1], [2, 2]])
    terminal = advance(first, jp.ones((2, 1)))
    np.testing.assert_array_equal(terminal.done, 1)
    np.testing.assert_array_equal(terminal.reward, -5.)
    np.testing.assert_array_equal(terminal.info[NAVIGATION_COUNTED_KEY], False)
    np.testing.assert_array_equal(navigation_count_snapshot(terminal.info), navigation_count_snapshot(first.info))
    # Both new scenes differ from their old populations: count the current
    # physical episode, not the scene chosen by the same terminal autoreset.
    assert np.any(np.asarray(terminal.info["pf_id"]) != np.asarray(first.info["pf_id"]))
    following = advance(terminal, -jp.ones((2, 1)))
    delta = navigation_rollout_metrics(navigation_count_snapshot(terminal.info), navigation_count_snapshot(following.info))
    assert delta["training/resolved_count"] == 2 and delta["training/goal_success_count"] == 2
    for index, label in enumerate(("cat", "ordinary_clutter", "hand_protection")):
        expected = sum(navigation_scene_groups(_NavigationTask.field_bank_manifest)[int(scene)] == index
                       for scene in terminal.info["pf_id"])
        assert delta[f"training/{label}_resolved_count"] == expected


def test_online_accounting_does_not_change_observations_rewards_dones_or_reset_draws():
    enabled, legacy = _wrapped_online(), _wrapped_online(False)
    keys = jp.array([[0, 7], [2, 7], [3, 7]], jp.uint32)
    a, b = enabled.reset(keys), legacy.reset(keys)
    assert NAVIGATION_COUNTS_KEY not in b.info and NAVIGATION_COUNTED_KEY not in b.info
    step_a, step_b = jax.jit(enabled.step), jax.jit(legacy.step)
    for actions in ([-1., 0., -1.], [0., 2., 0.], [0., -1., 1.], [0., 0., 0.]):
        a, b = step_a(a, jp.asarray(actions)[:, None]), step_b(b, jp.asarray(actions)[:, None])
        for name in ("obs", "data", "reward", "done"):
            _assert_tree_equal(getattr(a, name), getattr(b, name))
        np.testing.assert_array_equal(a.info["pf_id"], b.info["pf_id"])
        np.testing.assert_array_equal(a.info["rng"], b.info["rng"])


def test_serialized_mid_episode_resume_preserves_latch_and_cumulative_counts():
    env = _wrapped_online()
    advance = jax.jit(env.step)
    state = advance(env.reset(jp.array([[3, 7]], jp.uint32)), -jp.ones((1, 1)))
    restored = serialization.from_bytes(state, serialization.to_bytes(state))
    uninterrupted = advance(state, jp.zeros((1, 1)))
    resumed = advance(restored, jp.zeros((1, 1)))
    _assert_tree_equal(uninterrupted, resumed)
    np.testing.assert_array_equal(navigation_count_snapshot(resumed.info)[3], [1, 1])
    # A timeout after that arrival also cannot duplicate the success.
    for _ in range(2):
        resumed = advance(resumed, jp.zeros((1, 1)))
    np.testing.assert_array_equal(resumed.done, 1)
    np.testing.assert_array_equal(navigation_count_snapshot(resumed.info)[3], [1, 1])
    np.testing.assert_array_equal(resumed.info[NAVIGATION_COUNTED_KEY], False)
