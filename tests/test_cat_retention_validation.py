"""Regression metrics must count traversal, preserve failures, and gate forgetting."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from cat_ppo.furniture.retention_validation import (
    MODES, RetentionValidator, retention_selection, select_scenes, summarize_episodes,
)


def row(scene="hurdle", family="published_cat", *, success=True, **changes):
    result = dict(scene_id=scene, family=family, seed=0, goal_reached=success,
                  fall=False, obstacle=False, hand_violation=False, elbow_violation=False,
                  self_contact=False, numerical=False, outside_bounds=False, timeout=False,
                  seconds=2.0, min_hand_clearance=.2, **{"return": 1.0})
    result.update(changes)
    return result


def summaries():
    rows = [row(seed=seed) for seed in range(16)]
    rows += [row("room", "furniture", seed=seed) for seed in range(16)]
    return {mode: summarize_episodes(rows) for mode in MODES}


def test_scene_selection_by_identity_rejects_missing_duplicate_or_one_group():
    manifest = {"scenes": [dict(scene_id="room", family="furniture"),
                           dict(scene_id="hurdle", family="published_cat")]}
    assert select_scenes(manifest, ("hurdle", "room")) == [1, 0]
    for ids in (("missing", "room"), ("room", "room"), ("hurdle",), ("room",)):
        with pytest.raises(ValueError):
            select_scenes(manifest, ids)


def test_episode_rates_have_separate_cat_clutter_denominators():
    summary = summarize_episodes([
        row(), row(success=False, fall=True),
        row("room", "furniture", success=False, hand_violation=True, obstacle=True),
    ])
    assert summary["goal_success_rate"] == pytest.approx(1 / 3)
    assert summary["cat_goal_success_rate"] == .5
    assert summary["clutter_goal_success_rate"] == 0
    assert summary["clutter_hand_violation_rate"] == 1
    assert summary["hand_violation_rate"] == pytest.approx(1 / 3)
    assert summary["clutter_completion_time_count"] == 0
    assert summary["clutter_completion_time_seconds"] == 0


def test_selection_rejects_cat_regression_in_either_mode_despite_clutter_success():
    baseline = summaries()
    current = deepcopy(baseline)
    current["stochastic"]["cat_goal_success_rate"] = .9
    result = retention_selection(current, baseline)
    assert not result["eligible"]
    assert any("stochastic: aggregate" in reason for reason in result["reasons"])


def test_selection_per_scene_gate_catches_regression_hidden_by_aggregate():
    baseline = summaries()
    current = deepcopy(baseline)
    current["deterministic"]["scenes"]["hurdle"]["goal_success_rate"] = .75
    result = retention_selection(current, baseline)
    assert not result["eligible"]
    assert any("hurdle" in reason for reason in result["reasons"])


def test_selection_tolerance_boundaries_and_lexicographic_clutter_objective():
    baseline = summaries()
    current = deepcopy(baseline)
    current["deterministic"]["cat_goal_success_rate"] = .95
    current["deterministic"]["scenes"]["hurdle"]["goal_success_rate"] = .875
    result = retention_selection(current, baseline)
    assert result["eligible"]
    lower_success = deepcopy(current)
    lower_success["stochastic"]["clutter_goal_success_rate"] = .8
    assert retention_selection(lower_success, baseline)["score"] < result["score"]
    unsafe = deepcopy(current)
    unsafe["stochastic"]["clutter_hand_violation_rate"] = .1
    assert retention_selection(unsafe, baseline)["score"] < result["score"]
    numerical = deepcopy(current)
    numerical["deterministic"]["numerical_failure_rate"] = 1 / 32
    assert not retention_selection(numerical, baseline)["eligible"]


def test_selection_refuses_changed_populations_and_nonfinite_metrics():
    baseline = summaries()
    changed = deepcopy(baseline)
    changed["stochastic"]["scenes"]["hurdle"]["episode_count"] = 15
    with pytest.raises(ValueError, match="populations"):
        retention_selection(changed, baseline)
    invalid = deepcopy(baseline)
    invalid["deterministic"]["mean_return"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        retention_selection(invalid, baseline)


class ToyTraversalEnv:
    """Goal at steps 2/3; stepping beyond a goal would cause a hand failure."""

    action_size = 1
    dt = .02

    def __init__(self):
        import jax.numpy as jp
        self.field_bank_manifest = {"scenes": [dict(scene_id="hurdle", family="published_cat"),
                                                dict(scene_id="room", family="furniture")]}
        self._pf_scene_episode_lengths = jp.asarray([3, 5])
        self.sdf, self.bf, self.gf = (jp.ones((131072, width)) for width in (1, 3, 3))

    def observation_contract(self):
        return dict(actor_features=["state"], critic_features=["state"])

    def reset_with_pf_id(self, key, pf_id):
        del key
        from brax.envs import State
        import jax.numpy as jp
        value = self.sdf[pf_id, 0]
        episode = {name: jp.array(False) for name in (
            "goal_reached", "fall", "obstacle", "self_contact", "numerical",
            "outside_bounds", "hand_violation", "elbow_violation")}
        return State(None, {"state": jp.array([value]), "privileged_state": jp.array([value])},
                     jp.float32(0), jp.float32(0), {},
                     dict(pf_id=pf_id, age=jp.int32(0), wholebody_episode=episode,
                          wholebody_faults={"any": jp.array(False)}, handsdf=jp.array([[value], [value]])))

    def step(self, state, action):
        del action
        import jax.numpy as jp
        age = state.info["age"] + 1
        goal_step = state.info["pf_id"] + 2
        fail = age > goal_step
        episode = dict(state.info["wholebody_episode"], goal_reached=age == goal_step,
                       obstacle=fail, hand_violation=fail)
        value = self.sdf[state.info["pf_id"], 0]
        return state.replace(reward=jp.float32(1), done=fail.astype(jp.float32),
                             info=dict(state.info, age=age, wholebody_episode=episode,
                                       wholebody_faults={"any": fail},
                                       handsdf=jp.where(fail, -jp.ones((2, 1)), jp.full((2, 1), value))))


def toy_factory(shapes, action_size):
    del shapes
    import jax.numpy as jp
    from brax.training.distribution import NormalTanhDistribution
    def apply(normalizer, params, observations):
        del normalizer
        return jp.broadcast_to(params, (*observations["state"].shape[:-1], 2))
    return SimpleNamespace(policy_network=SimpleNamespace(apply=apply),
                           parametric_action_distribution=NormalTanhDistribution(action_size))


def test_validator_freezes_first_completion_and_uses_existing_dynamic_fields():
    import jax.numpy as jp
    environment = ToyTraversalEnv()
    fields = (environment.sdf, environment.bf, environment.gf)
    validator = RetentionValidator(environment, toy_factory, scene_ids=("hurdle", "room"),
                                   seeds=(0, 1), chunk_steps=2)
    result = validator.evaluate((None, jp.array([0., -2.]), None), step=0)
    for mode in MODES:
        assert result.modes[mode]["goal_success_rate"] == 1
        assert result.modes[mode]["hand_violation_rate"] == 0
        assert [row["length"] for row in result.episodes[mode]] == [2, 2, 3, 3]
        assert [row["return"] for row in result.episodes[mode]] == [2, 2, 3, 3]
    assert result.metrics["validation/goal_success_rate"] == 1
    assert result.metrics["validation/stochastic/goal_success_rate"] == 1
    assert all(getattr(environment, name) is value for name, value in zip(("sdf", "bf", "gf"), fields))
    # Existing executable sees changed field operands without loading a bank or
    # changing the raw environment. Large arrays must not become HLO literals.
    altered = (fields[0] * 2, *fields[1:])
    initial = validator._reset(altered)
    np.testing.assert_allclose(initial.info["handsdf"], 2)
    hlo = validator._reset.lower(fields).compiler_ir("hlo").as_serialized_hlo_module_proto()
    assert len(hlo) < 50000
