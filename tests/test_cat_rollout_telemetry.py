"""Latest-rollout outcomes and bounded exploration must not use stale windows."""

import copy
import functools

from brax.training import distribution
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cat_ppo.learning.policy.ppo.train import (
    TrainingMetricsLogger, _current_distribution_metrics,
)
from cat_ppo.learning.policy.ppo.wholebody_distribution import (
    WholeBodyNormalTanhDistribution, make_ppo_networks,
)
from test_cat_learner_continuity import ToyMixedEnv, run_updates


def _logger():
    emitted = []
    return TrainingMetricsLogger(buffer_size=1000, steps_between_logging=1,
        progress_fn=lambda step, metrics: emitted.append((step, metrics))), emitted


def _episode_metrics():
    return {
        "length": np.array([[4, 99, 8]]),
        "sum_reward": np.array([[2, 999, 3]]),
        "wb_goal_reached": np.array([[1, 1, 0]], dtype=float),
        "wb_fall": np.array([[0, 1, 1]], dtype=float),
        "wb_obstacle": np.array([[0, 1, 1]], dtype=float),
        "wb_hand_violation": np.array([[0, 1, 1]], dtype=float),
        "wb_elbow_violation": np.array([[0, 1, 0]], dtype=float),
        "wb_outside_bounds": np.array([[0, 1, 0]], dtype=float),
        "wb_self_contact": np.array([[0, 1, 0]], dtype=float),
        "wb_numerical": np.array([[0, 1, 0]], dtype=float),
        "wb_mean_upper_target_velocity": np.array([[.2, 999, .6]]),
    }


def test_latest_outcome_rates_use_only_this_rollouts_completed_episodes():
    logger, emitted = _logger()
    metrics = _episode_metrics()
    dones = np.array([[1, 0, 1]])
    logger.update_rollout_metrics(metrics, dones, np.array([[1, 0, 0]]))
    first = emitted[-1][1]
    assert first["training/completed_episode_count"] == 2
    assert first["training/goal_success_rate"] == .5
    assert first["training/timeout_rate"] == .5
    assert first["training/termination_rate"] == .5
    assert first["training/fall_rate"] == .5
    assert first["training/obstacle_failure_rate"] == .5
    assert first["training/hand_violation_rate"] == .5
    for name in ("elbow_violation", "outside_bounds", "self_contact", "numerical_failure"):
        assert first[f"training/{name}_rate"] == 0
    assert first["training/mean_upper_target_velocity"] == pytest.approx(.4)
    assert first["episode/length"] == 6

    # Only the previous failure row completes now: current outcome differs from
    # both the completed-episode and rollout-callback historical averages.
    logger.update_rollout_metrics(metrics, np.array([[0, 0, 1]]), np.zeros_like(dones))
    current = emitted[-1][1]
    assert current["training/completed_episode_count"] == 1
    assert current["training/goal_success_rate"] == 0
    assert current["training/fall_rate"] == 1
    assert current["training/timeout_rate"] == 0
    assert current["training/mean_upper_target_velocity"] == .6
    assert current["episode_metrics/wb_goal_reached"] == pytest.approx(1 / 3)
    assert current["rollout/timeout_rate"] == .25  # Existing buffered key unchanged.


def test_no_completed_episode_does_not_reuse_previous_outcome_rates_or_noise():
    logger, emitted = _logger()
    logger.update_rollout_metrics(_episode_metrics(), np.ones((1, 3)), np.zeros((1, 3)),
        action_std=.4, distribution_metrics={"training/upper_std_mean": .1})
    logger.update_rollout_metrics(_episode_metrics(), np.zeros((1, 3)), np.zeros((1, 3)),
        action_std=.2, distribution_metrics={"training/upper_std_mean": .05})
    current = emitted[-1][1]
    assert current["training/completed_episode_count"] == 0
    assert "training/goal_success_rate" not in current
    assert "training/fall_rate" not in current
    assert "training/timeout_rate" not in current
    assert "training/mean_upper_target_velocity" not in current
    assert current["training/upper_std_mean"] == .05
    assert current["training/action_std"] == pytest.approx(.3)


def test_latest_state_round_trips_and_older_metric_state_remains_loadable():
    logger, _ = _logger()
    logger.update_rollout_metrics(_episode_metrics(), np.array([[1, 0, 1]]), np.zeros((1, 3)),
        distribution_metrics={"training/upper_std_max": .1})
    saved = logger.state_dict()
    restored, emitted = _logger()
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    restored.log_metrics()
    assert emitted[-1][1]["training/upper_std_max"] == .1
    assert emitted[-1][1]["training/goal_success_rate"] == .5
    old = copy.deepcopy(saved)
    del old["latest_rollout_metrics"]
    restored.load_state_dict(old)
    assert restored.state_dict()["latest_rollout_metrics"] == {}


def test_scale_telemetry_uses_actual_distribution_and_returns_only_scalars():
    repaired = WholeBodyNormalTanhDistribution(29)
    parameters = jnp.concatenate([jnp.zeros((2, 29)), jnp.full((2, 29), 15.)], axis=-1)
    scales = repaired.create_dist(parameters).scale
    metrics = jax.jit(lambda x: _current_distribution_metrics(repaired, x))(scales)
    assert all(value.shape == () for value in metrics.values())
    assert float(metrics["training/upper_std_mean"]) == pytest.approx(.1)
    assert float(metrics["training/upper_std_max"]) == pytest.approx(.1)
    assert float(metrics["training/upper_std_min"]) == pytest.approx(.1)
    assert float(metrics["training/leg_std_mean"]) > 15
    assert float(metrics["training/upper_std_bounds_violation_rate"]) == 0
    invalid = scales.at[0, 12].set(.2).at[0, 13].set(.01).at[0, 14].set(jnp.nan)
    invalid_metrics = _current_distribution_metrics(repaired, invalid)
    assert float(invalid_metrics["training/upper_std_bounds_violation_rate"]) == pytest.approx(3 / 34)
    assert _current_distribution_metrics(distribution.NormalTanhDistribution(29), scales) == {}


def test_native_learner_logs_current_noise_and_rejects_changed_distribution_on_resume():
    class ToyWholeBodyEnv(ToyMixedEnv):
        action_size = 29

    def factory(max_std):
        return functools.partial(make_ppo_networks,
            policy_hidden_layer_sizes=(4,), value_hidden_layer_sizes=(4,),
            policy_obs_key="state", value_obs_key="privileged_state", upper_std_max=max_std)

    snapshots, logs, _ = run_updates(1, environment=ToyWholeBodyEnv(), network_factory=factory(.1))
    contract = snapshots[0]["contract"]["distribution"]
    assert contract == WholeBodyNormalTanhDistribution(29).config
    telemetry = [metrics for _, metrics in logs if "training/upper_std_max" in metrics]
    assert telemetry
    assert all(metrics["training/upper_std_max"] <= np.float32(.1) for metrics in telemetry)
    assert all(metrics["training/upper_std_bounds_violation_rate"] == 0 for metrics in telemetry)
    with pytest.raises(ValueError, match="configuration differs"):
        run_updates(1, restore=snapshots[0], environment=ToyWholeBodyEnv(),
                    network_factory=factory(.08))
