"""Exercise real training wrappers/callbacks across short synthetic episodes."""
import jax
import jax.numpy as jp
import numpy as np
from brax.envs.wrappers import training as brax_training

from mujoco_playground._src.mjx_env import State

from cat_ppo.furniture.training import wrap_for_furniture_training
from cat_ppo.learning.policy.ppo.train import TrainingMetricsLogger


class ShortContactTask:
    observation_size = 1
    action_size = 1

    @property
    def unwrapped(self):
        return self

    def reset(self, rng):
        del rng
        zero = jp.asarray(0.0)
        metrics = {name: zero for name in ("reward/progress", "success", "fall", "hand_contact",
            "forbidden_contact", "min_hand_clearance_m", "route_fraction", "cross_track_m", "map_age_seconds")}
        return State(data=jp.zeros(1), obs=jp.zeros(1), reward=zero, done=zero, metrics=metrics,
            info={"clock": jp.asarray(0), "contact": jp.asarray(False), "motor_target": zero,
                  "map_history": jp.zeros(2), "minimum_clearance": jp.asarray(1.0)})

    def step(self, state, action):
        position = state.data + action
        info = dict(state.info)
        info["clock"] += 1
        info["contact"] |= position[0] >= 2
        info["motor_target"] += action[0]
        info["map_history"] += 10
        info["minimum_clearance"] = jp.minimum(info["minimum_clearance"], 1 - .25 * position[0])
        metrics = {"reward/progress": action[0], "success": jp.asarray(0.0), "fall": jp.asarray(0.0),
            "hand_contact": info["contact"].astype(jp.float32),
            "forbidden_contact": info["contact"].astype(jp.float32),
            "min_hand_clearance_m": info["minimum_clearance"], "route_fraction": position[0] / 2,
            "cross_track_m": position[0], "map_age_seconds": .02 * info["clock"]}
        return state.replace(data=position, obs=position, reward=action[0],
            done=info["contact"].astype(jp.float32), metrics=metrics, info=info)


def test_complete_autoreset_preserves_terminal_info_then_resets_only_finished_slots():
    env = wrap_for_furniture_training(ShortContactTask(), episode_length=10)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 2))
    step = jax.jit(env.step)
    action = jp.asarray([[1.0], [.5]])
    state = step(state, action)
    state = step(state, action)
    np.testing.assert_array_equal(state.done, [1, 0])
    np.testing.assert_allclose(state.data[:, 0], [0, 1])
    np.testing.assert_array_equal(state.info["contact"], [True, False])
    np.testing.assert_array_equal(state.info["clock"], [2, 2])
    np.testing.assert_allclose(state.info["episode_metrics"]["min_hand_clearance_m"], [.5, .75])
    np.testing.assert_allclose(state.info["episode_metrics"]["cross_track_m"], [1.5, .75])
    state = step(state, action)
    np.testing.assert_array_equal(state.done, [0, 0])
    np.testing.assert_array_equal(state.info["contact"], [False, False])
    np.testing.assert_array_equal(state.info["clock"], [1, 3])
    np.testing.assert_allclose(state.info["motor_target"], [1, 1.5])
    np.testing.assert_allclose(state.info["map_history"], [[10, 10], [30, 30]])
    np.testing.assert_allclose(state.info["minimum_clearance"], [.75, .625])
    np.testing.assert_allclose(state.info["episode_metrics"]["sum_reward"], [1, 1.5])
    np.testing.assert_array_equal(state.info["episode_metrics"]["length"], [1, 3])


def test_episode_telemetry_emits_outcomes_and_clearance_after_completion_without_evaluation():
    env = wrap_for_furniture_training(ShortContactTask(), episode_length=10)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 1))
    emitted = []
    logger = TrainingMetricsLogger(buffer_size=100, steps_between_logging=1,
                                   progress_fn=lambda step, metrics: emitted.append((step, metrics)))
    step = jax.jit(env.step)
    for _ in range(4):
        state = step(state, jp.ones((1, 1)))
        logger.update_rollout_metrics(state.info["episode_metrics"], state.info["episode_done"],
                                      state.info["truncation"])
    assert emitted[0][1]["rollout/completed_episodes"] == 0
    assert "episode_metrics/success" not in emitted[0][1]
    assert "episode_metrics/min_hand_clearance_m" not in emitted[0][1]
    last_step, metrics = emitted[-1]
    assert last_step == 4
    assert metrics["rollout/completed_episodes"] == 2
    assert metrics["rollout/completed_episodes_in_buffer"] == 2
    assert metrics["episode_metrics/hand_contact"] == 1
    assert metrics["episode_metrics/forbidden_contact"] == 1
    assert metrics["episode_metrics/success"] == 0
    assert metrics["episode_metrics/fall"] == 0
    assert metrics["episode_metrics/min_hand_clearance_m"] == .5
    assert metrics["episode_metrics/route_fraction"] == 1
    assert metrics["episode_metrics/cross_track_m"] == 1.5
    np.testing.assert_allclose(metrics["episode_metrics/map_age_seconds"], .03)
    assert metrics["episode/sum_reward"] == 2
    assert metrics["episode/reward/progress"] == 2
    assert metrics["episode/length"] == 2


def test_wrapper_time_limit_resets_clock_without_inventing_contact():
    env = wrap_for_furniture_training(ShortContactTask(), episode_length=1)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 1))
    step = jax.jit(env.step)
    for _ in range(3):
        state = step(state, jp.ones((1, 1)))
        np.testing.assert_array_equal(state.done, [1])
        np.testing.assert_array_equal(state.info["truncation"], [1])
        np.testing.assert_array_equal(state.info["clock"], [1])
        np.testing.assert_array_equal(state.info["contact"], [False])
        np.testing.assert_array_equal(state.info["episode_metrics"]["length"], [1])


def test_optional_brax_evaluation_consumes_first_terminal_outcomes_only():
    # Exercise the optional validation wrapper contract using a synthetic task,
    # without running a robot policy or enabling evaluation in the launcher.
    env = brax_training.EvalWrapper(wrap_for_furniture_training(ShortContactTask(), episode_length=10))
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 1))
    step = jax.jit(env.step)
    for _ in range(4):
        state = step(state, jp.ones((1, 1)))
    metrics = state.info["eval_metrics"]
    np.testing.assert_array_equal(metrics.active_episodes, [0])
    np.testing.assert_array_equal(metrics.episode_steps, [2])
    np.testing.assert_array_equal(metrics.episode_metrics["hand_contact"], [1])
    np.testing.assert_array_equal(metrics.episode_metrics["forbidden_contact"], [1])
