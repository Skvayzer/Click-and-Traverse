"""Real CPU PPO obtains update-local navigation rates without evaluation."""
import jax.numpy as jp
import numpy as np
import pytest

from cat_ppo.furniture.generalist_runtime import atomic_save_runtime, load_runtime
from cat_ppo.furniture.hand_curriculum import (
    NAVIGATION_COUNTS_KEY, NAVIGATION_COUNTED_KEY, initial_navigation_outcomes,
    update_navigation_outcomes,
)
from test_cat_learner_continuity import ToyMixedEnv, assert_tree_equal, ppo, run_updates


class ToyNavigationEnv(ToyMixedEnv):
    """Two slots cycle original/clutter/hand scenes, with known first outcomes.

    Original scenes reach their goal at step one, ordinary clutter at step two,
    and hand scenes fail at step three. Every physical episode still lasts
    three steps, so success must be counted before physical reset and once only.
    """

    def reset(self, keys):
        state = super().reset(keys)
        state.info.update(initial_navigation_outcomes(state.done.shape))
        state.info["wholebody_episode"] = {
            "goal_reached": jp.zeros_like(state.done, dtype=bool),
            "body_collision": jp.zeros_like(state.done, dtype=bool),
        }
        return state

    def step(self, state, action):
        old_ids = state.info["pf_id"]
        age = state.info["age"] + 1
        next_state = super().step(state, action)
        done = next_state.done.astype(bool)
        goal = ((old_ids == 0) & (age >= 1)) | ((old_ids == 1) & (age >= 2))
        collision = (old_ids == 2) & done
        info = dict(next_state.info)
        info.update(pf_id=old_ids, truncation=(done & ~collision).astype(jp.float32),
                    wholebody_episode={"goal_reached": goal, "body_collision": collision})
        update_navigation_outcomes(info, done, jp.array([0, 1, 2]))
        # Mimic the outer scene wrapper: counters remain, episode latch resets,
        # and future observations use the new scene after the terminal outcome.
        info[NAVIGATION_COUNTED_KEY] = jp.where(done, False, info[NAVIGATION_COUNTED_KEY])
        info["pf_id"] = jp.where(done, (old_ids + 1) % 3, old_ids)
        obs = {key: value.at[..., 1].set(info["pf_id"].astype(jp.float32))
               for key, value in next_state.obs.items()}
        return next_state.replace(info=info, obs=obs)


def online_rows(logs):
    return [(step, metrics) for step, metrics in logs if "training/resolved_count" in metrics]


def saved_counts(snapshot):
    tree = snapshot["env_state"]
    matches = [value for path, value in zip(tree["paths"], tree["leaves"])
               if NAVIGATION_COUNTS_KEY in path]
    assert len(matches) == 1
    return np.asarray(matches[0]).reshape(-1, 4, 2)[0]


@pytest.fixture(scope="module")
def online_runs(tmp_path_factory):
    # Construction of an evaluator would violate this profile even if no
    # evaluation result happened to reach the logger.
    with pytest.MonkeyPatch.context() as patch:
        def forbidden_evaluation(*args, **kwargs):
            raise AssertionError("Training-only navigation metrics must not construct an evaluator")
        patch.setattr(ppo.acting, "Evaluator", forbidden_evaluation)
        whole, logs, result = run_updates(4, environment=ToyNavigationEnv())
        first, first_logs, _ = run_updates(2, environment=ToyNavigationEnv())
        path = tmp_path_factory.mktemp("online-success-resume") / "resume.msgpack"
        atomic_save_runtime(path, first[-1])
        resumed, resumed_logs, resumed_result = run_updates(
            2, restore=load_runtime(path), environment=ToyNavigationEnv())
    return whole, first, resumed, logs, first_logs, resumed_logs, result, resumed_result


def test_real_ppo_rates_use_each_update_counts_despite_donated_environment_buffers(online_runs):
    whole, _, _, logs, _, _, result, _ = online_runs
    rows = online_rows(logs)
    assert [step for step, _ in rows] == [12, 24, 36, 48]
    expected = [np.array([[1, 1], [2, 2], [1, 0], [4, 3]]),
                np.array([[2, 2], [1, 1], [1, 0], [4, 3]]),
                np.array([[1, 1], [1, 1], [2, 0], [4, 2]]),
                np.array([[1, 1], [2, 2], [1, 0], [4, 3]])]
    previous = np.zeros((4, 2), dtype=np.int32)
    for snapshot, (_, metrics), counts in zip(whole, rows, expected):
        current = saved_counts(snapshot)
        np.testing.assert_array_equal(current - previous, counts)
        previous = current.copy()
        for index, group in enumerate(("cat", "ordinary_clutter", "hand_protection", "")):
            prefix = "training/" + (group + "_" if group else "")
            assert metrics[prefix + "resolved_count"] == counts[index, 0]
            assert metrics[prefix + "goal_success_count"] == counts[index, 1]
            assert metrics[prefix + "goal_success_rate"] == counts[index, 1] / counts[index, 0]
    assert [row[1]["training/goal_success_rate"] for row in rows] == [.75, .75, .5, .75]
    assert result[2]["training/completed_steps"] == 48
    for _, metrics in logs:
        assert not any(key.startswith(("eval/", "validation/")) for key in metrics)
        assert all(np.all(np.isfinite(np.asarray(value))) for value in metrics.values())


def test_exact_resume_retains_online_counters_latches_and_update_local_denominators(online_runs):
    whole, first, resumed, logs, first_logs, resumed_logs, _, resumed_result = online_runs
    assert first[-1]["step"] == 24
    assert [step for step, _ in online_rows(resumed_logs)] == [36, 48]
    np.testing.assert_array_equal(saved_counts(whole[-1]), saved_counts(resumed[-1]))
    assert any(NAVIGATION_COUNTED_KEY in path for path in resumed[-1]["env_state"]["paths"])
    for key in ("training_state", "env_state", "local_key", "key_envs"):
        assert_tree_equal(whole[-1][key], resumed[-1][key])
    expected_rows = online_rows(logs)
    actual_rows = online_rows(first_logs) + online_rows(resumed_logs)
    for (expected_step, expected), (actual_step, actual) in zip(expected_rows, actual_rows):
        assert expected_step == actual_step
        keys = [key for key in expected if key.endswith(("goal_success_rate", "resolved_count", "goal_success_count"))]
        assert {key: expected[key] for key in keys} == {key: actual[key] for key in keys}
    assert resumed_result[2]["training/completed_steps"] == 48
