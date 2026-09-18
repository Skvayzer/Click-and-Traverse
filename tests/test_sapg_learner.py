"""Real tiny CPU SAPG updates, complete resume and native leader deployment."""

import copy
import functools

from brax.training.agents.ppo import checkpoint
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cat_ppo.furniture.generalist_runtime import atomic_save_runtime, load_runtime
from cat_ppo.learning.policy.sapg.networks import collapse_policy_params, make_sapg_networks
from test_cat_learner_continuity import ToyMixedEnv, assert_tree_equal, ppo, run_updates
from test_online_success_learner import ToyNavigationEnv, online_rows, saved_counts


SAPG_CONFIG = {"num_policies": 2, "embedding_dim": 3, "prepare_chunk_size": 2}
NETWORK_KWARGS = dict(policy_hidden_layer_sizes=(4,), value_hidden_layer_sizes=(4,),
                      policy_obs_key="state", value_obs_key="privileged_state")


def run_sapg_updates(count, restore=None, *, include_initial=True, **overrides):
    arguments = dict(
        environment=ToyNavigationEnv(), normalize_observations=False,
        max_devices_per_host=1, sapg_config=dict(SAPG_CONFIG),
        network_factory=functools.partial(make_sapg_networks, num_policies=2, embedding_dim=3,
                                          **NETWORK_KWARGS),
    )
    arguments.update(overrides)
    return run_updates(count, restore, include_initial=include_initial, **arguments)


@pytest.fixture(scope="module")
def sapg_continuity(tmp_path_factory):
    callbacks = []

    def scored(step, make_policy, params, config, metrics, source):
        callbacks.append((step, make_policy, jax.tree.map(lambda x: np.array(x, copy=True), params),
                          copy.deepcopy(config), dict(metrics), source))

    with pytest.MonkeyPatch.context() as patch:
        def forbidden_evaluation(*args, **kwargs):
            raise AssertionError("SAPG training-only mode must not construct an evaluator")
        patch.setattr(ppo.acting, "Evaluator", forbidden_evaluation)
        whole, logs, result = run_sapg_updates(4, scored_checkpoint_fn=scored)
        first, first_logs, _ = run_sapg_updates(2)
        path = tmp_path_factory.mktemp("sapg-resume") / "resume.msgpack"
        atomic_save_runtime(path, first[-1])
        resumed, resumed_logs, resumed_result = run_sapg_updates(2, restore=load_runtime(path))
    return dict(whole=whole, first=first, resumed=resumed, logs=logs, first_logs=first_logs,
                resumed_logs=resumed_logs, result=result, resumed_result=resumed_result,
                callbacks=callbacks, resume_path=path)


def _parameter_leaf(snapshot, *fragments):
    tree = snapshot["training_state"]
    matches = [np.asarray(value) for path, value in zip(tree["paths"], tree["leaves"])
               if path.startswith(".params.") and all(fragment in path for fragment in fragments)]
    assert len(matches) == 1
    return matches[0]


def test_real_sapg_updates_are_finite_and_learn_embeddings_and_both_conditioned_trunks(sapg_continuity):
    run = sapg_continuity
    initial, final = run["whole"][0], run["whole"][-1]
    assert [item["step"] for item in run["whole"]] == [0, 12, 24, 36, 48]
    assert final["contract"]["sapg"] == SAPG_CONFIG
    before_embedding = _parameter_leaf(initial, ".policy", "policy_embeddings")
    after_embedding = _parameter_leaf(final, ".policy", "policy_embeddings")
    assert before_embedding.shape == (1, 2, 3)
    assert np.any(after_embedding != before_embedding)
    assert not np.array_equal(after_embedding[0, 0], after_embedding[0, 1])
    for trunk in (".policy", ".value"):
        before = _parameter_leaf(initial, trunk, "hidden_0", "kernel")
        after = _parameter_leaf(final, trunk, "hidden_0", "kernel")
        assert before.shape == after.shape == (1, 5, 4)
        np.testing.assert_array_equal(before[:, -3:], 0)
        assert np.linalg.norm(after[:, -3:]) > 0
        assert np.any(after[:, :2] != before[:, :2])
    for _, metrics in run["logs"]:
        assert not any(key.startswith(("eval/", "validation/")) for key in metrics)
        assert all(np.all(np.isfinite(np.asarray(value))) for value in metrics.values())
    assert run["result"][2]["training/completed_steps"] == 48
    assert run["result"][2]["training/stopped_by_request"]


def test_sapg_exact_resume_preserves_optimizer_embedding_rng_environment_and_counters(sapg_continuity):
    run = sapg_continuity
    whole, resumed = run["whole"][-1], run["resumed"][-1]
    assert [item["step"] for item in run["resumed"]] == [36, 48]
    for key in ("training_state", "env_state", "local_key", "key_envs"):
        assert_tree_equal(whole[key], resumed[key])
    assert whole["contract"] == resumed["contract"]
    assert whole["metrics_logger"] == resumed["metrics_logger"]
    optimizer_paths = [path for path in whole["training_state"]["paths"]
                       if "optimizer_state" in path and "policy_embeddings" in path]
    assert len(optimizer_paths) >= 2  # Adam's first and second moments for the sole table.
    for original, restored in zip(jax.tree.leaves(run["result"][1]),
                                  jax.tree.leaves(run["resumed_result"][1])):
        np.testing.assert_array_equal(original, restored)
    assert min(step for step, _ in run["resumed_logs"]) > 24
    assert run["resumed_result"][2]["training/completed_steps"] == 48


def test_sapg_sharing_does_not_count_virtual_relabelled_samples_as_physical_successes(sapg_continuity):
    run = sapg_continuity
    rows = online_rows(run["logs"])
    assert [step for step, _ in rows] == [12, 24, 36, 48]
    assert [metrics["training/resolved_count"] for _, metrics in rows] == [4, 4, 4, 4]
    assert [metrics["training/goal_success_rate"] for _, metrics in rows] == [.75, .75, .5, .75]
    np.testing.assert_array_equal(saved_counts(run["whole"][-1]), saved_counts(run["resumed"][-1]))
    reconstructed = online_rows(run["first_logs"]) + online_rows(run["resumed_logs"])
    for (expected_step, expected), (actual_step, actual) in zip(rows, reconstructed):
        assert expected_step == actual_step
        keys = [key for key in expected if key.endswith(("goal_success_rate", "resolved_count", "goal_success_count"))]
        assert {key: expected[key] for key in keys} == {key: actual[key] for key in keys}


def test_returned_policy_defaults_to_leader_and_folded_callback_exports_load_in_native_brax(
    sapg_continuity, tmp_path
):
    run = sapg_continuity
    make_policy, params, _ = run["result"]
    observations = {"state": jnp.array([[.2, 0.], [-.1, 1.]]),
                    "privileged_state": jnp.array([[.2, 0.], [-.1, 1.]])}
    key = jax.random.PRNGKey(52)
    network = make_sapg_networks({"state": (2,), "privileged_state": (2,)}, 1,
                                 num_policies=2, embedding_dim=3, **NETWORK_KWARGS)
    returned_action, extras = make_policy(params, deterministic=True)(observations, key)
    leader_logits = network.policy_network.apply(params[0], params[1], observations, policy_ids=0)
    np.testing.assert_allclose(returned_action, network.parametric_action_distribution.mode(leader_logits), atol=1e-6)
    np.testing.assert_array_equal(extras["policy_id"], 0)
    assert params[1]["policy_embeddings"].shape == (2, 3)

    step, callback_make_policy, callback_params, config, _, source = run["callbacks"][-1]
    assert step == 48 and source == "training_proxy"
    assert callback_params[1]["policy_embeddings"].shape == (2, 3)
    exported = collapse_policy_params(callback_params, policy_id=0)
    assert exported[1]["params"]["hidden_0"]["kernel"].shape == (2, 4)
    assert exported[2]["params"]["hidden_0"]["kernel"].shape == (2, 4)
    checkpoint.save(tmp_path, step, exported, config)
    native_policy = checkpoint.load_policy(tmp_path / f"{step:012d}", deterministic=True)
    native_action, _ = native_policy(observations, key)
    callback_action, _ = callback_make_policy(callback_params, deterministic=True)(observations, key)
    np.testing.assert_allclose(native_action, callback_action, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(native_action, returned_action, rtol=3e-6, atol=3e-6)


class DistinctPolicyRewardEnv(ToyMixedEnv):
    """The leader slot earns -10 and follower +10; population mean is zero."""
    def reset(self, keys):
        state = super().reset(keys)
        state.info["fixed_slot"] = jnp.arange(len(keys), dtype=jnp.int32)
        return state

    def step(self, state, action):
        next_state = super().step(state, action)
        reward = jnp.where(state.info["fixed_slot"] == 0, -10., 10.)
        return next_state.replace(reward=reward, info={
            **next_state.info,
            "episode_metrics": {**next_state.info["episode_metrics"], "sum_reward": reward},
        })


def test_scored_callback_uses_leader_raw_reward_not_population_or_augmented_mean():
    scores = []
    def scored(step, make_policy, params, config, metrics, source):
        scores.append((step, float(metrics["training/rollout_reward_mean"]), source))
    _, logs, _ = run_sapg_updates(1, environment=DistinctPolicyRewardEnv(), scored_checkpoint_fn=scored)
    assert scores == [(12, -10., "training_proxy")]
    rows = [metrics for _, metrics in logs if "training/rollout_reward_mean" in metrics]
    assert len(rows) == 1
    assert float(rows[0]["training/rollout_reward_mean"]) == -10.


def test_sapg_resume_rejects_changed_algorithm_contract(sapg_continuity):
    changed = copy.deepcopy(sapg_continuity["first"][-1])
    changed["contract"]["sapg"]["prepare_chunk_size"] = 1
    with pytest.raises(ValueError, match="configuration differs"):
        run_sapg_updates(1, restore=changed)


@pytest.mark.parametrize("overrides", [
    {"normalize_observations": True},
    {"num_envs": 3},
    {"recovery_fn": lambda step: None},
    {"dagger_config": {"enable": True}},
])
def test_sapg_rejects_unsupported_training_combinations(overrides):
    with pytest.raises(ValueError, match="SAPG"):
        run_sapg_updates(1, **overrides)
