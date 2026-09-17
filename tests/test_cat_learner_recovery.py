"""Real CPU PPO rollback, continued learning, and exact recovery resume."""
import copy

from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import losses
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from cat_ppo.furniture.generalist_runtime import atomic_save_runtime, load_runtime
from test_cat_learner_continuity import assert_tree_equal, ppo, run_updates


def recovery_callbacks():
    """Score each completed update, then recover from update one at update two."""
    scored = {}

    def score(step, make_policy, params, config, metrics, source):
        scored[step] = jax.tree.map(lambda value: np.array(value, copy=True), params)

    def recover(step):
        if step == 24:
            # This also checks that recovery is called after the scoring hook.
            assert 24 in scored
            return dict(params=scored[12], learning_rate=1.5e-4, checkpoint="toy-best-update-one")
        return None

    return dict(scored_checkpoint_fn=score, recovery_fn=recover)


@pytest.fixture(scope="module")
def recovery_runs(tmp_path_factory):
    uninterrupted, logs, result = run_updates(4, normalize_observations=True, **recovery_callbacks())
    interrupted, _, _ = run_updates(2, normalize_observations=True, **recovery_callbacks())
    path = tmp_path_factory.mktemp("recovery-resume") / "resume.msgpack"
    atomic_save_runtime(path, interrupted[-1])
    resumed, resumed_logs, resumed_result = run_updates(
        2, restore=load_runtime(path), normalize_observations=True,
        recovery_fn=lambda step: None,
    )
    return uninterrupted, interrupted, resumed, logs, resumed_logs, result, resumed_result


def state_arrays(snapshot, prefix):
    state = snapshot["training_state"]
    return {path: value for path, value in zip(state["paths"], state["leaves"])
            if path.startswith(prefix)}


def assert_arrays_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def learning_rates(snapshot):
    return [value for path, value in state_arrays(snapshot, ".optimizer_state").items()
            if "hyperparams['learning_rate']" in path]


def test_rollback_restores_actor_critic_normalizer_and_resets_adam_and_scene(recovery_runs):
    uninterrupted = recovery_runs[0]
    before, recovered = uninterrupted[:2]
    assert [snapshot["step"] for snapshot in uninterrupted] == [12, 24, 36, 48]
    for prefix in (".params.policy", ".params.value", ".normalizer_params"):
        values = state_arrays(recovered, prefix)
        assert values
        assert_arrays_equal(state_arrays(before, prefix), values)
    counts = [value for path, value in state_arrays(recovered, ".optimizer_state").items()
              if path.endswith(".count")]
    assert counts and all(np.all(value == 0) for value in counts)
    rates = learning_rates(recovered)
    assert len(rates) == 1
    np.testing.assert_allclose(rates[0], 1.5e-4, rtol=1e-6)
    env = dict(zip(recovered["env_state"]["paths"], recovered["env_state"]["leaves"]))
    sampler = [value for path, value in env.items() if "pf_success_ema" in path]
    assert sampler and all(np.all(value == 0) for value in sampler)
    logger = recovered["metrics_logger"]
    assert logger["num_steps"] == 24
    assert not logger["metrics_buffer"] and not logger["rollout_buffer"]
    assert not logger["latest_rollout_metrics"]


def test_recovered_policy_continues_learning_at_reduced_rate_and_monotonic_steps(recovery_runs):
    uninterrupted, _, _, logs, _, result, _ = recovery_runs
    recovered, next_update = uninterrupted[1:3]
    before = state_arrays(recovered, ".params.policy")
    after = state_arrays(next_update, ".params.policy")
    assert any(not np.array_equal(before[key], after[key]) for key in before)
    for snapshot in uninterrupted[1:]:
        np.testing.assert_allclose(learning_rates(snapshot)[0], 1.5e-4, rtol=1e-6)
    assert [step for step, metrics in logs if "training/sps" in metrics] == [12, 24, 36, 48]
    assert result[2]["training/completed_steps"] == 48
    assert uninterrupted[-1]["metrics_logger"]["num_steps"] == 48


def test_exact_resume_after_rollback_keeps_lower_rate_rng_adam_and_normalizer(recovery_runs):
    uninterrupted, interrupted, resumed, _, resumed_logs, _, resumed_result = recovery_runs
    assert interrupted[-1]["step"] == 24
    assert [snapshot["step"] for snapshot in resumed] == [36, 48]
    for key in ("training_state", "env_state", "local_key", "key_envs"):
        assert_tree_equal(uninterrupted[-1][key], resumed[-1][key])
    assert uninterrupted[-1]["metrics_logger"] == resumed[-1]["metrics_logger"]
    np.testing.assert_allclose(learning_rates(resumed[-1])[0], 1.5e-4, rtol=1e-6)
    assert min(step for step, _ in resumed_logs) > 24
    assert resumed_result[2]["training/completed_steps"] == 48


def test_recovery_optimizer_cannot_silently_resume_as_legacy_adam(recovery_runs):
    recovered = recovery_runs[1][-1]
    with pytest.raises(ValueError, match="configuration differs"):
        run_updates(1, restore=recovered, normalize_observations=True)


def tiny_state(optimizer):
    params = losses.PPONetworkParams(
        policy={"weights": jnp.asarray([1., -2.], dtype=jnp.float32)},
        value={"weights": jnp.asarray([3.], dtype=jnp.float32)},
    )
    return ppo.TrainingState(
        params=params, optimizer_state=optimizer.init(params),
        normalizer_params=running_statistics.init_state(jnp.zeros((2,), dtype=jnp.float32)),
        env_steps=types.UInt64(hi=jnp.uint32(0), lo=jnp.uint32(96)),
    )


@pytest.mark.parametrize("clipped", [False, True])
def test_halved_injected_rate_halves_actual_adam_update_with_fresh_moments(clipped):
    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=3e-4)
    if clipped:
        optimizer = optax.chain(optax.clip_by_global_norm(1.), optimizer)
    state = tiny_state(optimizer)
    params = (state.normalizer_params, state.params.policy, state.params.value)
    full = ppo._recover_training_state(state, params, optimizer, 3e-4)
    half = ppo._recover_training_state(state, params, optimizer, 1.5e-4)
    grads = jax.tree.map(lambda value: jnp.full_like(value, .8), state.params)
    for _ in range(2):
        full_update, full_optimizer = optimizer.update(grads, full.optimizer_state, full.params)
        half_update, half_optimizer = optimizer.update(grads, half.optimizer_state, half.params)
        for a, b in zip(jax.tree.leaves(full_update), jax.tree.leaves(half_update)):
            np.testing.assert_allclose(b, .5 * a, rtol=1e-6, atol=0.)
        full = full.replace(optimizer_state=full_optimizer)
        half = half.replace(optimizer_state=half_optimizer)
    assert int(half.env_steps) == 96


@pytest.mark.parametrize("rate", [0., -1., np.inf, np.nan])
def test_recovery_rejects_invalid_learning_rate(rate):
    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=3e-4)
    state = tiny_state(optimizer)
    with pytest.raises(ValueError, match="finite and positive"):
        ppo._recover_training_state(state,
            (state.normalizer_params, state.params.policy, state.params.value), optimizer, rate)


def test_recovery_rejects_nonfinite_or_incompatible_params():
    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=3e-4)
    state = tiny_state(optimizer)
    params = (state.normalizer_params, state.params.policy, state.params.value)
    changed = copy.deepcopy(params)
    changed[1]["weights"] = jnp.asarray([np.nan, -2.], dtype=jnp.float32)
    with pytest.raises(ValueError, match="must be finite"):
        ppo._recover_training_state(state, changed, optimizer, 1.5e-4)
    changed[1]["weights"] = jnp.zeros((3,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        ppo._recover_training_state(state, changed, optimizer, 1.5e-4)
