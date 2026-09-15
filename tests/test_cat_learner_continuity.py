"""Small CPU-only learner tests; no robot simulator or remote training launch."""
import copy
import functools
import importlib

from brax import envs
from brax.training.agents.ppo import networks
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cat_ppo.furniture.generalist_runtime import atomic_save_runtime, load_runtime

ppo = importlib.import_module("cat_ppo.learning.policy.ppo.train")


class ToyMixedEnv:
    """Batched toy with per-episode scene resampling and adaptive sampler state."""
    action_size = 1

    def reset(self, keys):
        position = jax.vmap(lambda key: jax.random.normal(key, ()))(keys)
        zero = jnp.zeros_like(position)
        ids = jnp.arange(len(keys), dtype=jnp.int32) % 2
        obs = jnp.stack([position, ids.astype(jnp.float32)], axis=-1)
        return envs.State(None, {"state": obs, "privileged_state": obs}, zero, zero, {}, {
            "pf_id": ids, "age": jnp.zeros_like(ids), "truncation": zero,
            "episode_done": zero, "episode_metrics": {"sum_reward": zero, "length": zero},
            "pf_success_ema": jnp.zeros((len(keys), 2), dtype=jnp.float32),
        })

    def step(self, state, action):
        position = state.obs["state"][..., 0] + .1 * action[..., 0]
        age = state.info["age"] + 1
        done = age == 3
        ids = jnp.where(done, 1 - state.info["pf_id"], state.info["pf_id"])
        obs = jnp.stack([position, ids.astype(jnp.float32)], axis=-1)
        reward = 1 - position ** 2 - .01 * action[..., 0] ** 2
        return state.replace(obs={"state": obs, "privileged_state": obs}, reward=reward,
            done=done.astype(jnp.float32), info={
                **state.info, "pf_id": ids, "age": jnp.where(done, 0, age),
                "episode_done": done.astype(jnp.float32), "truncation": done.astype(jnp.float32),
                "episode_metrics": {"sum_reward": reward, "length": age.astype(jnp.float32)},
                "pf_success_ema": state.info["pf_success_ema"] + jax.nn.one_hot(state.info["pf_id"], 2) * done[..., None],
            })


def run_updates(count, restore=None, *, include_initial=False, **overrides):
    snapshots, logs = [], []
    arguments = dict(
        environment=ToyMixedEnv(), num_timesteps=0, continuous=True,
        training_steps_per_epoch=1, num_evals=0, num_eval_envs=0,
        num_resets_per_eval=0, wrap_env=False, num_envs=2,
        unroll_length=3, batch_size=2, num_minibatches=2,
        num_updates_per_batch=2, learning_rate=3e-4, seed=42,
        randomize_initial_episode_steps=False,
        log_training_metrics=True, training_metrics_steps=12,
        network_factory=functools.partial(networks.make_ppo_networks,
            policy_hidden_layer_sizes=(4,), value_hidden_layer_sizes=(4,),
            policy_obs_key="state", value_obs_key="privileged_state"),
        should_stop_fn=lambda: sum(value["step"] > 0 for value in snapshots) >= count,
        runtime_checkpoint_fn=lambda step, snapshot: snapshots.append(snapshot),
        progress_fn=lambda step, metrics: logs.append((step, metrics)),
        runtime_metadata={"scene_sha256": "test-mixed-scenes", "source": "test-v1"},
        restore_runtime_state=restore,
    )
    arguments.update(overrides)
    result = ppo.train(**arguments)
    return snapshots if include_initial else [value for value in snapshots if value["step"] > 0], logs, result


@pytest.fixture(scope="module")
def continuity(tmp_path_factory):
    uninterrupted, logs, result = run_updates(4)
    first, _, _ = run_updates(2)
    path = tmp_path_factory.mktemp("resume") / "resume.msgpack"
    atomic_save_runtime(path, first[-1])
    restored = load_runtime(path)
    resumed, resumed_logs, resumed_result = run_updates(2, restored)
    return uninterrupted, first, resumed, logs, resumed_logs, result, resumed_result, path


def assert_tree_equal(first, second):
    assert first["paths"] == second["paths"]
    for left, right in zip(first["leaves"], second["leaves"]):
        np.testing.assert_array_equal(left, right)


def test_exact_resume_retains_adam_rng_normalizer_environment_and_sampler(continuity):
    uninterrupted, first, resumed, _, _, result, resumed_result, _ = continuity
    assert [value["step"] for value in uninterrupted] == [12, 24, 36, 48]
    assert [value["step"] for value in resumed] == [36, 48]
    for key in ("training_state", "env_state", "local_key", "key_envs"):
        assert_tree_equal(uninterrupted[-1][key], resumed[-1][key])
    assert uninterrupted[-1]["metrics_logger"] == resumed[-1]["metrics_logger"]
    counts = [leaf for path, leaf in zip(resumed[-1]["training_state"]["paths"],
        resumed[-1]["training_state"]["leaves"]) if "optimizer_state" in path and path.endswith(".count")]
    assert counts and all(np.all(count == 16) for count in counts)
    assert any("pf_success_ema" in path for path in resumed[-1]["env_state"]["paths"])
    assert result[2]["training/completed_steps"] == resumed_result[2]["training/completed_steps"] == 48
    assert result[2]["training/stopped_by_request"]


def test_resume_log_axis_continues_and_updates_outlive_zero_timestep_budget(continuity):
    _, _, _, logs, resumed_logs, _, _, _ = continuity
    assert [step for step, metrics in logs if "training/sps" in metrics] == [12, 24, 36, 48]
    assert min(step for step, _ in resumed_logs) > 24
    assert all("eval/episode_reward" not in metrics for _, metrics in logs)


def test_fixed_ppo_batch_accumulates_when_parallel_environment_count_changes():
    snapshots, _, _ = run_updates(1, num_envs=4)
    assert snapshots[0]["step"] == 12
    assert snapshots[0]["contract"]["batch_size"] == 2


def test_initial_resume_snapshot_exists_before_any_optimizer_update():
    snapshots, logs, result = run_updates(0, include_initial=True)
    assert len(snapshots) == 1 and snapshots[0]["step"] == 0
    assert result[2]["training/completed_steps"] == 0
    assert logs == []
    resumed, _, _ = run_updates(1, restore=snapshots[0], include_initial=True)
    assert [snapshot["step"] for snapshot in resumed] == [12]


@pytest.mark.parametrize("inplace_info", [False, True])
def test_terminal_scene_id_is_from_before_autoreset(inplace_info):
    class MutatingToy(ToyMixedEnv):
        def step(self, state, action):
            next_state = super().step(state, action)
            if inplace_info:
                state.info["pf_id"] = next_state.info["pf_id"]
            return next_state
    env = MutatingToy()
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 2))
    state.info["age"] = jnp.full((2,), 2, dtype=jnp.int32)
    policy = lambda obs, key: (jnp.zeros((2, 1)), {})
    after, data = ppo._generate_unroll_with_scene_ids(env, state, policy,
        jax.random.PRNGKey(1), 1, ("episode_done", "truncation", "episode_metrics"))
    np.testing.assert_array_equal(data.extras["state_extras"]["pf_id"][0], [0, 1])
    np.testing.assert_array_equal(after.info["pf_id"], [1, 0])


def test_single_resume_file_replaced_atomically_and_failed_write_preserves_previous(continuity, monkeypatch):
    uninterrupted, _, _, _, _, _, _, path = continuity
    atomic_save_runtime(path, uninterrupted[-1])
    assert load_runtime(path)["step"] == 48
    assert list(path.parent.iterdir()) == [path]
    runtime = importlib.import_module("cat_ppo.furniture.generalist_runtime")
    def fail(*args):
        raise OSError("simulated disk error")
    monkeypatch.setattr(runtime.os, "replace", fail)
    with pytest.raises(OSError, match="disk error"):
        atomic_save_runtime(path, uninterrupted[0])
    assert load_runtime(path)["step"] == 48
    assert list(path.parent.iterdir()) == [path]


def test_resume_rejects_changed_scene_contract_and_leaf_shape(continuity):
    _, first, _, _, _, _, _, _ = continuity
    changed = copy.deepcopy(first[-1])
    changed["contract"]["metadata"]["scene_sha256"] = "different-scenes"
    with pytest.raises(ValueError, match="configuration differs"):
        run_updates(1, changed)
    changed = copy.deepcopy(first[-1])
    changed["env_state"]["leaves"][0] = np.zeros((123,), dtype=np.float32)
    with pytest.raises(ValueError, match="shape/dtype differs"):
        run_updates(1, changed)


@pytest.mark.parametrize("overrides", [
    {"num_evals": 1}, {"num_resets_per_eval": 1}, {"num_training_epochs": 3},
    {"training_steps_per_epoch": 0}, {"should_stop_fn": None},
    {"save_checkpoint_path": "unbounded"},
])
def test_continuous_rejects_eval_resets_and_unbounded_checkpoint_paths(overrides):
    with pytest.raises(ValueError):
        run_updates(1, **overrides)


def test_full_resume_cannot_silently_reset_adam_by_loading_params():
    with pytest.raises(ValueError, match="params-only"):
        run_updates(1, restore={}, restore_params=())
