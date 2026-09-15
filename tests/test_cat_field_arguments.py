"""CPU-only regression coverage for large field operands in reset and PPO."""

import functools
import importlib
from types import SimpleNamespace

from brax import envs
from brax.training.agents.ppo import networks
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cat_ppo.learning.policy.ppo.field_arguments import FieldArguments


class LookupEnv:
    def __init__(self, rows=1024, *, bank=True):
        if bank:
            self.field_bank_manifest = {"schema": "test-bank"}
        self.sdf, self.bf, self.gf = (
            jnp.asarray(np.random.default_rng(index).normal(size=(rows, channels)).astype(np.float32))
            for index, channels in enumerate((1, 3, 3)))

    @property
    def unwrapped(self):
        return self

    def lookup(self, index):
        return self.sdf[index, 0] + jnp.sum(self.bf[index], axis=-1) + jnp.sum(self.gf[index], axis=-1)


def test_binding_reaches_leaf_wrapper_and_restores_even_when_tracing_fails():
    env = LookupEnv()
    wrapper = SimpleNamespace(unwrapped=env)
    binding = FieldArguments(wrapper)
    original = binding.values
    alternate = tuple(value + 1 for value in original)
    with binding.bind(alternate):
        assert env.sdf is alternate[0]
        with binding.bind(original):
            assert env.gf is original[2]
        assert env.gf is alternate[2]
    assert all(getattr(env, name) is value for name, value in zip(binding.names, original))
    with pytest.raises(RuntimeError, match="trace failed"):
        with binding.bind(alternate):
            raise RuntimeError("trace failed")
    assert all(getattr(env, name) is value for name, value in zip(binding.names, original))


def test_nonbank_binding_is_empty_and_does_not_change_environment():
    env = LookupEnv(bank=False)
    binding = FieldArguments(env)
    before = env.sdf
    assert binding.values == ()
    with binding.bind(()):
        assert env.sdf is before
    with pytest.raises(ValueError, match="field tuple"):
        with binding.bind((before,)):
            pass


@pytest.mark.parametrize("rows", [1024, 131072])
def test_jit_reset_and_pmap_scan_lower_small_hlo_and_reuse_dynamic_values(rows):
    env = LookupEnv(rows)
    binding = FieldArguments(env)
    values = binding.values
    def reset(indices, fields):
        with binding.bind(fields):
            return jax.vmap(env.lookup)(indices)
    def epoch(indices, fields):
        with binding.bind(fields):
            def step(carry, _):
                return carry + env.lookup(indices), None
            return jax.lax.scan(step, jnp.float32(0), None, length=3)[0]
    reset_fn = jax.jit(reset)
    epoch_fn = jax.pmap(epoch, in_axes=(0, None))
    indices = jnp.arange(jax.local_device_count(), dtype=jnp.int32) + 7
    expected = jax.vmap(env.lookup)(indices)
    np.testing.assert_allclose(reset_fn(indices, values), expected, rtol=1e-6)
    np.testing.assert_allclose(epoch_fn(indices, values), 3 * expected, rtol=1e-6)
    # Changed buffers with identical shapes must affect existing executables;
    # restoring Python attributes cannot silently restore captured constants.
    alternate = tuple(value + 2 for value in values)
    np.testing.assert_allclose(reset_fn(indices, alternate), expected + 14, rtol=1e-6)
    np.testing.assert_allclose(epoch_fn(indices, alternate), 3 * (expected + 14), rtol=1e-6)
    for fn in (reset_fn, epoch_fn):
        hlo = fn.lower(indices, values).compiler_ir("hlo").as_serialized_hlo_module_proto()
        assert len(hlo) < 20000
    closed = jax.jit(lambda index: env.lookup(index))
    closed_hlo = closed.lower(jnp.int32(7)).compiler_ir("hlo").as_serialized_hlo_module_proto()
    assert len(closed_hlo) >= sum(value.nbytes for value in values)
    assert all(getattr(env, name) is value for name, value in zip(binding.names, values))


class ToyFieldEnv(LookupEnv):
    action_size = 1

    def reset(self, keys):
        index = jax.vmap(lambda key: jax.random.randint(key, (), 0, 128))(keys)
        position = self.lookup(index) * .01
        zeros = jnp.zeros_like(position)
        obs = jnp.stack([position, index.astype(jnp.float32) / 128], axis=-1)
        return envs.State(None, {"state": obs, "privileged_state": obs}, zeros, zeros, {}, {
            "index": index, "age": jnp.zeros_like(index), "truncation": zeros,
            "episode_done": zeros, "episode_metrics": {"sum_reward": zeros, "length": zeros},
        })

    def step(self, state, action):
        index = (state.info["index"] + 1) % 128
        position = state.obs["state"][..., 0] + .1 * action[..., 0] + .001 * self.lookup(index)
        age = state.info["age"] + 1
        done = age == 3
        obs = jnp.stack([position, index.astype(jnp.float32) / 128], axis=-1)
        reward = 1 - position ** 2 - .01 * action[..., 0] ** 2
        return state.replace(obs={"state": obs, "privileged_state": obs}, reward=reward,
            done=done.astype(jnp.float32), info={**state.info, "index": index,
                "age": jnp.where(done, 0, age), "truncation": done.astype(jnp.float32),
                "episode_done": done.astype(jnp.float32),
                "episode_metrics": {"sum_reward": reward, "length": age.astype(jnp.float32)}})


def _train_toy(environment, count=2, restore=None):
    ppo = importlib.import_module("cat_ppo.learning.policy.ppo.train")
    snapshots = []
    ppo.train(environment, num_timesteps=0, continuous=True, num_evals=0,
        num_eval_envs=0, num_resets_per_eval=0, training_steps_per_epoch=1,
        wrap_env=False, num_envs=2, unroll_length=3, batch_size=2,
        num_minibatches=2, num_updates_per_batch=2, seed=42,
        randomize_initial_episode_steps=False,
        should_stop_fn=lambda: sum(snapshot["step"] > (restore["step"] if restore else 0)
                                    for snapshot in snapshots) >= count,
        runtime_checkpoint_fn=lambda step, snapshot: snapshots.append(snapshot),
        restore_runtime_state=restore, runtime_metadata={"fixture": "field-equivalence"},
        network_factory=functools.partial(networks.make_ppo_networks,
            policy_hidden_layer_sizes=(4,), value_hidden_layer_sizes=(4,),
            policy_obs_key="state", value_obs_key="privileged_state"))
    return snapshots


def test_real_ppo_dynamic_bank_matches_closed_fields_and_keeps_runtime_schema():
    closed_env, dynamic_env = ToyFieldEnv(bank=False), ToyFieldEnv(bank=True)
    original = (dynamic_env.sdf, dynamic_env.bf, dynamic_env.gf)
    closed, dynamic = _train_toy(closed_env), _train_toy(dynamic_env)
    assert [snapshot["step"] for snapshot in dynamic] == [0, 12, 24]
    for expected, actual in zip(closed, dynamic):
        assert actual["contract"] == expected["contract"]
        for key in ("training_state", "env_state", "local_key", "key_envs"):
            assert actual[key]["paths"] == expected[key]["paths"]
            for first, last in zip(expected[key]["leaves"], actual[key]["leaves"]):
                np.testing.assert_allclose(last, first, rtol=1e-6, atol=1e-6)
        assert all(not any(name in path for name in (".sdf", ".bf", ".gf", "field_values"))
                   for path in actual["env_state"]["paths"])
    assert all(getattr(dynamic_env, name) is value for name, value in zip(("sdf", "bf", "gf"), original))
    resumed = _train_toy(ToyFieldEnv(bank=True), count=1, restore=dynamic[1])
    assert resumed[-1]["step"] == 24
    for key in ("training_state", "env_state", "local_key", "key_envs"):
        for expected, actual in zip(dynamic[-1][key]["leaves"], resumed[-1][key]["leaves"]):
            np.testing.assert_array_equal(actual, expected)
