"""Cooperative stopping finishes an optimizer epoch and checkpoint callback."""
import functools
import signal

import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks
from mujoco_playground._src.mjx_env import State

from cat_ppo.furniture.run_control import StopRequest
from cat_ppo.furniture.training import wrap_for_furniture_training
from cat_ppo.learning.policy.ppo import train as ppo


class ToyTask:
    action_size = 1

    @property
    def unwrapped(self):
        return self

    def reset(self, rng):
        del rng
        z = jp.asarray(0.)
        return State(data=jp.zeros(1), obs={"state": jp.zeros(1), "privileged_state": jp.zeros(1)},
                     reward=z, done=z, metrics={"progress": z}, info={})

    def step(self, state, action):
        position = state.data + .1 * action
        return state.replace(data=position, obs={"state": position, "privileged_state": position},
                             reward=1 - jp.square(action[0]), metrics={"progress": position[0]})


def test_manual_stop_waits_for_completed_epoch_and_preserves_candidate(tmp_path):
    stop_file = tmp_path / "STOP"
    candidates = []
    with StopRequest(stop_file) as stop:
        assert not stop.requested()

        def checkpoint(step, make_policy, params, config, metrics, source):
            del make_policy, config, metrics
            candidates.append((step, params, source))
            stop_file.touch()

        _, params, metrics = ppo.train(
            ToyTask(), num_timesteps=64, num_envs=4, episode_length=8,
            wrap_env_fn=wrap_for_furniture_training, randomize_initial_episode_steps=False,
            batch_size=1, num_minibatches=4, unroll_length=2, num_updates_per_batch=1,
            num_evals=0, num_training_epochs=4, should_stop_fn=stop.requested,
            scored_checkpoint_fn=checkpoint, network_factory=functools.partial(
                networks.make_ppo_networks, policy_hidden_layer_sizes=(8,), value_hidden_layer_sizes=(8,),
                policy_obs_key="state", value_obs_key="privileged_state"))
        assert metrics["training/completed_steps"] == 16
        assert metrics["training/stopped_by_request"]
        assert len(candidates) == 1 and candidates[0][0] == 16
        assert candidates[0][2] == "training_proxy"
        import jax
        for returned, selected in zip(jax.tree.leaves(params), jax.tree.leaves(candidates[0][1])):
            np.testing.assert_array_equal(returned, selected)


def test_signal_handler_is_cooperative_and_restored(tmp_path):
    previous = signal.getsignal(signal.SIGTERM)
    with StopRequest(tmp_path / "STOP") as stop:
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)
        assert stop.requested() and stop.reason == "SIGTERM"
    assert signal.getsignal(signal.SIGTERM) == previous
