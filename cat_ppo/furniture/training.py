"""Training wrappers for complete furniture episode resets and truthful metrics.

The generic Playground autoreset restores physics and observations only. This
task also needs its latched contacts, motor targets, perception history and
episode clock restored. Terminal task info remains available to collectors until
the next step; episode aggregates use snapshots for outcomes/clearance and sums
only for rewards, rather than summing every state metric indiscriminately.
"""
from __future__ import annotations

import jax
import jax.numpy as jp
from brax.envs.wrappers import training as brax_training
from mujoco_playground import wrapper


_CACHE = "furniture_training_reset"
_WRAPPER_KEYS = frozenset((_CACHE, "steps", "truncation", "episode_done", "episode_metrics"))
_MEAN_METRICS = frozenset(("cross_track_m", "map_age_seconds", "unknown_fraction"))


def _where(mask, first, second):
    """Select complete batched state leaves with a per-environment reset mask."""
    mask = jp.reshape(mask, mask.shape + (1,) * (first.ndim - mask.ndim))
    return jp.where(mask, first, second)


def _episode_metrics(previous, metrics, reward, steps, reset):
    aggregate = {
        "sum_reward": jp.where(reset, 0.0, previous["sum_reward"]) + reward,
        "length": steps,
    }
    for name, value in metrics.items():
        if name == "reward":
            continue
        before = jp.where(reset, 0.0, previous[name])
        if name.startswith("reward/"):
            aggregate[name] = before + value
        elif name in _MEAN_METRICS:
            aggregate[name] = (before * (steps - 1) + value) / steps
        else:
            # Outcomes and route progress are terminal snapshots; the task's
            # min_hand_clearance_m already contains the episode minimum.
            aggregate[name] = value
    return aggregate


class FurnitureTrainingWrapper(wrapper.Wrapper):
    """Wrap an already batched task, caching its complete nominal reset state.

    A terminal step returns its real reward, outcomes and metrics alongside
    reset physics/observations (the Brax training convention). Before the next
    action, every task info leaf is restored for the terminated environments.
    The nominal reset, including its sampled gait phase, is reused per vector
    slot, matching the existing cached-reset training convention.
    """

    def __init__(self, env, episode_length):
        super().__init__(env)
        self.episode_length = episode_length

    def reset(self, rng):
        state = self.env.reset(rng)
        zero = jp.zeros_like(state.done)
        episode_metrics = {name: zero for name in state.metrics if name != "reward"}
        episode_metrics.update(sum_reward=zero, length=zero)
        if _WRAPPER_KEYS.intersection(state.info):
            raise ValueError("Furniture training wrapper requires an unwrapped task info dictionary")
        cache = dict(data=state.data, obs=state.obs, info=state.info, metrics=state.metrics)
        info = dict(state.info)
        info.update({"steps": zero, "truncation": zero, "episode_done": zero,
                     "episode_metrics": episode_metrics, _CACHE: cache})
        return state.replace(info=info)

    def step(self, state, action):
        reset = state.done.astype(bool)
        cache = state.info[_CACHE]
        task_info = {key: value for key, value in state.info.items() if key not in _WRAPPER_KEYS}
        task_info = jax.tree.map(lambda a, b: _where(reset, a, b), cache["info"], task_info)
        metrics = jax.tree.map(lambda a, b: _where(reset, a, b), cache["metrics"], state.metrics)
        task_state = state.replace(info=task_info, metrics=metrics, done=jp.zeros_like(state.done))
        next_state = self.env.step(task_state, action)
        steps = jp.where(reset, 0, state.info["steps"]) + 1
        exhausted = steps >= self.episode_length
        done = jp.maximum(next_state.done, exhausted.astype(next_state.done.dtype))
        aggregates = _episode_metrics(state.info["episode_metrics"], next_state.metrics,
                                      next_state.reward, steps, reset)
        info = dict(next_state.info)
        info.update({"steps": steps, "truncation": jp.where(exhausted, 1 - next_state.done, 0.0),
                     "episode_done": done, "episode_metrics": aggregates, _CACHE: cache})
        data = jax.tree.map(lambda a, b: _where(done.astype(bool), a, b), cache["data"], next_state.data)
        obs = jax.tree.map(lambda a, b: _where(done.astype(bool), a, b), cache["obs"], next_state.obs)
        return next_state.replace(data=data, obs=obs, done=done, info=info)


def wrap_for_furniture_training(env, *, episode_length, action_repeat=1, randomization_fn=None):
    if action_repeat != 1:
        raise ValueError("Furniture training requires action_repeat=1 for physical contact timing")
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    return FurnitureTrainingWrapper(env, episode_length)
