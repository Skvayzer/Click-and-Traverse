"""CAT's adaptive reset stack with an explicit longer limit for added rooms."""
import jax
import jax.numpy as jp
from brax.envs.wrappers import training
from mujoco_playground._src import wrapper

from cat_ppo.learning.train.pf_utils import SamplePFWrapper


class SceneEpisodeWrapper(training.EpisodeWrapper):
    """Brax EpisodeWrapper arithmetic; only the scene's time limit differs.

    Original CAT scenes keep 1000 steps (20 s). Dense rooms have 4000 (80 s)
    because their traversable routes are much longer than CAT's 2 m scenes.
    A timeout sets truncation; a collision at the limit remains a termination.
    """
    def __init__(self, env, episode_length, action_repeat, scene_lengths):
        super().__init__(env, episode_length, action_repeat)
        self.scene_lengths = jp.asarray(scene_lengths, dtype=jp.int32)

    def reset(self, rng):
        state = super().reset(rng)
        metrics = state.info["episode_metrics"]
        for key, value in state.info.get("wholebody_episode", {}).items():
            metrics["wb_" + key] = jp.zeros_like(value, dtype=jp.float32)
        if "wholebody_telemetry" in state.info:
            metrics["wb_length"] = jp.zeros_like(state.done)
            for key, value in state.info["wholebody_telemetry"].items():
                metrics["wb_mean_" + key] = jp.zeros_like(value)
        return state

    def step(self, state, action):
        def f(state, _):
            next_state = self.env.step(state, action)
            return next_state, next_state.reward
        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jp.sum(rewards, axis=0))
        steps = state.info["steps"] + self.action_repeat
        limit = self.scene_lengths[state.info["pf_id"]]
        done = jp.where(steps >= limit, jp.ones_like(state.done), state.done)
        state.info["truncation"] = jp.where(steps >= limit, 1 - state.done, jp.zeros_like(state.done))
        state.info["steps"] = steps
        prev_done = state.info["episode_done"]
        state.info["episode_metrics"]["sum_reward"] += jp.sum(rewards, axis=0)
        state.info["episode_metrics"]["sum_reward"] *= 1 - prev_done
        state.info["episode_metrics"]["length"] += self.action_repeat
        state.info["episode_metrics"]["length"] *= 1 - prev_done
        for name in state.metrics:
            if name != "reward":
                state.info["episode_metrics"][name] += state.metrics[name]
                state.info["episode_metrics"][name] *= 1 - prev_done
        # Snapshot terminal sticky flags before auto-reset replaces raw info.
        # A goal event counts once per episode, never once per subsequent step.
        metrics = state.info["episode_metrics"]
        for key, value in state.info.get("wholebody_episode", {}).items():
            metrics["wb_" + key] = value.astype(jp.float32)
        if "wholebody_telemetry" in state.info:
            old_count = jp.where(prev_done, 0., metrics["wb_length"])
            count = old_count + self.action_repeat
            for key, value in state.info["wholebody_telemetry"].items():
                name = "wb_mean_" + key
                metrics[name] = (metrics[name] * old_count + value * self.action_repeat) / count
            metrics["wb_length"] = count
        state.info["episode_done"] = done
        return state.replace(done=done)


def wrap_for_cat_wholebody_training(env, episode_length=1000, action_repeat=1,
                                    randomization_fn=None, **kwargs):
    if randomization_fn is not None or kwargs.get("vision", False):
        raise ValueError("This launch follows CAT's released non-vision, non-model-randomized wrapper")
    limits = (env._pf_scene_episode_lengths if getattr(env, "_pf_expanded", False)
              else jp.where(env._pf_scene_original, episode_length, env._config.clutter_episode_length))
    env = training.VmapWrapper(env)
    env = SceneEpisodeWrapper(env, episode_length, action_repeat, limits)
    env = wrapper.BraxAutoResetWrapper(env)
    return SamplePFWrapper(env)
