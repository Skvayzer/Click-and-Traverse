import json
from types import SimpleNamespace

import pytest

from cat_ppo.furniture.generalist_config import batch_geometry, ppo_kwargs, released_config, training_config
from cat_ppo.furniture.generalist_logging import GeneralistLogger


def test_released_objective_and_batch_are_pinned():
    original = released_config()
    config = training_config()
    policy = ppo_kwargs(config)
    for key in ("entropy_cost", "discounting", "unroll_length", "num_minibatches",
                "num_updates_per_batch", "learning_rate", "clipping_epsilon",
                "gae_lambda", "normalize_observations", "randomize_initial_episode_steps"):
        assert policy[key] == original["policy_config"][key]
    assert config["env_config"] == original["env_config"]
    assert config["env_config"]["term_collision_threshold"] == 0
    assert len(config["env_config"]["pf_config"]["paths"]) == 37
    assert batch_geometry(original["policy_config"])["transitions_per_update"] == 4194304
    assert batch_geometry(config["policy_config"])["transitions_per_update"] == 524288
    assert policy["num_evals"] == 0


def test_simulator_parallelism_does_not_change_optimizer_batch():
    first = training_config(num_envs=2048)
    second = training_config(num_envs=4096)
    geometry1 = batch_geometry(first["policy_config"])
    geometry2 = batch_geometry(second["policy_config"])
    for key in ("transitions_per_update", "transitions_per_minibatch", "optimizer_steps_per_update"):
        assert geometry1[key] == geometry2[key]
    assert geometry1["rollout_chunks_per_update"] == 2 * geometry2["rollout_chunks_per_update"]
    with pytest.raises(ValueError, match="divide"):
        training_config(num_envs=3000)


class FakeRun:
    url = "https://example.invalid/run"
    def __init__(self):
        self.events = []
    def define_metric(self, *args, **kwargs):
        pass
    def log(self, value):
        self.events.append(value)
    def finish(self, **kwargs):
        pass


def test_one_wandb_identity_and_monotonic_recovery(tmp_path):
    calls, fake_run = [], FakeRun()
    def init(**kwargs):
        calls.append(kwargs)
        return fake_run
    module = SimpleNamespace(init=init)
    logger = GeneralistLogger(tmp_path, wandb_module=module)
    logger.log(100, {"episode/sum_reward": 2})
    logger.log(100, {"training/loss": .2})
    logger.finish()
    resumed = GeneralistLogger(tmp_path, resume=True, wandb_module=module)
    resumed.log(80, {"training/loss": .3})
    resumed.log(120, {"training/loss": .1})
    assert calls[0]["id"] == calls[1]["id"]
    assert calls[1]["resume"] == "must"
    assert [event["global_step"] for event in fake_run.events] == [100, 100, 120]
    assert json.loads((tmp_path / "wandb.json").read_text())["last_global_step"] == 120


def test_nonfinite_is_reported_and_fails_including_after_resume(tmp_path):
    logger = GeneralistLogger(tmp_path, mode="disabled")
    logger.log(100, {"training/loss": 1})
    with pytest.raises(FloatingPointError, match="training/loss"):
        logger.log(80, {"training/loss": float("nan")})
    last = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[-1])
    assert last["health/nonfinite"] == 1
    assert last["nonfinite_keys"] == ["training/loss"]
    assert last["source_step"] == 80


def test_resume_requires_same_logging_destination(tmp_path):
    GeneralistLogger(tmp_path, mode="disabled")
    with pytest.raises(ValueError, match="project"):
        GeneralistLogger(tmp_path, resume=True, mode="disabled", project="different")


def test_scene_timeouts_match_brax_and_do_not_mask_collisions():
    from copy import deepcopy
    import jax
    import jax.numpy as jp
    import numpy as np
    from brax.envs.base import State
    from brax.envs.wrappers.training import EpisodeWrapper
    from cat_ppo.furniture.generalist_training import SceneEpisodeWrapper

    class Toy:
        def reset(self, keys):
            zero = jp.zeros(len(keys))
            return State(None, zero, zero, zero, {"term": zero},
                         {"pf_id": jp.arange(len(keys)), "collision": zero})
        def step(self, state, action):
            return state.replace(reward=jp.ones(2), done=state.info["collision"])

    original = EpisodeWrapper(Toy(), episode_length=3, action_repeat=1)
    matched = SceneEpisodeWrapper(Toy(), 3, 1, [3, 3])
    keys = jp.zeros((2, 2), dtype=jp.uint32)
    a, b = original.reset(keys), matched.reset(keys)
    for _ in range(3):
        a, b = original.step(a, None), matched.step(b, None)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(x, y)
    longer = SceneEpisodeWrapper(Toy(), 3, 1, [3, 6])
    state = longer.reset(keys)
    for _ in range(3):
        state = longer.step(state, None)
    np.testing.assert_array_equal(state.done, [1, 0])
    np.testing.assert_array_equal(state.info["truncation"], [1, 0])
    state = longer.reset(keys)
    state.info["steps"] = jp.asarray([2, 5])
    state.info["collision"] = jp.ones(2)
    state = longer.step(state, None)
    np.testing.assert_array_equal(state.done, [1, 1])
    np.testing.assert_array_equal(state.info["truncation"], [0, 0])
