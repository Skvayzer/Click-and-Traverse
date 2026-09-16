"""Terminal collision transitions through the real CAT training wrapper stack."""
from types import SimpleNamespace

import flax.struct
import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.envs.wrappers import training
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env

from cat_ppo.furniture.generalist_training import SceneEpisodeWrapper
from cat_ppo.learning.train.pf_utils import SamplePFWrapper


@flax.struct.dataclass
class _Data:
    qpos: jax.Array
    qvel: jax.Array
    ctrl: jax.Array
    site_xpos: jax.Array


class _CollisionTask:
    """Two CAT and two room scenes with distinguishable reset histories."""

    def __init__(self, enabled):
        self._config = SimpleNamespace()
        if enabled is not None:
            self._config.wholebody = SimpleNamespace(
                body_collision=SimpleNamespace(enabled=enabled))

    @property
    def unwrapped(self):
        return self

    def reset(self, key):
        scene = (key[0] % 4).astype(jp.int32)
        base = scene.astype(jp.float32) * 10
        zero = jp.float32(0)
        vector = jp.full(2, base)
        sites = jp.full((2, 3), base)
        data = _Data(vector, vector + 1, jp.full(1, base + 2), sites)
        info = dict(
            rng=key, pf_id=scene, step=zero, command=vector,
            last_command=vector, last_act=jp.zeros(1), motor_targets=vector,
            stop_timestep=zero, phase=vector, phase_dt=zero,
            gait_freq=zero, foot_height=zero, head_pos=sites[0],
            head_vel=jp.zeros(3), odom_delay=vector,
            history_nested={"previous": jp.full((2, 2), base)},
            room_navigation={"enabled": scene >= 2, "violation": jp.array(False)},
            wholebody_body_collision=jp.array(False),
            wholebody_episode={"body_collision": jp.array(False)},
        )
        return mjx_env.State(data, {"state": vector, "privileged_state": sites},
                             zero, zero, {"collision_cost": zero}, info)

    def step(self, state, action):
        collision = action[0] > 0
        sites = state.data.site_xpos + .01
        info = dict(state.info)
        info.update(
            rng=state.info["rng"] + jp.array([1, 0], dtype=jp.uint32),
            step=state.info["step"] + 1,
            head_pos=sites[0], head_vel=(sites[0] - state.info["head_pos"]) / .02,
            odom_delay=state.data.qpos + .01,
            history_nested={"previous": state.info["history_nested"]["previous"] + 1},
            wholebody_body_collision=collision,
            wholebody_episode={"body_collision": collision},
        )
        data = state.data.replace(qpos=state.data.qpos + .01,
                                  qvel=state.data.qvel + .02,
                                  ctrl=state.data.ctrl + .03, site_xpos=sites)
        return state.replace(
            data=data, obs={"state": data.qpos, "privileged_state": sites},
            reward=jp.where(collision, -5., 1.), done=collision.astype(jp.float32),
            metrics={"collision_cost": jp.where(collision, -5., 0.)}, info=info)


def _wrapped(enabled=True, limits=(10, 10, 10, 10)):
    task = _CollisionTask(enabled)
    env = training.VmapWrapper(task)
    env = SceneEpisodeWrapper(env, max(limits), 1, limits)
    env = wrapper.BraxAutoResetWrapper(env)
    return task, SamplePFWrapper(env)


def _assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, b)


def test_collision_reward_and_fresh_history_survive_all_scene_transitions():
    task, env = _wrapped()
    # The first four collide and sample CAT->CAT, CAT->room, room->room,
    # room->CAT respectively. The last slot continues its current episode.
    keys = jp.array([[0, 7], [1, 7], [2, 7], [3, 7], [1, 9]], dtype=jp.uint32)
    state = env.reset(keys)
    actions = jp.array([[1.], [1.], [1.], [1.], [0.]])
    raw_step = jax.vmap(task.step)(state, actions)
    expected_reset = jax.vmap(task.reset)(keys + jp.array([1, 0], dtype=jp.uint32))
    advance = jax.jit(env.step)
    terminal = advance(state, actions)

    np.testing.assert_array_equal(terminal.reward, [-5., -5., -5., -5., 1.])
    np.testing.assert_array_equal(terminal.done, [1., 1., 1., 1., 0.])
    np.testing.assert_array_equal(terminal.info["episode_done"], terminal.done)
    np.testing.assert_array_equal(terminal.info["truncation"], 0.)
    np.testing.assert_array_equal(terminal.info["pf_id"], [1, 2, 3, 0, 1])
    np.testing.assert_array_equal(terminal.info["wholebody_body_collision"], False)
    np.testing.assert_array_equal(terminal.info["wholebody_episode"]["body_collision"], False)
    np.testing.assert_array_equal(
        terminal.info["episode_metrics"]["wb_body_collision"], [1., 1., 1., 1., 0.])
    np.testing.assert_array_equal(terminal.info["episode_metrics"]["sum_reward"], terminal.reward)

    for actual, fresh, ongoing in zip(jax.tree.leaves(terminal.data),
                                      jax.tree.leaves(expected_reset.data),
                                      jax.tree.leaves(raw_step.data)):
        np.testing.assert_array_equal(actual[:4], fresh[:4])
        np.testing.assert_array_equal(actual[4:], ongoing[4:])
    for name in ("head_pos", "head_vel", "odom_delay", "history_nested"):
        _assert_tree_equal(jax.tree.map(lambda x: x[:4], terminal.info[name]),
                           jax.tree.map(lambda x: x[:4], expected_reset.info[name]))
        _assert_tree_equal(jax.tree.map(lambda x: x[4:], terminal.info[name]),
                           jax.tree.map(lambda x: x[4:], raw_step.info[name]))
    _assert_tree_equal(jax.tree.map(lambda x: x[:4], terminal.obs),
                       jax.tree.map(lambda x: x[:4], expected_reset.obs))
    _assert_tree_equal(jax.tree.map(lambda x: x[:4], terminal.info["first_state"]),
                       jax.tree.map(lambda x: x[:4], expected_reset.data))
    _assert_tree_equal(jax.tree.map(lambda x: x[:4], terminal.info["first_obs"]),
                       jax.tree.map(lambda x: x[:4], expected_reset.obs))

    following = advance(terminal, jp.zeros_like(actions))
    np.testing.assert_array_equal(following.reward, 1.)
    np.testing.assert_array_equal(following.done, 0.)
    np.testing.assert_array_equal(following.info["wholebody_body_collision"], False)
    np.testing.assert_array_equal(following.info["episode_metrics"]["wb_body_collision"], 0.)
    # Fresh head history avoids a fictitious velocity from the previous scene.
    np.testing.assert_allclose(following.info["head_vel"], .5, atol=2e-5)


def test_collision_at_time_limit_is_terminal_and_timeout_remains_truncation():
    _, env = _wrapped(limits=(1, 1, 1, 1))
    keys = jp.array([[0, 7], [1, 7]], dtype=jp.uint32)
    state = env.reset(keys)
    result = jax.jit(env.step)(state, jp.array([[0.], [1.]]))
    np.testing.assert_array_equal(result.done, 1.)
    np.testing.assert_array_equal(result.reward, [1., -5.])
    np.testing.assert_array_equal(result.info["truncation"], [1., 0.])
    np.testing.assert_array_equal(result.info["episode_metrics"]["wb_body_collision"], [0., 1.])
    np.testing.assert_array_equal(result.info["wholebody_body_collision"], False)


@pytest.mark.parametrize("enabled", [False, None])
def test_disabled_collision_mode_retains_native_terminal_reward_and_history(enabled):
    task, env = _wrapped(enabled)
    keys = jp.array([[0, 7]], dtype=jp.uint32)
    state = env.reset(keys)
    raw = jax.vmap(task.step)(state, jp.ones((1, 1)))
    result = jax.jit(env.step)(state, jp.ones((1, 1)))
    np.testing.assert_array_equal(result.done, 1.)
    np.testing.assert_array_equal(result.reward, 0.)
    np.testing.assert_array_equal(result.info["pf_id"], 1)
    np.testing.assert_array_equal(result.info["head_pos"], raw.info["head_pos"])
    _assert_tree_equal(result.info["history_nested"], raw.info["history_nested"])
    # Native CAT-to-CAT resets only qpos/qvel after the cached auto-reset;
    # enabling the collision extension is what opts into a full coherent reset.
    np.testing.assert_array_equal(result.data.ctrl, state.data.ctrl)
    _assert_tree_equal(result.info["first_obs"], state.info["first_obs"])


def test_hand_curriculum_survives_actual_autoreset_and_state_roundtrip():
    from cat_ppo.furniture.hand_curriculum import STATE_KEYS, initial_curriculum_state

    class HandTask(_CollisionTask):
        _pf_hand_curriculum_levels = jp.array([-1, 0, 1, 2])

        def reset(self, key):
            state = super().reset(key)
            state.info.update(
                pf_success_ema=jp.zeros(4), pf_episode_ema=jp.zeros(4),
                pf_sampling_logits=jp.log(jp.array([.5, .5, 0., 0.])),
                pf_sampling_alpha=jp.float32(1), pf_sampling_ema_decay=jp.float32(.95),
                pf_sampling_group_ids=jp.full(4, 2, jp.int32),
                pf_sampling_group_masses=jp.array([0., 0., 1., 0.]),
                **initial_curriculum_state())
            state.info["wholebody_episode"]["goal_reached"] = jp.array(False)
            return state

        def step(self, state, action):
            result = super().step(state, action)
            result.info["wholebody_episode"]["goal_reached"] = action[0] < 0
            return result

    task = HandTask(True)
    env = SamplePFWrapper(wrapper.BraxAutoResetWrapper(
        SceneEpisodeWrapper(training.VmapWrapper(task), 1, 1, jp.ones(4))))
    state = env.reset(jp.array([[1, 7]], jp.uint32))
    state.info[STATE_KEYS[1]] = jp.array([[63, 0, 0]], jp.int32)
    state.info[STATE_KEYS[2]] = jp.array([[63, 0, 0]], jp.int32)
    advance = jax.jit(env.step)
    result = advance(state, -jp.ones((1, 1)))
    np.testing.assert_array_equal(result.done, 1.)
    np.testing.assert_array_equal(result.info[STATE_KEYS[0]], 1)
    np.testing.assert_array_equal(result.info[STATE_KEYS[1]], [[64, 0, 0]])
    np.testing.assert_array_equal(result.info[STATE_KEYS[2]], [[64, 0, 0]])
    # A learner checkpoint restores these ordinary state leaves with no extra
    # sampler-side hidden state. Both continuations must be identical.
    restored = jax.tree.map(lambda value: jp.asarray(np.array(value)), result)
    uninterrupted = advance(result, jp.ones((1, 1)))
    resumed = advance(restored, jp.ones((1, 1)))
    _assert_tree_equal(uninterrupted, resumed)
    np.testing.assert_array_equal(resumed.info[STATE_KEYS[0]], 1)
    assert np.all(np.asarray(resumed.info[STATE_KEYS[1]]) >= [64, 0, 0])
