"""Room route guidance reaches real CAT observations and terminal outcomes."""
from copy import deepcopy
import json
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest

from cat_ppo.envs.g1.env_cat import EPS, g1_loco_task_config, world_to_navi_vel
from cat_ppo.envs.g1.env_cat_wholebody import (
    G1CatWholeBodyEnv, _WholeBodyTask, wholebody_config,
)
from cat_ppo.envs.g1.room_navigation import RoomNavigationMixin
from cat_ppo.furniture import generalist_fields
from cat_ppo.furniture.generalist_fields import RaggedSceneMixin
from cat_ppo.furniture.room_navigation import pack_room_scenes, root_clearance


def _room():
    return dict(
        start=[0., 0., .8], goal=[1., 1., .75],
        route=[[0., 0.], [0., 1.], [1., 1.]],
        boxes=[dict(center=[.55, 0., .7], half_size=[.03, .3, .3], yaw=0.)],
    )


class _NativeRaggedWholeBody(RaggedSceneMixin, _WholeBodyTask):
    """Same compact whole-body task with the room extension omitted."""


@pytest.fixture(scope="module")
def environment_pair(tmp_path_factory):
    directory = tmp_path_factory.mktemp("room-navigation-integration")
    field_directory = directory / "fields"
    field_directory.mkdir()
    shape = (20, 21, 16)
    # The stored global field points east, directly toward the obstacle. The
    # certified room route first heads north. This catches accidental fallbacks
    # to the previously independent global navigation field.
    gf = np.broadcast_to(np.asarray([.6, 0., 0.], np.float32), (*shape, 3)).copy()
    for name, values in (("gf", gf), ("bf", np.zeros_like(gf)),
                         ("sdf", np.full(shape, 1., np.float32))):
        np.save(field_directory / f"{name}.npy", values)
    (field_directory / "scene.json").write_text(json.dumps(_room()))
    common = dict(path="fields", shape=list(shape), origin=[-2., -2., -.5], dx=.25,
                  start=[0., 0., .8], goal=[1., 1., .75], reset_yaw=0.,
                  sampling_weight=1.)
    cat = dict(common, family="original_cat", reset_xy_scale=[1., 1.])
    room = dict(common, family="furniture", reset_xy_scale=[.08, .08], source={
        "occupancy": "conservative-voxel-cell-OBB-intersection-v1",
        "room_navigation": "ordered-certified-route-v1",
    })
    manifest = dict(schema=generalist_fields.SCHEMA, scenes=[cat, room])
    base = g1_loco_task_config().env_config.copy_and_resolve_references()
    base.term_collision_threshold = 0.
    for term in ("headgf", "handsgf", "headdf", "handsdf", "feetdf", "kneesdf", "shldsdf"):
        base.reward_config.scales[term] = 1.
    base.reward_config.scales.feetgf = 2.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(generalist_fields, "load_generalist_manifest", lambda path: manifest)
        config = wholebody_config(base, bank_manifest=directory / "manifest.json")
        actual = G1CatWholeBodyEnv(config=config)
        native = _NativeRaggedWholeBody(config=config.copy_and_resolve_references())
    return actual, native


@pytest.fixture(scope="module")
def reset_states(environment_pair):
    actual, native = environment_pair
    key = jax.random.PRNGKey(71)
    reset = jax.jit(actual.reset_with_pf_id)
    cat = reset(key, jp.int32(0))
    room = reset(key, jp.int32(1))
    reference = jax.jit(native.reset_with_pf_id)(key, jp.int32(0))
    jax.block_until_ready(room.reward)
    return cat, room, reference


def _assert_tree_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for expected, actual in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_array_equal(actual, expected)


def _direction(vector):
    vector = np.asarray(vector)
    return vector / max(np.linalg.norm(vector), 1e-6)


def test_original_cat_reset_retains_identical_fields_commands_observations_and_rng(reset_states):
    actual, _, reference = reset_states
    assert not actual.info["room_navigation"]["enabled"]
    info = {key: value for key, value in actual.info.items() if key != "room_navigation"}
    _assert_tree_equal(actual.replace(info=info), reference)


def _assert_no_navigation_tracers(env):
    for name in ("_room_context", "_room_query_command", "_field_pf_id"):
        assert not any(isinstance(leaf, jax.core.Tracer)
                       for leaf in jax.tree.leaves(getattr(env, name, None))), name


def test_actual_room_jit_reset_and_step_use_ordered_route_without_growing_observations(
        environment_pair, reset_states):
    env, _ = environment_pair
    _, initial, _ = reset_states
    assert env.action_size == 29
    assert initial.obs["state"].shape == (222,)
    assert initial.obs["privileged_state"].shape == (310,)
    assert initial.info["command"][2] > .5
    assert initial.info["room_navigation"]["segment"] == 0
    _assert_no_navigation_tracers(env)
    step = jax.jit(env.step)
    state = step(deepcopy(initial), jp.zeros(29))
    # The second control tick holds previous odometry. Both queries must share
    # the current ordered waypoint, although their root positions differ.
    state = step(state, jp.zeros(29))
    jax.block_until_ready(state.reward)
    assert jax.tree.structure(state) == jax.tree.structure(initial)
    assert state.obs["state"].shape == (222,)
    assert state.obs["privileged_state"].shape == (310,)
    assert np.isfinite(state.obs["state"]).all()
    assert not bool(state.info["room_navigation"]["violation"])
    _assert_no_navigation_tracers(env)
    target = np.asarray(state.info["room_navigation"]["target"])
    for command_name, root in (("command", state.data.qpos[:2]),
                               ("command_delay", state.info["odom_delay"][:2])):
        expected = _direction(target - np.asarray(root))
        np.testing.assert_allclose(_direction(state.info[command_name][1:3]), expected, atol=1e-6)
    for actor, observation, command_name in (
            (False, "privileged_state", "command"), (True, "state", "command_delay")):
        direction = _direction(state.info[command_name][1:3])
        # CAT's vector normalization deliberately retains its EPS denominator.
        expected = jp.tile(jp.asarray([*direction, 0.]) * (.6 / (.6 + EPS)), (2, 1))
        if actor:
            expected = world_to_navi_vel(state.info["navi2world_pose"], expected)
        np.testing.assert_allclose(state.obs[observation][-14:-8], expected.reshape(-1), atol=1e-6)


def test_room_gait_resumes_after_stop_latch_without_changing_cat_gait(environment_pair, reset_states):
    env, native = environment_pair
    cat, room, reference = reset_states
    command = jp.asarray([.75, 0., .6, 0.])

    def stopped_copy(state):
        state = deepcopy(state)
        state.info.update(command=command, last_command=jp.zeros(4), stop_timestep=jp.int32(0))
        return state

    resumed = stopped_copy(room)
    env._update_phase(resumed)
    assert resumed.info["stop_timestep"] == 100
    assert resumed.info["command"][0] == 1.
    np.testing.assert_array_equal(resumed.info["command"][1:], command[1:])

    actual, expected = stopped_copy(cat), stopped_copy(reference)
    env._update_phase(actual)
    native._update_phase(expected)
    for name in ("stop_timestep", "command", "phase", "gait_mask"):
        np.testing.assert_array_equal(actual.info[name], expected.info[name])
    assert actual.info["command"][0] == 0.
    assert actual.info["stop_timestep"] == 0


def test_wrapped_autoreset_clears_sticky_route_state_and_previous_scene_sweep(environment_pair):
    from cat_ppo.furniture.generalist_training import wrap_for_cat_wholebody_training
    env, _ = environment_pair
    wrapped = wrap_for_cat_wholebody_training(env)
    keys = jax.random.split(jax.random.PRNGKey(31), 4)
    state = jax.jit(wrapped._reset_with_pf_id)(keys, jp.ones(4, dtype=jp.int32))
    state.info["room_navigation"]["violation"] = jp.ones(4, dtype=bool)
    state.info["room_navigation"]["segment"] = jp.ones(4, dtype=jp.int32)
    state.info["room_navigation"]["root_xy"] = jp.tile(jp.array([1., 0.]), (4, 1))
    advance = jax.jit(wrapped.step)
    reset = advance(state, jp.zeros((4, 29)))
    jax.block_until_ready(reset.reward)
    np.testing.assert_array_equal(reset.info["episode_done"], 1.)
    np.testing.assert_array_equal(reset.info["episode_metrics"]["wb_obstacle"], 1.)
    fresh = reset.info["room_navigation"]
    np.testing.assert_array_equal(fresh["violation"], False)
    np.testing.assert_array_equal(fresh["segment"], 0)
    np.testing.assert_array_equal(fresh["root_xy"], reset.data.qpos[:, :2])
    np.testing.assert_array_equal(fresh["enabled"], reset.info["pf_id"] == 1)
    np.testing.assert_array_equal(reset.info["wholebody_episode"]["obstacle"], False)
    for name, site_ids in (("head_pos", env._head_site_id),
                           ("feet_pos", env._feet_site_id),
                           ("hands_pos", env._hands_site_id)):
        np.testing.assert_array_equal(reset.info[name], reset.data.site_xpos[:, site_ids])
    # MJX normalizes the initial quaternion after CAT stores its odometry pose.
    np.testing.assert_allclose(reset.info["odom_delay"], reset.data.qpos[:, :7], atol=1e-7)
    # Exercise both a room reset and a change to the original CAT scene. This
    # seed is fixed so this coverage cannot disappear probabilistically.
    assert set(np.asarray(reset.info["pf_id"]).tolist()) == {0, 1}
    following = advance(reset, jp.zeros((4, 29)))
    jax.block_until_ready(following.reward)
    np.testing.assert_array_equal(following.info["episode_done"], 0.)
    np.testing.assert_array_equal(following.info["room_navigation"]["violation"], False)
    np.testing.assert_array_equal(following.info["wholebody_faults"]["room_root_field"], False)
    np.testing.assert_array_equal(following.info["room_navigation"]["root_xy"],
                                  following.data.qpos[:, :2])
    assert np.isfinite(following.obs["privileged_state"]).all()
    # Stale coordinates from the preceding room produced artificial velocities
    # of hundreds of m/s. A fresh standing reset has no such displacement.
    for name in ("head_vel", "feet_vel", "hands_vel"):
        assert np.max(np.abs(np.asarray(following.info[name]))) < 10., name
    _assert_no_navigation_tracers(env)


def test_swept_room_trunk_collision_terminates_immediately_and_disqualifies_goal(
        environment_pair, reset_states):
    env, _ = environment_pair
    _, initial, _ = reset_states
    env._field_pf_id = jp.int32(1)
    info = deepcopy(initial.info)
    info["step"] = jp.int32(1)  # Native sparse body samples still have their grace.
    info["room_navigation"]["root_xy"] = jp.asarray([-.4, 0.])
    data = initial.data.replace(qpos=initial.data.qpos.at[:2].set(jp.asarray([1., 0.])))
    meta = env._room_metadata()
    assert np.all(np.asarray(root_clearance(jp.asarray([[-.4, 0.], [1., 0.]]),
        meta["obstacles"], meta["obstacle_count"])) > 0.)
    info.update(env._update_navigation(data, info))
    assert bool(info["room_navigation"]["violation"])
    assert bool(env._get_termination(data, info))
    assert bool(info["wholebody_faults"]["room_root_field"])
    assert bool(info["wholebody_faults"]["obstacle"])
    assert not bool(info["wholebody_faults"]["hands_field"])
    assert not bool(info["wholebody_episode"]["goal_reached"])
    # Endpoint proximity cannot turn a route violation into a successful run.
    goal = jp.asarray([1., 1., .8])
    assert bool(_WholeBodyTask._crossed_goal(env, goal))
    assert not bool(env._crossed_goal(goal))


class _NativeHooks:
    def _sample_body_fields(self, positions, *, root_xy=None):
        return (jp.tile(jp.array([.6, 0., 0.]), (len(positions), 1)),
                jp.zeros_like(positions), jp.ones((len(positions), 1)))

    def compute_cmd_from_rtf(self, rtf, cgf, cbf):
        return jp.asarray([.75, .315, 0., 0.])

    def _crossed_goal(self, positions):
        return jp.linalg.norm(positions[..., :2] - jp.array([1., 1.]), axis=-1) < .5


class _NavigationHooks(RoomNavigationMixin, _NativeHooks):
    pass


def _hooks():
    env = object.__new__(_NavigationHooks)
    env._room_arrays = {key: jp.asarray(value) for key, value in pack_room_scenes([None, _room()]).items()}
    env._room_scene_index = jp.asarray([0, 1])
    env._field_pf_id = jp.int32(1)
    return env


def _query(env, root):
    gf, bf, sdf = env._sample_body_fields(jp.zeros((11, 3)), root_xy=jp.asarray(root))
    command = env.compute_cmd_from_rtf(gf[1], gf, bf)
    return gf, bf, sdf, command


def test_true_delayed_and_elbow_queries_share_one_ordered_waypoint():
    env = _hooks()
    info = env._update_navigation(SimpleNamespace(qpos=jp.asarray([0., .9])))
    assert info["room_navigation"]["segment"] == 1
    target = np.asarray(info["room_navigation"]["target"])
    commands = []
    for root in ([0., .9], [0., .7]):
        gf, bf, sdf, command = _query(env, root)
        commands.append(command)
        expected = _direction(target - root)
        np.testing.assert_allclose(_direction(command[1:3]), expected, atol=1e-7)
        np.testing.assert_allclose(_direction(gf[5, :2]), expected, atol=1e-7)
        np.testing.assert_array_equal(bf, 0.)
        np.testing.assert_array_equal(sdf, 1.)
        assert env._room_context["segment"] == 1
    info.update(command=commands[0], command_delay=commands[1])
    for actor in (False, True):
        elbow = env._room_elbow_guidance(jp.tile(jp.array([.6, 0., 0.]), (2, 1)),
            jp.zeros((2, 3)), jp.ones((2, 1)), info, actor=actor)
        np.testing.assert_allclose(_direction(elbow[0, :2]),
                                   _direction(commands[int(actor)][1:3]), atol=1e-7)


def test_goal_requires_ordered_completion_and_cat_mask_keeps_native_values():
    env = _hooks()
    info = env._update_navigation(SimpleNamespace(qpos=jp.asarray([0., 0.])))
    assert not env._crossed_goal(jp.asarray([1., 1., .8]))
    near_corner = SimpleNamespace(qpos=jp.asarray([0., .95]))
    info = env._update_navigation(near_corner, info)
    info = env._update_navigation(SimpleNamespace(qpos=jp.asarray([1., 1.])), info)
    assert info["room_navigation"]["route_complete"]
    assert env._crossed_goal(jp.asarray([1., 1., .8]))
    env._field_pf_id = jp.int32(0)
    env._update_navigation(SimpleNamespace(qpos=jp.asarray([0., 0.])))
    gf, bf, sdf, command = _query(env, [0., 0.])
    native = _NativeHooks()
    expected_fields = native._sample_body_fields(jp.zeros((11, 3)))
    _assert_tree_equal((gf, bf, sdf), expected_fields)
    np.testing.assert_array_equal(command, native.compute_cmd_from_rtf(gf[1], gf, bf))
    assert env._crossed_goal(jp.asarray([1., 1., .8]))


def test_batched_room_route_metadata_is_a_dynamic_runtime_operand():
    from cat_ppo.learning.policy.ppo.field_arguments import FieldArguments
    env = _hooks()
    binding = FieldArguments(env)
    original = binding.values

    def one(scene, root):
        env._field_pf_id = scene
        env._update_navigation(SimpleNamespace(qpos=root))
        return _query(env, root)[-1]

    def batch(scenes, roots, fields):
        with binding.bind(fields):
            return jax.vmap(one)(scenes, roots)

    navigate = jax.jit(batch)
    scenes, roots = jp.asarray([0, 1]), jp.zeros((2, 2))
    first = navigate(scenes, roots, original)
    changed_arrays = dict(original[0])
    changed_arrays["route"] = original[0]["route"].at[1].set(
        jp.asarray([[0., 0.], [-1., 0.], [-1., 1.]]))
    changed = navigate(scenes, roots, (changed_arrays, original[1]))
    np.testing.assert_array_equal(first[0], changed[0])
    np.testing.assert_allclose(first[1, 1:3], [0., .6], atol=1e-6)
    np.testing.assert_allclose(changed[1, 1:3], [-.6, 0.], atol=1e-6)
    assert all(getattr(env, name) is value for name, value in zip(binding.names, original))
