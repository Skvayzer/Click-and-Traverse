"""Numerical reference checks for CAT task inheritance and scoped extensions."""
from copy import deepcopy
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from cat_ppo.envs.g1 import constants
from cat_ppo.envs.g1.env_cat import G1CatEnv, g1_loco_task_config
from cat_ppo.envs.g1.env_cat_wholebody import (
    G1CatWholeBodyEnv, _WholeBodyTask, assemble_training_xml, wholebody_config,
)
from cat_ppo.furniture.control import JOINT_NAMES, legacy_observation_contract
from cat_ppo.furniture.learning import index_mapping


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    field_dir = tmp_path_factory.mktemp("cat-reference-field")
    grid = np.indices((20, 21, 16), dtype=np.float32)
    sdf = 1.0 + 0.005 * grid[0] + 0.003 * grid[1] + 0.007 * grid[2]
    gf = np.stack([0.7 + 0.01 * grid[0], 0.05 * grid[1], np.zeros_like(grid[0])], axis=-1)
    bf = np.stack([0.1 * grid[0], 0.1 * grid[1], 0.1 * grid[2]], axis=-1)
    for name, values in (("sdf", sdf), ("gf", gf), ("bf", bf)):
        np.save(field_dir / f"{name}.npy", values)
    result = g1_loco_task_config().env_config.copy_and_resolve_references()
    result.pf_config.path = str(field_dir)
    result.pf_config.origin = [-2.0, -2.0, -0.5]
    result.pf_config.dx = 0.25
    # The released checkpoint config uses zero, unlike the generic default.
    result.term_collision_threshold = 0.0
    for term in ("headgf", "handsgf", "headdf", "handsdf", "feetdf", "kneesdf", "shldsdf"):
        result.reward_config.scales[term] = 1.0
    result.reward_config.scales.feetgf = 2.0
    return result


@pytest.fixture(scope="module")
def reference(config):
    return G1CatEnv(config=config)


@pytest.fixture(scope="module")
def compatible(config):
    return _WholeBodyTask(config=wholebody_config(config, compatibility_mode=True))


@pytest.fixture(scope="module")
def wholebody(config):
    return _WholeBodyTask(config=wholebody_config(config))


@pytest.fixture(scope="module")
def reference_state(reference):
    return jax.jit(reference.reset)(jax.random.PRNGKey(42))


@pytest.fixture(scope="module")
def compatible_state(compatible):
    return jax.jit(compatible.reset)(jax.random.PRNGKey(42))


@pytest.fixture(scope="module")
def wholebody_state(wholebody):
    return jax.jit(wholebody.reset)(jax.random.PRNGKey(42))


def _assert_tree_equal(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_array_equal(a, b)


def test_compatibility_reset_is_exact_including_noise_randomization_and_rng(
        reference_state, compatible_state):
    _assert_tree_equal(reference_state, compatible_state)
    assert reference_state.obs["state"].shape == (162,)
    assert reference_state.obs["privileged_state"].shape == (250,)
    assert float(reference_state.info["kp_scale"]) != 1.0
    assert float(reference_state.info["kd_scale"]) != 1.0
    assert np.any(np.asarray(reference_state.info["rfi_lim_scale"]) != 0)
    assert np.any(np.asarray(reference_state.data.qvel[:6]) != 0)


def test_compatibility_real_step_matches_all_reference_outputs(
        reference, compatible, reference_state, compatible_state):
    action = jp.linspace(-0.2, 0.2, 12)
    expected = jax.jit(reference.step)(reference_state, action)
    actual = jax.jit(compatible.step)(compatible_state, action)
    jax.block_until_ready(actual.reward)
    _assert_tree_equal(expected, actual)


def test_full_cat_reward_settings_and_reset_randomization_preserved(config, wholebody):
    extended = wholebody._config
    for key, value in config.reward_config.scales.items():
        assert extended.reward_config.scales[key] == value
    for name in ("dm_rand_config", "push_config", "gait_config", "noise_config"):
        assert getattr(extended, name).to_dict() == getattr(config, name).to_dict()
    assert extended.term_collision_threshold == 0.0
    assert set(extended.reward_config.scales) - set(config.reward_config.scales) == {
        "wholebody_hand_clearance", "wholebody_arm_clearance"}


def test_dex3_training_model_retains_flat_physics_and_original_self_pairs(wholebody):
    root = ET.fromstring(assemble_training_xml())
    original = ET.parse(constants.ROOT_PATH / "g1_mjx_feetonly_torque.xml").getroot()
    original_pairs = {(p.get("geom1"), p.get("geom2")) for p in original.findall("./contact/pair")}
    expected_pairs = {tuple(name.replace("left_hand_collision", "furniture_left_hand_envelope")
                           .replace("right_hand_collision", "furniture_right_hand_envelope")
                           for name in pair) for pair in original_pairs}
    actual_pairs = {(p.get("geom1"), p.get("geom2")) for p in root.findall("./contact/pair")}
    assert actual_pairs == expected_pairs
    assert (wholebody.mj_model.nq, wholebody.mj_model.nv, wholebody.mj_model.nu) == (36, 35, 29)
    assert [wholebody.mj_model.actuator(i).name for i in range(29)] == JOINT_NAMES
    for i in range(wholebody.mj_model.ngeom):
        geom = wholebody.mj_model.geom(i)
        assert not (geom.name or "").startswith("furniture_object_")
        if (geom.name or "").endswith("hand_envelope"):
            assert geom.contype == 0 and geom.conaffinity == 0
    for side in ("left", "right"):
        assert wholebody.hand_envelope_validation[side]["checked_vertices"] > 80_000
        assert wholebody.hand_envelope_validation[side]["minimum_margin_m"] >= 0.005 - 2e-6


def test_wholebody_reset_extends_original_feature_values_without_remapping_noise(
        reference, wholebody, reference_state, wholebody_state):
    source = reference_state.obs
    target = wholebody_state.obs
    old_contract = legacy_observation_contract()
    new_contract = wholebody.observation_contract()
    assert target["state"].shape == (406,)
    assert target["privileged_state"].shape == (494,)
    for key, contract_key in (("state", "actor_features"), ("privileged_state", "critic_features")):
        mapping = index_mapping(old_contract[contract_key], new_contract[contract_key])
        # Hand geometry changes model inertia, not reset coordinates/site poses.
        np.testing.assert_allclose(source[key], np.asarray(target[key])[mapping], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(reference_state.info["rng"], wholebody_state.info["rng"])


def test_leg_target_math_exact_and_upper_slew_bounded(reference, wholebody):
    previous = wholebody._default_qpos + jp.linspace(-0.05, 0.05, 29)
    action = jp.linspace(-1.5, 1.5, 29)
    expected = reference._motor_targets(action[:12], previous)
    actual = wholebody._motor_targets(action, previous)
    np.testing.assert_array_equal(expected[:12], actual[:12])
    assert np.all(np.abs(np.asarray(actual[12:] - previous[12:])) <= 0.04 + 1e-6)


def test_base_reward_is_retained_and_extensions_are_additive(wholebody, wholebody_state):
    state = deepcopy(wholebody_state)
    action = jp.linspace(-0.1, 0.1, 29)
    contact = jp.zeros(2, dtype=bool)
    expected = G1CatEnv._get_reward(wholebody, state.data, action, state.info, state.done, contact)
    actual = wholebody._get_reward(state.data, action, state.info, state.done, contact)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)
    assert set(actual) - set(expected) == {"wholebody_hand_clearance", "wholebody_arm_clearance"}


def test_field_collision_uses_released_zero_threshold_and_50_step_grace(compatible, compatible_state):
    info = dict(compatible_state.info)
    # Put a sampled head point only one millimetre inside an obstacle.  This
    # distinguishes the released threshold zero from the generic .04 default.
    info["headdf"] = jp.asarray([[-0.001]])
    info["head_pos"] = jp.asarray([0.0, 0.0, 1.2])
    info["step"] = jp.asarray(49)
    assert not compatible._get_termination(compatible_state.data, info)
    info["step"] = jp.asarray(50)
    assert compatible._get_termination(compatible_state.data, info)
    info["headdf"] = jp.asarray([[0.0]])
    assert not compatible._get_termination(compatible_state.data, info)
    info["step"] = jp.asarray(1)
    info["head_pos"] = jp.asarray([0.0, 0.0, 0.6])
    assert compatible._get_termination(compatible_state.data, info)


def test_added_hand_collision_uses_same_grace(wholebody, wholebody_state):
    info = dict(wholebody_state.info)
    info["wholebody_probe_features"] = info["wholebody_probe_features"].at[0, 0].set(-0.001)
    info["step"] = jp.asarray(49)
    assert not wholebody._get_termination(wholebody_state.data, info)
    info["step"] = jp.asarray(50)
    assert wholebody._get_termination(wholebody_state.data, info)


def test_wholebody_real_step_keeps_reward_scaling_and_state_tree(wholebody, wholebody_state):
    state = jax.jit(wholebody.step)(wholebody_state, jp.zeros(29))
    jax.block_until_ready(state.reward)
    assert jax.tree_util.tree_structure(state) == jax.tree_util.tree_structure(wholebody_state)
    assert np.isfinite(state.obs["state"]).all()
    assert np.isfinite(state.data.qpos).all()
    expected = np.clip(sum(float(value) for key, value in state.metrics.items()
                           if key.startswith("reward/")) * wholebody.dt, 0.0, 10000.0)
    assert float(state.reward) == pytest.approx(expected, rel=1e-6)


def test_room_goal_and_episode_duration_do_not_change_original_scenes(config):
    task = object.__new__(_WholeBodyTask)
    task._config = wholebody_config(config)
    task._pf_scene_original = jp.asarray([True, False])
    task._pf_scene_goals = jp.asarray([[2.0, 0.0, .75], [8.0, 7.0, .75]])
    task._field_pf_id = jp.asarray(0)
    points = jp.asarray([[1.6, 0.0, 1.2], [7.8, 7.0, 1.2]])
    np.testing.assert_array_equal(task._crossed_goal(points), [True, True])
    assert task._episode_step_limit({"pf_id": jp.asarray(0)}) == 1000
    task._field_pf_id = jp.asarray(1)
    np.testing.assert_array_equal(task._crossed_goal(points), [False, True])
    assert task._episode_step_limit({"pf_id": jp.asarray(1)}) == 4000


def test_public_ragged_task_reset_matches_original_generalist(config, tmp_path, monkeypatch):
    from cat_ppo.envs.g1.env_cat_dagger import G1CatDaggerEnv
    from cat_ppo.furniture import generalist_fields
    scene = dict(path=config.pf_config.path, shape=[20, 21, 16],
                 origin=config.pf_config.origin, dx=config.pf_config.dx,
                 family="original_cat", start=[0.0, 0.0, .8], goal=[2.0, 0.0, .75],
                 reset_xy_scale=[1.0, 1.0], reset_yaw=0.0, sampling_weight=1.0)
    # Field provenance has separate download/hash tests; here use a small bank
    # so the actual public task's scene routing and randomization can be checked.
    monkeypatch.setattr(generalist_fields, "load_generalist_manifest", lambda path: {"scenes": [scene]})
    cfg = wholebody_config(config, bank_manifest=tmp_path / "bank.json", compatibility_mode=True)
    reference = G1CatDaggerEnv(config=cfg.copy_and_resolve_references())
    actual = G1CatWholeBodyEnv(config=cfg)
    key = jax.random.PRNGKey(73)
    _assert_tree_equal(jax.jit(reference.reset)(key), jax.jit(actual.reset)(key))
