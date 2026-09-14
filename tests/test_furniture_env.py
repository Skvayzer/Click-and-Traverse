"""CPU-only geometry, observation and one-control-step integration checks."""
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from cat_ppo.envs.g1.env_furniture import G1FurnitureEnv, assemble_scene_xml, default_config
from cat_ppo.furniture.control import JOINT_NAMES, PROBE_SPECS
from cat_ppo.furniture.scenes import generate_scene, write_scene_bundle


@pytest.fixture(scope='module')
def scene_dir(tmp_path_factory):
    scene = generate_scene(seed=3, split='train', family='table', difficulty='open_floor')
    return write_scene_bundle(scene, tmp_path_factory.mktemp('furniture-env') / 'scene', voxel_size=.2)


@pytest.fixture(scope='module')
def env(scene_dir):
    config = default_config()
    config.noise_config.level = 0.
    return G1FurnitureEnv(scene_dir, config)


@pytest.fixture(scope='module')
def reset_state(env):
    return jax.jit(env.reset)(jax.random.PRNGKey(3))


def test_mjcf_keeps_29_joints_and_physical_boxes_independent_of_fields():
    scene = generate_scene(seed=3, family='table', difficulty='pilot')
    root = ET.fromstring(assemble_scene_xml(scene))
    motors = root.findall('./actuator/motor')
    assert [m.get('name') for m in motors] == JOINT_NAMES
    world = root.find('worldbody')
    for geom in world.iter('geom'):
        if geom.get('class') == 'visual':
            assert geom.get('contype') == geom.get('conaffinity') == '0'
    for i, box in enumerate(scene['boxes']):
        geom = world.find(f"geom[@name='furniture_object_{i}']")
        np.testing.assert_allclose(np.fromstring(geom.get('pos'), sep=' '), box['center'])
        np.testing.assert_allclose(np.fromstring(geom.get('size'), sep=' '), box['half_size'])
        assert geom.get('contype') == '2' and geom.get('conaffinity') == '1'
    assert len(list(world.iter('site'))) >= len(PROBE_SPECS)
    for side in ('left', 'right'):
        hand = world.find(f".//geom[@name='furniture_{side}_hand_envelope']")
        assert hand is not None and hand.get('density') == '0'
    pairs = {(p.get('geom1'), p.get('geom2')) for p in root.findall('./contact/pair')}
    assert ('furniture_left_hand_envelope', 'furniture_torso') in pairs


def test_reset_observation_contract_and_no_initial_collision(env, reset_state):
    assert env.mj_model.nq == 36 and env.mj_model.nv == 35 and env.action_size == 29
    assert env.reset_validation['nominal_full_body_checked']
    contract = env.observation_contract()
    for key, feature_key in [('state', 'actor_features'), ('privileged_state', 'critic_features')]:
        assert reset_state.obs[key].shape == (len(contract[feature_key]),)
        assert np.isfinite(reset_state.obs[key]).all()
    assert reset_state.done == 0
    assert not reset_state.info['forbidden_contact']
    assert np.all(np.asarray(env._contacts(reset_state.data)) == 0)


def test_real_host_collision_classification_includes_hands_and_feet(env):
    data = mujoco.MjData(env.mj_model)
    data.qpos[:] = np.asarray(env._init_q)
    data.qpos[0] = .03  # physically put the body into the west wall
    mujoco.mj_forward(env.mj_model, data)
    contact = SimpleNamespace(geom=np.asarray(data.contact.geom), dist=np.asarray(data.contact.dist))
    actual = np.asarray(env._contacts(SimpleNamespace(contact=contact)))
    assert actual[0] == 1 and actual[3] == 1
    floor = env._floor_geom_id
    foot = int(env._feet_geom_id[0])
    furniture = int(env._furniture_geom_ids[0])
    hand = env.mj_model.geom('furniture_left_hand_envelope').id
    torso = env.mj_model.geom('furniture_torso').id
    def classify(pair):
        return np.asarray(env._contacts(SimpleNamespace(contact=SimpleNamespace(geom=np.asarray([pair]), dist=jp.asarray([-.001])))))
    assert np.all(classify((floor, foot)) == 0)
    assert classify((furniture, foot))[0] == 1  # kicking furniture is forbidden
    assert classify((hand, torso))[4] == 1 and classify((hand, torso))[3] == 0
    assert classify((hand, floor))[5] == 1


def test_one_jitted_physics_step_is_finite_and_keeps_brax_tree(env, reset_state):
    next_state = jax.jit(env.step)(reset_state, jp.zeros(29))
    jax.block_until_ready(next_state.reward)
    assert jax.tree_util.tree_structure(next_state) == jax.tree_util.tree_structure(reset_state)
    assert np.isfinite(next_state.data.qpos).all()
    assert np.isfinite(next_state.obs['state']).all()
    assert np.isfinite(next_state.reward)
    assert int(next_state.info['step']) == 1
    assert set(next_state.metrics) == set(reset_state.metrics)


def test_collision_wins_over_goal_on_first_control_step(env, reset_state, monkeypatch):
    at_goal = reset_state.data.replace(qpos=reset_state.data.qpos.at[:2].set(env._goal))
    def collision_step(data, targets, info):
        return at_goal, info['rng'], jp.asarray([1., 1., 0., 1., 0., 0.]), jp.asarray(0), jp.asarray(2)
    monkeypatch.setattr(env, '_physics_step', collision_step)
    state = env.step(reset_state, jp.zeros(29))
    assert state.info['goal_reached']
    assert state.done and not state.info['success']
    assert state.info['forbidden_contact'] and state.info['hand_contact']
    assert int(state.info['step']) == 1  # no 50-step grace
    assert state.metrics['completion_time'] == env.scene['time_budget']


@pytest.mark.parametrize('dofs', [12, 23])
def test_ablations_have_explicit_native_observation_shapes(scene_dir, dofs):
    config = default_config()
    config.action_dofs = dofs
    ablation = G1FurnitureEnv(scene_dir, config)
    shaped = jax.eval_shape(ablation.reset, jax.random.PRNGKey(7))
    contract = ablation.observation_contract()
    assert shaped.obs['state'].shape == (len(contract['actor_features']),)
    assert shaped.obs['privileged_state'].shape == (len(contract['critic_features']),)
    assert ablation.action_size == dofs


def test_map_latency_age_unknown_and_disabled_probe_channels(scene_dir):
    config = default_config()
    config.perception_mode = 'corrupted'
    config.map_latency_steps = 2
    config.map_update_interval_steps = 3
    config.unknown_probability = 1.
    config.map_position_noise_std = .02
    config.probe_features_enabled = False
    config.prediction_enabled = False
    delayed = G1FurnitureEnv(scene_dir, config)
    state = jax.jit(delayed.reset)(jax.random.PRNGKey(8))
    np.testing.assert_array_equal(state.obs['state'][-198:], 0.)
    np.testing.assert_array_equal(state.info['actor_probe_features'][:, 7], 1.)
    assert np.all(np.asarray(state.info['actor_probe_features'][:, :3]) < -1.)
    refresh = jax.jit(lambda data, info: delayed._update_features(data, info))
    info = dict(state.info)
    ages = []
    for step in range(1, 6):
        info['step'] = jp.asarray(step)
        info = refresh(state.data, info)
        ages.append(float(info['actor_probe_features'][0, 6]))
    np.testing.assert_allclose(ages, [.02, .04, .06, .08, .04], atol=1e-6)
    features = np.asarray(info['actor_probe_features'])
    np.testing.assert_allclose(features[:, 0], features[:, 1])
    np.testing.assert_allclose(features[:, 0], features[:, 2])


def test_raised_and_tucked_presets_improve_real_hand_geometry_without_self_contact(env):
    from cat_ppo.furniture.control import posture_actions, motor_targets
    from cat_ppo.envs.g1 import constants
    nominal = np.asarray(constants.DEFAULT_QPOS[7:])
    lower, upper = np.asarray(env._soft_lowers), np.asarray(env._soft_uppers)
    hand_sites = [env.mj_model.site('furniture_probe_' + name).id
                  for name, _, _, _ in PROBE_SPECS if 'corner' in name]
    measurements = {}
    for mode in ('nominal', 'raised', 'tucked'):
        action = posture_actions(np.zeros(29), JOINT_NAMES, mode)
        targets = nominal.copy()
        for _ in range(100):
            targets = motor_targets(action, targets, nominal, lower, upper, np.arange(29))
        # Float32 targets can round at a float64 model's soft-limit boundary.
        assert np.all(targets >= lower - 1e-6) and np.all(targets <= upper + 1e-6)
        data = mujoco.MjData(env.mj_model)
        data.qpos[:] = np.asarray(env._init_q)
        data.qpos[7:] = targets
        mujoco.mj_forward(env.mj_model, data)
        world = data.site_xpos[hand_sites]
        local = (world - data.qpos[:3]) @ data.xmat[env.mj_model.body('pelvis').id].reshape(3, 3)
        measurements[mode] = (world[:, 2].min(), np.ptp(local[:, 1]))
        contacts = SimpleNamespace(geom=np.asarray(data.contact.geom), dist=np.asarray(data.contact.dist))
        assert np.asarray(env._contacts(SimpleNamespace(contact=contacts)))[0] == 0
    assert measurements['raised'][0] > measurements['nominal'][0] + .25
    assert measurements['tucked'][1] < measurements['nominal'][1] - .10
    assert measurements['tucked'][0] < measurements['raised'][0] - .20


def test_final_integrated_pose_contact_is_checked_before_goal(env, reset_state, monkeypatch):
    from mujoco import mjx
    # Mimic an integration that updates qpos but leaves the preceding forward
    # pass's contacts in data, as native mjx.step does. Only final FK sees wall.
    def integrate_without_refresh(model, data):
        return data.replace(qpos=data.qpos.at[0].set(.03))
    monkeypatch.setattr(mjx, 'step', integrate_without_refresh)
    data, _, contacts, first_substep, _ = env._physics_step(
        reset_state.data, reset_state.info['motor_targets'], reset_state.info)
    assert contacts[0] and contacts[3]
    assert first_substep == env.n_substeps
    assert data.qpos[0] == pytest.approx(.03)
