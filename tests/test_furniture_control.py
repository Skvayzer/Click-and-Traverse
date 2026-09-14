import numpy as np
import pytest
import jax
import jax.numpy as jp

from cat_ppo.furniture.control import (JOINT_NAMES, LEG_NAMES, LEGACY_OBS_NAMES, joint_names,
    motor_targets, observation_contract, legacy_observation_contract, route_coordinate)
from cat_ppo.furniture.perception import sample_grid, probe_features, apply_uncertainty_margin


def test_joint_modes_and_warmstart_feature_identity():
    original = legacy_observation_contract()
    assert (len(original['actor_features']), len(original['critic_features'])) == (162, 250)
    for dofs, sizes in ((12, (360, 448)), (23, (382, 470)), (29, (406, 494))):
        contract = observation_contract(dofs)
        assert (len(contract['actor_features']), len(contract['critic_features'])) == sizes
        assert len(contract['action_names']) == dofs
        for key in ('actor_features', 'critic_features'):
            assert len(set(contract[key])) == len(contract[key])
            assert set(original[key]).issubset(contract[key])
        assert contract['action_names'][:12] == LEG_NAMES
    assert all('wrist' not in n for n in joint_names(23)[0])
    assert sum('wrist' in n for n in JOINT_NAMES) == 6
    with pytest.raises(ValueError):
        joint_names(28)


def test_bounded_leg_increments_and_upper_nominal_offsets_do_not_drift():
    nominal = np.zeros(29)
    lower, upper = np.full(29, -1.0), np.full(29, 1.0)
    previous = nominal.copy()
    action = np.ones(29)
    result = motor_targets(action, previous, nominal, lower, upper, np.arange(29))
    np.testing.assert_allclose(result[:12], .5)
    np.testing.assert_allclose(result[12:], .04)
    for _ in range(100):
        result = motor_targets(action * 10, result, nominal, lower, upper, np.arange(29))
    np.testing.assert_allclose(result[:12], 1.0)
    np.testing.assert_allclose(result[12:], .8)
    # Zero upper action returns toward nominal under the same slew bound.
    next_result = motor_targets(np.zeros(29), result, nominal, lower, upper, np.arange(29))
    np.testing.assert_allclose(next_result[12:], .76)
    np.testing.assert_allclose(next_result[:12], 1.0)


def test_23_actions_leave_all_wrist_targets_nominal():
    ids = np.asarray([JOINT_NAMES.index(n) for n in LEGACY_OBS_NAMES])
    nominal = np.linspace(-.1, .1, 29)
    result = motor_targets(np.ones(23), nominal, nominal, np.full(29, -2.), np.full(29, 2.), ids)
    wrist = [i for i, name in enumerate(JOINT_NAMES) if 'wrist' in name]
    np.testing.assert_array_equal(result[wrist], nominal[wrist])
    fn = jax.jit(lambda action, prev: motor_targets(action, prev, jp.asarray(nominal),
        jp.full(29, -2.), jp.full(29, 2.), jp.asarray(ids), xp=jp))
    np.testing.assert_allclose(fn(jp.ones(23), jp.asarray(nominal)), result, atol=1e-7)


def test_sampler_reproduces_affine_xyz_fields_and_flags_outside():
    grid = np.stack(np.meshgrid(np.arange(3), np.arange(4), np.arange(5), indexing='ij'), axis=-1)
    points = jp.asarray([[.2, .4, .8], [1.9, 2.9, 3.9], [2., 3., 4.], [-.01, 1, 1]])
    values, known = jax.jit(sample_grid)(jp.asarray(grid, dtype=jp.float32), points, jp.zeros(3), 1.)
    np.testing.assert_allclose(values[:3], points[:3], atol=1e-6)
    np.testing.assert_array_equal(known, [True, True, True, False])


def test_forecast_and_latency_uncertainty_ablations_change_clearance():
    sdf = jp.broadcast_to(jp.arange(8, dtype=jp.float32)[:, None, None], (8, 4, 4))
    bf = jp.broadcast_to(jp.asarray([1., 0, 0]), (8, 4, 4, 3))
    args = (sdf, bf, jp.asarray([[3., 1., 1.]]), jp.asarray([[-2., 0, 0]]), jp.asarray([.1]), jp.zeros(3), 1.)
    predictive = probe_features(*args, age=.2, uncertainty=.03)
    current_only = probe_features(*args, prediction_enabled=False, age=.2, uncertainty=.03)
    np.testing.assert_allclose(predictive[0, :3], [2., 2., 2.], atol=1e-6)  # display clip is 2m
    # Query nearer obstacle to see distinct .2/.4s predictions.
    args = args[:2] + (jp.asarray([[1.5, 1., 1.]]),) + args[3:]
    predictive = probe_features(*args, age=.2, uncertainty=.03)
    current_only = probe_features(*args, prediction_enabled=False, age=.2, uncertainty=.03)
    np.testing.assert_allclose(predictive[0, :3], [1.4, 1., .6], atol=1e-6)
    np.testing.assert_allclose(current_only[0, :3], [1.4] * 3, atol=1e-6)
    robust = apply_uncertainty_margin(predictive, fixed_margin=.01, std_multiplier=2., relative_motion_bound=1.)
    np.testing.assert_allclose(robust[0, :3], np.asarray(predictive[0, :3]) - .27, atol=1e-6)
    np.testing.assert_array_equal(apply_uncertainty_margin(predictive, enabled=False), predictive)
    unknown = probe_features(*args, unknown=jp.asarray([True]))
    assert unknown[0, 7] == 1 and np.all(np.asarray(unknown[0, :3]) < 0)


def test_route_progress_is_not_hand_motion_or_world_x():
    route = np.asarray([[0., 0.], [0., 2.], [2., 2.]])
    progress, error = route_coordinate(np.asarray([1., 2.1]), route)
    assert progress == pytest.approx(3.)
    assert error == pytest.approx(.1)


def test_posture_baselines_keep_legs_and_choose_tuck_under_overhead():
    from cat_ppo.furniture.control import posture_actions, PROBE_SPECS
    action = np.linspace(-.5, .5, 29)
    probes = np.zeros((len(PROBE_SPECS), 9))
    probes[:, :3] = .1
    for mode in ('nominal', 'raised', 'tucked', 'contextual'):
        actual = posture_actions(action, JOINT_NAMES, mode, probe_features=probes)
        np.testing.assert_array_equal(actual[:12], action[:12])
        assert np.max(np.abs(actual)) <= 1
    raised = posture_actions(action, JOINT_NAMES, 'contextual', probe_features=probes)
    probes[:, 5] = -1.0
    tucked = posture_actions(action, JOINT_NAMES, 'contextual', probe_features=probes)
    assert raised[15] < tucked[15]  # shoulder pitches: tuck lifts less under overhang.
    with pytest.raises(ValueError):
        posture_actions(action, JOINT_NAMES, 'unknown')
