"""Opt-in clearance pressure and arm freedom without loosening waist control."""
import jax
import jax.numpy as jp
import numpy as np
import pytest

from cat_ppo.furniture.wholebody_stability import hand_clearance_pressure, upper_stability_terms


def costs(target, *, previous=None, prior=None, hands=(.4, .4), elbows=(.4, .4), enabled=True):
    target = jp.asarray(target)
    return upper_stability_terms(
        target, jp.zeros(17) if previous is None else previous,
        jp.zeros(17) if prior is None else prior, jp.zeros(17),
        jp.asarray(hands).reshape(2, 1), jp.asarray(elbows), dt=.02,
        hand_protection=enabled)


def test_pressure_is_bounded_monotonic_and_gives_earlier_clearance_gradient():
    distances = jp.array([-.1, 0., .01, .02, .034, .04, .10, .12, .19, .20, .4])
    values = jax.jit(jax.vmap(lambda d: hand_clearance_pressure(jp.array([d, d]))[0]))(distances)
    assert np.all(np.isfinite(values)) and np.all(np.diff(values) <= 0)
    assert np.min(values) == 0. and np.max(values) == 1.
    assert float(values[0]) == float(values[1]) == 1.
    # Unlike the old 12 cm hinge, a clearance-improving move at 15 cm is useful.
    derivative = jax.grad(lambda d: hand_clearance_pressure(jp.array([d, .4]))[0])(jp.array(.15))
    assert derivative < 0
    assert hand_clearance_pressure(jp.array([.2, .5]))[0] == 0


def test_certified_raised_and_tucked_pose_clearances_are_not_treated_as_contact():
    raised, _ = hand_clearance_pressure(jp.full(2, .208))
    tucked, _ = hand_clearance_pressure(jp.full(2, .034))
    contact, _ = hand_clearance_pressure(jp.zeros(2))
    assert raised == 0
    assert 0 < tucked < .16 * contact
    # The native reward still applies dt; at 10 cm the new early pressure is
    # materially stronger than the old squared metric deficit.
    early, _ = hand_clearance_pressure(jp.full(2, .10))
    new_step_cost = .5 * .02 * float(early)
    old_step_cost = 5. * .02 * (.12 - .10) ** 2
    assert new_step_cost == pytest.approx(.0005)
    assert new_step_cost > 10. * old_step_cost
    assert .5 * .02 * float(contact) == pytest.approx(.01)


def test_one_unsafe_hand_contributes_without_cancelling_a_clear_hand():
    cost, telemetry = hand_clearance_pressure(jp.array([0., .4]))
    assert float(cost) == pytest.approx(.5)
    assert telemetry['left_hand_clearance_pressure'] == 1.
    assert telemetry['right_hand_clearance_pressure'] == 0.
    assert telemetry['hand_protection_active_fraction'] == .5


def test_only_arm_motion_costs_are_reduced_and_waist_normalization_is_unchanged():
    for joint in (0, 1, 2, 3, 6, 10, 16):
        target = jp.zeros(17).at[joint].set(.04)
        old, _ = costs(target, enabled=False)
        new, _ = costs(target)
        velocity_ratio, acceleration_ratio = ((1., 1.) if joint < 3 else (.5, .1))
        assert float(new['wholebody_upper_target_velocity']) == pytest.approx(float(old['wholebody_upper_target_velocity']) * velocity_ratio)
        assert float(new['wholebody_upper_target_acceleration']) == pytest.approx(float(old['wholebody_upper_target_acceleration']) * acceleration_ratio)
        assert float(new['wholebody_upper_clear_posture']) == pytest.approx(float(old['wholebody_upper_clear_posture']))


def test_each_arm_posture_releases_early_but_waist_posture_stays_regularized():
    left = jp.zeros(17).at[3].set(.4)
    right = jp.zeros(17).at[10].set(.4)
    waist = jp.zeros(17).at[0].set(.4)
    for hands, elbows in (((.19, .4), (.4, .4)), ((.4, .4), (.08, .4))):
        left_cost, telemetry = costs(left, hands=hands, elbows=elbows)
        right_cost, _ = costs(right, hands=hands, elbows=elbows)
        waist_cost, _ = costs(waist, hands=hands, elbows=elbows)
        assert telemetry['left_arm_posture_gate'] == 0
        assert telemetry['right_arm_posture_gate'] == 1
        assert left_cost['wholebody_upper_clear_posture'] == 0
        assert right_cost['wholebody_upper_clear_posture'] > 0
        assert waist_cost['wholebody_upper_clear_posture'] > 0
    _, halfway = costs(left, hands=(.25, .4))
    assert float(halfway['left_arm_posture_gate']) == pytest.approx(.5, abs=1e-6)


def test_open_space_prefers_nominal_without_rewarding_any_hand_height():
    nominal, _ = costs(jp.zeros(17))
    raised, _ = costs(jp.zeros(17).at[3].set(-.8).at[6].set(-.8))
    tucked, _ = costs(jp.zeros(17).at[3].set(-.3).at[6].set(.7))
    assert nominal['wholebody_upper_clear_posture'] == 0
    assert raised['wholebody_upper_clear_posture'] > 0
    assert tucked['wholebody_upper_clear_posture'] > 0
    assert hand_clearance_pressure(jp.full(2, .4))[0] == 0


def test_motion_and_protection_telemetry_compile_without_changing_old_terms():
    target = jp.linspace(-.04, .04, 17)
    default, _ = upper_stability_terms(target, jp.zeros(17), jp.zeros(17), jp.zeros(17),
                                       jp.full((2, 1), .4), jp.full(2, .4), dt=.02)
    explicit, _ = costs(target, enabled=False)
    for key in default:
        np.testing.assert_array_equal(default[key], explicit[key])
    result, telemetry = jax.jit(jax.vmap(lambda t: costs(t, hands=(.1, .3))))(jp.stack([target, -target]))
    assert all(np.isfinite(x).all() for x in (*result.values(), *telemetry.values()))
    np.testing.assert_allclose(telemetry['left_shoulder_pitch_target_offset'], np.array([target[3], -target[3]]))
    np.testing.assert_allclose(telemetry['right_elbow_target_offset'], np.array([target[13], -target[13]]))


@pytest.mark.parametrize('target,early,weight', [(0., .2, .8), (.3, .2, .8), (.04, .2, 2.)])
def test_invalid_clearance_pressure_parameters_fail(target, early, weight):
    with pytest.raises(ValueError):
        hand_clearance_pressure(jp.ones(2), target_clearance=target,
                                anticipation_distance=early, near_weight=weight)
