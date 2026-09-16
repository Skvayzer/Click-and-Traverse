"""Physical smoothing and terminal/goal semantics, independent of PPO scores."""
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest

from cat_ppo.furniture.wholebody_stability import (
    goal_status, native_fault_flags, upper_stability_terms,
)


def terms(target, previous, previous_previous, hand=.4, elbow=.4):
    return upper_stability_terms(jp.asarray(target), jp.asarray(previous),
        jp.asarray(previous_previous), jp.zeros(17), jp.full((2, 1), hand),
        jp.full(2, elbow), dt=.02)


def test_stationary_nominal_targets_have_no_stability_cost():
    costs, telemetry = terms(jp.zeros(17), jp.zeros(17), jp.zeros(17))
    assert all(float(value) == 0. for value in costs.values())
    assert telemetry["clear_posture_gate"] == 1.


def test_constant_target_velocity_has_no_acceleration_cost_but_reversal_does():
    constant, telemetry = terms(jp.full(17, .04), jp.full(17, .02), jp.zeros(17))
    assert float(telemetry["upper_target_velocity_rms"]) == pytest.approx(1.)
    assert float(constant["wholebody_upper_target_velocity"]) == pytest.approx(.25)
    assert float(constant["wholebody_upper_target_acceleration"]) == pytest.approx(0.)
    reversal, reverse_telemetry = terms(jp.zeros(17), jp.full(17, .02), jp.zeros(17))
    assert float(reverse_telemetry["upper_target_acceleration_rms"]) == pytest.approx(100.)
    assert float(reversal["wholebody_upper_target_acceleration"]) == pytest.approx(25.)


def test_nearby_hand_or_elbow_releases_nominal_posture_without_disabling_smoothing():
    args = (jp.full(17, .2), jp.full(17, .18), jp.full(17, .16))
    comfortable, clear = terms(*args)
    near_hand, hand = terms(*args, hand=.12)
    near_elbow, elbow = terms(*args, elbow=.08)
    half, halfway = terms(*args, hand=.18)
    assert comfortable["wholebody_upper_clear_posture"] > 0
    assert hand["clear_posture_gate"] == elbow["clear_posture_gate"] == 0
    assert near_hand["wholebody_upper_clear_posture"] == near_elbow["wholebody_upper_clear_posture"] == 0
    assert float(halfway["clear_posture_gate"]) == pytest.approx(.5)
    assert float(half["wholebody_upper_clear_posture"]) == pytest.approx(float(comfortable["wholebody_upper_clear_posture"]) * .5)
    assert near_hand["wholebody_upper_target_velocity"] == comfortable["wholebody_upper_target_velocity"]


def test_identical_single_joint_displacement_costs_more_at_waist():
    waist, _ = terms(jp.zeros(17).at[0].set(.2), jp.zeros(17), jp.zeros(17))
    arm, _ = terms(jp.zeros(17).at[5].set(.2), jp.zeros(17), jp.zeros(17))
    for name in waist:
        assert float(waist[name]) == pytest.approx(4. * float(arm[name]))


def fake_scene(*, room=False):
    return SimpleNamespace(
        _field_pf_id=jp.array(0), _pf_expanded=True,
        _pf_origins=jp.array([[-.5, -1., 0.]]), _pf_shapes=jp.array([[75, 50, 38]]),
        _pf_dxs=jp.array([.04]), _pf_crossed_is_x_plane=jp.array([not room]),
        _crossed_goal=(lambda positions: jp.linalg.norm(positions[..., :2] - jp.array([2., 0.]), axis=-1) <= .5)
                      if room else (lambda positions: positions[..., 0] > 1.5))


def positions(root_x, root_y=0.):
    data = SimpleNamespace(qpos=jp.array([root_x, root_y, .8]), qvel=jp.zeros(35))
    return data, {"feet_pos": jp.array([[root_x, root_y-.1, .05], [root_x, root_y+.1, .05]])}


def test_cat_valid_approach_before_field_and_lateral_bypass_are_distinct():
    env = fake_scene()
    data, info = positions(-.8)
    approach = goal_status(env, data, info, jp.array(False))
    assert not approach["outside_bounds"]
    data, info = positions(1.7)
    clean = goal_status(env, data, info, approach["outside_bounds"])
    assert clean["raw_goal"] and clean["goal_reached"]
    data, info = positions(.8, 1.1)
    bypass = goal_status(env, data, info, jp.array(False))
    data, info = positions(1.7)
    returned = goal_status(env, data, info, bypass["outside_bounds"])
    assert returned["raw_goal"] and not returned["goal_reached"]


def test_room_requires_both_feet_near_goal_and_keeps_xy_exit_history():
    env = fake_scene(room=True)
    data, info = positions(1.8)
    assert goal_status(env, data, info, jp.array(False))["goal_reached"]
    info["feet_pos"] = info["feet_pos"].at[0, 0].set(1.3)
    assert not goal_status(env, data, info, jp.array(False))["raw_goal"]
    data, info = positions(-.8)
    assert goal_status(env, data, info, jp.array(False))["outside_bounds"]


def test_faults_preserve_grace_and_overlapping_fall_field_causes(monkeypatch):
    from cat_ppo.furniture import wholebody_stability as helper
    monkeypatch.setattr(helper.collision, "geoms_colliding", lambda *args: jp.array(False))
    env = SimpleNamespace(_config=SimpleNamespace(term_collision_threshold=.04,
        terminate_on_elbow_collision=True), get_gravity=lambda *args: jp.array([0., 0., 1.]),
        _right_foot_geom_id=0, _left_foot_geom_id=1, _right_shin_geom_id=2, _left_shin_geom_id=3)
    data, info = positions(0.)
    info.update({key: jp.array([[.2]]) for key in ("headdf", "pelvdf", "torsdf", "feetdf", "handsdf", "kneesdf", "shldsdf")})
    info.update(step=jp.array(49), head_pos=jp.array([0., 0., 1.]), wholebody_elbow_clearance=jp.array([.1, .1]))
    info["handsdf"] = jp.array([[-.05]])
    assert not native_fault_flags(env, data, info)["any"]
    info["step"] = jp.array(50)
    info["head_pos"] = jp.array([0., 0., .6])
    flags = native_fault_flags(env, data, info)
    assert flags["fall"] and flags["obstacle"] and flags["hands_field"] and flags["any"]
    assert not flags["self_contact"] and not flags["numerical"]


def test_stability_costs_jit_and_vmap_remain_finite():
    costs, _ = jax.jit(jax.vmap(lambda x: terms(x, jp.zeros(17), jp.zeros(17))))(jp.ones((3, 17)) * .04)
    assert all(np.isfinite(value).all() for value in costs.values())
