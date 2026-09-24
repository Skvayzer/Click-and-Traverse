"""CPU checks for the posture terms and the hand-vs-own-leg clearance penalty."""
import math
import torch
from cat_mjlab.task_math import posture_terms, segment_distance, self_clearance_terms


def _posture(pitch_deg, head_z, gf_z=0., sdf=1.):
    terms, crouch = posture_terms(torch.tensor([math.radians(pitch_deg)]), torch.tensor([head_z]),
                                  torch.tensor([[0., 0., gf_z]]), torch.tensor([sdf]), torch.zeros(1, 3))
    return {k: float(v) for k, v in terms.items()}, bool(crouch)


def test_upright_and_tall_reward_the_nominal_pose_and_punish_the_measured_hunch():
    good, crouch = _posture(0., 1.24)
    assert not crouch and good['upright'] > .99 and good['stand_tall'] > .99
    hunch, _ = _posture(40., 1.0)          # what the videos showed
    assert hunch['upright'] < .01 and hunch['stand_tall'] < .01
    assert _posture(8., 1.18)[0]['upright'] > .5   # tolerance, not a razor


def test_crouch_is_free_only_for_a_reason():
    hurdle, crouch = _posture(40., 0.9, gf_z=-.6)       # head field points down
    assert crouch and hurdle['upright'] == 0. and hurdle['stand_tall'] == 0.
    overhead, crouch = _posture(40., 0.9, sdf=.1)        # something 10 cm above the head
    assert crouch
    assert not _posture(40., 0.9, gf_z=-.1, sdf=.5)[1]   # neither: the loophole is closed


def test_segment_distance_and_self_clearance_geometry():
    hands = torch.tensor([[[0., .3, 0.], [0., -.3, 0.]]])            # 30 cm either side
    a = torch.tensor([[[0., .06, 0.], [0., -.06, 0.]]]); b = torch.tensor([[[0., .06, -.4], [0., -.06, -.4]]])
    d = segment_distance(hands, a, b)
    assert torch.allclose(d[0, 0, 0], torch.tensor(.24)) and torch.allclose(d[0, 1, 1], torch.tensor(.24))
    radii = torch.tensor([.1034, .1034]); caps = torch.tensor([.06, .06])
    penalty, gap = self_clearance_terms(hands, radii, a, b, caps)
    assert abs(float(gap) - (.24 - .1034 - .06)) < 1e-6 and float(penalty) == 0.    # 7.7 cm clear: outside the 4 cm ramp
    touching = hands.clone(); touching[0, 0, 1] = .06 + .1034 + .06                 # exactly on the surface
    penalty2, gap2 = self_clearance_terms(touching, radii, a, b, caps)
    assert abs(float(gap2)) < 1e-6 and abs(float(penalty2) - 1.) < 1e-6              # touching: full cost for that pair
    inside = hands.clone(); inside[0, 0, 1] = .2      # 2.3 cm into the left thigh envelope only
    assert float(self_clearance_terms(inside, radii, a, b, caps)[1]) < 0
    assert abs(float(self_clearance_terms(inside, radii, a, b, caps)[0]) - 1.) < 1e-6    # bounded inside


def test_config_flags():
    from cat_mjlab.config import wholebody_config
    c = wholebody_config(upright_weight=1., stand_tall_weight=1., torso_rate_weight=-.5, self_clearance_weight=-20.,
                         upper_posture_weight=-.5, upper_home_shoulder_pitch=-.3, terminate_on_hand_self_contact=True)
    s = c['reward_config']['scales']
    assert s['upright'] == 1. and s['stand_tall'] == 1. and s['torso_rate'] == -.5 and s['self_clearance'] == -20.
    assert s['wholebody_upper_clear_posture'] == -.5 and c['upper_home_shoulder_pitch'] == -.3 and c['terminate_on_hand_self_contact']
    off = wholebody_config(c, upright_weight=0, self_clearance_weight=0)
    assert 'upright' not in off['reward_config']['scales'] and 'self_clearance' not in off['reward_config']['scales']
    import pytest
    with pytest.raises(ValueError):
        wholebody_config(self_clearance_weight=5.)


def test_stand_still_cost_prices_motion_only_at_zero_command():
    from cat_mjlab.task_math import stand_still_cost
    v = torch.tensor([[0., 0., 0.], [.3, 0., 0.], [.3, 0., 0.]]); w = torch.zeros(3, 3); move = torch.tensor([0., 0., 1.])
    c = stand_still_cost(v, w, move)
    assert float(c[0]) == 0. and float(c[1]) > .99 and float(c[2]) == 0.
    assert 0. < float(stand_still_cost(torch.tensor([[.05, 0., 0.]]), torch.zeros(1, 3), torch.zeros(1))) < .5


def test_sdf_rate_contract_and_function_preserving_widening():
    from cat_mjlab.observation_contract import ACTOR_SIZE, actor_size
    from cat_ppo.furniture.control import mjlab_observation_contract
    from cat_mjlab.runner import _expand_actor_input
    assert actor_size(False) == 222 and actor_size(True) == 226
    assert len(mjlab_observation_contract(True)['actor_features']) == 226
    assert mjlab_observation_contract(True)['actor_features'][:222] == mjlab_observation_contract(False)['actor_features']
    w = {'actor.layers.0.weight': torch.randn(512, 222), 'actor.layers.0.bias': torch.randn(512)}
    wide = _expand_actor_input(w, 226)
    assert wide['actor.layers.0.weight'].shape == (512, 226)
    x = torch.randn(3, 222); extra = torch.randn(3, 4)
    assert torch.allclose(x @ w['actor.layers.0.weight'].T, torch.cat((x, extra), 1) @ wide['actor.layers.0.weight'].T, atol=1e-5)
    assert _expand_actor_input(w, 222) is w
