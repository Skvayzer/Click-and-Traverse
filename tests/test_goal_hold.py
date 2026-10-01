"""Config checks for ending room episodes after a short stand at the goal, and for route recovery."""
import pytest


def test_goal_hold_and_route_recovery_flags():
    from cat_mjlab.config import wholebody_config
    c = wholebody_config(goal_hold_seconds=1.0)
    assert c['goal_hold_seconds'] == 1.0 and 'route_recovery' not in c
    assert 'goal_hold_seconds' not in wholebody_config(c, goal_hold_seconds=0)
    c = wholebody_config(route_recovery_speed=.4)
    assert c['route_recovery'] == dict(speed=.4, lost_seconds=5.)
    assert 'route_recovery' not in wholebody_config(c, route_recovery_speed=0)
    for bad in (dict(goal_hold_seconds=-1.), dict(route_recovery_speed=-.1), dict(route_recovery_speed=.4, route_lost_seconds=0),
                dict(route_lost_seconds=3.)):
        with pytest.raises(ValueError):
            wholebody_config(**bad)


def test_defaults_leave_existing_runs_unchanged():
    from cat_mjlab.config import wholebody_config
    c = wholebody_config()
    assert 'goal_hold_seconds' not in c and 'route_recovery' not in c


def test_smoothness_flags():
    from cat_mjlab.config import wholebody_config
    c = wholebody_config(action_rate_weight=-.02, joint_acc_weight=-5e-6)
    s = c['reward_config']['scales']
    assert s['action_rate'] == -.02 and s['smoothness_joint'] == -5e-6
    assert 'action_rate' not in wholebody_config(c, action_rate_weight=0)['reward_config']['scales']
    with pytest.raises(ValueError):
        wholebody_config(action_rate_weight=.1)


def test_hand_body_contact_model_and_flag():
    import mujoco
    from cat_mjlab.config import wholebody_config
    from cat_mjlab.model import assemble_training_xml, hand_body_pairs
    pairs = hand_body_pairs()
    regions = [r for _, _, r in pairs]
    assert regions.count('hand') == 1 and regions.count('trunk') == 8 and regions.count('head') == 2 and regions.count('arm') == 14
    # a hand is never paired with its own arm chain
    assert not any(a.startswith('furniture_left') and 'arm_left_' in b for a, b, _ in pairs)
    base = mujoco.MjModel.from_xml_string(assemble_training_xml())
    m = mujoco.MjModel.from_xml_string(assemble_training_xml(hand_body_collision=True))
    assert m.npair == base.npair + len(pairs) and abs(m.body_mass.sum() - base.body_mass.sum()) < 1e-9
    d = mujoco.MjData(m); d.qpos[2] = .79; mujoco.mj_forward(m, d)
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for i in range(d.ncon) for g in (d.contact[i].geom1, d.contact[i].geom2)}
    assert not any('envelope' in (n or '') for n in names)          # nominal pose: no hand-body contact
    c = wholebody_config(hand_body_contact_weight=-2.)
    assert c['hand_body_contact'] and c['reward_config']['scales']['hand_body_contact'] == -2.
    assert 'hand_body_contact' not in wholebody_config(hand_body_contact_weight=0.)['reward_config']['scales']


def test_heading_hand_probe_flag():
    from cat_mjlab.config import wholebody_config
    c = wholebody_config(heading_align_weight=.4, heading_hand_probes=True)
    assert c['heading_align']['hand_probe'] == dict(half_width=.22, radius=.10)
    assert 'hand_probe' not in wholebody_config(c, heading_hand_probes=False)['heading_align']
    with pytest.raises(ValueError):
        wholebody_config(heading_hand_probes=True)


def test_teleop_flag():
    from cat_mjlab.config import wholebody_config
    c = wholebody_config(teleop_fraction=.3)
    assert c['teleop']['fraction'] == .3 and c['teleop']['episode_seconds'] == 20.
    assert 'teleop' not in wholebody_config(c, teleop_fraction=0)
    with pytest.raises(ValueError):
        wholebody_config(teleop_fraction=1.5)


def test_body_joystick_flags_and_yaw_tracking():
    import math
    import torch
    from cat_mjlab.config import wholebody_config
    from cat_mjlab import task_math as tm
    c = wholebody_config(teleop_fraction=.25, teleop_body_commands=True)
    assert c['teleop']['mode'] == 'body' and c['yaw_command'] and c['reward_config']['scales']['tracking_yaw'] == 1.
    with pytest.raises(ValueError):
        wholebody_config(teleop_body_commands=True)
    assert float(tm.yaw_tracking(torch.tensor([.5]), torch.tensor([.5]))) == 1.
    assert float(tm.yaw_tracking(torch.tensor([.5]), torch.tensor([-.5]))) < .02
    # The heading frame removes the body's yaw: the body's world yaw must come from navi itself.
    R = torch.tensor([[[math.cos(1.), -math.sin(1.), 0.], [math.sin(1.), math.cos(1.), 0.], [0., 0., 1.]]])
    navi = tm.navi_rotation(R)
    assert abs(float(tm.matrix_rpy(navi.transpose(-1, -2) @ R)[0, 2])) < 1e-6
    assert abs(float(torch.atan2(navi[0, 1, 0], navi[0, 0, 0])) - 1.) < 1e-6
