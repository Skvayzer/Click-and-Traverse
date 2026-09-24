"""CPU checks for the clearance-gated forward-facing bonus."""
import math
import torch
from cat_mjlab.task_math import heading_align_reward, heading_probe_points


def _reward(yaw, sdf, move=1.):
    direction = torch.tensor([[1., 0.]])
    return float(heading_align_reward(direction, torch.tensor([yaw]), torch.tensor([move]), torch.tensor([[sdf]*4])))


def test_open_space_prefers_forward_and_never_penalises():
    assert math.isclose(_reward(0., .30), 1., abs_tol=1e-6)            # facing the command, room to spare
    assert math.isclose(_reward(math.pi/2, .30), .5, abs_tol=1e-6)     # sideways
    assert math.isclose(_reward(math.pi, .30), 0., abs_tol=1e-6)       # backwards
    assert min(_reward(y, .30) for y in (0., 1., 2., 3.)) >= 0.   # bonus form: never negative


def test_narrow_gap_switches_off_for_every_heading():
    # 0.40 m corridor: probes at +-0.16 m sit 0.04 m from the walls, under the 0.05 m margin.
    for yaw in (0., math.pi/2, math.pi):
        assert _reward(yaw, .04) == 0.


def test_gate_is_continuous_and_uses_the_worst_probe():
    assert 0. < _reward(0., .10) < 1.
    worst = heading_align_reward(torch.tensor([[1., 0.]]), torch.zeros(1), torch.ones(1),
                                 torch.tensor([[.30, .30, .30, .02]]))
    assert float(worst) == 0.


def test_idle_and_no_command_earn_nothing():
    assert _reward(0., .30, move=0.) == 0.
    zero = heading_align_reward(torch.zeros(1, 2), torch.zeros(1), torch.ones(1), torch.full((1, 4), .3))
    assert float(zero) == 0.


def test_probes_are_lateral_to_the_command_and_one_stride_ahead():
    root = torch.tensor([[1., 2.]]); direction = torch.tensor([[0., 1.]]); height = torch.tensor([1.3])
    p = heading_probe_points(root, direction, height)
    assert p.shape == (1, 4, 3) and torch.allclose(p[..., 2], torch.full((1, 4), 1.3))
    assert torch.allclose(p[0, :2, :2], torch.tensor([[.84, 2.], [1.16, 2.]]))
    assert torch.allclose(p[0, 2:, :2], torch.tensor([[.84, 2.3], [1.16, 2.3]]))


def test_config_registers_scale_and_zero_removes_it():
    from cat_mjlab.config import wholebody_config
    on = wholebody_config(heading_align_weight=.4)
    assert on['reward_config']['scales']['heading_align'] == .4 and on['heading_align']['half_width'] == .16
    off = wholebody_config(on, heading_align_weight=0)
    assert 'heading_align' not in off['reward_config']['scales'] and 'heading_align' not in off
