import torch

from cat_mjlab.refusal_training import advance_clock, apply_terms, refusal_failure


def test_progress_marker_and_command_release_reset_stall_clock():
    progress = torch.tensor([.04, .01, .04, .03])
    seconds = torch.tensor([1., 2., 3., 4.])
    moving = torch.tensor([True, True, False, True])
    advance = torch.tensor([.02, .01, .80, -.02])
    travel, time = advance_clock(progress, seconds, moving, advance, .02, .05)
    torch.testing.assert_close(travel, torch.tensor([0., .02, 0., .01]))
    torch.testing.assert_close(time, torch.tensor([0., 2.02, 0., 4.02]))


def test_distance_scales_only_passage_clearance_costs_and_preserves_stall_penalties():
    terms = {
        "handsdf": torch.tensor([-3., -3., -3., 1.]),
        "track_pos": torch.tensor([3., 3., -2., 3.]),
        "stall": torch.tensor([-1., -1., -1., -1.]),
    }
    apply_terms(terms, torch.tensor([True, True, False, True]), torch.tensor([True, False, True, False]),
                torch.tensor([0., .012, 0., .012]), .02, .3)
    torch.testing.assert_close(terms["handsdf"], torch.tensor([0., -6., -3., 1.]))
    torch.testing.assert_close(terms["track_pos"], torch.tensor([0., 3., -2., 3.]))
    torch.testing.assert_close(terms["stall"], torch.full((4,), -1.))


def test_refusal_timeout_excludes_goals_obstacles_falls_and_other_modes():
    b = torch.tensor
    moving = b([1, 1, 1, 1, 0, 1, 1], dtype=torch.bool)
    clock = b([8., 3., 9., 9., 9., 9., 9.])
    timeout = b([0, 1, 0, 0, 1, 0, 0], dtype=torch.bool)
    terminated = b([0, 0, 0, 0, 0, 0, 1], dtype=torch.bool)
    obstacle = b([0, 0, 1, 0, 0, 0, 0], dtype=torch.bool)
    fall = b([0, 0, 0, 1, 0, 0, 0], dtype=torch.bool)
    reached = b([0, 0, 0, 0, 0, 1, 0], dtype=torch.bool)
    assert refusal_failure(moving, clock, timeout, terminated, obstacle, fall, reached, 8.).tolist() == [
        True, True, False, False, False, False, False,
    ]
