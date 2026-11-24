"""Value-guided candidate selection and value arbitration in OnlinePlanner.plan_block (CPU, no checkpoint)."""
import math

import torch

import cat_mjlab.planner.online as online
from cat_mjlab.planner.data import FRAME_DIM, H


class FixedScorer:
    def __init__(self, clear):
        self.clear = clear

    def score(self, chunks, grid, goal, batch_frames=None):
        return dict(min_clear=self.clear.clone())


def planner(clear, select="clear", arbitrate=None, k=4):
    op = online.OnlinePlanner.__new__(online.OnlinePlanner)
    op.device = torch.device("cpu"); op.samples = k
    op.clear_cap, op.side_penalty, op.switch_penalty = .10, .02, .1
    op.clearance_margins = None; op.capture = False; op._parts = []
    op.select, op.safe_m, op.arbitrate = select, .05, arbitrate
    op.interval = (.3, 1.); op.horizon = 2.; op.grid_chunk = 64; op.score_frames = 4000
    op.offsets = torch.tensor(online.heading_offsets(k))
    op.generator = torch.Generator().manual_seed(0)
    op.scorer = FixedScorer(clear)

    def sample(state, goal, grid):
        m, kk = goal.shape[:2]
        x = torch.zeros(m, kk, H, FRAME_DIM)
        x[..., 0] = torch.arange(kk, dtype=torch.float32)[None, :, None] * .01   # candidate j moves j cm forward
        return x
    op.sample = sample
    return op


def run(op, m, rank, still=None):
    saved, online.scene_grid = online.scene_grid, (lambda *a, **kw: torch.zeros(m, 1))
    try:
        return _run(op, m, rank, still)
    finally:
        online.scene_grid = saved


def _run(op, m, rank, still):
    qpos = torch.zeros(m, 36); qpos[:, 3] = 1.; qpos[:, 2] = .78
    qvel = torch.zeros(m, 35)
    still = torch.zeros(m, dtype=torch.bool) if still is None else still
    out = op.plan_block(qpos, qvel, torch.zeros(m, dtype=torch.long), None, torch.ones(m, 2), torch.zeros(m),
                        torch.full((m,), .3), still, rank=rank, rows=torch.arange(m))
    return out


def chosen(out):
    return torch.round(out[0][:, 1, 0] * 100).long()        # world x of the first planned frame = j cm


def test_value_safe_picks_highest_value_among_safe():
    clear = torch.tensor([[.08, .06, .01, .07]])
    value = torch.tensor([[1., 2., 9., 3.]])
    rank = lambda rows, xy, yaw, z, up: (value, torch.zeros(1))
    out = run(planner(clear, "value_safe"), 1, rank)
    assert chosen(out).tolist() == [3]                       # candidate 2 has the most value but 1 cm clearance
    assert out[5]["value_chosen"].tolist() == [3.]


def test_value_ignores_clearance_and_clear_rule_unchanged():
    clear = torch.tensor([[.08, .06, .01, .07]])
    value = torch.tensor([[1., 2., 9., 3.]])
    rank = lambda rows, xy, yaw, z, up: (value, torch.zeros(1))
    assert chosen(run(planner(clear, "value"), 1, rank)).tolist() == [2]
    out = run(planner(clear, "clear"), 1, rank)
    assert chosen(out).tolist() == [0] and "value_chosen" not in out[5]


def test_value_safe_relaxes_to_near_clearest_when_none_is_safe():
    clear = torch.tensor([[.02, .035, .03, .01]])
    value = torch.tensor([[5., 1., 2., 9.]])
    rank = lambda rows, xy, yaw, z, up: (value, torch.zeros(1))
    assert chosen(run(planner(clear, "value_safe"), 1, rank)).tolist() == [2]   # within 1 cm of the clearest (3.5 cm)


def test_arbitration_rejects_low_value_plans_but_not_standing():
    clear = torch.full((3, 4), .08)
    value = torch.tensor([[1., 2., 3., 4.], [0., 0., 0., -2.], [0., 0., 0., -2.]])
    none = torch.tensor([0., 1., 1.])
    rank = lambda rows, xy, yaw, z, up: (value, none)
    still = torch.tensor([False, False, True])
    out = run(planner(clear, "value_safe", arbitrate=-.5), 3, rank, still)
    assert out[5]["margin_accepted"].tolist() == [True, False, True]


def test_rank_receives_world_frame_candidates():
    clear = torch.full((1, 4), .08)
    seen = {}

    def rank(rows, xy, yaw, z, up):
        seen.update(xy=xy, z=z, up=up)
        return torch.zeros(1, 4), torch.zeros(1)
    run(planner(clear, "value"), 1, rank)
    assert seen["xy"].shape == (1, 4, H + 1, 2) and seen["up"].shape == (1, 4, H + 1, 17)
    assert math.isclose(float(seen["xy"][0, 3, 5, 0]), .03, abs_tol=1e-6)
    assert float(seen["xy"][0, 3, 0, 0]) == 0. and math.isclose(float(seen["z"][0, 0, 0]), .78, abs_tol=1e-6)
