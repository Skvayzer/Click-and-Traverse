"""Planner v1 in the training loop: the reference tracker's 'plan' source (cat_mjlab/tracking.py).

For the worlds that are due, OnlinePlanner builds the planner's inputs from the simulation -- the measured state, the
robot-centred scene grid sampled from the bank's SDF, and a goal `horizon` s ahead on the guidance path at the route
speed --, samples K chunks (bf16, few Euler steps, classifier-free guidance), scores them with the 497-sphere body
cover against the same grid, and returns the chosen chunk in the world frame with the start pose prepended (the
tracker's PLAN_FRAMES = 21 frames, every 0.1 s).

Goal headings: sample k aims at the path heading plus HEADING_OFFSETS (0 for all but the last two samples, which turn
+90 and -90 deg), so the planner can propose a sideways pass where the route is tight. The choice maximises clearance
counted up to clear_cap, minus side_penalty for a sideways pass and switch_penalty per metre of mean disagreement with
the chunk being replaced over the next second (pelvis position, plus 0.2 m per radian of heading, so that a plan does
not flip between facing forward and sideways from one replan to the next). Stand-still worlds (reactive standing episodes, or a route speed under
5 cm/s) get the goal "stay here, same heading" in every sample.

6-8 Euler steps in bf16 plan as well as the 16 steps in fp32 the planner was evaluated with (600 held-out situations:
100% vs 99.8% clear, goal error 1.5 vs 1.6 cm, same smoothness). Memory: the training task already fills a 32 GB card
(V7r peaked at 30.9 GB), so worlds are planned in blocks of `block` and the grid and the scoring run in small pieces;
200 worlds then take about 0.2 s on an RTX 6000 Ada (grid 36 ms, sampling 148 ms with 8 steps, scoring 46 ms).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .data import FRAME_DIM, GOAL_DIM, H, STATE_DIM, euler_zyx
from .grid import heading_of, scene_grid, wrap
from .model import FlowPlanner, Normalizer


def planner_state(qpos, qvel):
    """The planner's state features [n, 67] (as planner/data.py `features` builds them) for qpos [n, 36], qvel [n, 35]."""
    roll, pitch, yaw = euler_zyx(qpos[:, 3:7])
    c, s = yaw.cos(), yaw.sin()
    lin = torch.stack((c * qvel[:, 0] + s * qvel[:, 1], -s * qvel[:, 0] + c * qvel[:, 1], qvel[:, 2]), -1)
    return torch.cat((qpos[:, 7:], qvel[:, 6:], qpos[:, 2:3], roll[:, None], pitch[:, None], lin, qvel[:, 3:6]), -1)


def heading_offsets(samples):
    k = int(samples)
    return [0.] * k if k < 3 else [0.] * (k - 2) + [math.pi / 2, -math.pi / 2]


class OnlinePlanner:
    def __init__(self, path, mj_model, device, *, samples=4, steps=6, guidance=2.0, horizon=2.0, interval=(0.3, 1.0),
                 clear_cap=0.10, side_penalty=0.02, switch_penalty=0.1, bf16=True, seed=0, block=256, grid_chunk=64,
                 score_frames=4000, clearance_margins=None):
        self.device = torch.device(device)
        ck = torch.load(path, map_location='cpu', weights_only=True)
        self.model = FlowPlanner(**ck['config']).to(self.device).eval()
        self.model.load_state_dict(ck['model'])
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.norm = Normalizer(np.zeros((2, STATE_DIM)), np.zeros((2, GOAL_DIM)), np.zeros((2, H, FRAME_DIM))).to(self.device)
        self.norm.load_state_dict(ck['norm'])
        from .metrics import ChunkScorer
        self.scorer = ChunkScorer(mj_model, self.device)
        self.samples, self.steps, self.guidance = int(samples), int(steps), float(guidance)
        self.horizon, self.interval = float(horizon), tuple(map(float, interval))
        self.clear_cap, self.side_penalty, self.switch_penalty = float(clear_cap), float(side_penalty), float(switch_penalty)
        self.bf16 = bool(bf16) and self.device.type == 'cuda'
        self.block, self.grid_chunk, self.score_frames = int(block), int(grid_chunk), int(score_frames)
        self.clearance_margins = clearance_margins
        self.capture, self.last, self._parts = False, None, []     # capture=True keeps the chunks of the last call (recordings)
        self.offsets = torch.tensor(heading_offsets(self.samples), device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.identity = dict(path=str(path), step=int(ck.get('step', -1)), samples=self.samples, steps=self.steps,
                             guidance=self.guidance, horizon=self.horizon, interval=list(self.interval),
                             clear_cap=self.clear_cap, side_penalty=self.side_penalty, switch_penalty=self.switch_penalty)

    @torch.no_grad()
    def sample(self, state, goal, grid):
        """Denormalized chunks [m, K, H, 35] for state [m, 67], per-sample goals [m, K, 5], grids [m, ...]."""
        m, k = goal.shape[:2]
        st = self.norm.norm('state', state).repeat_interleave(k, 0)
        gl = self.norm.norm('goal', goal.reshape(m * k, GOAL_DIM))
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
            tokens = self.model.scene_tokens(grid, torch.ones(m, dtype=torch.bool, device=self.device)).repeat_interleave(k, 0)
            null = (self.model.scene_tokens(None, torch.zeros(m * k, dtype=torch.bool, device=self.device))
                    if self.guidance != 1.0 else None)
            x = torch.randn((m * k, H, FRAME_DIM), device=self.device, generator=self.generator)
            for i in range(self.steps):
                t = torch.full((m * k,), i / self.steps, device=self.device)
                v = self.model(x, t, st, gl, tokens)
                if null is not None:
                    v0 = self.model(x, t, st, gl, null)
                    v = v0 + self.guidance * (v - v0)
                x = x + v.float() / self.steps
        return self.norm.denorm('chunk', x.float()).reshape(m, k, H, FRAME_DIM)

    @torch.no_grad()
    def plan(self, qpos, qvel, scene_ids, bank, goal_xy, goal_yaw, route_speed, still, prev_xy=None, prev_yaw=None, prev_ok=None):
        """plan_block over blocks of at most `block` worlds (bounded memory); same arguments and results."""
        m = len(qpos)
        self._parts = []
        if m <= self.block:
            out = self.plan_block(qpos, qvel, scene_ids, bank, goal_xy, goal_yaw, route_speed, still, prev_xy, prev_yaw, prev_ok)
            self._keep()
            return out
        parts = []
        for a in range(0, m, self.block):
            sl = slice(a, a + self.block)
            cut = lambda x: None if x is None else x[sl]
            parts.append(self.plan_block(qpos[sl], qvel[sl], scene_ids[sl], bank, goal_xy[sl], goal_yaw[sl], route_speed[sl],
                                         still[sl], cut(prev_xy), cut(prev_yaw), cut(prev_ok)))
        xy, yaw, z, upper, interval = (torch.cat([p[i] for p in parts]) for i in range(5))
        stats = {k: torch.cat([p[5][k] for p in parts]) for k in parts[0][5]}
        self._keep()
        return xy, yaw, z, upper, interval, stats

    def _keep(self):
        """With capture on, last = the latest plan() call's chunks: state [m, 67], chosen chunk [m, H, 35] in the chunk
        frame, pose [m, 3] (x, y, heading of the chunk frame), as planner_render.world_qpos takes them, and every
        candidate: chunks [m, K, H, 35], clear [m, K] (body-sphere clearance, m), score [m, K], offset [m, K] (rad)."""
        if self.capture and self._parts:
            self.last = {k: torch.cat([p[k] for p in self._parts]).detach().cpu() for k in self._parts[0]}
        self._parts = []

    @torch.no_grad()
    def plan_block(self, qpos, qvel, scene_ids, bank, goal_xy, goal_yaw, route_speed, still, prev_xy=None, prev_yaw=None,
                   prev_ok=None):
        """Chosen chunks for m worlds. goal_xy [m, 2] / goal_yaw [m]: the route point `horizon` s ahead (world);
        still [m] bool; prev_xy [m, 10, 2] / prev_yaw [m, 10]: the replaced chunk 0.1-1.0 s ahead (world), used where
        prev_ok [m].
        Returns world-frame xy [m, 21, 2], yaw [m, 21], z [m, 21], upper [m, 21, 17], interval [m], stats (with the
        full body state per frame for continuing the plan: roll, pitch [m, 21], legs [m, 21, 12])."""
        m, k = len(qpos), self.samples
        origin, yaw0 = qpos[:, :2], heading_of(qpos[:, 3:7])
        c, s = yaw0.cos(), yaw0.sin()
        local = lambda d: torch.stack((c.reshape(-1, *([1] * (d.dim() - 2))) * d[..., 0] + s.reshape(-1, *([1] * (d.dim() - 2))) * d[..., 1],
                                       -s.reshape(-1, *([1] * (d.dim() - 2))) * d[..., 0] + c.reshape(-1, *([1] * (d.dim() - 2))) * d[..., 1]), -1)
        state = planner_state(qpos, qvel)
        grid = scene_grid(bank, scene_ids, origin, yaw0, chunk=self.grid_chunk)
        still = still | (route_speed < .05)
        end = torch.where(still[:, None], torch.zeros_like(goal_xy), local(goal_xy - origin))
        heading = torch.where(still, torch.zeros_like(yaw0), wrap(goal_yaw - yaw0))
        offset = self.offsets[None].expand(m, k) * (~still)[:, None]
        h = heading[:, None] + offset
        speed = torch.linalg.vector_norm(end, dim=-1) / self.horizon
        goal = torch.stack((end[:, None, 0].expand(m, k), end[:, None, 1].expand(m, k), h.sin(), h.cos(),
                            speed[:, None].expand(m, k)), -1)
        chunks = self.sample(state, goal, grid)
        sc = self.scorer.score(chunks, grid, goal[:, 0], batch_frames=self.score_frames)
        score = sc['min_clear'].clamp(max=self.clear_cap) - self.side_penalty * offset.sin().abs()
        if prev_xy is not None and prev_ok is not None and bool(prev_ok.any()):
            old = local(prev_xy - origin[:, None])                                            # [m, 10, 2]
            gap = torch.linalg.vector_norm(chunks[:, :, :10, :2] - old[:, None], dim=-1).mean(-1)   # [m, K]
            if prev_yaw is not None:
                turn = wrap(yaw0[:, None, None] + chunks[:, :, :10, 3] - prev_yaw[:, None]).abs().mean(-1)
                gap = gap + .2 * turn
            score = score - self.switch_penalty * gap * prev_ok[:, None].float()
        accepted = None
        if self.clearance_margins is not None:
            high, low = self.clearance_margins
            strong = sc['min_clear'] >= high
            eligible = torch.where(strong.any(-1, keepdim=True), strong, sc['min_clear'] >= low)
            accepted = eligible.any(-1)
            score = torch.where(accepted[:, None] & ~eligible, -torch.inf, score)
        best = score.argmax(-1)
        rows = torch.arange(m, device=self.device)
        pick = chunks[rows, best]                                                              # [m, H, 35]
        if self.capture:
            # recordings: the chosen chunk plus every candidate with its clearance, score and goal-heading offset
            self._parts.append(dict(state=state, chunk=pick, pose=torch.cat((origin, yaw0[:, None]), -1), chunks=chunks,
                                    clear=sc['min_clear'], score=score, offset=offset))
        xy = origin[:, None] + torch.stack((c[:, None] * pick[..., 0] - s[:, None] * pick[..., 1],
                                            s[:, None] * pick[..., 0] + c[:, None] * pick[..., 1]), -1)
        xy = torch.cat((origin[:, None], xy), 1)
        yaw = torch.cat((yaw0[:, None], yaw0[:, None] + pick[..., 3]), 1)
        z = torch.cat((qpos[:, None, 2], pick[..., 2]), 1)
        upper = torch.cat((qpos[:, None, 19:36], pick[..., 18:35]), 1)
        roll0, pitch0, _ = euler_zyx(qpos[:, 3:7])
        roll = torch.cat((roll0[:, None], pick[..., 4]), 1); pitch = torch.cat((pitch0[:, None], pick[..., 5]), 1)
        legs = torch.cat((qpos[:, None, 7:19], pick[..., 6:18]), 1)
        lo, hi = self.interval
        interval = lo + (hi - lo) * torch.rand(m, device=self.device, generator=self.generator)
        stats = dict(min_clear=sc['min_clear'][rows, best], sideways=(offset[rows, best] != 0).float(),
                     any_clear=(sc['min_clear'].amax(-1) >= -.01).float(), roll=roll, pitch=pitch, legs=legs)
        if accepted is not None:
            stats['margin_accepted'] = accepted
        return xy, yaw, z, upper, interval, stats
