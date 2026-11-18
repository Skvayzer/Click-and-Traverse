"""Conditional flow-matching planner over whole-body chunks.

x1 is a normalized chunk [H, 35]; x0 ~ N(0, I); training regresses the straight-path velocity x1 - x0 at
x_t = (1 - t) x0 + t x1 (rectified flow / conditional flow matching). Sampling integrates dx/dt = v(x, t) from
t = 0 to 1 with Euler steps.

Conditioning tokens: 20 scene tokens (a 2D CNN over the robot-centred grid with its 36 height slices as channels,
as in Gallant), one state+goal token and one time token, followed by the 20 frame tokens; a pre-norm transformer
encoder attends over all of them and the frame tokens are read out as velocities. Scene dropout trains a learned
"no scene" token set, so the same network also plans blind (and supports classifier-free guidance).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .data import FRAME_DIM, GOAL_DIM, H, STATE_DIM
from .grid import NZ, SDF_HI, SDF_LO


def grid_input(grid):
    """uint8 grids [B, NX, NY, NZ] -> float [B, NZ, NX, NY] in about [-1, 1]."""
    sdf = grid.float() / 255 * (SDF_HI - SDF_LO) + SDF_LO
    return ((sdf - (SDF_LO + SDF_HI) / 2) / ((SDF_HI - SDF_LO) / 2)).permute(0, 3, 1, 2)


class SceneEncoder(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(NZ, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GELU(),          # 20 x 16
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GELU(),         # 10 x 8
            nn.Conv2d(128, width, 3, stride=2, padding=1), nn.GELU())      # 5 x 4 -> 20 tokens
        self.pos = nn.Parameter(torch.zeros(1, 20, width))

    def forward(self, grid):
        f = self.net(grid_input(grid))
        return f.flatten(2).transpose(1, 2) + self.pos


def time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    angle = (t.float()[:, None] * 1000.) * freqs[None]
    return torch.cat((angle.sin(), angle.cos()), -1)


class FlowPlanner(nn.Module):
    def __init__(self, width=256, layers=6, heads=8):
        super().__init__()
        self.width = width
        self.scene = SceneEncoder(width)
        self.null_scene = nn.Parameter(torch.zeros(1, 20, width))
        self.cond = nn.Sequential(nn.Linear(STATE_DIM + GOAL_DIM, width), nn.GELU(), nn.Linear(width, width))
        self.time = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.frame_in = nn.Linear(FRAME_DIM, width)
        self.frame_pos = nn.Parameter(torch.zeros(1, H, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, FRAME_DIM))
        nn.init.zeros_(self.out[1].weight); nn.init.zeros_(self.out[1].bias)
        for p in (self.frame_pos, self.scene.pos, self.null_scene):
            nn.init.normal_(p, std=0.02)

    def scene_tokens(self, grid, use_scene):
        b = len(use_scene)
        null = self.null_scene.expand(b, -1, -1)
        if grid is None or not bool(use_scene.any()):
            return null
        tokens = self.scene(grid)
        return torch.where(use_scene[:, None, None], tokens, null)

    def forward(self, x, t, state, goal, scene_tokens):
        c = self.cond(torch.cat((state, goal), -1))[:, None]
        e = self.time(time_embedding(t, self.width))[:, None]
        f = self.frame_in(x) + self.frame_pos + e
        h = self.transformer(torch.cat((scene_tokens, c, e, f), 1))
        return self.out(h[:, -x.shape[1]:])


def flow_loss(model, x1, state, goal, grid, scene_dropout=0.15):
    b = len(x1)
    x0 = torch.randn_like(x1)
    t = torch.sigmoid(torch.randn(b, device=x1.device))          # logit-normal: more weight to mid times
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    use_scene = torch.rand(b, device=x1.device) >= scene_dropout
    v = model(xt, t, state, goal, model.scene_tokens(grid, use_scene))
    return ((v - (x1 - x0)) ** 2).mean()


@torch.no_grad()
def sample(model, state, goal, grid, *, use_scene=True, samples=1, steps=16, guidance=1.0, generator=None):
    """Normalized chunks [B, samples, H, 35]. use_scene=False plans blind; guidance > 1 is classifier-free guidance."""
    b = len(state)
    rep = lambda x: None if x is None else x.repeat_interleave(samples, 0)
    st, gl = rep(state), rep(goal)
    mask = torch.full((b * samples,), bool(use_scene), device=state.device)
    tokens = model.scene_tokens(rep(grid) if use_scene else None, mask)
    null = model.scene_tokens(None, torch.zeros_like(mask)) if (use_scene and guidance != 1.0) else None
    x = torch.randn((b * samples, H, FRAME_DIM), device=state.device, generator=generator)
    for i in range(steps):
        t = torch.full((b * samples,), i / steps, device=state.device)
        v = model(x, t, st, gl, tokens)
        if null is not None:
            v0 = model(x, t, st, gl, null)
            v = v0 + guidance * (v - v0)
        x = x + v / steps
    return x.reshape(b, samples, H, FRAME_DIM)


class Normalizer(nn.Module):
    """Per-feature mean/std for state [67], goal [5] and chunk [H, 35]; stored with the model."""

    def __init__(self, state, goal, chunk):
        super().__init__()
        for name, x in (("state", state), ("goal", goal), ("chunk", chunk)):
            x = torch.as_tensor(x, dtype=torch.float32)
            self.register_buffer(name + "_mean", x.mean(0))
            self.register_buffer(name + "_std", x.std(0).clamp_min(1e-3))

    def norm(self, name, x):
        return (x - getattr(self, name + "_mean")) / getattr(self, name + "_std")

    def denorm(self, name, x):
        return x * getattr(self, name + "_std") + getattr(self, name + "_mean")
