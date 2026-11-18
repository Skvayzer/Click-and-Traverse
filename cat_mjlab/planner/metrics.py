"""Kinematic evaluation of whole-body chunks against the stored local scene grid.

For every frame, forward kinematics places the 497-sphere cover of the collision body; its clearance is the
scene's signed distance at each sphere centre minus the sphere radius. A chunk collides when any clearance over its
20 frames is below the tolerance. The cover is conservative (spheres enclose the proxy) and the field comes from a
conservative 4 cm occupancy, so even the policy's real, contact-free motion can read a few millimetres of overlap;
the executed ("ground-truth") chunks are scored the same way and are the reference.
"""
from __future__ import annotations

import torch

from .data import chunk_qpos
from .grid import sample_local_grid


class ChunkScorer:
    def __init__(self, model, device):
        from .fk import BodySpheres
        self.spheres = BodySpheres(model, device)
        self.device = device
        names = [model.body(i).name for i in range(model.nbody)]
        self.feet = torch.tensor([names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")], device=device)

    @torch.no_grad()
    def score(self, chunks, grid, goal, *, batch_frames=60000):
        """chunks [B, K, H, 35] (chunk frame, denormalized), grid uint8 [B, ...], goal [B, 5] -> dict of [B, K]."""
        b, k, h, _ = chunks.shape
        qpos = chunk_qpos(chunks).reshape(b, k * h, -1)
        clear = torch.empty(b, k * h, device=self.device)
        hand_clear = torch.empty_like(clear)
        low = torch.empty_like(clear)
        feet = torch.empty(b, k * h, 2, 3, device=self.device)
        per = max(1, batch_frames // (k * h))
        for s in range(0, b, per):
            q = qpos[s:s + per]
            centres = self.spheres(q)                                           # [b', KH, S, 3]
            bb, kh, n, _ = centres.shape
            sdf = sample_local_grid(grid[s:s + per], centres.reshape(bb, kh * n, 3)).reshape(bb, kh, n)
            c = sdf - self.spheres.radii
            clear[s:s + per] = c.amin(-1)
            hand_clear[s:s + per] = c[..., self.spheres.hand_mask].amin(-1)
            low[s:s + per] = (centres[..., 2] - self.spheres.radii).amin(-1)
            xpos, _ = self.spheres.fk(q)
            feet[s:s + per] = xpos[..., self.feet, :]
        clear, hand_clear, low = (x.reshape(b, k, h) for x in (clear, hand_clear, low))
        feet = feet.reshape(b, k, h, 2, 3)
        step = torch.linalg.vector_norm(feet[:, :, 1:, :, :2] - feet[:, :, :-1, :, :2], dim=-1)       # [B, K, H-1, 2]
        ground = feet[..., 2].amin(2, keepdim=True)                                                # [B, K, 1, 2]
        planted = feet[:, :, 1:, :, 2] < ground + 0.02
        skate = (step * planted).sum((-1, -2)) / planted.sum((-1, -2)).clamp_min(1)
        end = chunks[:, :, -1, :2]
        goal_err = torch.linalg.vector_norm(end - goal[:, None, :2], dim=-1)
        return dict(min_clear=clear.amin(-1), min_hand_clear=hand_clear.amin(-1), min_low=low.amin(-1),
                    skate=skate, goal_err=goal_err)


def diversity(chunks):
    """Mean pairwise root-path distance (m, averaged over frames) between the K samples of each context [B]."""
    path = chunks[..., :2]                                                       # [B, K, H, 2]
    d = torch.linalg.vector_norm(path[:, :, None] - path[:, None], dim=-1).mean(-1)    # [B, K, K]
    k = path.shape[1]
    return d.sum((-1, -2)) / max(1, k * (k - 1))
