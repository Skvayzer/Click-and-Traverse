"""Batched, shared-bank sampling preserving released CAT interpolation."""
from __future__ import annotations

import torch


def sample_ragged_field(field, pos, *, origin, dx, shape, offset):
    """Sample [B,Q,3] positions, or [Q,3] for a single scene.

    Preserve the original x-fast corner order paired with z-fast weight order.
    Correcting this historical mismatch would silently change checkpoint inputs.
    Scene arrays have shapes [B,3], [B], [B,3], [B], respectively.
    """
    single = pos.ndim == 2
    if single:
        pos = pos.unsqueeze(0)
        origin = origin.reshape(1, 3)
        dx = dx.reshape(1)
        shape = shape.reshape(1, 3)
        offset = offset.reshape(1)
    xyz = (pos - origin[:, None]) / dx[:, None, None]
    xyz = torch.minimum(xyz.clamp_min(0), (shape[:, None] - 2).to(xyz.dtype))
    base = torch.floor(xyz).long()
    f = xyz - base
    corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                           device=pos.device)
    ix = base[:, :, None] + corners
    addresses = offset[:, None, None] + (
        ix[..., 0] * shape[:, None, None, 1] + ix[..., 1]) * shape[:, None, None, 2] + ix[..., 2]
    values = field[addresses.long()]
    wx, wy, wz = (torch.stack((1 - f[..., i], f[..., i]), dim=-1) for i in range(3))
    weights = (wx[..., :, None, None] * wy[..., None, :, None] * wz[..., None, None, :]).flatten(-3)
    result = (weights[..., None] * values).sum(dim=-2)
    return result[0] if single else result
