"""2-D blocking footprint of every scene: what stops a walking body, as opposed to what it ducks under or
steps over.

A cell column blocks when the scene is occupied anywhere in the body band (default 0.25-0.95 m above the
floor). Hurdles below the band (CAT: <= 0.20 m) and beams above it (CAT: bottom >= 1.00 m) do not block,
walls, table tops (0.70-0.75 m), boxes and chairs do. Per scene a 2-D signed distance to the blocking
cells (metres, negative inside) and its unit gradient (pointing away from the nearest blocking cell) are
precomputed from the scene's 3-D SDF already in the bank, on the same grid (cell i at origin + i*dx).

Why: the obstacle-safe velocity target and the heading gate used the 3-D field at body points. Its
horizontal part, rescaled to unit length, read a beam overhead or a hurdle underfoot as an obstacle
straight ahead, so a keyboard-driven robot was rewarded for stopping in front of every beam and hurdle
(measured on the demo course: field-driven it crouched under a 1.02 m beam, keyboard-driven it stalled
0.5 m before it for 40 s).
"""
from __future__ import annotations

import numpy as np
import torch

BAND = (.25, .95)


def blocking_sdf2d(sdf3, origin_z, dx, band=BAND, also=None):
    """[X,Y,Z] scene SDF -> ([X,Y] 2-D signed distance in metres, [X,Y,2] unit gradient away from blocking).

    ``also``: a second band the column must ALSO be occupied in (e.g. (1.0, 1.4) for full-height walls only:
    a table blocks the body band but not shoulder height, a beam the reverse)."""
    from scipy import ndimage
    z = origin_z + dx * np.arange(sdf3.shape[2])

    def occupied(b):
        k = (z >= b[0]) & (z <= b[1])
        return (sdf3[:, :, k] <= 0).any(-1) if k.any() else np.zeros(sdf3.shape[:2], bool)
    blocked = occupied(band)
    if also is not None:
        blocked = blocked & occupied(also)
    if blocked.any():
        outside = ndimage.distance_transform_edt(~blocked) * dx
        inside = ndimage.distance_transform_edt(blocked) * dx
        dist = np.where(blocked, -inside, outside).astype(np.float32)
    else:
        dist = np.full(sdf3.shape[:2], 100., np.float32)
    gx, gy = np.gradient(dist, dx, dx) if min(dist.shape) > 1 else (np.zeros_like(dist), np.zeros_like(dist))
    grad = np.stack((gx, gy), -1)
    grad /= np.linalg.norm(grad, axis=-1, keepdims=True) + 1e-9
    return dist, grad.astype(np.float32)


class BlockingMap:
    """Per-scene 2-D blocking distance + direction, sampled bilinearly at world xy."""

    def __init__(self, bank, band=BAND, also=None):
        device = bank.offsets.device
        shapes, offsets, origins, dxs = (bank.shapes.cpu().numpy(), bank.offsets.cpu().numpy(),
                                         bank.origins.cpu().numpy(), bank.dxs.cpu().numpy())
        field = bank.fields["sdf"]
        dists, grads, offsets2 = [], [], []
        total = 0
        for i in range(len(shapes)):
            X, Y, Z = (int(v) for v in shapes[i])
            sdf3 = field[int(offsets[i]):int(offsets[i]) + X * Y * Z].reshape(X, Y, Z).float().cpu().numpy()
            d, g = blocking_sdf2d(sdf3, float(origins[i, 2]), float(dxs[i]), band, also)
            dists.append(d.reshape(-1)); grads.append(g.reshape(-1, 2)); offsets2.append(total); total += X * Y
        self.dist = torch.as_tensor(np.concatenate(dists), device=device)
        self.grad = torch.as_tensor(np.concatenate(grads), device=device)
        self.offsets = torch.as_tensor(np.asarray(offsets2), device=device, dtype=torch.long)
        self.shapes = bank.shapes[:, :2].clone()
        self.origins = bank.origins[:, :2].clone()
        self.dxs = bank.dxs.clone()
        self.band = band

    def sample(self, xy, scene_ids):
        """xy [N,P,2] world positions -> (distance [N,P], away [N,P,2] unit, from the nearest blocking cell)."""
        origin, dx = self.origins[scene_ids][:, None], self.dxs[scene_ids][:, None, None]
        shape, offset = self.shapes[scene_ids][:, None], self.offsets[scene_ids][:, None, None]
        f = ((xy - origin) / dx).clamp_min(0)
        f = torch.minimum(f, (shape - 2).to(f.dtype))
        base = f.floor().long(); w = f - base
        corners = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], device=xy.device)
        ix = base[:, :, None] + corners
        address = offset + ix[..., 0] * shape[:, :, None, 1] + ix[..., 1]
        wx = torch.stack((1 - w[..., 0], w[..., 0]), -1); wy = torch.stack((1 - w[..., 1], w[..., 1]), -1)
        weights = (wx[..., :, None] * wy[..., None, :]).flatten(-2)                   # order (0,0),(0,1),(1,0),(1,1)
        weights = weights[..., [0, 2, 1, 3]]                                           # match corner order above
        dist = (weights * self.dist[address]).sum(-1)
        away = (weights[..., None] * self.grad[address]).sum(-2)
        away = away / (torch.linalg.vector_norm(away, dim=-1, keepdim=True) + 1e-9)
        return dist, away
