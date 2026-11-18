"""Robot-centred scene grid for the planner.

The grid is fixed to the robot's floor frame at the start of a chunk: origin at the pelvis' floor projection, x
along the pelvis heading, y to its left, z up from the floor. Cells are 5 cm: x from -0.5 to 1.5 m (40), y from
-0.8 to 0.8 m (32), z from 0 to 1.8 m (36) -- 46,080 cells, about 2 s of walking ahead at 0.8 m/s.

Each cell stores the scene's signed distance to the nearest obstacle at its centre (floor excluded, as in the
policy's fields), clipped to [-0.10, 0.40] m and quantized to uint8 (about 2 mm steps).

`sample_bank_sdf` reads a SceneBank's flat float32 SDF with standard trilinear interpolation. It deliberately does
NOT use the bank's own sampler, which pairs x-fast corners with z-fast weights (kept there to match the trained
policies' inputs) and so blends the wrong corners between nodes.
"""
from __future__ import annotations

import math

import torch

CELL = 0.05
X0, NX = -0.50, 40
Y0, NY = -0.80, 32
Z0, NZ = 0.00, 36
SDF_LO, SDF_HI = -0.10, 0.40
SHAPE = (NX, NY, NZ)


def cell_centres(device="cpu", dtype=torch.float32):
    """Local cell centres [NX, NY, NZ, 3] in the chunk frame."""
    xs = X0 + CELL * (torch.arange(NX, device=device, dtype=dtype) + .5)
    ys = Y0 + CELL * (torch.arange(NY, device=device, dtype=dtype) + .5)
    zs = Z0 + CELL * (torch.arange(NZ, device=device, dtype=dtype) + .5)
    return torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), -1)


def to_world(local, origin_xy, yaw):
    """Chunk-frame points [B, ..., 3] to world: rotate xy by yaw [B] and shift by origin_xy [B, 2]; z unchanged."""
    c, s = yaw.cos(), yaw.sin()
    shape = (-1,) + (1,) * (local.dim() - 2)
    c, s = c.reshape(shape), s.reshape(shape)
    x, y = local[..., 0], local[..., 1]
    wx = c * x - s * y + origin_xy[:, 0].reshape(shape)
    wy = s * x + c * y + origin_xy[:, 1].reshape(shape)
    return torch.stack((wx, wy, local[..., 2].expand_as(wx)), -1)


def to_local(world, origin_xy, yaw):
    """World points [B, ..., 3] to the chunk frame (inverse of to_world)."""
    c, s = yaw.cos(), yaw.sin()
    shape = (-1,) + (1,) * (world.dim() - 2)
    c, s = c.reshape(shape), s.reshape(shape)
    dx = world[..., 0] - origin_xy[:, 0].reshape(shape)
    dy = world[..., 1] - origin_xy[:, 1].reshape(shape)
    return torch.stack((c * dx + s * dy, -s * dx + c * dy, world[..., 2]), -1)


def sample_bank_sdf(bank, points, scene_ids):
    """Standard trilinear sample of bank.fields['sdf'] at world points [B, Q, 3] for scenes [B] -> [B, Q]."""
    field = bank.fields["sdf"].reshape(-1)
    origin, dx = bank.origins[scene_ids], bank.dxs[scene_ids]
    shape, offset = bank.shapes[scene_ids], bank.offsets[scene_ids]
    u = (points - origin[:, None]) / dx[:, None, None]
    top = (shape[:, None] - 1).to(u.dtype)
    u = torch.minimum(u.clamp_min(0), top)                       # outside the scene grid: the border value
    i0 = torch.minimum(torch.floor(u), top - 1).long()           # the last cell interpolates up to its far node
    f = u - i0
    out = torch.zeros(points.shape[:2], dtype=points.dtype, device=points.device)
    ny, nz = shape[:, 1, None], shape[:, 2, None]
    for cx in (0, 1):
        wx = f[..., 0] if cx else 1 - f[..., 0]
        for cy in (0, 1):
            wy = f[..., 1] if cy else 1 - f[..., 1]
            for cz in (0, 1):
                wz = f[..., 2] if cz else 1 - f[..., 2]
                address = offset[:, None] + ((i0[..., 0] + cx) * ny + (i0[..., 1] + cy)) * nz + (i0[..., 2] + cz)
                out = out + field[address].to(points.dtype) * wx * wy * wz
    return out


def quantize(sdf):
    return ((sdf.clamp(SDF_LO, SDF_HI) - SDF_LO) / (SDF_HI - SDF_LO) * 255).round().to(torch.uint8)


def dequantize(q):
    return q.to(torch.float32) / 255 * (SDF_HI - SDF_LO) + SDF_LO


def scene_grid(bank, scene_ids, origin_xy, yaw, chunk=256):
    """Quantized local SDF grids [B, NX, NY, NZ] (uint8) for chunk frames (origin_xy [B, 2], yaw [B])."""
    local = cell_centres(origin_xy.device).reshape(1, -1, 3)
    out = []
    for start in range(0, len(scene_ids), chunk):
        stop = start + chunk
        world = to_world(local.expand(len(scene_ids[start:stop]), -1, -1), origin_xy[start:stop], yaw[start:stop])
        out.append(quantize(sample_bank_sdf(bank, world, scene_ids[start:stop])).reshape(-1, *SHAPE))
    return torch.cat(out)


def sample_local_grid(grid, points):
    """Trilinear sample of dequantized local grids [B, NX, NY, NZ] at chunk-frame points [B, Q, 3] -> [B, Q] (m).

    Points outside the grid read SDF_HI (free): the grid covers everything a 2 s chunk can reach in practice, and
    below the floor is not an obstacle in these fields."""
    sdf = dequantize(grid) if grid.dtype == torch.uint8 else grid
    u = torch.stack(((points[..., 0] - X0) / CELL - .5, (points[..., 1] - Y0) / CELL - .5, (points[..., 2] - Z0) / CELL - .5), -1)
    size = torch.tensor(SHAPE, device=points.device, dtype=u.dtype)
    inside = ((u >= 0) & (u <= size - 1)).all(-1)
    u = torch.minimum(u.clamp_min(0), size - 1.0001)
    i0 = torch.floor(u).long(); f = u - i0
    b = torch.arange(len(grid), device=points.device)[:, None].expand_as(i0[..., 0])
    out = torch.zeros(points.shape[:2], dtype=sdf.dtype, device=points.device)
    for cx in (0, 1):
        wx = f[..., 0] if cx else 1 - f[..., 0]
        for cy in (0, 1):
            wy = f[..., 1] if cy else 1 - f[..., 1]
            for cz in (0, 1):
                wz = f[..., 2] if cz else 1 - f[..., 2]
                out = out + sdf[b, i0[..., 0] + cx, i0[..., 1] + cy, i0[..., 2] + cz] * wx * wy * wz
    return torch.where(inside, out, torch.full_like(out, SDF_HI))


def heading_of(quat):
    """Yaw [...] of MuJoCo quaternions [..., 4] (w, x, y, z)."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(angle):
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi
