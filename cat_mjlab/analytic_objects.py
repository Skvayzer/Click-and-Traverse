"""Exact point queries for moving primitives, without any voxel allocations.

This is perception infrastructure, NOT a no-contact scene admission mechanism.
Kinds: 0 sphere, 1 box, 2 capped cylinder (also the 16 mm rod). Rotation maps
local to world coordinates; cylinder axis is local Z. Sizes are sphere radius,
box half extents, or cylinder radius/half length in entries 0/1.
"""
from __future__ import annotations

import torch


def primitive_distance_normal(points, centers, rotations, sizes, kinds):
    """Return exact signed point distances and outward normals [B,Q,M,(3)].

    Nondifferentiable ties choose a deterministic outward subgradient. Queries
    inside a primitive remain signed; normals at a sphere center choose +X.
    """
    delta = points[:, :, None] - centers[:, None]
    p = torch.einsum('bmji,bqmj->bqmi', rotations, delta)
    norm = torch.linalg.vector_norm(p, dim=-1)
    fallback = torch.zeros_like(p); fallback[..., 0] = 1
    sphere_n = torch.where((norm > 1e-12)[..., None], p / norm.clamp_min(1e-12)[..., None], fallback)
    sphere_d = norm - sizes[:, None, :, 0]

    offset = p.abs() - sizes[:, None]
    outside = offset.clamp_min(0)
    length = torch.linalg.vector_norm(outside, dim=-1)
    axis = offset.argmax(-1)
    inside_n = torch.nn.functional.one_hot(axis, 3).to(p.dtype)
    sign = torch.where(p >= 0, 1., -1.)
    box_n = torch.where((length > 1e-12)[..., None], outside / length.clamp_min(1e-12)[..., None], inside_n) * sign
    box_d = length + offset.amax(-1).clamp_max(0)

    radial = torch.linalg.vector_norm(p[..., :2], dim=-1)
    radial_n = torch.where((radial > 1e-12)[..., None], p[..., :2] / radial.clamp_min(1e-12)[..., None], fallback[..., :2])
    edge = torch.stack((radial - sizes[:, None, :, 0], p[..., 2].abs() - sizes[:, None, :, 1]), -1)
    outside2 = edge.clamp_min(0)
    length2 = torch.linalg.vector_norm(outside2, dim=-1)
    edge_n = torch.where((length2 > 1e-12)[..., None], outside2 / length2.clamp_min(1e-12)[..., None],
                         torch.nn.functional.one_hot(edge.argmax(-1), 2).to(p.dtype))
    cylinder_n = torch.cat((radial_n * edge_n[..., :1], sign[..., 2:] * edge_n[..., 1:]), -1)
    cylinder_d = length2 + edge.amax(-1).clamp_max(0)
    kind = kinds[:, None]
    distance = torch.where(kind == 0, sphere_d, torch.where(kind == 1, box_d, cylinder_d))
    normal = torch.where((kind == 0)[..., None], sphere_n, torch.where((kind == 1)[..., None], box_n, cylinder_n))
    return distance, torch.einsum('bmij,bqmj->bqmi', rotations, normal)


def nearest(points, centers, rotations, sizes, kinds, valid):
    distance, normal = primitive_distance_normal(points, centers, rotations, sizes, kinds)
    distance = torch.where(valid[:, None], distance, torch.inf)
    value, index = distance.min(-1)
    selected = normal.gather(2, index[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
    return value[..., None], selected


def merge_fields(gf, bf, sdf, dynamic_distance, dynamic_normal, radii):
    """SDF inputs are surface clearances; dynamic input is a point distance.

    Static wins ties. Only closer dynamic objects alter normals or guidance.
    Guidance is projected tangentially within 0.5 m; no global replan is claimed.
    """
    clearance = dynamic_distance - radii
    closer = clearance < sdf
    inward = (gf * dynamic_normal).sum(-1, keepdim=True).clamp_max(0)
    guidance = gf - inward * dynamic_normal
    return (torch.where(closer & (clearance < .5), guidance, gf),
            torch.where(closer, dynamic_normal, bf), torch.minimum(sdf, clearance))


class AnalyticObjects:
    """Immutable per-environment approach/hold/retreat query specification.

    Time is supplied by the task episode clock; there is no hidden mutable
    phase. Construction does not certify robot/floor clearance. This class is
    intentionally not wired into training CLI or bank admission yet.
    """
    def __init__(self, *, start, end, rotations, sizes, kinds, valid, speed, hold):
        self.state = dict(start=start, end=end, rotations=rotations, sizes=sizes,
                          kinds=kinds, valid=valid, speed=speed, hold=hold)
        b, m, xyz = start.shape
        expected = dict(end=(b,m,3), rotations=(b,m,3,3), sizes=(b,m,3),
                        kinds=(b,m), valid=(b,m), speed=(b,m), hold=(b,m))
        if xyz != 3 or m < 1:
            raise ValueError('Expected at least one object slot')
        for key, shape in expected.items():
            if self.state[key].shape != shape:
                raise ValueError(f'Invalid object shape: {key}')
        if any(v.device != start.device for v in self.state.values()):
            raise ValueError('Object tensors must share device')
        if not all(torch.isfinite(v).all() for v in self.state.values()):
            raise ValueError('Nonfinite object state')
        if kinds.dtype != torch.long or valid.dtype != torch.bool or ((kinds < 0) | (kinds > 2)).any():
            raise ValueError('Invalid kind or validity dtype')
        if (sizes <= 0).any() or (speed <= 0).any() or (hold < 0).any():
            raise ValueError('Sizes/speed must be positive and hold nonnegative')
        eye = torch.eye(3, device=start.device, dtype=start.dtype)
        if not torch.allclose(rotations.transpose(-1,-2) @ rotations, eye.expand_as(rotations), atol=1e-5):
            raise ValueError('Object rotation must be orthonormal')

    def centers(self, ids, time):
        s = self.state
        start, end = s['start'][ids], s['end'][ids]
        length = torch.linalg.vector_norm(end-start, dim=-1)
        duration = length / s['speed'][ids]
        elapsed = time[:, None]
        progress = torch.minimum(elapsed, duration)
        progress = (progress - (elapsed-duration-s['hold'][ids]).clamp_min(0)).clamp_min(0)
        fraction = progress / duration.clamp_min(1e-12)
        return start + fraction[..., None] * (end-start)

    def query(self, points, ids, time):
        s = self.state
        return nearest(points, self.centers(ids, time), s['rotations'][ids],
                       s['sizes'][ids], s['kinds'][ids], s['valid'][ids])

    def merge(self, points, ids, time, gf, bf, sdf, radii):
        return merge_fields(gf, bf, sdf, *self.query(points, ids, time), radii)
