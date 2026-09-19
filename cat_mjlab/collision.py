"""Exact Torch ports of the approved primitive-volume terminal checks.

No obstacle impulses, added policy inputs, candidate truncation or mesh probes.
All scene arrays are shared, with temporary narrow-phase tiles of eight.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

REGIONS = ("feet", "legs", "trunk", "head", "arms", "hands")
ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "docs/assets/collision-proxy-proposal-20260916/proposal.json"


def _local(point, center, rotation):
    return torch.einsum("...ji,...j->...i", rotation, point - center)


def sphere_box_separation(center, radius, box_center, box_rotation, box_half_size):
    offset = _local(center, box_center, box_rotation).abs() - box_half_size
    return torch.linalg.vector_norm(offset.clamp_min(0), dim=-1) + offset.amax(-1).clamp_max(0) - radius


def segment_box_squared_distance_local(start, end, half_size):
    start, end, half_size = torch.broadcast_tensors(start, end, half_size)
    delta = end - start
    moving = delta != 0
    divisor = torch.where(moving, delta, 1.)
    crossings = torch.cat(((-half_size - start) / divisor, (half_size - start) / divisor), -1)
    crossings = torch.where(torch.cat((moving, moving), -1), crossings, 0.).clamp(0, 1)
    zero = torch.zeros_like(start[..., :1])
    boundaries = torch.cat((zero, zero + 1, crossings), -1).sort(dim=-1).values
    lower, upper = boundaries[..., :-1], boundaries[..., 1:]
    middle = .5 * (lower + upper)
    midpoint = start[..., None, :] + middle[..., :, None] * delta[..., None, :]
    active = midpoint.abs() > half_size[..., None, :]
    offset = start[..., None, :] - torch.where(midpoint >= 0, 1., -1.) * half_size[..., None, :]
    quadratic = torch.where(active, delta[..., None, :] ** 2, 0.).sum(-1)
    linear = torch.where(active, delta[..., None, :] * offset, 0.).sum(-1)
    optimum = torch.where(quadratic > 0, -linear / torch.where(quadratic > 0, quadratic, 1.), middle)
    optimum = torch.maximum(lower, torch.minimum(upper, optimum))
    position = start[..., None, :] + optimum[..., :, None] * delta[..., None, :]
    return ((position.abs() - half_size[..., None, :]).clamp_min(0) ** 2).sum(-1).amin(-1)


def capsule_box_separation(start, end, radius, box_center, box_rotation, box_half_size):
    squared = segment_box_squared_distance_local(_local(start, box_center, box_rotation),
                                                _local(end, box_center, box_rotation), box_half_size)
    return squared.clamp_min(0).sqrt() - radius


def box_box_separation(center, rotation, half_size, box_center, box_rotation,
                       box_half_size, *, parallel_axis_epsilon=1e-7):
    relative = torch.einsum("...ji,...jk->...ik", rotation, box_rotation)
    translation = _local(box_center, center, rotation)
    absolute = relative.abs()
    first = translation.abs() - half_size - torch.einsum("...ij,...j->...i", absolute, box_half_size)
    second = torch.einsum("...ij,...i->...j", relative, translation).abs() - box_half_size \
        - torch.einsum("...ij,...i->...j", absolute, half_size)
    one, two = [1, 2, 0], [2, 0, 1]
    row_one, row_two = relative[..., one, :], relative[..., two, :]
    projected = (translation[..., two, None] * row_one - translation[..., one, None] * row_two).abs()
    first_radius = half_size[..., one, None] * row_two.abs() + half_size[..., two, None] * row_one.abs()
    second_radius = box_half_size[..., None, one] * absolute[..., two] + box_half_size[..., None, two] * absolute[..., one]
    length = (row_one ** 2 + row_two ** 2).sqrt()
    usable = length > parallel_axis_epsilon
    cross = torch.where(usable, (projected - first_radius - second_radius) /
                        torch.where(usable, length, 1.), -torch.inf)
    return torch.maximum(torch.maximum(first.amax(-1), second.amax(-1)), cross.amax((-2, -1)))


def transform_primitives(xpos, xmat, compiled):
    positions = xpos[:, compiled["body_ids"]]
    rotations = xmat.reshape(xmat.shape[0], -1, 3, 3)[:, compiled["body_ids"]]
    return dict(centers=positions + torch.einsum("bsij,sj->bsi", rotations, compiled["local_centers"]),
                rotations=torch.einsum("bsij,sjk->bsik", rotations, compiled["local_rotations"]),
                endpoints=positions[:, :, None] + torch.einsum("bsij,skj->bski", rotations, compiled["local_endpoints"]))


def lookup_collision_candidates(arrays, scene_id, shape_centers, *, max_candidates):
    shape = arrays["scene_grid_shapes"][scene_id][:, None]
    origin = arrays["scene_grid_origins"][scene_id][:, None]
    index = torch.floor((shape_centers - origin) / arrays["cell_size"]).long()
    inside = ((index >= 0) & (index < shape)).all(-1)
    index = torch.minimum(index.clamp_min(0), shape - 1)
    cell = arrays["scene_grid_offsets"][scene_id][:, None] + (index[..., 0] * shape[..., 1] + index[..., 1]) * shape[..., 2] + index[..., 2]
    count, start = arrays["cell_counts"][cell], arrays["cell_starts"][cell]
    offsets = torch.arange(max_candidates, device=shape_centers.device)
    addresses = (start[..., None] + offsets).clamp_max(arrays["candidate_ids"].shape[0] - 1)
    valid = inside[..., None] & (offsets < count[..., None])
    return torch.where(valid, arrays["candidate_ids"][addresses], 0).long(), valid


def body_collisions(compiled, bank, scene_id, xpos, xmat, *, max_candidates):
    world = transform_primitives(xpos, xmat, compiled)
    result = torch.zeros((len(scene_id), len(compiled["body_ids"])), dtype=torch.bool, device=xpos.device)
    for kind, indices in compiled["indices"].items():
        if not len(indices):
            continue
        centers = world["centers"][:, indices]
        ids, valid = lookup_collision_candidates(bank, scene_id, centers, max_candidates=max_candidates)
        collided = torch.zeros_like(valid[..., 0])
        for begin in range(0, max_candidates, 8):
            item, mask = ids[..., begin:begin + 8], valid[..., begin:begin + 8]
            obstacle = (bank["centers"][item], bank["rotations"][item], bank["half_sizes"][item])
            if kind == "box":
                separation = box_box_separation(centers[:, :, None], world["rotations"][:, indices, None],
                    compiled["half_sizes"][indices][None, :, None], *obstacle)
            elif kind == "capsule":
                ends = world["endpoints"][:, indices]
                separation = capsule_box_separation(ends[:, :, None, 0], ends[:, :, None, 1],
                    compiled["radii"][indices][None, :, None], *obstacle)
            else:
                separation = sphere_box_separation(centers[:, :, None], compiled["radii"][indices][None, :, None], *obstacle)
            collided |= (mask & (separation <= 0)).any(-1)
        result[:, indices] = collided
    return result


def compile_proposal(proposal, model, device):
    """Compile only immutable geometry; no JAX runtime dependency."""
    import mujoco
    values = {k: [] for k in ("body_ids", "local_centers", "local_rotations", "local_endpoints", "half_sizes", "radii")}
    indices = {k: [] for k in ("box", "capsule", "sphere")}
    names = [s["id"] for s in proposal["shapes"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate collision shape names")
    for i, s in enumerate(proposal["shapes"]):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s["body_name"])
        if body <= 0 or s["kind"] not in indices:
            raise ValueError("Unknown robot body or primitive")
        q = np.asarray(s["quat"], dtype=float); q /= np.linalg.norm(q)
        mat = np.empty(9); mujoco.mju_quat2Mat(mat, q)
        for name, value in (("body_ids", body), ("local_centers", s["center"]),
                            ("local_rotations", mat.reshape(3, 3)),
                            ("local_endpoints", s.get("endpoints", [s["center"], s["center"]])),
                            ("half_sizes", s.get("half_size", [0., 0., 0.])), ("radii", s.get("radius", 0.))):
            values[name].append(value)
        indices[s["kind"]].append(i)
    compiled = {k: torch.as_tensor(np.asarray(v), device=device, dtype=torch.long if k == "body_ids" else torch.float32)
                for k, v in values.items()}
    compiled["indices"] = {k: torch.tensor(v, device=device, dtype=torch.long) for k, v in indices.items()}
    compiled["region_mask"] = torch.tensor([[s["group"] == r for s in proposal["shapes"]] for r in REGIONS], device=device)
    return compiled


class CollisionChecker:
    def __init__(self, model, bank_manifest, *, field_manifest, device, proposal_path=PROPOSAL):
        from cat_ppo.furniture.body_collision_bank import load_body_collision_bank
        self.proposal_path = Path(proposal_path)
        self.proposal_hash = hashlib.sha256(self.proposal_path.read_bytes()).hexdigest()
        arrays, self.metadata = load_body_collision_bank(bank_manifest,
            expected_field_manifest=field_manifest, expected_proxy_sha256=self.proposal_hash)
        self.bank = {k: torch.as_tensor(v, device=device, dtype=torch.long if v.dtype.kind in "iu" else torch.float32)
                     for k, v in arrays.items()}
        self.compiled = compile_proposal(json.loads(self.proposal_path.read_text()), model, device)
        self._kernel = self._compute

    def enable_compilation(self, *, backend="inductor"):
        self._kernel = torch.compile(self._compute, backend=backend, fullgraph=True,
                                     dynamic=True, mode="default")

    def _compute(self, scene_ids, xpos, xmat):
        flags = body_collisions(self.compiled, self.bank, scene_ids, xpos, xmat,
                                max_candidates=self.metadata["static_candidate_count"])
        return (flags[:, None] & self.compiled["region_mask"][None]).any(-1)

    def __call__(self, scene_ids, data):
        return self._kernel(scene_ids, data.xpos, data.xmat)
