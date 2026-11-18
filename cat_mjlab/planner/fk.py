"""Batched forward kinematics in torch for the G1 training model, and the body sphere cover.

`TorchFK(model)` reproduces MuJoCo's mj_kinematics for this model class: one free root joint and hinge joints, at
most one joint per body, any joint anchor (jnt_pos) and reference (qpos0). It is checked against MuJoCo in
tests/test_planner.py. `BodySpheres` places the conservative sphere cover of the approved 35-part collision proxy
(cat_mjlab/reactive.BodyEnvelope: 9 spheres per capsule, 27 per box, the original hand spheres) from the FK poses.
"""
from __future__ import annotations

import numpy as np
import torch


def quat_to_mat(q):
    """MuJoCo quaternions [..., 4] (w, x, y, z) to rotation matrices [..., 3, 3]; normalizes first."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(*q.shape[:-1], 3, 3)


def axis_angle_mat(axis, angle):
    """Rotation about unit `axis` [3] by `angle` [...] (Rodrigues), [..., 3, 3]."""
    a = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
    k = torch.zeros(3, 3, dtype=angle.dtype, device=angle.device)
    k[0, 1], k[0, 2], k[1, 0], k[1, 2], k[2, 0], k[2, 1] = -a[2], a[1], a[2], -a[0], -a[1], a[0]
    s, c = angle.sin()[..., None, None], angle.cos()[..., None, None]
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    return eye + s * k + (1 - c) * (k @ k)


class TorchFK:
    """xpos [..., nbody, 3] and xmat [..., nbody, 3, 3] from qpos [..., nq] (root position, root quaternion, hinges)."""

    def __init__(self, model, device="cpu", dtype=torch.float32):
        import mujoco
        self.device, self.dtype = torch.device(device), dtype
        self.nbody, self.nq = int(model.nbody), int(model.nq)
        tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)
        self.parent = [int(p) for p in model.body_parentid]
        self.body_pos, self.body_rot = tensor(model.body_pos), quat_to_mat(tensor(model.body_quat))
        self.joint = []                       # per body: None, ('free', qadr) or ('hinge', qadr, axis, anchor, q0)
        for b in range(self.nbody):
            n = int(model.body_jntnum[b])
            if n == 0:
                self.joint.append(None)
                continue
            if n > 1:
                raise ValueError(f"body {b} has {n} joints; TorchFK supports at most one")
            j = int(model.body_jntadr[b]); kind = int(model.jnt_type[j]); adr = int(model.jnt_qposadr[j])
            if kind == mujoco.mjtJoint.mjJNT_FREE:
                self.joint.append(("free", adr))
            elif kind == mujoco.mjtJoint.mjJNT_HINGE:
                self.joint.append(("hinge", adr, tensor(model.jnt_axis[j]), tensor(model.jnt_pos[j]), float(model.qpos0[adr])))
            else:
                raise ValueError(f"joint type {kind} is not supported (free and hinge only)")

    def __call__(self, qpos):
        qpos = qpos.to(self.device, self.dtype)
        batch = qpos.shape[:-1]
        eye = torch.eye(3, dtype=self.dtype, device=self.device).expand(*batch, 3, 3)
        pos = [torch.zeros(*batch, 3, dtype=self.dtype, device=self.device)]
        rot = [eye]
        for b in range(1, self.nbody):
            joint = self.joint[b]
            if joint is not None and joint[0] == "free":
                adr = joint[1]
                pos.append(qpos[..., adr:adr + 3]); rot.append(quat_to_mat(qpos[..., adr + 3:adr + 7]))
                continue
            p, r = pos[self.parent[b]], rot[self.parent[b]]
            x = p + (r @ self.body_pos[b, :, None])[..., 0]
            m = r @ self.body_rot[b]
            if joint is not None:
                _, adr, axis, anchor, q0 = joint
                xanchor = x + (m @ anchor[:, None])[..., 0]
                m = m @ axis_angle_mat(axis, qpos[..., adr] - q0)
                x = xanchor - (m @ anchor[:, None])[..., 0]
            pos.append(x); rot.append(m)
        return torch.stack(pos, -2), torch.stack(rot, -3)


class BodySpheres:
    """World centres [..., S, 3] and radii [S] of the conservative sphere cover of the 35-part collision proxy."""

    def __init__(self, model, device="cpu", dtype=torch.float32):
        from cat_mjlab.reactive import BodyEnvelope
        envelope = BodyEnvelope(model, "cpu")
        self.bodies = envelope.bodies.to(device)
        self.local = envelope.local.to(device, dtype)
        self.radii = envelope.radii.to(device, dtype)
        self.hand_mask = envelope.hand_mask.to(device)
        self.fk = TorchFK(model, device, dtype)

    def __call__(self, qpos):
        xpos, xmat = self.fk(qpos)
        body_pos, body_rot = xpos[..., self.bodies, :], xmat[..., self.bodies, :, :]
        return body_pos + (body_rot @ self.local[:, :, None])[..., 0]
