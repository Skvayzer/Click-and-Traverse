"""Exact-geometry clearance of whole-body poses against a scene's collision boxes (full-path study, docs/fullpath_plan.md).

The episode-ending check (cat_mjlab/collision.py) tests the 35 primitives of the approved collision proxy (boxes, capsules,
spheres on the robot's links) against the scene's oriented collision boxes. Here the same separations are evaluated against
ALL boxes of one scene (no grid candidate lookup) and returned as numbers: separation < 0 means the pose touches an obstacle.
Box-box separations are the separating-axis value, a lower bound of the true distance when apart (so margins are
conservative); capsule-box and sphere-box separations are exact distances.

Poses are MuJoCo qpos [N, 36] of the training model (free root + 29 hinges); forward kinematics is cat_mjlab/planner/fk.TorchFK.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .collision import PROPOSAL, REGIONS, box_box_separation, capsule_box_separation, compile_proposal, \
    sphere_box_separation, transform_primitives


class ExactGeometry:
    def __init__(self, device="cpu", model=None):
        import mujoco
        from .model import assemble_training_xml
        from .planner.fk import TorchFK
        self.model = model if model is not None else mujoco.MjModel.from_xml_string(assemble_training_xml())
        self.device = torch.device(device)
        self.fk = TorchFK(self.model, self.device)
        proposal = json.loads(PROPOSAL.read_text())
        self.compiled = compile_proposal(proposal, self.model, self.device)
        self.shape_names = [s["id"] for s in proposal["shapes"]]
        self.region_of_shape = self.compiled["region_mask"].int().argmax(0)            # [S] index into REGIONS
        self.n_shapes = len(self.shape_names)
        names = [self.model.body(i).name for i in range(self.model.nbody)]
        self.body = {n: i for i, n in enumerate(names)}
        hands = [i for i, s in enumerate(proposal["shapes"]) if s["group"] == "hands"]
        self.hand_shapes = torch.tensor(hands, device=self.device)                    # left, right (proposal order)
        self.feet_bodies = torch.tensor([self.body["left_ankle_roll_link"], self.body["right_ankle_roll_link"]], device=self.device)

    def primitives(self, qpos):
        xpos, xmat = self.fk(torch.as_tensor(qpos, dtype=torch.float32, device=self.device))
        return transform_primitives(xpos, xmat, self.compiled), xpos

    def keypoints(self, qpos):
        """pelvis [N, 3], hands [N, 2, 3] (proxy hand-sphere centres), feet [N, 2, 3] (ankle-roll bodies)."""
        world, xpos = self.primitives(qpos)
        return xpos[:, self.body["pelvis"]], world["centers"][:, self.hand_shapes], xpos[:, self.feet_bodies]

    def separation(self, qpos, boxes, chunk=4096):
        """Per-pose, per-primitive, per-box separation reduced to: min over boxes [N, S] and the argmin box [N, S]."""
        center, rot, half = (torch.as_tensor(boxes[k], dtype=torch.float32, device=self.device)
                             for k in ("box_center", "box_rot", "box_half"))
        qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        n, m = len(qpos), len(center)
        if m == 0:
            return torch.full((n, self.n_shapes), 10., device=self.device), torch.full((n, self.n_shapes), -1, device=self.device)
        out_v, out_i = [], []
        c = self.compiled
        for a in range(0, n, chunk):
            world, _ = self.primitives(qpos[a:a + chunk])
            b = len(world["centers"])
            sep = torch.empty((b, self.n_shapes, m), device=self.device)
            ob = (center[None, None], rot[None, None], half[None, None])
            for kind, idx in c["indices"].items():
                if not len(idx):
                    continue
                if kind == "box":
                    s = box_box_separation(world["centers"][:, idx, None], world["rotations"][:, idx, None],
                                           c["half_sizes"][idx][None, :, None], *ob)
                elif kind == "capsule":
                    e = world["endpoints"][:, idx]
                    s = capsule_box_separation(e[:, :, None, 0], e[:, :, None, 1], c["radii"][idx][None, :, None], *ob)
                else:
                    s = sphere_box_separation(world["centers"][:, idx, None], c["radii"][idx][None, :, None], *ob)
                sep[:, idx] = s
            v, i = sep.min(-1)
            out_v.append(v); out_i.append(i)
        return torch.cat(out_v), torch.cat(out_i)

    def min_separation(self, qpos, boxes):
        """[N] min separation over all primitives and boxes."""
        return self.separation(qpos, boxes)[0].amin(-1)


def interpolate_frames(frames, t, dt=0.1):
    """Pose at times t [n] (s) of a keyframe chunk frames [F, 36] (0.1 s apart): linear, quaternion nlerp."""
    frames = np.asarray(frames, dtype=np.float64)
    u = np.clip(np.asarray(t, dtype=np.float64) / dt, 0., len(frames) - 1.)
    i = np.minimum(np.floor(u).astype(int), len(frames) - 2)
    a = (u - i)[:, None]
    q = (1 - a) * frames[i] + a * frames[i + 1]
    qa, qb = frames[i, 3:7], frames[i + 1, 3:7]
    qb = np.where((qa * qb).sum(-1, keepdims=True) < 0, -qb, qb)
    quat = (1 - a) * qa + a * qb
    q[:, 3:7] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    return q.astype(np.float32)


def box_kind(boxes, overhead_z=0.15, low_top=0.5):
    """0 low (hurdle), 1 wall, 2 overhead (beam / table top) per box, as cat_mjlab/diagnose.classify (axis-aligned extents
    from the rotated half sizes)."""
    rot, half, cen = np.asarray(boxes["box_rot"]), np.asarray(boxes["box_half"]), np.asarray(boxes["box_center"])
    ext = np.abs(rot) @ half[..., None] if len(half) else np.zeros((0, 3, 1))
    ext = ext[..., 0]
    lo, hi = cen[:, 2] - ext[:, 2], cen[:, 2] + ext[:, 2]
    kind = np.ones(len(cen), dtype=np.int64)
    kind[hi <= low_top] = 0
    kind[lo >= overhead_z] = 2
    return kind


def box_far_end(boxes, j, u):
    """Largest projection of box j's corners on the unit direction u [2] (xy)."""
    rot, half, cen = np.asarray(boxes["box_rot"][j]), np.asarray(boxes["box_half"][j]), np.asarray(boxes["box_center"][j])
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    corners = cen + (signs * half) @ rot.T
    return float((corners[:, :2] @ u).max())
