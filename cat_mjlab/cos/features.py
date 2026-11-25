"""Per-frame kinematic features of G1 motion (qpos [T, 36], training model) from the approved 35-primitive collision proxy.

All heights are of the proxy surface (box corners, capsule ends + radius, sphere + radius), so they are the heights an
obstacle would have to clear, not joint-centre heights.
"""
from __future__ import annotations

import numpy as np
import torch

from ..exact_geometry import ExactGeometry


def _extent(world, compiled):
    """Per-primitive axis-aligned lower / upper world corners [T, S, 3]."""
    c = world["centers"]
    lo = torch.empty_like(c); hi = torch.empty_like(c)
    idx = compiled["indices"]
    if len(idx["box"]):
        b = idx["box"]
        ext = (world["rotations"][:, b].abs() @ compiled["half_sizes"][b][None, :, :, None])[..., 0]
        lo[:, b] = c[:, b] - ext; hi[:, b] = c[:, b] + ext
    if len(idx["capsule"]):
        k = idx["capsule"]; e = world["endpoints"][:, k]; r = compiled["radii"][k][None, :, None]
        lo[:, k] = e.amin(2) - r; hi[:, k] = e.amax(2) + r
    if len(idx["sphere"]):
        s = idx["sphere"]; r = compiled["radii"][s][None, :, None]
        lo[:, s] = c[:, s] - r; hi[:, s] = c[:, s] + r
    return lo, hi


def yaw_of(mat):
    """Heading (rad) of a body's x axis from rotation matrices [..., 3, 3]."""
    return np.arctan2(mat[..., 1, 0], mat[..., 0, 0])


class FrameFeatures:
    def __init__(self, geom: ExactGeometry | None = None):
        self.g = geom or ExactGeometry()
        names = self.g.shape_names
        self.groups = {}
        import json
        from ..collision import PROPOSAL
        shapes = json.loads(PROPOSAL.read_text())["shapes"]
        for i, s in enumerate(shapes):
            self.groups.setdefault(s["group"], []).append(i)
        self.knee_shapes = [i for i, n in enumerate(names) if "knee" in n]
        self.left_foot = [i for i, n in enumerate(names) if n in ("left_sole", "left_upper_foot", "left_ankle_pitch_link")]
        self.right_foot = [i for i, n in enumerate(names) if n in ("right_sole", "right_upper_foot", "right_ankle_pitch_link")]
        self.torso_body = self.g.body["torso_link"]
        self.knee_bodies = [self.g.body["left_knee_link"], self.g.body["right_knee_link"]]

    def __call__(self, qpos, chunk=20000):
        out = {k: [] for k in ("top", "top_body", "pelvis", "pelvis_yaw", "torso_yaw", "hand_z", "hand_xyz", "knee_z", "foot_z",
                               "foot_xy", "hand_lat", "elbow_lat")}
        for a in range(0, len(qpos), chunk):
            q = torch.as_tensor(np.asarray(qpos[a:a + chunk]), dtype=torch.float32)
            world, xpos = self.g.primitives(q)
            xmat = self.g.fk(q)[1]
            lo, hi = _extent(world, self.g.compiled)
            out["top"].append(hi[..., 2].amax(1).numpy())
            tb = self.groups["head"] + self.groups["trunk"]
            out["top_body"].append(hi[:, tb, 2].amax(1).numpy())          # head + trunk (arms excluded)
            out["pelvis"].append(xpos[:, self.g.body["pelvis"]].numpy())
            out["pelvis_yaw"].append(yaw_of(xmat[:, self.g.body["pelvis"]].numpy()))
            tyaw = yaw_of(xmat[:, self.torso_body].numpy()); out["torso_yaw"].append(tyaw)
            hands = world["centers"][:, self.g.hand_shapes]                # [T, 2, 3]
            hl = lo[:, self.g.hand_shapes, 2]
            out["hand_z"].append(hl.numpy()); out["hand_xyz"].append(hands.numpy())
            out["knee_z"].append(xpos[:, self.knee_bodies, 2].amin(1).numpy())     # knee joint centres
            out["foot_z"].append(torch.stack([lo[:, self.left_foot, 2].amin(1), lo[:, self.right_foot, 2].amin(1)], 1).numpy())
            out["foot_xy"].append(xpos[:, self.g.feet_bodies, :2].numpy())
            # lateral half-width of the arms in the torso heading frame (max |y| of arm/hand primitive corners)
            t = xpos[:, self.torso_body]
            cy, sy = np.cos(tyaw), np.sin(tyaw)
            left = torch.as_tensor(np.stack([-sy, cy], -1), dtype=torch.float32)            # [T, 2] lateral unit
            arm = self.groups["arms"] + self.groups["hands"]
            for key, ids in (("hand_lat", self.groups["hands"]), ("elbow_lat", arm)):
                pts = torch.cat([lo[:, ids, :2], hi[:, ids, :2]], 1) - t[:, None, :2]
                out[key].append((pts * left[:, None]).sum(-1).abs().amax(1).numpy())
        return {k: np.concatenate(v) for k, v in out.items()}
