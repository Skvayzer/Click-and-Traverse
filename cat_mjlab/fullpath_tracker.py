"""Tracker v2 inputs, plan feed and plan-relative terms for full start->goal plans (S3 of docs/fullpath_plan.md). GPU, batched.

FullPathBank holds P oracle plans (cat_mjlab/fullpath.py / scripts/fullpath_oracle*.py), each a dense 50 Hz whole-body qpos
trajectory in one scene from one start, padded to a common length; at load it derives per frame: the 0.5 s-smoothed pelvis
path and its arc length, the posture modes, both feet's positions, and the planned clearance at the policy's 13 body points
(the scene's SDF at the plan pose's sites, so that the robot's clearance can be compared with the plan's).

FullPathFeed keeps, per world, the plan id (-1: none) and the matched plan index (searched forward from the last match by
nearest smoothed pelvis), and provides
  * chunks for the tracker's existing 'plan' source (21 frames re-timed to the operator speed, as OracleFeeder),
  * the extended reference block (FEATURES = 46, appended at the end of the observation; zero when there is no plan):
      path: 6 points at arc lengths AHEAD m past the match: dx, dy (heading frame), dz, sin/cos dyaw                 (30)
      modes at the match (sideways, crouch, high step, hands raised, hands tucked) + distance to the next high step
      and the next crouch frame (/3 m, clipped to 1)                                                                  (7)
      legs: each foot's planned position LEG_HORIZON s ahead relative to the pelvis (heading frame), the planned
      swing apex of the next second (foot height), and the leg mask (a high step within LEG_WINDOW m)                (9)
  * foot tracking (reward, under the leg mask), the planned clearance at the 13 points (plan-relative costs, deviation
    reflex).
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch

DT = 0.02
AHEAD = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
LEG_HORIZON = 0.3
LEG_WINDOW = 1.0
FEATURES = 46


def feature_names():
    n = []
    for d in AHEAD:
        n += [f'fp.path.{d:g}.{k}' for k in ('dx', 'dy', 'dz', 'sin', 'cos')]
    n += [f'fp.mode.{k}' for k in ('sideways', 'crouch', 'high_step', 'hands_raised', 'hands_tucked')]
    n += ['fp.next_high_step', 'fp.next_crouch']
    for f in ('left', 'right'):
        n += [f'fp.foot.{f}.{k}' for k in ('dx', 'dy', 'dz', 'apex')]
    n += ['fp.leg_mask']
    assert len(n) == FEATURES
    return n


def _yaw(q):
    w, x, y, z = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class FullPathBank:
    def __init__(self, files, mj_model, bank, scene_index, device, *, site_names, max_len=None):
        """files: oracle npz paths (each with qpos and meta incl. scene); scene_index: scene id -> bank index."""
        from .planner.fk import TorchFK
        from .exact_geometry import ExactGeometry
        plans, scenes, metas = [], [], []
        for f in files:
            d = np.load(f)
            meta = json.loads(str(d["meta"]))
            if not meta.get("ok") or meta["scene"] not in scene_index:
                continue
            plans.append(np.asarray(d["qpos"], np.float32)); scenes.append(scene_index[meta["scene"]]); metas.append(meta)
        if not plans:
            raise ValueError("no usable full-path plans")
        T = max(len(p) for p in plans) if max_len is None else int(max_len)
        P = len(plans)
        q = np.zeros((P, T, 36), np.float32)
        length = np.zeros(P, np.int64)
        for i, p in enumerate(plans):
            p = p[:T]; q[i, :len(p)] = p; q[i, len(p):] = p[-1]; length[i] = len(p)
        self.device = torch.device(device)
        self.qpos = torch.as_tensor(q, device=self.device)
        self.length = torch.as_tensor(length, device=self.device)
        self.scene = torch.as_tensor(scenes, device=self.device)
        self.metas = metas
        self.P, self.T = P, T
        # smoothed pelvis path (25-step box filter, edge-padded) and its arc length
        xy = self.qpos[..., :2]
        k = 12
        pad = torch.cat((xy[:, :1].expand(-1, k, -1), xy, xy[:, -1:].expand(-1, k, -1)), 1)
        csum = torch.cumsum(torch.cat((torch.zeros_like(pad[:, :1]), pad), 1), 1)
        self.smooth = (csum[:, 2 * k + 1:] - csum[:, :-2 * k - 1]) / (2 * k + 1)
        step = torch.linalg.vector_norm(torch.diff(self.smooth, dim=1), dim=-1)
        self.s = torch.cat((torch.zeros_like(step[:, :1]), torch.cumsum(step, 1)), 1)        # [P, T]
        self.yaw = torch.as_tensor(np.unwrap(_yaw(self.qpos).cpu().numpy(), axis=1), device=self.device)
        # kinematics: feet, hands, sites
        fk = TorchFK(mj_model, self.device)
        names = [mj_model.body(i).name for i in range(mj_model.nbody)]
        feet = [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")]
        site_ids = [mj_model.site(n).id for n in site_names]
        site_body = torch.as_tensor(np.asarray(mj_model.site_bodyid)[site_ids], device=self.device)
        site_pos = torch.as_tensor(np.asarray(mj_model.site_pos)[site_ids], dtype=torch.float32, device=self.device)
        self.feet = torch.zeros((P, T, 2, 3), device=self.device)
        self.sites = torch.zeros((P, T, len(site_ids), 3), device=self.device)
        flat = self.qpos.reshape(-1, 36)
        feet_out, site_out = [], []
        for a in range(0, len(flat), 8192):
            xpos, xmat = fk(flat[a:a + 8192])
            feet_out.append(xpos[:, feet])
            site_out.append(xpos[:, site_body] + torch.einsum("nsij,sj->nsi", xmat[:, site_body], site_pos))
        self.sites = torch.cat(site_out).reshape(P, T, len(site_ids), 3)
        self.feet = self.sites[:, :, 3:5]              # the feet sites (task.site_names order), as the policy's positions
        # planned clearance at the sites: the scene SDF (the policy's field) at the plan pose
        sdf = []
        for i in range(P):
            sdf.append(bank.sample('sdf', self.sites[i], self.scene[i].expand(T))[..., 0])
        self.clear = torch.stack(sdf)                                                             # [P, T, S]
        # posture modes per frame (from the clip labels when present, else from kinematics)
        z = self.qpos[..., 2]
        foot_lift = self.feet[..., 2].amax(-1) > .12
        self.modes = torch.stack((torch.zeros_like(z, dtype=torch.bool), z < .70, foot_lift,
                                  torch.zeros_like(z, dtype=torch.bool), torch.zeros_like(z, dtype=torch.bool)), -1)
        motion = torch.diff(self.smooth, dim=1, append=self.smooth[:, -1:])
        heading = self.yaw
        moving = torch.linalg.vector_norm(motion, dim=-1) > .002
        ang = torch.remainder(torch.atan2(motion[..., 1], motion[..., 0]) - heading + math.pi, 2 * math.pi) - math.pi
        self.modes[..., 0] = moving & (ang.abs() > math.pi / 4) & (ang.abs() < 3 * math.pi / 4)
        # distance (along s) to the next high-step / crouch frame, per frame
        self.next_high = self._next_distance(self.modes[..., 2])
        self.next_crouch = self._next_distance(self.modes[..., 1])

    def _next_distance(self, flag):
        big = torch.full_like(self.s, 1e3)
        s_at = torch.where(flag, self.s, big)
        nxt = torch.flip(torch.cummin(torch.flip(s_at, [1]), 1).values, [1])        # min s of flagged frames at or after t
        return (nxt - self.s).clamp(0, 3.)

    def starts(self):
        return self.qpos[:, 0]


class FullPathFeed:
    def __init__(self, bank, num_envs, device, *, lead=0.0, search_back=.5, search_ahead=2.0):
        self.bank, self.n, self.device = bank, int(num_envs), torch.device(device)
        self.plan = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        self.idx = torch.zeros(self.n, device=self.device)
        self.lead = float(lead)
        self.back, self.ahead = int(search_back / DT), int(search_ahead / DT)
        self.v_plan = (bank.s[:, -1] / ((bank.length - 1).clamp_min(1) * DT)).clamp_min(.05)       # [P]

    @property
    def active(self):
        return self.plan >= 0

    def assign(self, ids, plan_ids):
        self.plan[ids] = plan_ids; self.idx[ids] = 0.

    def match(self, ids, root_xy):
        """Advance the matched plan index of worlds ids (with a plan) to the smoothed pelvis nearest the robot."""
        p = self.plan[ids]
        ok = p >= 0
        if not bool(ok.any()):
            return
        ids, p, xy = ids[ok], p[ok], root_xy[ok]
        lo = (self.idx[ids].long() - self.back).clamp_min(0)
        win = lo[:, None] + torch.arange(self.back + self.ahead, device=self.device)[None]
        win = torch.minimum(win, (self.bank.length[p] - 1)[:, None])
        pts = self.bank.smooth[p[:, None], win]                                        # [m, W, 2]
        j = torch.linalg.vector_norm(pts - xy[:, None], dim=-1).argmin(-1)
        new = win.gather(1, j[:, None])[:, 0].float()
        self.idx[ids] = torch.maximum(new, self.idx[ids] - 10)

    def _at(self, x, p, t):
        """Frame-interpolated plan buffer x [P, T, ...] at fractional indices t [m, K] for plans p [m]."""
        t = torch.minimum(t, (self.bank.length[p] - 1).float()[:, None]).clamp_min(0)
        a = t.floor().long().clamp_max(self.bank.T - 2); w = t - a
        x0 = x[p[:, None], a]; x1 = x[p[:, None], (a + 1)]
        w = w.reshape(*w.shape, *([1] * (x0.dim() - 2)))
        return x0 + w * (x1 - x0)

    def chunk(self, ids, route_speed):
        """21 world frames for the tracker's plan source: xy [m, 21, 2], yaw, z [m, 21], upper [m, 21, 17], roll/pitch, legs."""
        p = self.plan[ids]
        rate = (route_speed / self.v_plan[p]).clamp(.2, 1.5)
        t = self.idx[ids][:, None] + (self.lead + .1 * torch.arange(21, device=self.device)[None]) * rate[:, None] / DT
        q = self._at(self.bank.qpos, p, t)
        yaw = self._at(self.bank.yaw, p, t)
        w, x, y, z = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
        return q[..., :2], yaw, q[..., 2], q[..., 19:36], roll, pitch, q[..., 7:19]

    def features(self, ids, root_xy, root_yaw, root_z, navi, route_speed):
        out = torch.zeros((len(ids), FEATURES), device=self.device)
        p = self.plan[ids]
        ok = p >= 0
        if not bool(ok.any()):
            return out
        sel = torch.nonzero(ok)[:, 0]
        p = p[ok]; i0 = self.idx[ids[ok]]
        b = self.bank
        s0 = b.s[p, i0.long().clamp_max(b.T - 1)]
        cols = []
        rot = navi[sel]
        for d in AHEAD:
            target = s0 + d
            j = torch.searchsorted(b.s[p], target[:, None]).clamp_max(b.T - 1)[:, 0]
            j = torch.minimum(j, b.length[p] - 1)
            q = b.qpos[p, j]
            off = torch.cat((q[:, :2] - root_xy[ok], torch.zeros_like(q[:, :1])), -1)
            loc = torch.einsum("bij,bi->bj", rot, off)[:, :2]
            dyaw = b.yaw[p, j] - root_yaw[ok]
            cols += [loc, (q[:, 2] - root_z[ok])[:, None], dyaw.sin()[:, None], dyaw.cos()[:, None]]
        k = i0.long().clamp_max(b.T - 1)
        cols += [b.modes[p, k].float(), (b.next_high[p, k] / 3.)[:, None], (b.next_crouch[p, k] / 3.)[:, None]]
        rate = (route_speed[ok] / self.v_plan[p]).clamp(.2, 1.5)
        th = (i0 + LEG_HORIZON * rate / DT)[:, None]
        feet = self._at(b.feet, p, th)[:, 0]                                                  # [m, 2, 3]
        horizon = (i0[:, None] + torch.arange(0, int(1. / DT), 5, device=self.device)[None] * rate[:, None]).clamp_max(b.T - 1)
        apex = self._at(b.feet, p, horizon)[..., 2].amax(1)                                   # [m, 2]
        pel = torch.cat((root_xy[ok], root_z[ok][:, None]), -1)
        for f in range(2):
            off = feet[:, f] - pel
            loc = torch.einsum("bij,bi->bj", rot, torch.cat((off[:, :2], torch.zeros_like(off[:, :1])), -1))[:, :2]
            cols += [loc, off[:, 2:3], apex[:, f:f + 1]]
        mask = (b.next_high[p, k] <= LEG_WINDOW).float()[:, None]
        cols.append(mask)
        block = torch.cat(cols, -1)
        # legs only near a high step
        leg = slice(FEATURES - 9, FEATURES - 1)
        block[:, leg] = block[:, leg] * mask
        out[sel] = block
        return out

    def planned(self, ids):
        """Plan pose at the match for worlds ids: planned clearance at the sites [m, S] (inf without a plan), feet [m, 2, 3],
        the leg mask [m] (a high step within LEG_WINDOW)."""
        p = self.plan[ids]
        ok = p >= 0
        b = self.bank
        S = b.clear.shape[-1]
        clear = torch.full((len(ids), S), float('inf'), device=self.device)
        feet = torch.zeros((len(ids), 2, 3), device=self.device)
        mask = torch.zeros(len(ids), device=self.device)
        if bool(ok.any()):
            k = self.idx[ids[ok]].long().clamp_max(b.T - 1)
            clear[ok] = b.clear[p[ok], k]; feet[ok] = b.feet[p[ok], k]
            mask[ok] = (b.next_high[p[ok], k] <= LEG_WINDOW).float()
        return clear, feet, mask
