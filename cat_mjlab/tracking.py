"""Tracker v1: time-indexed whole-body references for the reference-tracking policy (SWAP-style interface).

The reactive policy keeps every input it has (proprioception, command, humanoid potential field at 13 body points)
and gets one more block, the reference, so the same network can track a plan when it has one and react when it
does not (valid = 0). Per world the reference is a path in the scene (world xy polyline, uniform spacing H) with an
arc-length progress s and a speed v, plus optional heading, pelvis height and upper-body (waist + arms) targets.

Progress is a clock that waits: s advances by v * dt only while the robot is within MAX_LAG behind it, so a slow
reference stays slow and a robot that falls behind is not dragged. A robot ahead of its reference is charged
(track_lead), so it cannot rush.

Sources (chosen per episode):
    'field'   the scene's guidance streamline from the pelvis (the geodesic toward the goal, as the potential field
              already gives it), played at the operator speed; heading, height and arms free (masked). This is the
              slow-traversal reference that needs no planner.
    'random'  a procedural smooth path (piecewise curvature) with a piecewise speed (stops included), and optional
              heading (sideways segments), pelvis height (crouches) and upper-body targets: general tracking.
    'none'    no reference: valid = 0, the policy runs on the potential field and the command as before.
Planner chunks and stored rollout windows (cat_mjlab/planner, data pose field) plug in as more sources later: they set
the same tensors (path, speed, targets, masks).

Features (FEATURES = 41), in the robot's heading frame:
    valid, then for each horizon tau in HORIZONS: dx, dy, sin dyaw, cos dyaw, dz (pelvis height minus NOMINAL_Z),
    then mask_heading, mask_height, the 17 upper-body targets minus nominal at UPPER_HORIZON, mask_upper.
"""
from __future__ import annotations

import math

import torch

FEATURES = 41
HORIZONS = (0.1, 0.3, 0.6, 1.0)          # s, base waypoints
UPPER_HORIZON = 0.3                       # s, upper-body target time
H = 0.05                                  # m, path spacing
P = 160                                   # path points kept per world (8 m)
MAX_LAG = 0.15                            # m, progress waits while the robot is farther behind
LOST = 1.0                                # m, farther than this from the reference point: re-anchor
NOMINAL_Z = 0.78                          # m, pelvis height of the nominal stance
SOURCES = ('field', 'random', 'none')


def feature_names():
    names = ['ref.valid']
    for tau in HORIZONS:
        t = f'{tau:.1f}'
        names += [f'ref.base.{t}.dx', f'ref.base.{t}.dy', f'ref.base.{t}.sin_dyaw', f'ref.base.{t}.cos_dyaw', f'ref.base.{t}.dz']
    names += ['ref.mask.heading', 'ref.mask.height']
    names += [f'ref.upper.{k}' for k in range(17)] + ['ref.mask.upper']
    assert len(names) == FEATURES
    return names


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


class ReferenceTracker:
    """Reference state for N worlds. Torch only; every method is batched over world ids."""

    def __init__(self, num_envs, device, *, mix=(('field', .5), ('random', .3), ('none', .2)), random_speed=(0., .6),
                 stop_share=.15, sideways_share=.3, crouch_share=.25, upper_share=.5, upper_amplitude=.4,
                 upper_rate=1.0, nominal_upper=None, generator=None):
        self.n, self.device, self.generator = int(num_envs), device, generator
        names = [m[0] for m in mix]
        if not set(names) <= set(SOURCES):
            raise ValueError(f'tracker sources must be among {SOURCES}, got {names}')
        weights = torch.tensor([float(m[1]) for m in mix], device=device)
        if weights.sum() <= 0:
            raise ValueError('tracker source weights must sum to more than 0')
        self.mix_names, self.mix_cdf = names, (weights / weights.sum()).cumsum(0)
        self.random_speed, self.stop_share = tuple(map(float, random_speed)), float(stop_share)
        self.sideways_share, self.crouch_share, self.upper_share = float(sideways_share), float(crouch_share), float(upper_share)
        self.upper_amplitude, self.upper_rate = float(upper_amplitude), float(upper_rate)
        z = lambda *s, dtype=torch.float32: torch.zeros(s, device=device, dtype=dtype)
        self.path = z(self.n, P, 2)                     # world xy
        self.count = z(self.n, dtype=torch.long)        # valid points in path
        self.s = z(self.n)                              # progress, m along the path (from path[0])
        self.speed = z(self.n)                          # m/s
        self.source = z(self.n, dtype=torch.long)       # index into SOURCES
        self.yaw_offset = z(self.n)                     # heading reference = path tangent + offset (random source)
        self.z_ref = torch.full((self.n,), NOMINAL_Z, device=device)
        self.mask_heading, self.mask_height, self.mask_upper = z(self.n), z(self.n), z(self.n)
        self.nominal_upper = (torch.zeros(17, device=device) if nominal_upper is None
                              else torch.as_tensor(nominal_upper, dtype=torch.float32, device=device).reshape(17))
        self.upper_ref, self.upper_goal = z(self.n, 17), z(self.n, 17)
        self.timer = z(self.n)                          # s until the random source re-draws its profiles
        self.curvature = z(self.n)                      # 1/m, random source
        self.lost = z(self.n, dtype=torch.bool)
        self._start_yaw = z(self.n)                     # robot heading at the reference start (random paths)

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape, low=0., high=1.):
        return torch.rand(shape, device=self.device, generator=self.generator) * (high - low) + low

    @property
    def valid(self):
        return self.source != SOURCES.index('none')

    def _point(self, ids, s):
        """Path point at arc length s [n] (linear interpolation, clamped to the last valid point)."""
        u = (s / H).clamp_min(0.)
        top = (self.count[ids] - 1).clamp_min(0).float()
        u = torch.minimum(u, top)
        k0 = u.floor().long(); k1 = torch.minimum(k0 + 1, top.long()); a = (u - k0.float())[:, None]
        p0 = self.path[ids, k0]; p1 = self.path[ids, k1]
        return p0 + a * (p1 - p0)

    def _tangent(self, ids, s):
        u = (s / H).clamp_min(0.); top = (self.count[ids] - 1).clamp_min(1)
        k0 = torch.minimum(u.floor().long(), top - 1); d = self.path[ids, k0 + 1] - self.path[ids, k0]
        return torch.atan2(d[:, 1], d[:, 0])

    def project(self, ids, xy):
        """Arc length of the nearest path point and the distance to it, for robot positions xy [n, 2]."""
        d = torch.linalg.vector_norm(self.path[ids] - xy[:, None], dim=-1)
        k = torch.arange(P, device=self.device)[None]
        d = torch.where(k < self.count[ids][:, None], d, torch.full_like(d, 1e6))
        best = d.argmin(-1)
        return best.float() * H, d.gather(1, best[:, None])[:, 0]

    # ------------------------------------------------------------------ path building
    def _extend_field(self, ids, sample_dir, steps):
        """Append `steps` points along the unit guidance direction sample_dir(xy [n, 2], ids) -> [n, 2]
        (zero where the field ends: the goal seed, or no direction)."""
        for _ in range(int(steps)):
            last = self.path[ids, (self.count[ids] - 1).clamp_min(0)]
            room = self.count[ids] < P
            d = sample_dir(last, ids)
            ok = room & (torch.linalg.vector_norm(d, dim=-1) > .5)
            nxt = last + H * d
            k = self.count[ids].clamp_max(P - 1)
            self.path[ids, k] = torch.where(ok[:, None], nxt, self.path[ids, k])
            self.count[ids] = self.count[ids] + ok.long()

    def _extend_random(self, ids, steps):
        for _ in range(int(steps)):
            c = self.count[ids]; room = c < P
            last = self.path[ids, (c - 1).clamp_min(0)]
            prev = self.path[ids, (c - 2).clamp_min(0)]
            yaw = torch.where(c >= 2, torch.atan2(last[:, 1] - prev[:, 1], last[:, 0] - prev[:, 0]), self._start_yaw[ids])
            yaw = yaw + self.curvature[ids] * H
            nxt = last + H * torch.stack((yaw.cos(), yaw.sin()), -1)
            k = c.clamp_max(P - 1)
            self.path[ids, k] = torch.where(room[:, None], nxt, self.path[ids, k])
            self.count[ids] = c + room.long()

    def _shift(self, ids):
        """Drop the points more than 1 m behind the progress (keeps long episodes inside P points)."""
        drop = ((self.s[ids] - 1.0) / H).floor().long().clamp_min(0)
        if not bool((drop > 0).any()):
            return
        idx = (torch.arange(P, device=self.device)[None] + drop[:, None]).clamp_max(P - 1)
        self.path[ids] = self.path[ids].gather(1, idx[..., None].expand(-1, -1, 2))
        self.count[ids] = (self.count[ids] - drop).clamp_min(1)
        self.s[ids] = self.s[ids] - drop.float() * H

    # ------------------------------------------------------------------ episode and step
    def reset(self, ids, root_xy, root_yaw, *, sample_dir, op_speed, room_scene=None):
        """New references for worlds ids at the robot's pose. op_speed [n]: the operator speed (field source).
        room_scene [n] bool: the random source only runs in open rooms (other scenes fall back to 'field')."""
        if len(ids) == 0:
            return
        n = len(ids)
        pick = torch.searchsorted(self.mix_cdf, self._rand(n)).clamp_max(len(self.mix_names) - 1)
        source = torch.tensor([SOURCES.index(m) for m in self.mix_names], device=self.device)[pick]
        if room_scene is not None:
            source = torch.where((source == SOURCES.index('random')) & ~room_scene, torch.full_like(source, SOURCES.index('field')), source)
        self.source[ids] = source
        self.path[ids] = root_xy[:, None].expand(-1, P, -1)
        self.count[ids] = 1; self.s[ids] = 0.; self.lost[ids] = False
        self._start_yaw[ids] = root_yaw
        self.speed[ids] = torch.where(source == SOURCES.index('field'), op_speed, torch.zeros_like(op_speed))
        self.mask_heading[ids] = 0.; self.mask_height[ids] = 0.; self.mask_upper[ids] = 0.
        self.yaw_offset[ids] = 0.; self.z_ref[ids] = NOMINAL_Z
        self.upper_ref[ids] = self.nominal_upper; self.upper_goal[ids] = self.nominal_upper
        self.timer[ids] = 0.
        rnd = ids[source == SOURCES.index('random')]
        if len(rnd):
            self._redraw(rnd, first=True)
            self._extend_random(rnd, 30)
        fld = ids[source == SOURCES.index('field')]
        if len(fld):
            self._extend_field(fld, sample_dir, 30)

    def _redraw(self, ids, first=False):
        m = len(ids)
        lo, hi = self.random_speed
        speed = self._rand(m, low=lo, high=hi)
        self.speed[ids] = torch.where(self._rand(m) < self.stop_share, torch.zeros_like(speed), speed)
        self.curvature[ids] = torch.where(self._rand(m) < .4, torch.zeros(m, device=self.device), self._rand(m, low=-1.5, high=1.5))
        side = self._rand(m) < self.sideways_share
        self.yaw_offset[ids] = torch.where(side, torch.where(self._rand(m) < .5, torch.full((m,), math.pi / 2, device=self.device),
                                                              torch.full((m,), -math.pi / 2, device=self.device)), torch.zeros(m, device=self.device))
        crouch = self._rand(m) < self.crouch_share
        self.z_ref[ids] = torch.where(crouch, self._rand(m, low=.58, high=.72), torch.full((m,), NOMINAL_Z, device=self.device))
        upper = self._rand(m) < self.upper_share
        offset = self._rand(m, 17, low=-self.upper_amplitude, high=self.upper_amplitude)
        self.upper_goal[ids] = self.nominal_upper + offset * upper[:, None]
        if first:
            self.mask_heading[ids] = 1.; self.mask_height[ids] = 1.; self.mask_upper[ids] = upper.float()
        else:
            self.mask_upper[ids] = torch.maximum(self.mask_upper[ids], upper.float())
        self.timer[ids] = self._rand(m, low=2., high=5.)

    def step(self, root_xy, dt, *, sample_dir, op_speed):
        """Advance every world's progress (the clock waits for a lagging robot), extend paths, re-draw random profiles."""
        ids = torch.arange(self.n, device=self.device)
        valid = self.valid
        s_robot, dist = self.project(ids, root_xy)
        lag = self.s - s_robot
        fld = self.source == SOURCES.index('field')
        self.speed = torch.where(fld, op_speed, self.speed)
        go = valid & (lag < MAX_LAG)
        end = (self.count - 1).float() * H
        self.s = torch.where(go, torch.minimum(self.s + self.speed * dt, end), self.s)
        # upper-body reference moves toward its goal at a limited rate
        step = self.upper_rate * dt
        self.upper_ref = self.upper_ref + (self.upper_goal - self.upper_ref).clamp(-step, step)
        # random source: re-draw profiles when the timer runs out
        rnd = self.source == SOURCES.index('random')
        self.timer = torch.where(rnd, self.timer - dt, self.timer)
        due = torch.nonzero(rnd & (self.timer <= 0), as_tuple=False)[:, 0]
        if len(due):
            self._redraw(due)
        # keep at least ~1.5 m of path ahead of the progress
        self._shift(ids)
        short = ((self.count - 1).float() * H - self.s) < 1.5
        ext_f = torch.nonzero(short & fld, as_tuple=False)[:, 0]
        if len(ext_f):
            self._extend_field(ext_f, sample_dir, 2)
        ext_r = torch.nonzero(short & rnd, as_tuple=False)[:, 0]
        if len(ext_r):
            self._extend_random(ext_r, 2)
        here = self._point(ids, self.s)
        self.lost = valid & (torch.linalg.vector_norm(here - root_xy, dim=-1) > LOST)
        return dict(lag=lag, distance=dist, lead=(s_robot - self.s).clamp_min(0.))

    def reanchor(self, ids, root_xy, root_yaw, *, sample_dir, op_speed):
        """Lost worlds: a fresh reference of the same source from where the robot is."""
        if len(ids) == 0:
            return
        keep = self.source[ids].clone()
        self.path[ids] = root_xy[:, None].expand(-1, P, -1); self.count[ids] = 1; self.s[ids] = 0.
        self._start_yaw[ids] = root_yaw
        rnd = ids[keep == SOURCES.index('random')]
        if len(rnd):
            self._extend_random(rnd, 30)
        fld = ids[keep == SOURCES.index('field')]
        if len(fld):
            self._extend_field(fld, sample_dir, 30)

    # ------------------------------------------------------------------ what the policy sees, and the rewards
    def targets(self, ids, tau):
        s = self.s[ids] + self.speed[ids] * tau
        xy = self._point(ids, s)
        yaw = self._tangent(ids, s) + self.yaw_offset[ids]
        return xy, yaw

    def features(self, ids, root_xy, root_yaw, navi):
        """[n, 41] in the heading frame `navi` [n, 3, 3] (world -> heading frame, as world_to_navi)."""
        n = len(ids); out = []
        valid = self.valid[ids].float()
        out.append(valid[:, None])
        for tau in HORIZONS:
            xy, yaw = self.targets(ids, tau)
            d = torch.cat((xy - root_xy, torch.zeros(n, 1, device=self.device)), -1)
            local = torch.einsum('bij,bi->bj', navi, d)[:, :2]
            dyaw = wrap(yaw - root_yaw) * self.mask_heading[ids]
            dz = (self.z_ref[ids] - NOMINAL_Z) * self.mask_height[ids]
            block = torch.cat((local, dyaw.sin()[:, None], dyaw.cos()[:, None], dz[:, None]), -1)
            out.append(block * valid[:, None])
        out.append((self.mask_heading[ids] * valid)[:, None]); out.append((self.mask_height[ids] * valid)[:, None])
        upper = (self.upper_ref[ids] - self.nominal_upper) * (self.mask_upper[ids] * valid)[:, None]
        out.append(upper); out.append((self.mask_upper[ids] * valid)[:, None])
        f = torch.cat(out, -1)
        assert f.shape[-1] == FEATURES
        return f

    def rewards(self, root_xy, root_yaw, root_z, q_upper, info, *, sigma=.15):
        """Per-world reward terms (unscaled). info: the dict step() returned."""
        ids = torch.arange(self.n, device=self.device)
        valid = self.valid.float()
        here = self._point(ids, self.s)
        d = torch.linalg.vector_norm(root_xy - here, dim=-1)
        yaw_ref = self._tangent(ids, self.s) + self.yaw_offset
        e_yaw = wrap(root_yaw - yaw_ref)
        e_up = (q_upper - self.upper_ref).square().mean(-1)
        return dict(
            track_pos=torch.exp(-(d / sigma).square()) * valid,
            track_lead=(info['lead'] - .1).clamp_min(0.) * valid,
            track_heading=torch.exp(-(e_yaw / .3).square()) * self.mask_heading * valid,
            track_height=torch.exp(-((root_z - self.z_ref) / .05).square()) * self.mask_height * valid,
            track_upper=torch.exp(-e_up / .2 ** 2) * self.mask_upper * valid,
        ), dict(track_error_m=d * valid, track_valid=valid, track_lost=self.lost.float())
