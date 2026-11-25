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
    'plan'    planner v1 in the loop (cat_mjlab/planner/online.py, called by the task): 2 s whole-body chunks from the
              robot's measured state toward a goal 2 s ahead on the guidance path, replanned every 0.3-1.0 s. The chunk
              is time-indexed (PLAN_FRAMES frames every PLAN_DT, the start pose first) with heading, pelvis height and
              upper-body targets, and its own clock that waits for a lagging robot. The guidance path is still built
              for these worlds: it is where the planner's goal comes from. Reactive standing worlds can get a
              stand-still plan instead of no reference (hold_plan).
Stored rollout windows (planner data pose field) plug in as more sources later: they set the same tensors.

The reflex rule (rewards): an arm whose hand is inside the reflex band of an obstacle (the task passes `released`) drops
out of the upper-body tracking term, and the pelvis term tolerates `deadband` m of error near obstacles; the potential-
field and contact terms are untouched, so near obstacles the field acts and away from them the reference does.

fallback (the monitor's stall rule): a plan world whose robot makes less than `min_speed` m/s of progress along the
route for `stall_s` s while the route says go drops its plan for `hold_s` s -- the same network then acts on the
potential field alone, as without a planner -- and replans afterwards. Measured motivation: without a plan the
planner-reflex policy passes narrow cabinet passages as the reactive policy does (0.83 clean), with plans it refuses
in 43% of them.

continuity (COMBO4): a replan continues the executing plan instead of restarting it at the robot. The new chunk is
planned from the plan's own state at the current plan time (its pose, joints and velocities, see plan_state), toward a
goal taken ahead of the plan's own route progress, and its first frame is that state. A robot that stalls therefore
falls behind a reference that keeps moving (standing still keeps about 20% of track_pos at 0.15 m/s instead of 87-91%);
only a robot more than `reanchor` m from the plan's current point gets a plan from its measured state. Measured
motivation: plans for a standing robot start slowly (4 cm in their first 0.3 s against 8 cm elsewhere), and replans
every 0.3-1.0 s restarted them at the robot, so only the slow start was ever played (HALO and PASSAGE continue the
executing plan instead).

commit=True (COMBO2): a plan's clock no longer waits for a lagging robot, and the deadband forgives only the sideways
part of the pelvis error, not lag along the reference. Measured on COMBO1 in a hurdle course where it stalled for 60 s:
replans restart the chunk at the robot and the clock waited, so the reference point stayed 4 cm away and standing still
earned 91% of track_pos; with the deadband near obstacles, falling behind there was free as well.

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
SOURCES = ('field', 'random', 'none', 'plan')
PLAN_FRAMES = 21                          # the start pose plus the planner's 20 frames
PLAN_DT = 0.1                             # s between plan frames
UPPER_GROUPS = ((0, 3), (3, 10), (10, 17))    # waist, left arm, right arm in the 17 upper-body joints


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
                 upper_rate=1.0, nominal_upper=None, hold_plan=0., commit=False, fallback=None, continuity=None,
                 generator=None, stall_plan=None):
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
        self.hold_plan = float(hold_plan)
        self.commit = bool(commit)
        self.fallback = dict(fallback) if fallback else None
        self.continuity = dict(continuity) if continuity else None
        # Stall-triggered planning (eval-only): plan worlds act without a reference by default; after stall_s s without
        # progress_m of route progress the plan is engaged for hold_s s (a fresh chunk), then dropped again.
        self.stall_plan = dict(stall_plan) if stall_plan else None
        if self.stall_plan and self.fallback:
            raise ValueError('stall_plan and fallback are exclusive')
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
        # plan source: time-indexed chunk in the world frame
        self.plan_xy = z(self.n, PLAN_FRAMES, 2)
        self.plan_yaw = z(self.n, PLAN_FRAMES)
        self.plan_z = torch.full((self.n, PLAN_FRAMES), NOMINAL_Z, device=device)
        self.plan_upper = z(self.n, PLAN_FRAMES, 17)
        self.plan_roll, self.plan_pitch = z(self.n, PLAN_FRAMES), z(self.n, PLAN_FRAMES)
        self.plan_legs = z(self.n, PLAN_FRAMES, 12)       # leg joints: kept only to continue the plan (never tracked)
        self.has_full = z(self.n, dtype=torch.bool)     # the chunk carries roll, pitch and legs (continuity)
        self.plan_t = z(self.n)                         # s since the chunk started
        self.replan_in = z(self.n)                      # s until the next replan
        self.has_plan = z(self.n, dtype=torch.bool)
        self.still = z(self.n, dtype=torch.bool)        # stand-still plans (reactive standing worlds)
        self.route_speed = z(self.n)                    # operator speed along the guidance path (plan goals)
        self.stall_t = z(self.n)                        # s without progress along the route (fallback rule)
        self.fallback_t = z(self.n)                     # s left without a plan (fallback rule)
        self.prev_s = z(self.n)
        self.progress_mark = z(self.n)
        self.plan_retry = z(self.n)
        self.plan_on_t = z(self.n)                      # s of engaged plan left (stall_plan rule)

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape, low=0., high=1.):
        return torch.rand(shape, device=self.device, generator=self.generator) * (high - low) + low

    @property
    def valid(self):
        ok = ((self.source != SOURCES.index('none')) & ((self.source != SOURCES.index('plan')) | self.has_plan)
              & (self.fallback_t <= 0))
        if self.stall_plan:
            ok = ok & (~self.is_plan | self.still | (self.plan_on_t > 0))
        return ok

    @property
    def is_plan(self):
        return self.source == SOURCES.index('plan')

    def plan_at(self, ids, t):
        """Plan pose at plan time t [n] (s, clamped to the chunk): xy [n, 2], yaw [n], z [n], upper [n, 17]."""
        u = (t / PLAN_DT).clamp(0., PLAN_FRAMES - 1.)
        k0 = u.floor().long().clamp_max(PLAN_FRAMES - 2); a = u - k0.float()
        def lerp(x):
            x0, x1 = x[ids, k0], x[ids, k0 + 1]
            w = a.reshape(-1, *([1] * (x0.dim() - 1)))
            return x0 + w * (x1 - x0)
        y0, y1 = self.plan_yaw[ids, k0], self.plan_yaw[ids, k0 + 1]
        return lerp(self.plan_xy), y0 + a * wrap(y1 - y0), lerp(self.plan_z), lerp(self.plan_upper)

    def set_plan(self, ids, xy, yaw, z, upper, interval, roll=None, pitch=None, legs=None):
        """A new chunk for worlds ids (world frame, PLAN_FRAMES frames from now): xy [m, F, 2], yaw/z [m, F],
        upper [m, F, 17]; the next replan comes after interval [m] s. roll/pitch [m, F] and legs [m, F, 12] complete the
        body state so that a later replan can continue the plan (continuity)."""
        if len(ids) == 0:
            return
        self.plan_xy[ids], self.plan_yaw[ids], self.plan_z[ids], self.plan_upper[ids] = xy, yaw, z, upper
        full = roll is not None and pitch is not None and legs is not None
        if full:
            self.plan_roll[ids], self.plan_pitch[ids], self.plan_legs[ids] = roll, pitch, legs
        self.has_full[ids] = full
        self.plan_t[ids] = 0.; self.replan_in[ids] = interval; self.has_plan[ids] = True
        self.mask_heading[ids] = 1.; self.mask_height[ids] = 1.; self.mask_upper[ids] = 1.

    def due(self):
        """Plan worlds that need a new chunk: no chunk yet, or their replan time has come."""
        due = self.is_plan & (self.fallback_t <= 0) & (self.plan_retry <= 0) & (~self.has_plan | (self.replan_in <= 0))
        if self.stall_plan:
            due = due & (self.still | (self.plan_on_t > 0))
        return torch.nonzero(due, as_tuple=False)[:, 0]

    def route_goal(self, ids, horizon, from_xy=None):
        """The point horizon s ahead of the robot's progress along the guidance path (or, with from_xy [n, 2], of that
        point's progress), at the route speed: xy [n, 2], path heading there [n]. Stand-still worlds should ignore it."""
        s0 = self.s[ids] if from_xy is None else self.project(ids, from_xy)[0]
        s = s0 + self.route_speed[ids] * horizon
        return self._point(ids, s), self._tangent(ids, s)

    def _interp(self, x, ids, t):
        """Frame-interpolated value of a plan buffer x [n, F, ...] at plan time t [m] for worlds ids."""
        u = (t / PLAN_DT).clamp(0., PLAN_FRAMES - 1.)
        k0 = u.floor().long().clamp_max(PLAN_FRAMES - 2); a = u - k0.float()
        x0, x1 = x[ids, k0], x[ids, k0 + 1]
        return x0 + a.reshape(-1, *([1] * (x0.dim() - 1))) * (x1 - x0)

    def continuing(self, ids, root_xy):
        """Worlds among ids whose next chunk continues the executing plan: plan source with a full chunk, and the robot
        within `reanchor` m of the plan's current point (otherwise the plan restarts from the robot)."""
        if self.continuity is None or len(ids) == 0:
            return torch.zeros(len(ids), dtype=torch.bool, device=self.device)
        here = self.plan_at(ids, self.plan_t[ids])[0]
        near = torch.linalg.vector_norm(root_xy - here, dim=-1) < float(self.continuity.get('reanchor', .5))
        return self.is_plan[ids] & self.has_plan[ids] & self.has_full[ids] & near

    def plan_state(self, ids):
        """The executing plan's state at the current plan time as MuJoCo-style qpos [m, 36] and qvel [m, 35] (free root:
        world position, quaternion w-x-y-z, world linear velocity, body angular velocity), velocities by finite
        difference over the next 0.1 s of the chunk. Leg joints come first, then the 17 upper-body joints."""
        t0 = self.plan_t[ids]
        t1 = (t0 + PLAN_DT).clamp_max((PLAN_FRAMES - 1) * PLAN_DT)
        t0 = torch.minimum(t0, t1 - 1e-3)
        dt = (t1 - t0).clamp_min(1e-3)
        def at(t):
            xy, yaw, zz, up = self.plan_at(ids, t)
            return xy, yaw, zz, self._interp(self.plan_roll, ids, t), self._interp(self.plan_pitch, ids, t), \
                torch.cat((self._interp(self.plan_legs, ids, t), up), -1)
        xy0, y0, z0, r0, p0, q0 = at(t0)
        xy1, y1, z1, r1, p1, q1 = at(t1)
        cy, sy, cp, sp, cr, sr = (torch.cos(y0 / 2), torch.sin(y0 / 2), torch.cos(p0 / 2), torch.sin(p0 / 2),
                                  torch.cos(r0 / 2), torch.sin(r0 / 2))
        quat = torch.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1)
        qpos = torch.cat((xy0, z0[:, None], quat, q0), -1)
        lin = torch.cat(((xy1 - xy0) / dt[:, None], ((z1 - z0) / dt)[:, None]), -1)
        ang = torch.stack(((r1 - r0) / dt, (p1 - p0) / dt, wrap(y1 - y0) / dt), -1)
        qvel = torch.cat((lin, ang, (q1 - q0) / dt[:, None]), -1)
        return qpos, qvel

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
    def reset(self, ids, root_xy, root_yaw, *, sample_dir, op_speed, room_scene=None, hold=None):
        """New references for worlds ids at the robot's pose. op_speed [n]: the operator speed (field source).
        room_scene [n] bool: the random source only runs in open rooms (other scenes fall back to 'field').
        hold [n] bool: no reference for these worlds (reactive standing episodes keep the standing skill)."""
        if len(ids) == 0:
            return
        n = len(ids)
        pick = torch.searchsorted(self.mix_cdf, self._rand(n)).clamp_max(len(self.mix_names) - 1)
        source = torch.tensor([SOURCES.index(m) for m in self.mix_names], device=self.device)[pick]
        if room_scene is not None:
            source = torch.where((source == SOURCES.index('random')) & ~room_scene, torch.full_like(source, SOURCES.index('field')), source)
        still = torch.zeros(n, dtype=torch.bool, device=self.device)
        if hold is not None:
            still = hold & (self._rand(n) < self.hold_plan)
            source = torch.where(hold, torch.where(still, torch.full_like(source, SOURCES.index('plan')),
                                                   torch.full_like(source, SOURCES.index('none'))), source)
        self.source[ids] = source; self.still[ids] = still
        self.has_plan[ids] = False; self.has_full[ids] = False; self.plan_t[ids] = 0.; self.replan_in[ids] = 0.
        self.stall_t[ids] = 0.; self.fallback_t[ids] = 0.
        self.progress_mark[ids] = 0.; self.plan_retry[ids] = 0.; self.prev_s[ids] = 0.; self.plan_on_t[ids] = 0.
        self.route_speed[ids] = torch.where(still, torch.zeros_like(op_speed), op_speed)
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
        route = ids[(source == SOURCES.index('plan')) & ~still]
        if len(route):
            self._extend_field(route, sample_dir, 50)

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
        plan = self.is_plan
        self.speed = torch.where(fld, op_speed, self.speed)
        self.route_speed = torch.where(plan & ~self.still, op_speed, self.route_speed)
        go = valid & ~plan & (lag < MAX_LAG)
        end = (self.count - 1).float() * H
        self.s = torch.where(go, torch.minimum(self.s + self.speed * dt, end), self.s)
        # plan worlds: route progress is the robot's own projection (the goal is taken ahead of it); the plan clock
        # advances unless the robot is MAX_LAG behind the plan point and nearer an earlier one
        self.s = torch.where(plan, torch.minimum(s_robot, end), self.s)
        plan_lead = torch.zeros_like(self.s)
        if bool(plan.any()):
            here_xy, _, _, _ = self.plan_at(ids, self.plan_t)
            off = root_xy - here_xy
            gap = torch.linalg.vector_norm(off, dim=-1)
            t_robot = torch.linalg.vector_norm(self.plan_xy - root_xy[:, None], dim=-1).argmin(-1).float() * PLAN_DT
            behind = (gap > MAX_LAG) & (t_robot < self.plan_t - .15)
            run = plan & self.has_plan & (~behind | self.commit)
            self.plan_t = torch.where(run, (self.plan_t + dt).clamp_max((PLAN_FRAMES - 1) * PLAN_DT), self.plan_t)
            self.replan_in = torch.where(plan, self.replan_in - dt, self.replan_in)
            ahead_xy, _, _, _ = self.plan_at(ids, self.plan_t + .3)
            motion = ahead_xy - here_xy; length = torch.linalg.vector_norm(motion, dim=-1)
            self.speed = torch.where(plan, length / .3, self.speed)
            along = (off * motion).sum(-1) / length.clamp_min(1e-6)
            plan_lead = torch.where(plan & self.has_plan & (length > .03), along.clamp_min(0.), plan_lead)
        advance = self.s - self.prev_s
        self.plan_retry = (self.plan_retry - dt).clamp_min(0.)
        if self.stall_plan:
            sp = self.stall_plan
            moving = plan & ~self.still & (self.route_speed > .05)
            active = self.plan_on_t > 0.
            self.progress_mark = torch.where(moving & ~active, self.progress_mark + advance, 0.)
            progressed = self.progress_mark >= float(sp.get('progress_m', .05))
            self.stall_t = torch.where(moving & ~active & ~progressed, self.stall_t + dt, 0.)
            self.progress_mark = torch.where(progressed, 0., self.progress_mark)
            start = self.stall_t >= float(sp.get('stall_s', 2.)) - 1e-5
            ended = active & (self.plan_on_t - dt <= 0.)
            self.plan_on_t = torch.where(start, float(sp.get('hold_s', 4.)), (self.plan_on_t - dt).clamp_min(0.))
            self.stall_t = torch.where(start, 0., self.stall_t)
            self.has_plan &= ~(start | ended)              # a fresh chunk on engagement; none after release
        elif self.fallback and self.fallback.get('until_progress'):
            was = self.fallback_t > 0.
            moving = plan & ~self.still & (self.route_speed > .05) & (self.has_plan | was)
            self.progress_mark = torch.where(moving, self.progress_mark + advance, 0.)
            resumed = self.progress_mark >= float(self.fallback.get('progress_m', .05))
            self.stall_t = torch.where(moving & ~was & ~resumed, self.stall_t + dt, 0.)
            start = self.stall_t >= float(self.fallback.get('stall_s', 2.)) - 1e-5
            self.fallback_t = (start | (was & ~resumed)).float()
            self.has_plan &= ~start
            self.stall_t = torch.where(start, 0., self.stall_t)
            self.progress_mark = torch.where(start | resumed, 0., self.progress_mark)
        elif self.fallback is not None and bool(plan.any()):
            fb = self.fallback
            moving = plan & ~self.still & (self.route_speed > .05)
            progress = (self.s - self.prev_s) / dt
            was = self.fallback_t > 0
            stalled = moving & (progress < float(fb.get('min_speed', .03))) & ~was
            self.stall_t = torch.where(stalled, self.stall_t + dt, torch.zeros_like(self.stall_t))
            start = self.stall_t > float(fb.get('stall_s', 2.))
            self.fallback_t = torch.where(start, torch.full_like(self.fallback_t, float(fb.get('hold_s', 4.))),
                                          (self.fallback_t - dt).clamp_min(0.))
            self.stall_t = torch.where(start, torch.zeros_like(self.stall_t), self.stall_t)
            ended = plan & was & (self.fallback_t <= 0)
            self.replan_in = torch.where(ended, torch.zeros_like(self.replan_in), self.replan_in)
            self.has_plan = torch.where(start, torch.zeros_like(self.has_plan), self.has_plan)
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
        self.prev_s = self.s.clone()            # after the shift, so progress is measured in the same frame
        reach = torch.where(plan, torch.full_like(self.s, 2.5), torch.full_like(self.s, 1.5))
        short = ((self.count - 1).float() * H - self.s) < reach
        ext_f = torch.nonzero(short & (fld | (plan & ~self.still)), as_tuple=False)[:, 0]
        if len(ext_f):
            self._extend_field(ext_f, sample_dir, 2)
        ext_r = torch.nonzero(short & rnd, as_tuple=False)[:, 0]
        if len(ext_r):
            self._extend_random(ext_r, 2)
        here = self._point(ids, self.s)
        if bool(plan.any()):
            here = torch.where(plan[:, None], self.plan_at(ids, self.plan_t)[0], here)
        self.lost = valid & (torch.linalg.vector_norm(here - root_xy, dim=-1) > LOST)
        lead = torch.where(plan, plan_lead, (s_robot - self.s).clamp_min(0.))
        return dict(lag=lag, distance=dist, lead=lead, route_advance=advance,
                    route_moving=plan & ~self.still & (self.route_speed > .05))

    def reanchor(self, ids, root_xy, root_yaw, *, sample_dir, op_speed):
        """Lost worlds: a fresh reference of the same source from where the robot is."""
        if len(ids) == 0:
            return
        keep = self.source[ids].clone()
        self.path[ids] = root_xy[:, None].expand(-1, P, -1); self.count[ids] = 1; self.s[ids] = 0.; self.prev_s[ids] = 0.
        self._start_yaw[ids] = root_yaw
        rnd = ids[keep == SOURCES.index('random')]
        if len(rnd):
            self._extend_random(rnd, 30)
        fld = ids[keep == SOURCES.index('field')]
        if len(fld):
            self._extend_field(fld, sample_dir, 30)
        route = ids[(keep == SOURCES.index('plan')) & ~self.still[ids]]
        if len(route):
            self._extend_field(route, sample_dir, 50)
        self.replan_in[ids] = torch.where(keep == SOURCES.index('plan'), torch.zeros_like(self.replan_in[ids]), self.replan_in[ids])

    # ------------------------------------------------------------------ what the policy sees, and the rewards
    def targets(self, ids, tau):
        s = self.s[ids] + self.speed[ids] * tau
        xy = self._point(ids, s)
        yaw = self._tangent(ids, s) + self.yaw_offset[ids]
        plan = self.is_plan[ids]
        if bool(plan.any()):
            pxy, pyaw, _, _ = self.plan_at(ids, self.plan_t[ids] + tau)
            xy = torch.where(plan[:, None], pxy, xy); yaw = torch.where(plan, pyaw, yaw)
        return xy, yaw

    def features(self, ids, root_xy, root_yaw, navi):
        """[n, 41] in the heading frame `navi` [n, 3, 3] (world -> heading frame, as world_to_navi)."""
        n = len(ids); out = []
        valid = self.valid[ids].float()
        out.append(valid[:, None])
        plan = self.is_plan[ids]
        for tau in HORIZONS:
            xy, yaw = self.targets(ids, tau)
            d = torch.cat((xy - root_xy, torch.zeros(n, 1, device=self.device)), -1)
            local = torch.einsum('bij,bi->bj', navi, d)[:, :2]
            dyaw = wrap(yaw - root_yaw) * self.mask_heading[ids]
            z_ref = self.z_ref[ids]
            if bool(plan.any()):
                z_ref = torch.where(plan, self.plan_at(ids, self.plan_t[ids] + tau)[2], z_ref)
            dz = (z_ref - NOMINAL_Z) * self.mask_height[ids]
            block = torch.cat((local, dyaw.sin()[:, None], dyaw.cos()[:, None], dz[:, None]), -1)
            out.append(block * valid[:, None])
        out.append((self.mask_heading[ids] * valid)[:, None]); out.append((self.mask_height[ids] * valid)[:, None])
        ahead = self.upper_rate * UPPER_HORIZON   # rad the rate-limited target moves in UPPER_HORIZON
        target = self.upper_ref[ids] + (self.upper_goal[ids] - self.upper_ref[ids]).clamp(-ahead, ahead)
        if bool(plan.any()):
            target = torch.where(plan[:, None], self.plan_at(ids, self.plan_t[ids] + UPPER_HORIZON)[3], target)
        upper = (target - self.nominal_upper) * (self.mask_upper[ids] * valid)[:, None]
        out.append(upper); out.append((self.mask_upper[ids] * valid)[:, None])
        f = torch.cat(out, -1)
        assert f.shape[-1] == FEATURES
        return f

    def rewards(self, root_xy, root_yaw, root_z, q_upper, info, *, sigma=.15, released=None, deadband=None):
        """Per-world reward terms (unscaled). info: the dict step() returned. The reflex rule: released [n, 2] bool
        (left, right hand inside the reflex band) drops that arm from track_upper, which then pays only for the
        groups still tracked (no reward for being released); deadband [n] m is subtracted from the pelvis error."""
        ids = torch.arange(self.n, device=self.device)
        valid = self.valid.float()
        here = self._point(ids, self.s)
        yaw_ref = self._tangent(ids, self.s) + self.yaw_offset
        z_ref, upper_ref = self.z_ref, self.upper_ref
        plan = self.is_plan
        if bool(plan.any()):
            pxy, pyaw, pz, pup = self.plan_at(ids, self.plan_t)
            here = torch.where(plan[:, None], pxy, here); yaw_ref = torch.where(plan, pyaw, yaw_ref)
            z_ref = torch.where(plan, pz, z_ref); upper_ref = torch.where(plan[:, None], pup, upper_ref)
        d = torch.linalg.vector_norm(root_xy - here, dim=-1)
        d_paid = d if deadband is None else (d - deadband).clamp_min(0.)
        if deadband is not None and self.commit:
            # the deadband forgives sideways error only: lag along the reference's direction of motion stays paid
            tangent = self._tangent(ids, self.s)
            direction = torch.stack((tangent.cos(), tangent.sin()), -1)
            moving = ~plan
            if bool(plan.any()):
                motion = self.plan_at(ids, self.plan_t + .3)[0] - here
                length = torch.linalg.vector_norm(motion, dim=-1)
                direction = torch.where(plan[:, None], motion / length.clamp_min(1e-6)[:, None], direction)
                moving = moving | (length > .03)
            off = root_xy - here
            along = (off * direction).sum(-1)
            cross = torch.linalg.vector_norm(off - along[:, None] * direction, dim=-1)
            split = torch.sqrt(along.square() + (cross - deadband).clamp_min(0.).square())
            d_paid = torch.where(moving, split, d_paid)
        e_yaw = wrap(root_yaw - yaw_ref)
        err = (q_upper - upper_ref).square()
        upper = torch.zeros_like(d)
        for g, (a, b) in enumerate(UPPER_GROUPS):
            keep = torch.ones_like(d) if (released is None or g == 0) else (~released[:, g - 1]).float()
            upper = upper + (b - a) / 17 * keep * torch.exp(-err[:, a:b].mean(-1) / .2 ** 2)
        return dict(
            track_pos=torch.exp(-(d_paid / sigma).square()) * valid,
            track_lead=(info['lead'] - .1).clamp_min(0.) * valid,
            track_heading=torch.exp(-(e_yaw / .3).square()) * self.mask_heading * valid,
            track_height=torch.exp(-((root_z - z_ref) / .05).square()) * self.mask_height * valid,
            track_upper=upper * self.mask_upper * valid,
        ), dict(track_error_m=d * valid, track_valid=valid, track_lost=self.lost.float(),
                track_plan=(plan & self.has_plan).float(), track_fallback=((self.fallback_t > 0) | (self.plan_retry > 0)).float(),
                track_released=(torch.zeros_like(d) if released is None else released.float().mean(-1)) * valid)
