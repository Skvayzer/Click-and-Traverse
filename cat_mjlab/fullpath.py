"""Oracle full-path planning on a known scene (S2 of docs/fullpath_plan.md) and its feed into the tracker's plan source.

Posture library (ClipLibrary): 0.4 s whole-body clips (21 frames at 50 Hz) cut every 0.1 s from clean traversals of our own
policies (scripts/fullpath_rollout.py recordings). Each clip is stored in its start frame (pelvis floor projection, x along
the pelvis heading) with a start feature for continuity (leg and arm joints, pelvis height / roll / pitch, local velocity)
and mode labels: sideways (heading vs motion direction > 45 deg), crouch (pelvis below 0.70 m), high step (a foot more than
12 cm above the floor), hands raised (a hand above 1.05 m) / tucked (both hands within 25 cm of the pelvis axis).

Search (plan_scene): weighted A* over clip sequences. A node is the end state of a clip placed at the previous end's pelvis
frame; its successors are the k nearest library clips by start feature (cKDTree), each placed there and accepted only if
the exact-geometry separation (cat_mjlab/exact_geometry.py: the 35 proxy primitives against every collision box of the
scene) is at least `margin` at every frame. g = time (0.4 s per clip) + lateral penalty; h = route distance left / 0.6 m/s.
The closed set is keyed by (x, y, yaw, leg phase) cells. The result is a dense start->goal qpos trajectory at 50 Hz with its
mode schedule, foot positions, and its end-to-end exact minimum separation (checked again at every frame).

Feed (OracleFeeder): replaces task._replan for the single CPU world. Every 0.1 s it finds the oracle time whose pelvis is
nearest the robot (searching forward from the last match), and sets a 21-frame chunk from there, re-timed to the operator
speed (distance-indexed: the reference never runs ahead of the robot's own progress along the path), with pelvis
waypoints, heading, height and upper-body targets -- exactly the inputs the current network gets from planner v1.
"""
from __future__ import annotations

import heapq
import json
import math
import time

import numpy as np
import torch

CLIP = 20                # steps per clip (0.4 s at 50 Hz)
STRIDE = 5               # library windows every 0.1 s
DT = 0.02


def yaw_of(q):
    w, x, y, z = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_yaw_mul(q, dyaw):
    """Rotate MuJoCo quaternions q [..., 4] by a world-z rotation dyaw [...] (left multiplication)."""
    c, s = np.cos(dyaw / 2), np.sin(dyaw / 2)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack((c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w), -1)


def to_local(traj):
    """qpos [n, 36] -> the same motion in the frame of its first pelvis pose (floor projection, heading)."""
    x0, y0, a0 = traj[0, 0], traj[0, 1], yaw_of(traj[0])
    c, s = math.cos(a0), math.sin(a0)
    out = traj.copy()
    dx, dy = traj[:, 0] - x0, traj[:, 1] - y0
    out[:, 0], out[:, 1] = c * dx + s * dy, -s * dx + c * dy
    out[:, 3:7] = quat_yaw_mul(traj[:, 3:7], np.full(len(traj), -a0))
    return out


def place(local, x0, y0, a0):
    """Inverse of to_local for a clip [.., n, 36] placed at pelvis frame (x0, y0, a0)."""
    c, s = math.cos(a0), math.sin(a0)
    out = local.copy()
    out[..., 0] = x0 + c * local[..., 0] - s * local[..., 1]
    out[..., 1] = y0 + s * local[..., 0] + c * local[..., 1]
    out[..., 3:7] = quat_yaw_mul(local[..., 3:7], np.full(local.shape[:-1], a0))
    return out


def roll_pitch(q):
    w, x, y, z = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


def feature(seq):
    """Continuity feature of the state at seq[-1] (needs >= 3 frames): legs, arms/waist (x0.5), pelvis z (x2), roll, pitch,
    local planar velocity (x0.5), yaw rate (x0.2)."""
    q = seq[-1]
    loc = to_local(seq[-3:])
    v = (loc[-1, :2] - loc[0, :2]) / (2 * DT)
    # velocity in the frame of the last pose
    a = yaw_of(seq[-1]) - yaw_of(seq[-3])
    c, s = math.cos(-a), math.sin(-a)
    v = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
    r, p = roll_pitch(q)
    yr = (yaw_of(seq[-1]) - yaw_of(seq[-2])) / DT
    return np.concatenate((q[7:19], .5 * q[19:36], [2 * q[2], r, p], .5 * v, [.2 * yr])).astype(np.float32)


class ClipLibrary:
    def __init__(self, clips, feats, modes, sources):
        from scipy.spatial import cKDTree
        self.clips, self.feats, self.modes, self.sources = clips, feats, modes, sources
        self.tree = cKDTree(feats)
        # end features and end-frame displacement for quick successor placement
        self.end_feats = np.stack([feature(c[-3:]) for c in clips]) if len(clips) else np.zeros((0, feats.shape[1]))

    @staticmethod
    def build(recordings, geometry, *, max_clips=60000, rng=0):
        clips, feats, modes, sources = [], [], [], []
        for path in recordings:
            d = np.load(path)
            meta = json.loads(str(d["meta"]))
            if meta["outcome"]["result"] != "clean":
                continue
            q = d["qpos"].astype(np.float64)
            for a in range(25, len(q) - CLIP - 1, STRIDE):
                seq = q[a - 2:a + CLIP + 1]
                local = to_local(seq[2:])
                clips.append(local.astype(np.float32))
                feats.append(feature(seq[:3]))
                sources.append((meta["scene"], meta["speed"], meta["trial"], meta["mix"], int(a)))
        clips = np.stack(clips); feats = np.stack(feats)
        if len(clips) > max_clips:
            keep = np.random.default_rng(rng).choice(len(clips), max_clips, replace=False)
            clips, feats, sources = clips[keep], feats[keep], [sources[i] for i in keep]
        modes = clip_modes(clips, geometry)
        return ClipLibrary(clips, feats, modes, sources)

    def save(self, path):
        np.savez_compressed(path, clips=self.clips, feats=self.feats, modes=self.modes,
                            sources=np.array([json.dumps(s) for s in self.sources]))

    @staticmethod
    def load(path):
        d = np.load(path)
        return ClipLibrary(d["clips"], d["feats"], d["modes"], [json.loads(s) for s in d["sources"]])


MODE_NAMES = ("sideways", "crouch", "high_step", "hands_raised", "hands_tucked")


def clip_modes(clips, geometry):
    """[n, 5] bool mode labels of local clips [n, 21, 36]."""
    n = len(clips)
    flat = clips.reshape(-1, 36)
    out = []
    for a in range(0, len(flat), 20000):
        p, h, f = geometry.keypoints(flat[a:a + 20000])
        out.append((p.numpy(), h.numpy(), f.numpy()))
    p = np.concatenate([o[0] for o in out]).reshape(n, -1, 3); h = np.concatenate([o[1] for o in out]).reshape(n, -1, 2, 3)
    f = np.concatenate([o[2] for o in out]).reshape(n, -1, 2, 3)
    motion = clips[:, -1, :2] - clips[:, 0, :2]
    heading = yaw_of(clips[:, CLIP // 2])
    ang = np.abs(np.angle(np.exp(1j * (np.arctan2(motion[:, 1], motion[:, 0]) - heading))))
    moving = np.linalg.norm(motion, axis=-1) > .02
    sideways = moving & (ang > math.pi / 4) & (ang < 3 * math.pi / 4)
    crouch = clips[:, :, 2].min(1) < .70
    high = f[..., 2].max((1, 2)) > .12
    raised = h[..., 2].max((1, 2)) > 1.05
    lateral = np.linalg.norm(h[..., :2] - p[:, :, None, :2], axis=-1)
    tucked = lateral.max((1, 2)) < .25
    return np.stack((sideways, crouch, high, raised, tucked), -1)


def plan_scene(lib, geometry, boxes, start_qpos, goal_xy, route_dir, *, margin=.05, k=32, max_feat_dist=1.5, eps=4.0,
               budget_s=120., max_expansions=6000, lateral_w=.5, goal_radius=None, goal_beyond=.15, v_cap=.6,
               corridor=None, bounds=None, bound_pad=.15, verbose=False):
    """Weighted A* over clip sequences from start_qpos [36] to past the goal along route_dir [2] (or within goal_radius of
    goal_xy). Returns dict(ok, qpos [n, 36] dense 50 Hz, clip ids, modes [n_clips, 5], min_sep, expansions, seconds)."""
    t0 = time.time()
    u = np.asarray(route_dir, dtype=np.float64); u = u / np.linalg.norm(u)
    nrm = np.array([-u[1], u[0]])
    start = np.asarray(start_qpos, dtype=np.float64)
    s0 = float(start[:2] @ u)
    goal_s = float(np.asarray(goal_xy) @ u)
    line0 = float(start[:2] @ nrm)
    centers = np.asarray(boxes["box_center"])
    sub = lambda xy: {kk: np.asarray(boxes[kk])[np.linalg.norm(centers[:, :2] - xy, axis=-1) - np.linalg.norm(np.asarray(boxes['box_half'])[:, :2], axis=-1) < 1.4] for kk in ("box_center", "box_rot", "box_half")} \
        if len(centers) else boxes

    def done(q):
        if goal_radius is not None:
            return np.linalg.norm(q[:2] - goal_xy) <= goal_radius
        return float(q[:2] @ u) >= goal_s + goal_beyond

    def h(q):
        if goal_radius is not None:
            return max(0., np.linalg.norm(q[:2] - goal_xy) - goal_radius) / v_cap
        return max(0., goal_s + goal_beyond - float(q[:2] @ u)) / v_cap

    def key(q):
        a = yaw_of(q)
        phase = int(q[7 + 0] > q[7 + 6])               # left vs right hip pitch: which leg leads
        return (int(round(q[0] / .04)), int(round(q[1] / .04)), int(round(a / (math.pi / 18))), phase)

    # start: a 3-frame history at rest
    hist0 = np.stack([start, start, start])
    nodes = [dict(q=start, feat=feature(hist0), parent=-1, clip=-1, g=0., traj=None)]
    heap = [(eps * h(start), 0)]
    closed = set()
    expansions = 0
    best_i, best_h = 0, h(start)
    min_seen = []
    while heap and expansions < max_expansions and time.time() - t0 < budget_s:
        f, i = heapq.heappop(heap)
        n = nodes[i]
        kq = key(n["q"])
        if kq in closed:
            continue
        closed.add(kq)
        if done(n["q"]):
            best_i = i
            break
        expansions += 1
        dist, idx = lib.tree.query(n["feat"], k=k)
        ok = np.isfinite(dist) & (dist < max_feat_dist)
        idx = idx[ok]
        if not len(idx):
            continue
        q = n["q"]; x0, y0, a0 = q[0], q[1], yaw_of(q)
        placed = place(lib.clips[idx].astype(np.float64), x0, y0, a0)           # [m, 21, 36]
        # z from the clip (absolute floor height); joints from the clip
        near_boxes = sub(q[:2])
        # two stages: 4 frames first (most rejections), then every 2nd frame of the survivors
        quick = placed[:, [5, 10, 15, 20]]
        sep = geometry.min_separation(quick.reshape(-1, 36), near_boxes).reshape(len(idx), -1).amin(-1).numpy()
        good = sep >= margin
        if good.any():
            sv = np.nonzero(good)[0]
            frames = placed[sv][:, 1::2]
            s2 = geometry.min_separation(frames.reshape(-1, 36), near_boxes).reshape(len(sv), -1).amin(-1).numpy()
            sep[sv] = np.minimum(sep[sv], s2)
            good = sep >= margin
        if corridor is not None:
            lat = np.abs(placed[:, :, :2] @ nrm - line0).max(1)
            good &= lat <= corridor
        if bounds is not None:
            # the scene's field grid (task._goal_status: pelvis and feet inside it laterally for plane-goal scenes, the
            # root inside it in xy otherwise); the pelvis is kept bound_pad inside, frames behind the start are exempt
            xlo, xhi, ylo, yhi = bounds
            px, py = placed[:, 1:, 0], placed[:, 1:, 1]
            inside = (py >= ylo + bound_pad) & (py <= yhi - bound_pad)
            if goal_radius is not None:
                inside &= (px >= xlo + .05) & (px <= xhi - .05)
            good &= inside.all(1)
        for j in np.nonzero(good)[0]:
            end = placed[j, -1]
            # no going backwards along the route by more than 5 cm per clip
            if float(end[:2] @ u) < float(q[:2] @ u) - .05:
                continue
            g = n["g"] + CLIP * DT + lateral_w * abs(float(end[:2] @ nrm) - line0) * CLIP * DT
            nodes.append(dict(q=end, feat=lib.end_feats[idx[j]], parent=i, clip=int(idx[j]), g=g, traj=placed[j], sep=float(sep[j])))
            hh = h(end)
            if hh < best_h:
                best_h, best_i = hh, len(nodes) - 1
            heapq.heappush(heap, (g + eps * hh, len(nodes) - 1))
    reached = done(nodes[best_i]["q"])
    # backtrack
    chain = []
    i = best_i
    while nodes[i]["parent"] >= 0:
        chain.append(i); i = nodes[i]["parent"]
    chain.reverse()
    if chain:
        traj = np.concatenate([start[None]] + [nodes[c]["traj"][1:] for c in chain])
    else:
        traj = start[None]
    clips = [nodes[c]["clip"] for c in chain]
    full_sep = float(geometry.min_separation(traj, boxes).min()) if len(traj) > 1 else float("nan")
    return dict(ok=bool(reached), qpos=traj.astype(np.float32), clips=np.array(clips, dtype=np.int64),
                modes=lib.modes[clips] if clips else np.zeros((0, len(MODE_NAMES)), bool), min_sep=full_sep, margin=margin,
                expansions=expansions, nodes=len(nodes), seconds=round(time.time() - t0, 1),
                progress=round(float(nodes[best_i]["q"][:2] @ u) - s0, 3), route=round(goal_s - s0, 3))


class OracleFeeder:
    """Distance-indexed feed of an oracle trajectory into the tracker's plan source for world 0 (CPU rollouts).

    The trajectory is indexed by progress u = arc length of its 0.5 s-smoothed pelvis path + 0.15 m per radian of heading
    change (a turn in place still makes progress), so dwell and hesitation in the oracle's own timing do not matter. Every
    `interval` s the robot's progress u* is its nearest smoothed-path point (searched in [u - 0.3 m, u + 1.0 m] of the last
    match, never more than 5 cm backwards), and the chunk's 21 frames are the oracle poses at u* + v_op (lead + 0.1 k).
    """

    def __init__(self, data, task, *, lead=0.0, interval=0.1, back_m=.3, ahead_m=1.0, turn_m=.15, leash=None):
        self.q = np.asarray(data["qpos"], dtype=np.float64)
        self.task, self.lead, self.interval = task, float(lead), float(interval)
        n = len(self.q)
        k = 12
        pad = np.concatenate([np.repeat(self.q[:1, :2], k, 0), self.q[:, :2], np.repeat(self.q[-1:, :2], k, 0)])
        kernel = np.ones(2 * k + 1) / (2 * k + 1)
        self.smooth = np.stack([np.convolve(pad[:, c], kernel, mode="valid") for c in range(2)], -1)   # [n, 2]
        self.yaw = np.unwrap(yaw_of(self.q))
        step = np.linalg.norm(np.diff(self.smooth, axis=0), axis=-1) + turn_m * np.abs(np.diff(self.yaw))
        self.u = np.concatenate(([0.], np.cumsum(step)))                    # progress per frame, non-decreasing
        self.n, self.match_u, self.clock_u = n, 0., 0.
        self.back, self.ahead = float(back_m), float(ahead_m)
        # leash (m): the reference clock advances at the operator speed but at most `leash` m ahead of the robot's match
        # (the plan clocks the network was trained with keep running); None: purely distance-indexed (clock = match)
        self.leash = None if leash is None else float(leash)

    def at(self, u):
        """Oracle poses at progress values u [m] (frame interpolation, quaternion nlerp) and their unwrapped yaw."""
        t = np.interp(np.clip(u, 0., self.u[-1]), self.u, np.arange(self.n, dtype=np.float64))
        a = np.floor(t).astype(int).clip(0, self.n - 2); w = (t - a)[:, None]
        qa, qb = self.q[a], self.q[a + 1].copy()
        qb[:, 3:7] = np.where((qa[:, 3:7] * qb[:, 3:7]).sum(-1, keepdims=True) < 0, -qb[:, 3:7], qb[:, 3:7])
        q = (1 - w) * qa + w * qb
        q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=-1, keepdims=True)
        yaw = (1 - w[:, 0]) * self.yaw[a] + w[:, 0] * self.yaw[a + 1]
        return q, yaw

    def replan(self, ids, events, step):
        t = self.task.tracker
        if len(ids) == 0 or not bool(t.is_plan[0]):
            return
        robot = self.task.data.qpos[0, :2].cpu().numpy().astype(np.float64)
        win = np.nonzero((self.u >= self.match_u - self.back) & (self.u <= self.match_u + self.ahead))[0]
        if len(win):
            j = win[int(np.linalg.norm(self.smooth[win] - robot, axis=-1).argmin())]
            self.match_u = max(float(self.u[j]), self.match_u)          # progress never goes back
        speed = float(t.route_speed[0])
        if self.leash is None:
            self.clock_u = self.match_u
        else:
            self.clock_u = min(max(self.clock_u + speed * self.interval, self.match_u), self.match_u + self.leash)
        q, yaw = self.at(self.clock_u + speed * (self.lead + .1 * np.arange(21)))
        dev = t.device
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)[None]
        roll, pitch = roll_pitch(q)
        t.set_plan(torch.tensor([0], device=dev), f(q[:, :2]), f(yaw), f(q[:, 2]), f(q[:, 19:36]),
                   torch.tensor([self.interval], device=dev), roll=f(roll), pitch=f(pitch), legs=f(q[:, 7:19]))
        t.has_full[0] = False                                     # no continuity: every chunk comes from the oracle
        task = self.task
        if hasattr(task, "plan_clear"):
            task.plan_clear[0] = .1; task.plan_side[0] = 0.; task.plan_fresh[0] = 1.; task.plan_cont[0] = 0.
        events["t"].append(step)
        events["chosen"].append(q.astype(np.float32)); events["cont"].append(0.); events["interval"].append(self.interval)
