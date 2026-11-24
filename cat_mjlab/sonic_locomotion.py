"""SONIC's own kinematic locomotion generator (planner_sonic.onnx) as the reference source for the frozen SONIC tracker.

Port of the official deployment loop (gear_sonic_deploy g1_deploy_onnx_ref: localmotion_kplanner.hpp,
localmotion_kplanner_onnx.hpp and the planner/control threads of g1_deploy_onnx_ref.cpp):
  * command per world: mode (SLOW_WALK = 1 or IDLE = 0), target speed, movement and facing directions;
  * replan every `replan_s` (deployment: 1.0 s while walking) and at once when the command changes (mode change, direction
    turned by more than `turn_deg`, speed changed by more than 0.05 m/s);
  * context = 4 frames at 30 Hz of the executing motion, sampled from 2 control ticks ahead (motion_look_ahead_steps);
  * the output (up to 64 frames at 30 Hz, MuJoCo qpos) is resampled to 50 Hz and blended into the executing motion over
    8 frames starting 2 frames ahead (the deployment's splice); joint velocities by finite difference at 50 Hz.
Each call is canonicalized: context frame 0 is moved to the origin with zero yaw (the deployment's initialization frame),
directions are rotated with it, and the output is transformed back to the world. The generator is open-loop in global
position (as deployed); the caller closes the loop through the directions it commands from the measured robot pose.

The released ONNX graph has batch 1 baked in, so worlds are planned one call at a time on a pool of sessions in threads.
"""
from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

GENERATOR_SHA256 = "39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea"
DEFAULT_HEIGHT = 0.788740
IDLE, SLOW_WALK = 0, 1
BUFFER = 200                    # 50 Hz frames kept per world (generator output <= 107 frames + splice)
LOOKAHEAD = 2                   # control ticks (motion_look_ahead_steps)
BLEND = 8                       # frames of linear blend


def _yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _qz(yaw):
    out = np.zeros(yaw.shape + (4,), np.float32)
    out[..., 0] = np.cos(yaw / 2); out[..., 3] = np.sin(yaw / 2)
    return out


def _qmul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0); bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)


def _nlerp(q0, q1, w):
    q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
    q = (1 - w) * q0 + w * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _sample(frames, t50):
    """frames [T, 36] at 50 Hz, fractional frame index t50 [k] -> [k, 36] (linear, quaternion nlerp)."""
    f0 = np.clip(np.floor(t50).astype(int), 0, len(frames) - 1)
    f1 = np.minimum(f0 + 1, len(frames) - 1)
    w = (t50 - np.floor(t50))[:, None].astype(np.float32)
    out = (1 - w) * frames[f0] + w * frames[f1]
    out[:, 3:7] = _nlerp(frames[f0, 3:7], frames[f1, 3:7], w)
    return out


class SonicGenerator:
    """Per-world executing motion from planner_sonic.onnx. `command(ids)` -> (mode [m] int, speed [m], move [m, 2],
    face [m, 2]) in world frame; `reference()` returns (pose [n, 10, 36], joint velocity [n, 10, 29]) for SonicTracker and
    advances every world's cursor by one control tick."""

    def __init__(self, task, onnx_path, command, *, sessions=4, threads=2, provider="cpu", replan_s=1.0, turn_deg=20.,
                 seed=1234):
        import onnxruntime as ort

        digest = hashlib.sha256()
        with Path(onnx_path).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 24), b""):
                digest.update(block)
        if digest.hexdigest() != GENERATOR_SHA256:
            raise ValueError("planner_sonic.onnx differs from the pinned release")
        self.task, self.command, self.n = task, command, task.num_envs
        self.replan_steps = int(round(replan_s / task.dt))
        self.turn_cos = math.cos(math.radians(turn_deg))
        self.seed = seed
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        providers = ["CUDAExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
        self.sessions = [ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
                         for _ in range(sessions)]
        self.pool = ThreadPoolExecutor(max_workers=sessions)
        n = self.n
        self.buf = np.zeros((n, BUFFER, 36), np.float32)
        self.length = np.zeros(n, int)
        self.cursor = np.zeros(n, int)
        self.since = np.arange(n) % self.replan_steps          # stagger replans across worlds
        self.last_mode = np.full(n, -1)
        self.last_speed = np.zeros(n, np.float32)
        self.last_move = np.zeros((n, 2), np.float32)
        self.ready = np.zeros(n, bool)
        self.calls = 0

    # ---- generator call -------------------------------------------------------------------------------------
    def _generate(self, session, context, mode, speed, move, face):
        """context [4, 36] world frame -> generated motion [T50, 36] world frame (starting at context frame 0)."""
        p0 = context[0, :2].copy(); yaw0 = float(_yaw(context[0, 3:7]))
        c, s = math.cos(-yaw0), math.sin(-yaw0)
        rot = np.array([[c, -s], [s, c]], np.float32)
        ctx = context.copy()
        ctx[:, :2] = (ctx[:, :2] - p0) @ rot.T
        ctx[:, 3:7] = _qmul(_qz(np.full(4, -yaw0)), ctx[:, 3:7])
        dirs = np.zeros((2, 3), np.float32)
        dirs[0, :2] = rot @ move; dirs[1, :2] = rot @ face
        allowed = np.zeros((1, 11), np.int64); allowed[0, :6] = 1
        feed = dict(context_mujoco_qpos=ctx[None], target_vel=np.array([speed], np.float32),
                    mode=np.array([mode], np.int64), movement_direction=dirs[:1], facing_direction=dirs[1:],
                    random_seed=np.array([self.seed], np.int64), has_specific_target=np.zeros((1, 1), np.int64),
                    specific_target_positions=np.zeros((1, 4, 3), np.float32),
                    specific_target_headings=np.zeros((1, 4), np.float32), allowed_pred_num_tokens=allowed,
                    height=np.array([-1.], np.float32))
        qpos, frames = session.run(None, feed)
        frames = int(frames[0])
        if not 2 <= frames <= 64 or not np.isfinite(qpos[0, :frames]).all():
            raise RuntimeError(f"SONIC generator returned {frames} frames or non-finite values")
        out = qpos[0, :frames].astype(np.float32)
        inv = rot.T
        out[:, :2] = out[:, :2] @ inv.T + p0
        out[:, 3:7] = _qmul(_qz(np.full(frames, yaw0)), out[:, 3:7])
        t50 = np.arange(int(math.floor(frames / 30 * 50))) * 30 / 50          # 50 Hz sample times in 30 Hz frames
        f0 = np.floor(t50).astype(int); f1 = np.minimum(f0 + 1, frames - 1)
        w = (t50 - f0)[:, None].astype(np.float32)
        res = (1 - w) * out[f0] + w * out[f1]
        res[:, 3:7] = _nlerp(out[f0, 3:7], out[f1, 3:7], w)
        return res

    def _replan(self, i, session, mode, speed, move, face):
        if not self.ready[i]:
            q = self.task.data.qpos[i].detach().cpu().numpy().astype(np.float32)
            context = np.repeat(q[None], 4, 0)
            context[:, 2] = DEFAULT_HEIGHT
            gen = self._generate(session, context, mode, speed, move, face)
            self.buf[i, :len(gen)] = gen; self.length[i] = len(gen); self.cursor[i] = 0
            self.ready[i] = True
            return
        cur, old = self.cursor[i], self.buf[i, :self.length[i]]
        start = cur + LOOKAHEAD
        context = _sample(old, start + np.arange(4) * 50 / 30)
        gen = self._generate(session, context, mode, speed, move, face)
        new_len = min(LOOKAHEAD + len(gen), BUFFER)
        f = np.arange(new_len)
        a = _sample(old, np.minimum(f + cur, len(old) - 1).astype(np.float64))
        b = gen[np.clip(f - LOOKAHEAD, 0, len(gen) - 1)]
        w = np.clip((f - LOOKAHEAD) / BLEND, 0., 1.)[:, None].astype(np.float32)
        spliced = (1 - w) * a + w * b
        spliced[:, 3:7] = _nlerp(a[:, 3:7], b[:, 3:7], w)
        self.buf[i, :new_len] = spliced; self.length[i] = new_len; self.cursor[i] = 0

    # ---- per control tick ------------------------------------------------------------------------------------
    @torch.no_grad()
    def reference(self):
        task, n = self.task, self.n
        fresh = (task.info["step"] == 0).detach().cpu().numpy()
        self.ready[fresh] = False
        mode, speed, move, face = (np.asarray(x.detach().cpu() if torch.is_tensor(x) else x)
                                   for x in self.command(task.all_ids))
        turned = (self.last_move * move).sum(-1) < self.turn_cos * np.linalg.norm(move, axis=-1) * np.linalg.norm(
            self.last_move, axis=-1)
        due = (~self.ready | (self.since >= self.replan_steps) | (mode != self.last_mode)
               | (np.abs(speed - self.last_speed) > .05) | turned | (self.cursor >= self.length - 12))
        ids = np.flatnonzero(due)
        if len(ids):
            groups = [ids[k::len(self.sessions)] for k in range(len(self.sessions))]

            def work(k):
                for i in groups[k]:
                    self._replan(i, self.sessions[k], int(mode[i]), float(speed[i]), move[i], face[i])
            list(self.pool.map(work, range(len(self.sessions))))
            self.calls += len(ids)
            self.since[ids] = 0
            self.last_mode[ids] = mode[ids]; self.last_speed[ids] = speed[ids]; self.last_move[ids] = move[ids]
        idx = np.minimum(self.cursor[:, None] + np.arange(10)[None] * 5, self.length[:, None] - 1)
        pose = self.buf[np.arange(n)[:, None], idx]                                  # [n, 10, 36]
        nxt = self.buf[np.arange(n)[:, None], np.minimum(idx + 1, self.length[:, None] - 1)]
        prv = self.buf[np.arange(n)[:, None], np.maximum(idx - 1, 0)]
        span = (np.minimum(idx + 1, self.length[:, None] - 1) - np.maximum(idx - 1, 0)).clip(min=1)[..., None]
        qd = (nxt[..., 7:] - prv[..., 7:]) / (span * task.dt)
        self.cursor = np.minimum(self.cursor + 1, self.length - 1)
        self.since += 1
        dev = task.device
        return torch.from_numpy(pose).to(dev), torch.from_numpy(qd.astype(np.float32)).to(dev)


def route_command(task, *, lookahead=.6, min_speed=.05, speed=None, generator_speed=(.3, .8)):
    """Command from the tracker's guidance route: walk toward the route point `lookahead` m ahead of the robot's
    projection, facing that way, at the operator's commanded speed (or a fixed `speed`), clipped to `generator_speed`.
    The clip: at a 0.15 m/s SLOW_WALK command the generator produces a shuffle that SONIC does not execute (0.02 m in
    12 s on flat ground, G1); at 0.30 m/s SONIC walks at ~0.21 m/s. Returns numpy arrays."""
    t = task.tracker
    ids = task.all_ids
    xy = task.data.qpos[:, :2]
    s, _ = t.project(ids, xy)
    goal = t._point(ids, s + lookahead)
    move = goal - xy
    move = move / torch.linalg.vector_norm(move, dim=-1, keepdim=True).clamp_min(1e-6)
    if speed is None:
        v = torch.linalg.vector_norm(task.info["command"][:, 1:3], dim=-1)
    else:
        v = torch.full((task.num_envs,), float(speed), device=task.device)
    mode = torch.where(v >= min_speed, SLOW_WALK, IDLE)
    v = torch.where(v >= min_speed, v.clamp(*generator_speed), torch.full_like(v, -1.))
    return mode.cpu().numpy(), v.cpu().numpy(), move.cpu().numpy(), move.cpu().numpy()
