#!/usr/bin/env python3
"""Motion library for the full-body tracker (cat_mjlab/mimic.py): the policies' own clean motion, as continuous clips.

A planner dataset holds overlapping 2 s windows of policy rollouts: the start state, 20 chunk frames at 10 Hz and the
chunk frame's pose in the scene. Windows of one world and scene whose starts are at most 2 s apart and whose frames
agree are one continuous stretch of clean motion (no contact or fall). Each stretch of at least --min-seconds becomes
a clip: resampled to 50 Hz with cubic splines, turned into mjlab's G1 body states, and written back to back with the
clip starts (segment_starts), the format mjlab's MotionLoader reads plus the clip table.

    python scripts/mimic_library.py --data RUN/data_v7 RUN/data_e2 RUN/data_t2 --split train --minutes 40 --out LIB/train.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

FPS = 50
STEP = 5                 # control steps between chunk frames (10 Hz)


def keyframes(state, chunk, pose):
    """[21, 35] world keyframes: x, y, z, roll, pitch, yaw, 29 joints (start, then the 20 chunk frames)."""
    x0, y0, yaw0 = (float(v) for v in pose)
    c, s = np.cos(yaw0), np.sin(yaw0)
    out = np.empty((21, 35))
    out[0, :6] = (x0, y0, state[58], state[59], state[60], yaw0); out[0, 6:] = state[:29]
    lx, ly = chunk[:, 0], chunk[:, 1]
    out[1:, 0] = x0 + c * lx - s * ly; out[1:, 1] = y0 + s * lx + c * ly; out[1:, 2] = chunk[:, 2]
    out[1:, 3] = chunk[:, 4]; out[1:, 4] = chunk[:, 5]; out[1:, 5] = yaw0 + chunk[:, 3]; out[1:, 6:] = chunk[:, 6:]
    return out


def segments(data, min_frames):
    """Continuous stretches (lists of 10 Hz keyframes [T, 35]) from one dataset's windows."""
    meta, state, chunk, pose = data["meta"], data["state"], data["chunk"], data["pose"]
    groups = defaultdict(list)
    for i in range(len(meta)):
        groups[(int(meta[i, 2]), int(meta[i, 0]))].append(i)            # (world, scene)
    out = []
    for idx in groups.values():
        idx.sort(key=lambda i: int(meta[i, 3]))
        open_segs = []                                                  # [t0 of last window, last keyframes, {t: frame}]
        for i in idx:
            t0, kf = int(meta[i, 3]), keyframes(state[i], chunk[i], pose[i])
            home = None
            for seg in open_segs:
                gap = t0 - seg[0]
                if 0 < gap <= 20 * STEP and gap % STEP == 0:
                    ref = seg[1][gap // STEP]
                    if np.abs(ref[:2] - kf[0, :2]).max() < 0.01 and np.abs(ref[6:] - kf[0, 6:]).max() < 0.01:
                        home = seg; break
            if home is None:
                home = [t0, kf, {}]; open_segs.append(home)
            for k in range(21):
                home[2].setdefault(t0 + k * STEP, kf[k])
            home[0], home[1] = t0, kf
        for seg in open_segs:
            ts = sorted(seg[2])
            frames = np.stack([seg[2][t] for t in ts])
            # keep only gap-free runs (the dictionary can skip a time if windows were dropped)
            breaks = np.nonzero(np.diff(ts) != STEP)[0] + 1
            for run in np.split(frames, breaks):
                if len(run) >= min_frames:
                    out.append(run)
    return out


def resample(frames):
    """10 Hz keyframes [T, 35] -> 50 Hz qpos [N, 36] (MuJoCo order, quaternion wxyz) and qvel [N, 35]."""
    from scipy.interpolate import CubicSpline
    t = np.arange(len(frames)) * STEP * 0.02
    f = frames.copy(); f[:, 5] = np.unwrap(f[:, 5])
    spline = CubicSpline(t, f, axis=0)
    ts = np.arange(0, t[-1] + 1e-9, 1.0 / FPS)
    p, v = spline(ts), spline(ts, 1)
    roll, pitch, yaw = p[:, 3], p[:, 4], p[:, 5]
    cr, sr, cp, sp, cy, sy = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2), np.cos(yaw / 2), np.sin(yaw / 2)
    quat = np.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1)
    # world angular velocity of R = Rz(yaw) Ry(pitch) Rx(roll), then the free joint's body-frame angular velocity
    rd, pd, yd = v[:, 3], v[:, 4], v[:, 5]
    cyaw, syaw, cpit, spit = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    w_world = np.stack((-syaw * pd + cyaw * cpit * rd, cyaw * pd + syaw * cpit * rd, yd - spit * rd), -1)
    R = rotation(roll, pitch, yaw)
    w_body = np.einsum("nji,nj->ni", R, w_world)
    qpos = np.concatenate((p[:, :3], quat, p[:, 6:]), -1)
    qvel = np.concatenate((v[:, :3], w_body, v[:, 6:]), -1)
    return qpos, qvel


def rotation(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    R = np.empty((len(roll), 3, 3))
    R[:, 0, 0] = cy * cp; R[:, 0, 1] = cy * sp * sr - sy * cr; R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp; R[:, 1, 1] = sy * sp * sr + cy * cr; R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp; R[:, 2, 1] = cp * sr; R[:, 2, 2] = cp * cr
    return R


def body_states(model, qpos, qvel):
    """mjlab G1 link states of every robot body for each frame."""
    import mujoco
    data = mujoco.MjData(model)
    bodies = [b for b in range(model.nbody) if model.body(b).name.startswith("robot/")]
    n, nb = len(qpos), len(bodies)
    pos, quat = np.empty((n, nb, 3), np.float32), np.empty((n, nb, 4), np.float32)
    lin, ang = np.empty((n, nb, 3), np.float32), np.empty((n, nb, 3), np.float32)
    res = np.zeros(6)
    for i in range(n):
        data.qpos[:] = qpos[i]; data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        pos[i] = data.xpos[bodies]; quat[i] = data.xquat[bodies]
        for j, b in enumerate(bodies):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
            ang[i, j], lin[i, j] = res[:3], res[3:]
    return pos, quat, lin, ang


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, nargs="+", required=True, help="planner dataset roots (one per policy)")
    p.add_argument("--split", default="train", choices=("train", "test"))
    p.add_argument("--minutes", type=float, default=40.0, help="motion kept per dataset")
    p.add_argument("--min-seconds", type=float, default=3.0)
    p.add_argument("--max-seconds", type=float, default=20.0, help="longer stretches are cut into clips of this length")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    from cat_mjlab.planner.data import load_split
    from mjlab.scene import Scene
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
    started = time.time()
    model = Scene(unitree_g1_flat_tracking_env_cfg().scene, device="cpu").compile()
    rng = np.random.default_rng(args.seed)
    min_frames = int(round(args.min_seconds * 10)) + 1
    cut = int(round(args.max_seconds * 10))
    parts = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    starts, sources, report, total = [], [], [], 0
    for root in args.data:
        info = json.loads((root / "info.json").read_text())
        data = load_split(root, args.split, grids=False)
        segs = segments(data, min_frames)
        clips = [s[a:a + cut + 1] for s in segs for a in range(0, len(s) - 1, cut) if len(s[a:a + cut + 1]) >= min_frames]
        rng.shuffle(clips)
        budget, kept = args.minutes * 60.0, 0.0
        for clip in clips:
            if kept >= budget:
                break
            qpos, qvel = resample(clip)
            pos, quat, lin, ang = body_states(model, qpos, qvel)
            starts.append(total); sources.append(int(info["source_id"])); total += len(qpos)
            for k, v in zip(parts, (qpos[:, 7:], qvel[:, 6:], pos, quat, lin, ang)):
                parts[k].append(np.asarray(v, np.float32))
            kept += (len(qpos) - 1) / FPS
        report.append(dict(data=str(root), source_id=int(info["source_id"]), windows=int(len(data["meta"])), stretches=len(segs),
                           clips_available=len(clips), clips_kept=sum(1 for s in sources if s == int(info["source_id"])),
                           minutes_kept=round(kept / 60, 1)))
        print(json.dumps(report[-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, fps=np.array([FPS]), segment_starts=np.array(starts, np.int64), segment_source=np.array(sources, np.int64),
             **{k: np.concatenate(v) for k, v in parts.items()})
    print("LIBRARY DONE", json.dumps(dict(out=str(args.out), frames=total, minutes=round(total / FPS / 60, 1), clips=len(starts),
                                          seconds=round(time.time() - started, 1))), flush=True)


if __name__ == "__main__":
    main()
