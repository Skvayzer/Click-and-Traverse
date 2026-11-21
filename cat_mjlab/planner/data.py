"""Planner examples from policy rollouts, and the on-disk dataset.

A rollout of N worlds over T control steps (50 Hz) is recorded as
    qpos [T+1, N, 36], qvel [T+1, N, 35]   state before each step (index T: after the last step)
    scene [T+1, N] (int), epstep [T+1, N]  scene and episode step of that state
    ended [T, N], bad [T, N]               step t ended the episode (any reason / contact or fall)
A reset happens inside the step, so state t+1 of an ended world belongs to its next episode.

An example is a window that starts at state t0 and spans H frames every STRIDE steps (default 20 frames x 5 steps
= 2.0 s at 10 Hz), all inside one episode, starting at least SETTLE steps into it, and with no contact or fall
before MARGIN steps after its end. Features, all in the chunk frame (floor projection of the pelvis at t0, x along
its heading):
    state [67]  29 joint angles, 29 joint velocities, pelvis height, roll, pitch, pelvis linear velocity (chunk
                frame, 3), angular velocity (body frame, 3)
    chunk [H, 35] per frame: pelvis x, y, z, heading change, roll, pitch, 29 joint angles
    goal [5]    last frame's x, y, sin and cos of its heading change, mean speed (straight-line distance / 2 s)
    pose [3]    the chunk frame in the scene: world x, y of its origin and its heading (to replay the window as a
                tracker reference in the same scene)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .grid import heading_of, wrap

H, STRIDE, SETTLE, MARGIN = 20, 5, 25, 25
DT = 0.02
STATE_DIM, FRAME_DIM, GOAL_DIM = 67, 35, 5
FILES = ("grid", "state", "chunk", "goal", "meta", "pose")
META = ("scene", "bucket", "world", "t0", "source", "split")


def euler_zyx(quat):
    """roll, pitch, yaw [...] of MuJoCo quaternions (w, x, y, z); R = Rz(yaw) Ry(pitch) Rx(roll)."""
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
    return roll, pitch, heading_of(quat)


def quat_from_euler(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (torch.cos(roll / 2), torch.sin(roll / 2), torch.cos(pitch / 2), torch.sin(pitch / 2),
                              torch.cos(yaw / 2), torch.sin(yaw / 2))
    return torch.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1)


def valid_starts(ended, bad, epstep, *, horizon=H * STRIDE, settle=SETTLE, margin=MARGIN, every=10):
    """Window starts [K, 2] (t0, world) for rollout flags ended/bad [T, N] and epstep [T+1, N] (numpy)."""
    T, N = ended.shape
    t0 = np.arange(0, T - horizon + 1, every)
    ended_c = np.concatenate([np.zeros((1, N), np.int64), np.cumsum(ended, 0)])      # [T+1, N]
    bad_c = np.concatenate([np.zeros((1, N), np.int64), np.cumsum(bad, 0)])
    same_episode = ended_c[t0 + horizon] - ended_c[t0] == 0                           # no reset in [t0, t0+horizon)
    stop = np.minimum(t0 + horizon + margin, T)
    clean = bad_c[stop] - bad_c[t0] == 0
    # A window may also end within `margin` of a CLEAN episode end (goal or timeout): then `clean` holds, but the
    # steps after the reset are not part of the window anyway.
    ok = same_episode & clean & (epstep[t0] >= settle)
    tt, nn = np.nonzero(ok)
    return np.stack((t0[tt], nn), 1)


def features(qpos, qvel, starts, *, horizon=H, stride=STRIDE):
    """state [K, 67], chunk [K, H, 35], goal [K, 5], origin [K, 2] and yaw [K] for window starts [K, 2] (t0, world).

    qpos/qvel are torch tensors [T+1, N, nq/nv] (any device)."""
    t0 = torch.as_tensor(starts[:, 0], device=qpos.device, dtype=torch.long)
    n = torch.as_tensor(starts[:, 1], device=qpos.device, dtype=torch.long)
    q0, v0 = qpos[t0, n], qvel[t0, n]
    roll0, pitch0, yaw0 = euler_zyx(q0[:, 3:7])
    origin = q0[:, :2]
    c, s = yaw0.cos(), yaw0.sin()
    rot = lambda v: torch.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]), -1)
    lin = torch.cat((rot(v0[:, :2]), v0[:, 2:3]), -1)
    state = torch.cat((q0[:, 7:], v0[:, 6:], q0[:, 2:3], roll0[:, None], pitch0[:, None], lin, v0[:, 3:6]), -1)
    k = torch.arange(1, horizon + 1, device=qpos.device) * stride
    qf = qpos[t0[:, None] + k[None], n[:, None]]                                       # [K, H, nq]
    roll, pitch, yaw = euler_zyx(qf[..., 3:7])
    d = qf[..., :2] - origin[:, None]
    cc, ss = c[:, None], s[:, None]
    local = torch.stack((cc * d[..., 0] + ss * d[..., 1], -ss * d[..., 0] + cc * d[..., 1]), -1)
    dyaw = wrap(yaw - yaw0[:, None])
    chunk = torch.cat((local, qf[..., 2:3], dyaw[..., None], roll[..., None], pitch[..., None], qf[..., 7:]), -1)
    end = local[:, -1]
    speed = torch.linalg.vector_norm(end, dim=-1) / (horizon * stride * DT)
    goal = torch.stack((end[:, 0], end[:, 1], dyaw[:, -1].sin(), dyaw[:, -1].cos(), speed), -1)
    return state.float(), chunk.float(), goal.float(), origin.float(), yaw0.float()


def chunk_qpos(chunk):
    """Chunk-frame qpos [..., H, 36] (root position, root quaternion, joints) from chunk frames [..., H, 35]."""
    quat = quat_from_euler(chunk[..., 4], chunk[..., 5], chunk[..., 3])
    return torch.cat((chunk[..., 0:3], quat, chunk[..., 6:]), -1)


def state_qpos(state):
    """Chunk-frame qpos [..., 36] of the start pose (root at the frame origin, heading 0)."""
    zeros = torch.zeros_like(state[..., :1])
    quat = quat_from_euler(state[..., 59], state[..., 60], zeros[..., 0])
    return torch.cat((zeros, zeros, state[..., 58:59], quat, state[..., :29]), -1)


class ShardWriter:
    """Accumulates examples per split and writes shards of `size` examples: <root>/<split>/<k:03d>_<file>.npy."""

    def __init__(self, root, size=20000):
        self.root, self.size = Path(root), int(size)
        self.buffers, self.counts = {}, {}

    def add(self, split, **arrays):
        buf = self.buffers.setdefault(split, {f: [] for f in FILES})
        for f in FILES:
            buf[f].append(np.asarray(arrays[f]))
        if sum(len(a) for a in buf["state"]) >= self.size:
            self.flush(split)

    def flush(self, split=None):
        for name in ([split] if split else list(self.buffers)):
            buf = self.buffers.get(name)
            if not buf or not buf["state"]:
                continue
            k = self.counts.get(name, 0)
            directory = self.root / name; directory.mkdir(parents=True, exist_ok=True)
            for f in FILES:
                np.save(directory / f"{k:03d}_{f}.npy", np.concatenate(buf[f]))
                buf[f] = []
            self.counts[name] = k + 1


def load_split(root, split, *, grids=True, keep=None):
    """Concatenated arrays of one split, in memory: dict(file -> array). grids=False skips the large grid file.
    keep(meta, goal) -> bool [n] selects examples shard by shard (the grid is read only for the kept rows)."""
    directory = Path(root) / split
    shards = sorted({p.name.split("_")[0] for p in directory.glob("*_state.npy")})
    if not shards:
        raise FileNotFoundError(f"no shards in {directory}")
    masks = None
    if keep is not None:
        masks = [np.asarray(keep(np.load(directory / f"{k}_meta.npy"), np.load(directory / f"{k}_goal.npy")), bool) for k in shards]
    out = {}
    for f in FILES:
        if f == "grid" and not grids:
            continue
        parts = [np.load(directory / f"{k}_{f}.npy", mmap_mode="r") for k in shards]
        if masks is not None:
            parts = [part[mask] for part, mask in zip(parts, masks)]
        out[f] = np.concatenate(parts) if len(parts) > 1 else np.asarray(parts[0])
    return out


def drop_stopped(info, speed=0.05, buckets=("procedural_cat", "original_cat", "published_cat", "narrow_passage",
                                              "protected_passage", "transition_passage", "table_edges", "open_passage")):
    """keep() for load_split: drops windows whose mean speed is below `speed` m/s in the given scene types (info.json
    scene_types). Measured on planner v1's data: 38-55% of the clean windows in CAT courses and 43% in side gaps were the
    policy standing still (rooms 8%), because windows ending in contact were filtered out; the planner then proposed
    hesitant, slow-starting plans in front of hard obstacles (4 cm in the first 0.3 s against 8 cm elsewhere)."""
    types = list(info["scene_types"])
    ids = np.array([types.index(b) for b in buckets if b in types])

    def keep(meta, goal):
        return ~((goal[:, 4] < speed) & np.isin(meta[:, 1].astype(int), ids))
    return keep


def write_info(root, info):
    (Path(root) / "info.json").write_text(json.dumps(info, indent=1) + "\n")
