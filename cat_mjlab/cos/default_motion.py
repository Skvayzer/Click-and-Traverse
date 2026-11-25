"""Default (counterfactual) motion D(M): same start, same along-path progress over time, upright straight walking with
neutral arm swing, facing along the chord (docs/cos_obstacles.md, G2).

Gait source: periodic stride cycles cut from steady, straight `walk` stretches of the motion library (human walking
retargeted to the G1), one cycle per take, indexed by speed. D is distance-indexed: the cycle phase advances with the
station s(t) = running max of M's pelvis progress along the chord, so D stands where M stands and passes every station at
the same time as M.
"""
from __future__ import annotations

import numpy as np

from .events import EXCLUDE, smooth


# ---------- quaternion helpers (w, x, y, z) ----------
def qmul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0); bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def qyaw(yaw):
    return np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], -1)


def yaw_of_q(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_frame(qpos, origin_xy, yaw):
    """Express qpos [T, 36] in the frame with origin (x, y, 0) and heading yaw (rotation about z)."""
    q = np.array(qpos, dtype=np.float64)
    c, s = np.cos(-yaw), np.sin(-yaw)
    d = q[:, :2] - origin_xy
    q[:, 0], q[:, 1] = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]
    q[:, 3:7] = qmul(qyaw(np.full(len(q), -yaw)), q[:, 3:7])
    return q


# ---------- gait bank ----------
def build_gait_bank(library_qpos, clip, feats, meta_clips, *, min_s=2.0, max_cycles=600, seed=0):
    """Stride cycles from steady straight upright walking: list of dicts with joints [P, 29], z [P], rot (quat relative to
    the travel heading) [P, 4], stride (m per cycle), speed (m/s). Walking = pelvis speed > 0.3 m/s, head/trunk top > 1.22 m,
    both hands below 0.75 m, pelvis > 0.6 m (features from cat_mjlab/cos/features.py)."""
    rng = np.random.default_rng(seed)
    v = np.gradient(feats["pelvis"][:, :2], .02, axis=0)
    sp_all = np.linalg.norm(smooth(v, 25), axis=1)
    walk = (sp_all > 0.3) & (feats["top_body"] > 1.22) & (feats["hand_z"].max(1) < 0.75) & (feats["pelvis"][:, 2] > 0.6)
    starts = np.r_[0, np.where(np.diff(clip) != 0)[0] + 1]; ends = np.r_[starts[1:], len(clip)]
    cand = []
    for s, e in zip(starts, ends):
        c = meta_clips[int(clip[s])]
        if c["source"] == "omniretarget" or EXCLUDE.search(c["clip_id"]):
            continue
        m = walk[s:e]
        d = np.diff(np.r_[0, m.astype(int), 0]); a, b = np.where(d == 1)[0], np.where(d == -1)[0]
        for x, y in zip(a, b):
            if (y - x) * .02 >= min_s:
                cand.append((s + x, s + y))
    rng.shuffle(cand)
    bank = []
    for a, b in cand:
        b = min(b, a + int(4.0 / .02))
        q = library_qpos[a:b].astype(np.float64)
        p = q[:, :2]; v = smooth(np.gradient(p, .02, axis=0), 25); sp = np.linalg.norm(v, axis=1)
        if sp.min() < 0.25 or sp.std() > 0.30 * sp.mean():
            continue
        head = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        if np.ptp(head) > np.deg2rad(30):                             # straight only
            continue
        hip = q[:, 7]                                                 # left hip pitch
        h = hip - hip.mean()
        ac = np.correlate(h, h, "full")[len(h) - 1:]
        lo, hi = int(0.7 / .02), min(int(1.8 / .02), len(h) - 2)
        if hi <= lo:
            continue
        P = lo + int(np.argmax(ac[lo:hi]))
        j0 = int(np.argmax(hip[:P]))
        if j0 + P >= len(q):
            continue
        cyc = q[j0:j0 + P + 1]
        if np.abs(cyc[-1, 7:] - cyc[0, 7:]).mean() > 0.08:
            continue
        hd = head[j0:j0 + P + 1]
        rot = qmul(qyaw(-hd), cyc[:, 3:7])
        stride = float(np.linalg.norm(cyc[-1, :2] - cyc[0, :2]))
        bank.append(dict(joints=cyc[:-1, 7:], z=cyc[:-1, 2], rot=rot[:-1], stride=stride, speed=stride / (P * .02),
                         period=P * .02, src=(int(a + j0), int(a + j0 + P))))
        if len(bank) >= max_cycles:
            break
    return bank


def pick_cycle(bank, speed, rng):
    sp = np.array([c["speed"] for c in bank])
    order = np.argsort(np.abs(sp - speed))[:5]
    return bank[int(rng.choice(order))]


def station(qc):
    """Monotone along-chord progress of the pelvis [T] (chord frame: x forward)."""
    x = smooth(qc[:, 0], 5)
    return np.maximum.accumulate(x - x[0]) + x[0]


def default_motion(qc, cycle, *, phase0=0.):
    """D(M) [T, 36] in the chord frame for M = qc (chord frame, starts at the origin, goal on +x)."""
    s = station(qc)
    T = len(qc)
    P = len(cycle["joints"])
    u = (phase0 + (s - s[0]) / max(cycle["stride"], 1e-3)) % 1.0 * P
    i = np.floor(u).astype(int) % P; j = (i + 1) % P; f = (u - np.floor(u))[:, None]
    D = np.zeros((T, 36))
    D[:, 0] = s; D[:, 1] = 0.
    D[:, 2] = (1 - f[:, 0]) * cycle["z"][i] + f[:, 0] * cycle["z"][j]
    D[:, 7:] = (1 - f) * cycle["joints"][i] + f * cycle["joints"][j]
    qa, qb = cycle["rot"][i], cycle["rot"][j]
    qb = np.where((qa * qb).sum(-1, keepdims=True) < 0, -qb, qb)
    r = (1 - f) * qa + f * qb
    D[:, 3:7] = r / np.linalg.norm(r, axis=-1, keepdims=True)
    return D
