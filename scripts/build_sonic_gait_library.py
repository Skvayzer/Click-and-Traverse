#!/usr/bin/env python3
"""Gait library from SONIC's own released locomotion generator (planner_sonic.onnx) for a GPU-batched reference source.

For each SLOW_WALK target speed the generator is rolled out open-loop exactly as deployed (standing context, replan
every 1.0 s from its own motion, 2-tick look-ahead, 8-frame blend; cat_mjlab/sonic_locomotion.SonicGenerator._replan
logic) for --seconds along +x. The steady part (after --settle s) gives the stride period (joint autocorrelation) and
one stride, phase-aligned across speeds (phase 0 = maximum left hip flexion), with joints, pelvis roll/pitch in the
heading frame, heading drift and forward speed. An IDLE rollout gives the standing pose.
Output npz: speeds [S], period [S] s, forward_speed [S] m/s, joints [S, P, 29], roll/pitch [S, P], stand [29].
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cat_mjlab.sonic_locomotion import BLEND, BUFFER, DEFAULT_HEIGHT, LOOKAHEAD, SLOW_WALK, IDLE, _nlerp, _sample, _yaw  # noqa: E402
from cat_mjlab.sonic_tracker import DEFAULT_ANGLES  # noqa: E402

P = 100


class _Standalone:
    """SonicGenerator's call/splice logic for one world without a simulator."""

    def __init__(self, onnx_path, seed=1234):
        import onnxruntime as ort
        from cat_mjlab.sonic_locomotion import SonicGenerator

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.seed = seed
        self._generate = SonicGenerator._generate.__get__(self)

    def rollout(self, mode, speed, seconds):
        q = np.zeros(36, np.float32); q[2] = DEFAULT_HEIGHT; q[3] = 1.; q[7:] = DEFAULT_ANGLES
        move = np.array([1., 0.], np.float32)
        buf = self._generate(self.session, np.repeat(q[None], 4, 0), mode, speed, move, move)
        out, cursor = [], 0
        for step in range(int(seconds / .02)):
            if step and step % 50 == 0:
                start = cursor + LOOKAHEAD
                gen = self._generate(self.session, _sample(buf, start + np.arange(4) * 50 / 30), mode, speed, move, move)
                new_len = min(LOOKAHEAD + len(gen), BUFFER)
                f = np.arange(new_len)
                a = _sample(buf, np.minimum(f + cursor, len(buf) - 1).astype(np.float64))
                b = gen[np.clip(f - LOOKAHEAD, 0, len(gen) - 1)]
                w = np.clip((f - LOOKAHEAD) / BLEND, 0., 1.)[:, None].astype(np.float32)
                buf = (1 - w) * a + w * b
                buf[:, 3:7] = _nlerp(a[:, 3:7], b[:, 3:7], w)
                cursor = 0
            out.append(buf[min(cursor, len(buf) - 1)].copy())
            cursor += 1
        return np.array(out)


def roll_pitch(q):
    w, x, y, z = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


def stride(motion, settle, dt=.02):
    """Phase-fold every stride of the steady segment: period T from the joint autocorrelation, phase 0 at maximum
    left hip flexion, joints / pelvis roll and pitch averaged per phase bin (the slow generator gait is only
    quasi-periodic; folding keeps its mean stride). Also returns the folded residual (rad) as a periodicity check."""
    m = motion[int(settle / dt):]
    j = m[:, 7:] - m[:, 7:].mean(0)
    lags = np.arange(int(.6 / dt), int(2.0 / dt))
    corr = [np.sum(j[:-l] * j[l:]) / np.sqrt(np.sum(j[:-l] ** 2) * np.sum(j[l:] ** 2)) for l in lags]
    lag = int(lags[int(np.argmax(corr))])
    # refine the period to a fractional value by maximizing fold consistency around the integer lag
    t = np.arange(len(m)) * dt
    t0 = t[int(np.argmin(m[:lag, 7]))]
    best = None
    for period in np.linspace((lag - 2) * dt, (lag + 2) * dt, 41):
        phase = ((t - t0) / period) % 1.
        bins = np.minimum((phase * P).astype(int), P - 1)
        mean = np.stack([np.bincount(bins, m[:, 7 + k], P) / np.maximum(np.bincount(bins, None, P), 1) for k in range(29)], -1)
        resid = np.sqrt(np.mean((m[:, 7:] - mean[bins]) ** 2))
        if best is None or resid < best[0]:
            best = (resid, period, bins, mean)
    resid, period, bins, joints = best
    filled = np.bincount(bins, None, P) > 0
    count = np.maximum(np.bincount(bins, None, P), 1)
    r, p = roll_pitch(m[:, 3:7])
    roll = np.bincount(bins, r, P) / count; pitch = np.bincount(bins, p, P) / count

    def fill(x):
        """Empty phase bins (a 1.0 s stride at 50 Hz fills only 50 of 100) by circular linear interpolation."""
        idx = np.flatnonzero(filled)
        ext = np.concatenate((idx - P, idx, idx + P))
        vals = np.concatenate((x[idx], x[idx], x[idx]))
        return np.interp(np.arange(P), ext, vals)
    joints = np.stack([fill(joints[:, k]) for k in range(29)], -1)
    roll, pitch = fill(roll), fill(pitch)
    disp = np.linalg.norm(m[-1, :2] - m[0, :2]) / ((len(m) - 1) * dt)
    return float(period), float(disp), joints.astype(np.float32), roll.astype(np.float32), pitch.astype(np.float32), \
        float(np.ptp(_yaw(m[:, 3:7]))), float(resid)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generator", type=Path, default=Path("/home/konstantinsmirnov/robotics/sonic/hf_gear_sonic/planner_sonic.onnx"))
    p.add_argument("--speeds", type=float, nargs="+", default=[.3, .4, .5, .6, .8])
    p.add_argument("--seconds", type=float, default=14.)
    p.add_argument("--settle", type=float, default=5.)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    gen = _Standalone(a.generator)
    rows = []
    for v in a.speeds:
        motion = gen.rollout(SLOW_WALK, v, a.seconds)
        period, fwd, joints, roll, pitch, heading_ptp, periodicity = stride(motion, a.settle)
        print(f"speed {v:.2f}: stride {period:.2f} s, forward {fwd:.3f} m/s, heading range {heading_ptp:.3f} rad, "
              f"folded rms residual {periodicity:.3f} rad", flush=True)
        rows.append((period, fwd, joints, roll, pitch))
    stand = gen.rollout(IDLE, -1., 3.)[-1, 7:]
    np.savez(a.out, speeds=np.array(a.speeds, np.float32), period=np.array([r[0] for r in rows], np.float32),
             forward_speed=np.array([r[1] for r in rows], np.float32), joints=np.stack([r[2] for r in rows]),
             roll=np.stack([r[3] for r in rows]), pitch=np.stack([r[4] for r in rows]), stand=stand.astype(np.float32))
    print("saved", a.out)


if __name__ == "__main__":
    main()
