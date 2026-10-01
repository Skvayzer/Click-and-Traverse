#!/usr/bin/env python3
"""How well does a checkpoint follow joystick velocity commands? Open flat scene, CPU MuJoCo.

Segments (directions relative to the robot's initial facing; world-frame velocity commands, as the
demo sends them): stand, forward 0.3/0.5/0.8 m/s, stop, left sideways 0.4, backward 0.3, diagonal
0.5, a 90 deg direction change while walking, stop. Per segment, over its second half: speed along
the command, sideways drift, velocity-direction error, facing error (body vs command direction),
time to reach 80% of the commanded speed; for stops: time to < 0.1 m/s and distance travelled.
Uses the demo's Session (no pushes, no observation noise, nominal PD gains, deterministic policy).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DT = .02


def load_demo():
    spec = importlib.util.spec_from_file_location("teleop_demo", ROOT / "scripts/demo/teleop_demo.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def wrap(a):
    return math.degrees(math.remainder(a, 2 * math.pi))


def body_segments(s):
    """Joystick commands in the body frame (forward, sideways, turn rate): v7+ checkpoints."""
    plan = [("stand", (0., 0., 0.), 2.), ("turn in place left 0.8 rad/s", (0., 0., .8), 4.), ("stand", (0., 0., 0.), 2.),
            ("turn in place right 0.8 rad/s", (0., 0., -.8), 4.), ("forward 0.5", (.5, 0., 0.), 4.),
            ("forward 0.5 + turn left 0.5", (.5, 0., .5), 4.), ("backward 0.3", (-.3, 0., 0.), 4.),
            ("sideways left 0.3", (0., .3, 0.), 4.), ("sideways right 0.3", (0., -.3, 0.), 4.), ("stop", (0., 0., 0.), 3.)]
    rows = []
    for name, command, seconds in plan:
        s.body_command = list(command)
        start_xy = s.sim.raw.qpos[:2].copy(); start_yaw = s.yaw(); falls0 = s.falls
        fwd, side, rates = [], [], []
        steps = int(seconds / DT); previous = s.yaw()
        for _ in range(steps):
            s.step()
            yaw = s.yaw(); v = np.asarray(s.sim.raw.qvel[:2], dtype=float)
            fwd.append(float(v @ [math.cos(yaw), math.sin(yaw)])); side.append(float(v @ [-math.sin(yaw), math.cos(yaw)]))
            rates.append(math.remainder(yaw - previous, 2 * math.pi) / DT); previous = yaw
        half = slice(steps // 2, steps)
        rows.append(dict(segment=name, command=command, forward=round(float(np.mean(fwd[half])), 3),
                         sideways=round(float(np.mean(side[half])), 3), turn_rate=round(float(np.mean(rates[half])), 3),
                         turned_deg=round(wrap(s.yaw() - start_yaw), 1),
                         displacement_m=round(float(np.linalg.norm(s.sim.raw.qpos[:2] - start_xy)), 2), falls=s.falls - falls0))
    return rows


def run(checkpoint, scene, body_mode=False):
    demo = load_demo()
    s = demo.Session(argparse.Namespace(checkpoint=checkpoint, bank="data/furniture/table_edges_v1", scenes=[scene], speed=.5,
                                        body_mode=body_mode))
    s.load(s.scenes[0])
    for _ in range(50):
        s.step()
    yaw0 = s.yaw()
    plan = [("stand", 0., 0., 2.), ("forward 0.3", 0., .3, 4.), ("forward 0.5", 0., .5, 4.), ("forward 0.8", 0., .8, 4.),
            ("stop", 0., 0., 3.), ("left sideways 0.4", 90., .4, 4.), ("backward 0.3", 180., .3, 4.),
            ("diagonal 45 deg 0.5", 45., .5, 4.), ("forward 0.5 (before turn)", 0., .5, 3.),
            ("turn to +90 deg at 0.5", 90., .5, 4.), ("stop", 0., 0., 3.)]
    rows = []
    if s.body_mode:
        rows += body_segments(s)
        s.body_mode = False                     # then the world-velocity segments below, for comparison
    for name, angle, speed, seconds in plan:
        heading = yaw0 + math.radians(angle)
        s.heading, s.speed = heading, speed
        u = np.array([math.cos(heading), math.sin(heading)]); n = np.array([-u[1], u[0]])
        start_xy = s.sim.raw.qpos[:2].copy(); falls0 = s.falls
        along, side, facing, speeds, t80 = [], [], [], [], None
        steps = int(seconds / DT)
        for k in range(steps):
            s.step()
            v = np.asarray(s.sim.raw.qvel[:2], dtype=float)
            along.append(float(v @ u)); side.append(float(v @ n)); speeds.append(float(np.linalg.norm(v)))
            facing.append(wrap(s.yaw() - heading))
            if speed and t80 is None and along[-1] >= .8 * speed:
                t80 = (k + 1) * DT
        half = slice(steps // 2, steps)
        row = dict(segment=name, commanded=speed)
        if speed:
            a, sd = np.mean(along[half]), np.mean(np.abs(side[half]))
            vdir = [math.degrees(math.atan2(y, x)) for x, y in zip(along[half], side[half])]   # 0 = along the command
            row.update(speed_along=round(float(a), 3), speed_ratio=round(float(a / speed), 2), sideways_drift=round(float(sd), 3),
                       direction_error_deg=round(float(np.mean(np.abs(vdir))), 1),
                       facing_error_deg=round(float(np.mean(np.abs(facing[half]))), 1),
                       time_to_80pct_s=None if t80 is None else round(t80, 2))
        else:
            below = next((k for k, v in enumerate(speeds) if v < .1), None)
            row.update(time_to_stop_s=None if below is None else round(below * DT, 2),
                       distance_after_command_m=round(float(np.linalg.norm(s.sim.raw.qpos[:2] - start_xy)), 2),
                       mean_speed_second_half=round(float(np.mean(speeds[half])), 3))
        row["falls"] = s.falls - falls0
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scene", default="flat-balance-walk-v1")
    p.add_argument("--output")
    p.add_argument("--body-mode", action="store_true", help="Force body-frame joystick segments (automatic for v7+)")
    args = p.parse_args()
    rows = run(args.checkpoint, args.scene, args.body_mode)
    for r in rows:
        print(json.dumps(r), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(dict(checkpoint=args.checkpoint, scene=args.scene, rows=rows), indent=1) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
