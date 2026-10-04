#!/usr/bin/env python3
"""Obstacle-course regression check for a checkpoint (CPU, the keyboard demo's exact simulation).

Three drives on course-mixed-obstacles-v1 (scripts/demo/build_obstacle_course.py):
  field     follow the scene's CAT field to the end (no keyboard)
  keyboard  hold W at --speed (heading mode, reference heading straight down the course)
  idle      no keys for 20 s: foot touchdowns, drift, yaw change
--per-obstacle: instead, each obstacle on its own -- start 1 m before it, facing it, 12 s -- for the field and keyboard
drives: "clean" (cleared without contact), "contact" (cleared, touched), "refused" (not cleared).
For each drive: the obstacles cleared (root more than 0.3 m past the obstacle), contacts by section and body
part, falls. A drive stops early once the robot has not advanced 5 cm in --stall-seconds.
Prints one JSON line per drive; --compare FILE prints which obstacles a previous result cleared and this one did not.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

# runtime x of each obstacle (build_obstacle_course.py positions minus the 1.5 m start shift)
OBSTACLES = [(1.5, "hurdle 10cm"), (3.5, "hurdle 20cm"), (5.5, "beam 1.16m"), (7.5, "beam 1.02m"), (9.5, "gap 0.44m"),
             (11.5, "gap 0.34m"), (13.8, "double hurdle"), (15.5, "hurdle+beam"), (17.5, "gap+hurdle"), (19.7, "tables 0.60m")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bank", default="data/furniture/course_v1")
    p.add_argument("--speed", type=float, default=.4)
    p.add_argument("--seconds", type=float, default=90.)
    p.add_argument("--stall-seconds", type=float, default=15.)
    p.add_argument("--drives", nargs="+", default=["field", "keyboard", "idle"])
    p.add_argument("--per-obstacle", action="store_true", help="Each obstacle separately, starting 1 m before it")
    p.add_argument("--output", help="Write the results (JSON) here")
    p.add_argument("--compare", help="A previous --output to compare against")
    args = p.parse_args()
    import mujoco
    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location("demo", ROOT / "scripts/demo/teleop_demo.py")
    demo = importlib.util.module_from_spec(spec); spec.loader.exec_module(demo)
    s = demo.Session(argparse.Namespace(checkpoint=args.checkpoint, bank=args.bank, scenes=["course"], speed=args.speed, body_mode=False))
    task = s.task
    feet = [mujoco.mj_name2id(s.sim.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    results = {}

    def section(x):
        return ([n for b, n in OBSTACLES if x > b - .6] or ["start"])[-1]

    def place(x):
        task.data.qpos[0, 0] = x; task.data.qpos[0, 1] = 0.
        task.sim.forward(task.all_ids)

    if args.per_obstacle:
        for drive in [d for d in args.drives if d != "idle"]:
            table = {}
            for bx, name in OBSTACLES:
                if drive == "field":
                    task.joystick = torch.zeros((1, 2)); task.joystick_body = torch.zeros((1, 3)) if s.body_mode else None
                    s.load(s.scenes[0]); task.joystick = None; task.joystick_body = None
                else:
                    if task.joystick is None:
                        task.joystick = torch.zeros((1, 2)); task.joystick_body = torch.zeros((1, 3)) if s.body_mode else None
                    s.load(s.scenes[0]); s.hold_mode = True; s.held = {"up"}; s.speed_setting = args.speed
                place(bx - 1.)
                touched, cleared = set(), False
                for k in range(int(12. / .02)):
                    if drive == "field":
                        with torch.no_grad():
                            task.step(s.learner.act(task.obs, policy_ids=0, deterministic=True)["action"])
                    else:
                        s.step()
                    x = float(s.sim.raw.qpos[0])
                    if abs(x - bx) < .8:
                        touched |= {r for r, v in zip(demo.REGIONS, s.last["regions"].tolist()) if v}
                    if x > bx + .3:
                        cleared = True; break
                table[name] = ("clean" if not touched else "contact:" + "+".join(sorted(touched))) if cleared else \
                              ("refused" if not touched else "refused,touched:" + "+".join(sorted(touched)))
            results[drive] = table
            print(json.dumps(dict(drive=drive, **table)), flush=True)
        args.drives = []
    for drive in args.drives:
        if drive == "field":
            task.joystick = torch.zeros((1, 2)); task.joystick_body = torch.zeros((1, 3)) if s.body_mode else None
            s.load(s.scenes[0]); task.joystick = None; task.joystick_body = None
        else:
            if task.joystick is None:
                task.joystick = torch.zeros((1, 2)); task.joystick_body = torch.zeros((1, 3)) if s.body_mode else None
            s.load(s.scenes[0])
            s.hold_mode = True; s.held = {"up"} if drive == "keyboard" else set(); s.speed_setting = args.speed
        steps = int((20. if drive == "idle" else args.seconds) / .02)
        best, best_step, contacts, falls, touchdowns = 0., 0, {}, 0, 0
        x0 = s.sim.raw.qpos[:2].copy(); yaw0 = s.yaw(); base = None; up_prev = None
        for k in range(steps):
            if drive == "field":
                with torch.no_grad():
                    task.step(s.learner.act(task.obs, policy_ids=0, deterministic=True)["action"])
            else:
                s.step()
            x = float(s.sim.raw.qpos[0])
            regions = [r for r, v in zip(demo.REGIONS, s.last["regions"].tolist()) if v]
            if regions:
                key = section(x); contacts.setdefault(key, {})
                for r in regions:
                    contacts[key][r] = contacts[key].get(r, 0) + 1
            if s.last["flags"].get("fall"):
                falls += 1
            if drive == "idle":
                h = s.sim.raw.xpos[feet, 2]
                if k == 50:
                    base = [float(v) for v in h]
                if base is not None:
                    up = [bool(v > b + .02) for v, b in zip(h, base)]
                    if up_prev is not None:
                        touchdowns += sum(1 for a, b in zip(up_prev, up) if a and not b)
                    up_prev = up
                continue
            if x > best + .05:
                best, best_step = x, k
            if (k - best_step) * .02 > args.stall_seconds:
                break
        if drive == "idle":
            results[drive] = dict(foot_touchdowns=touchdowns, drift_m=round(float(np.linalg.norm(s.sim.raw.qpos[:2] - x0)), 3),
                                  yaw_change_deg=round(math.degrees(s.yaw() - yaw0), 1), contacts=contacts, falls=falls)
        else:
            cleared = [n for b, n in OBSTACLES if best > b + .3]
            stuck = next((n for b, n in OBSTACLES if best <= b + .3), "end")
            results[drive] = dict(cleared=cleared, n_cleared=len(cleared), stuck_at=stuck, furthest_x=round(best, 2),
                                  seconds=round((k + 1) * .02, 1), contacts=contacts, falls=falls)
        print(json.dumps(dict(drive=drive, **results[drive])), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(dict(checkpoint=args.checkpoint, speed=args.speed, results=results), indent=1) + "\n")
    if args.compare and args.per_obstacle:
        before = json.loads(Path(args.compare).read_text())["results"]
        for drive in results:
            if drive in before:
                rows = {n: (before[drive].get(n), results[drive].get(n)) for _, n in OBSTACLES}
                print(json.dumps(dict(compare=drive, **{n: f"{a} -> {b}" for n, (a, b) in rows.items() if a != b})))
    elif args.compare:
        before = json.loads(Path(args.compare).read_text())["results"]
        for drive in ("field", "keyboard"):
            if drive in before and drive in results:
                lost = [n for n in before[drive]["cleared"] if n not in results[drive]["cleared"]]
                gained = [n for n in results[drive]["cleared"] if n not in before[drive]["cleared"]]
                print(json.dumps(dict(compare=drive, before=before[drive]["n_cleared"], now=results[drive]["n_cleared"], lost=lost, gained=gained)))


if __name__ == "__main__":
    main()
