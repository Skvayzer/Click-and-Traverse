#!/usr/bin/env python3
"""Scripted run of the browser demo's control logic (teleop_demo.Session, headless), for presentation videos.

  goals     goal points as the browser click sets them (Session.set_goal): the next goal is placed when the robot has
            reached the current one (WaypointField in obstacle-course scenes: guidance fields toward the goal)
  keyboard  held keys as the browser key page sends them (W/S forward/back, A/D turn, Q/E sideways)

Writes a recording directory as scripts/record_mjlab_rollout.py does (model.xml, scene.json, trajectory.npz,
metadata.json) plus events.json (goals / keys over time) for scripts/demo/render_control_video.py.

  record_control_video.py --mode goals --scene course-mixed-obstacles-v1 --out DIR
  record_control_video.py --mode keyboard --scene sidegap2-0162-plain-w0.47 --keys "up:7,none:1,left:2,up:2,strafe_left:2,none:2" --out DIR
"""
import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'scripts' / 'demo')]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('goals', 'keyboard'), required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--checkpoint', default=str(ROOT / 'outputs/saved_checkpoints/generalist_robust_delay_v1_u150.pt'))
    p.add_argument('--bank', default='data/furniture/demo_v1')
    p.add_argument('--speed', type=float, default=.5)
    p.add_argument('--seconds', type=float, default=90.)
    p.add_argument('--goal-spacing', type=float, default=2.5, help='goals mode: distance between goals along start -> goal')
    p.add_argument('--goals', type=float, nargs='*', help='goals mode: explicit goal points x1 y1 x2 y2 ...')
    p.add_argument('--hold', type=float, default=1.0, help='goals mode: seconds standing at a goal before the next')
    p.add_argument('--keys', default='up:7,none:1,left:2,up:2,strafe_left:2,none:2',
                   help='keyboard mode: comma list key[+key]:seconds; keys up, down, left, right, strafe_left, strafe_right, none')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    os.chdir(ROOT)
    import teleop_demo as demo
    from record_mjlab_rollout import write_geometry, sha256
    args = SimpleNamespace(checkpoint=a.checkpoint, bank=a.bank, scenes=[a.scene], speed=a.speed, body_mode=False,
                           keep_robustness=False, upper_average=True)
    s = demo.Session(args)
    s.load(s.scenes[0])
    record = s.manifest['scenes'][s.index[s.scene]]
    dt = float(s.task.dt)
    raw = s.sim.raw

    def pose():
        return dict(qpos=np.array(raw.qpos, dtype=np.float64), qvel=np.array(raw.qvel, dtype=np.float64), time=float(raw.time))

    frames, events, outcome = [pose()], [], dict(falls=0, contact_steps=0, contact_parts={})
    steps = int(a.seconds / dt)
    if a.mode == 'goals':
        if a.goals:
            goals = list(np.asarray(a.goals, dtype=np.float64).reshape(-1, 2))
        else:
            start = np.asarray(raw.qpos[:2], dtype=np.float64)
            end = np.asarray(record.get('goal', record.get('goal_xy'))[:2], dtype=np.float64)
            length = float(np.linalg.norm(end - start)); direction = (end - start) / max(length, 1e-9)
            goals = [start + direction * d for d in np.arange(a.goal_spacing, length, a.goal_spacing)] + [end]
        s.mode = 'waypoint'
        current, reached_at, active = -1, None, None
        for k in range(steps):
            xy = np.asarray(raw.qpos[:2], dtype=np.float64)
            w = s.waypoint
            # The goal's fields are computed on a background thread: count arrival only once the new goal is active.
            live = w.goal is not None and active is not None and float(np.linalg.norm(np.asarray(w.goal) - active)) < 1e-6
            arrived = live and current >= 0 and float(np.linalg.norm(w.goal - xy)) < w.REACHED
            if current < 0 or (arrived and reached_at is None):
                if current >= 0:
                    reached_at = k
                    events.append(dict(t=k * dt, kind='reached', index=current, xy=[float(v) for v in w.goal]))
            if current < 0 or (reached_at is not None and (k - reached_at) * dt >= a.hold):
                current += 1; reached_at = None
                if current >= len(goals):
                    outcome['all_goals_reached'] = True
                    break
                used = s.set_goal(goals[current])
                active = np.asarray(used if used is not None else goals[current], dtype=np.float64)
                events.append(dict(t=k * dt, kind='goal', index=current,
                                   xy=[float(v) for v in (used if used is not None else goals[current])]))
            status = s.step()
            frames.append(pose())
            outcome['contact_steps'] += bool(status['touching'])
            for part in status['touching']:
                outcome['contact_parts'][part] = outcome['contact_parts'].get(part, 0) + 1
            if status['fell']:
                outcome['falls'] += 1; frames.pop(); break
        outcome.update(goals=len(goals), goals_reached=sum(e['kind'] == 'reached' for e in events))
    else:
        schedule = []
        for item in a.keys.split(','):
            keys, seconds = item.split(':')
            schedule.append((set() if keys == 'none' else set(keys.split('+')), float(seconds)))
        s.hold_mode = True
        k = 0
        for keys, seconds in schedule:
            s.held = set(keys)
            events.append(dict(t=k * dt, kind='keys', keys=sorted(keys)))
            for _ in range(int(seconds / dt)):
                status = s.step(); frames.append(pose()); k += 1
                outcome['contact_steps'] += bool(status['touching'])
                for part in status['touching']:
                    outcome['contact_parts'][part] = outcome['contact_parts'].get(part, 0) + 1
                if status['fell']:
                    outcome['falls'] += 1; frames.pop(); break
            if outcome['falls']:
                break
        events.append(dict(t=k * dt, kind='keys', keys=[]))
    times = np.array([f['time'] for f in frames])
    keep = np.concatenate(([True], np.diff(times) > 0))
    arrays = {key: np.stack([f[key] for f, ok in zip(frames, keep) if ok]) for key in ('qpos', 'qvel', 'time')}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.control-', dir=a.out.parent) as tmp:
        d = Path(tmp)
        scene, is_cat = write_geometry(d, Path(a.bank + '_packed/manifest.json'), record)
        np.savez_compressed(d / 'trajectory.npz', **arrays)
        (d / 'events.json').write_text(json.dumps(events, indent=1) + '\n')
        success = outcome['falls'] == 0 and (a.mode == 'keyboard' or outcome.get('all_goals_reached', False))
        metadata = dict(schema='cat-control-recording-v1', mode=a.mode, scene_id=record['scene_id'], family=record['family'],
                        scene_seed=scene.get('seed'), seed=0, checkpoint_path=a.checkpoint, checkpoint_steps=None,
                        dt=dt, upper_target_average=True, speed=a.speed, keys=a.keys if a.mode == 'keyboard' else None,
                        outcome=dict(outcome, goal_reached=success, success=success),
                        trajectory_sha256=sha256(d / 'trajectory.npz'), is_cat=bool(is_cat),
                        conditions='browser-demo session: clean (no observation noise, gain or torque noise, pushes, delay)')
        (d / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        os.rename(d, a.out)
    print(json.dumps(dict(out=str(a.out), frames=len(arrays['time']), outcome=metadata['outcome']), indent=1))


if __name__ == '__main__':
    main()
