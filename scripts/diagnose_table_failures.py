#!/usr/bin/env python3
"""Why narrow table passages fail: posture and placement at the moment an episode ends, and while between the tables.

Runs the forward_protected table-edge scenes with a gap <= --max-gap (the original 0.44 m pairs and the narrow end of
the width curriculum) for N deterministic episodes each, no pushes, nominal conditions (as eval_table_style.py).
Per episode it records how it ended (goal, timeout, fall, obstacle and which body region / hand violation), and at the
last step: progress along the route, pelvis offset from the route line, body yaw relative to the route, palm heights,
and the site closest to an obstacle. While between the tables (pelvis within 0.40 m of an obstacle, walking) it
accumulates the mean |lateral offset|, mean |yaw| and the share of steps with both palms above the tabletop band.

  diagnose_table_failures.py CKPT --bank data/furniture/tablegap_v1 --episodes 8 --out diag.json
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
SITES = ('head', 'pelvis', 'torso', 'left_foot', 'right_foot', 'left_hand', 'right_hand', 'left_knee', 'right_knee',
         'left_shoulder', 'right_shoulder')


def diagnose(checkpoint, episodes, bank, seed, max_steps, max_gap):
    from record_mjlab_rollout import load_policy
    from cat_mjlab.runner import create_task
    dev = 'cuda:0'
    contract = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)['contract']
    env = copy.deepcopy(contract['environment_config'])
    for key in ('side_gap_curriculum', 'sgs', 'reactive_stationary', 'robustness'):
        env.pop(key, None)
    env['push_config']['enable'] = False
    if 'teleop' in env:
        env['teleop'].update(fraction=0., cat_fraction=0., cat_scenes=False)
    B = ROOT / bank
    scenes = json.load(open(str(B) + '_packed/manifest.json'))['scenes']

    def gap_of(s):
        src = s['source']
        if src.get('table_gap_curriculum'): return float(src['table_gap_curriculum']['gap_m'])
        return float(src['hand_contrast']['table_gap_m'])
    ids = [i for i, s in enumerate(scenes) if (s.get('source') or {}).get('kind') == 'table-edge-passage'
           and s['source']['hand_contrast']['role'] == 'forward_protected' and gap_of(s) <= max_gap]
    n_envs = len(ids) * episodes
    ns = SimpleNamespace(bank_manifest=Path(str(B) + '_packed/manifest.json'), body_collision_bank=Path(str(B) + '_collision/manifest.json'),
                         body_collision_resets=Path(str(B) + '_resets/manifest.json'), device=dev, seed=seed, num_envs=n_envs,
                         compile_task=False, nconmax=contract['nconmax'], njmax=contract['njmax'], reactive_bank=None, reactive_row=None)
    task, _, _ = create_task(ns, environment_config=env)
    job = torch.tensor(ids, device=dev).repeat(episodes)
    original_reset = task.reset
    task.reset = lambda env_ids=None, scene_ids=None: original_reset(
        task.all_ids if env_ids is None else env_ids, job[(task.all_ids if env_ids is None else env_ids).long()])
    task.reset()
    learner, _ = load_policy(checkpoint, device=dev, policy_id=0)
    start = torch.tensor([scenes[int(s)]['start'][:2] for s in job], device=dev)
    goal = torch.tensor([scenes[int(s)]['goal'][:2] for s in job], device=dev)
    route = (goal - start) / torch.linalg.vector_norm(goal - start, dim=-1, keepdim=True)
    normal = torch.stack((-route[:, 1], route[:, 0]), -1)
    top = torch.tensor([float(scenes[int(s)]['source']['hand_contrast']['tabletop_top_z_m']) for s in job], device=dev)
    gaps = [gap_of(scenes[int(s)]) for s in job]
    threshold = -float(env.get('term_collision_threshold', .04))
    live = torch.ones(n_envs, dtype=torch.bool, device=dev)
    z = lambda: torch.zeros(n_envs, device=dev)
    acc = dict(steps=z(), lateral=z(), yaw=z(), sideways=z(), both_up=z(), one_up=z(), knee_close=z())
    last = {}
    flags = {k: torch.zeros(n_envs, dtype=torch.bool, device=dev) for k in
             ('done', 'goal', 'obstacle', 'fall', 'hand_violation', 'elbow_violation', 'body_collision',
              'body_collision_feet', 'body_collision_legs', 'body_collision_trunk', 'body_collision_head',
              'body_collision_arms', 'body_collision_hands')}
    end_step = torch.full((n_envs,), max_steps, device=dev)
    prev = None
    for t in range(max_steps):
        with torch.no_grad():
            out = task.step(learner.act(task.obs, policy_ids=0, deterministic=True)['action'])
        i, d = task.info, task.data
        q = d.qpos[:, 3:7]
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        rel_yaw = torch.remainder(yaw - torch.atan2(route[:, 1], route[:, 0]) + math.pi, 2 * math.pi) - math.pi
        pelvis = i['positions'][:, 1, :2] - start
        progress = (pelvis * route).sum(-1); lateral = (pelvis * normal).sum(-1)
        palm_z = i['positions'][:, 5:7, 2]
        sdf = i['sdf'][:, :11, 0]
        speed = torch.linalg.vector_norm(d.qvel[:, :2], dim=-1)
        between = live & (sdf[:, 1] < .40) & (speed > .1)
        up = palm_z > (top[:, None] + .08)
        acc['steps'] += between
        acc['lateral'] += between * lateral.abs()
        acc['yaw'] += between * rel_yaw.abs()
        acc['sideways'] += between & (rel_yaw.abs() > math.pi / 4)
        acc['both_up'] += between & up.all(-1)
        acc['one_up'] += between & up.any(-1) & ~up.all(-1)
        acc['knee_close'] += between & (sdf[:, 7:9].amin(-1) < .03)
        # snapshot of the step before termination (the terminal step's state is the reset one)
        snap = dict(progress=progress, lateral=lateral, yaw_deg=rel_yaw * 180 / math.pi, palm_z=palm_z, sdf=sdf)
        done = out['done'].bool() & live
        if bool(done.any()):
            m = out['metrics']; zero = torch.zeros_like(done)
            flags['done'] |= done
            flags['goal'] |= done & m.get('episode/goal_reached', m['successful']).bool()
            for k in flags:
                if k not in ('done', 'goal'):
                    flags[k] |= done & m.get('episode/' + k, zero).bool()
            end_step = torch.where(done, torch.full_like(end_step, t), end_step)
            for k, v in (prev or snap).items():
                last.setdefault(k, torch.zeros_like(v))
                last[k] = torch.where(done.view(-1, *([1] * (v.dim() - 1))), v, last[k])
            live &= ~done
        prev = snap
        if not bool(live.any()):
            break
    for k, v in prev.items():                                            # still running at max_steps (timeouts)
        last.setdefault(k, torch.zeros_like(v))
        last[k] = torch.where(live.view(-1, *([1] * (v.dim() - 1))), v, last[k])
    rows = []
    for e in range(n_envs):
        f = {k: bool(v[e]) for k, v in flags.items()}
        steps = float(acc['steps'][e])
        sd = last['sdf'][e].tolist()
        rows.append(dict(scene=int(job[e]), gap=round(gaps[e], 3), end_step=int(end_step[e]),
                         outcome='goal' if f['goal'] else 'fall' if f['fall'] else 'obstacle' if f['obstacle'] else 'timeout' if not f['done'] else 'other',
                         touched=[k.replace('body_collision_', '') for k, v in f.items() if k.startswith('body_collision_') and v]
                                 + (['hand_violation'] if f['hand_violation'] else []) + (['elbow_violation'] if f['elbow_violation'] else []),
                         end=dict(progress=round(float(last['progress'][e]), 3), lateral=round(float(last['lateral'][e]), 3),
                                  yaw_deg=round(float(last['yaw_deg'][e]), 1), palm_z=[round(x, 3) for x in last['palm_z'][e].tolist()],
                                  closest_site=SITES[min(range(11), key=lambda k: sd[k])], closest_sdf=round(min(sd), 3),
                                  below_threshold=[SITES[k] for k in range(11) if sd[k] < threshold]),
                         between=dict(steps=int(steps), mean_abs_lateral=round(float(acc['lateral'][e]) / steps, 4) if steps else None,
                                      mean_abs_yaw_deg=round(float(acc['yaw'][e]) / steps * 180 / math.pi, 1) if steps else None,
                                      sideways=round(float(acc['sideways'][e]) / steps, 3) if steps else None,
                                      both_palms_above=round(float(acc['both_up'][e]) / steps, 3) if steps else None,
                                      one_palm_above=round(float(acc['one_up'][e]) / steps, 3) if steps else None,
                                      knee_within_3cm=round(float(acc['knee_close'][e]) / steps, 3) if steps else None)))
    return dict(checkpoint=str(checkpoint), episodes_per_scene=episodes, max_gap=max_gap, threshold=threshold, rows=rows)


def summary(result):
    """Aggregate per gap bin and outcome."""
    out = {}
    bins = (('original_0.44', 0., .45), ('0.46-0.50', .45, .50), ('0.50-0.54', .50, .545))
    for name, lo, hi in bins:
        rows = [r for r in result['rows'] if lo < r['gap'] <= hi]
        if not rows: continue
        agg = dict(episodes=len(rows))
        for o in ('goal', 'obstacle', 'timeout', 'fall'):
            agg[o] = round(sum(r['outcome'] == o for r in rows) / len(rows), 3)
        fails = [r for r in rows if r['outcome'] == 'obstacle']
        if fails:
            touched = {}
            for r in fails:
                for k in r['touched']: touched[k] = touched.get(k, 0) + 1
            agg['failures_touched'] = {k: round(v / len(fails), 3) for k, v in sorted(touched.items(), key=lambda kv: -kv[1])}
            closest = {}
            for r in fails: closest[r['end']['closest_site']] = closest.get(r['end']['closest_site'], 0) + 1
            agg['failures_closest_site'] = {k: round(v / len(fails), 3) for k, v in sorted(closest.items(), key=lambda kv: -kv[1])}
            mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
            agg['failures_at_end'] = dict(abs_yaw_deg=mean([abs(r['end']['yaw_deg']) for r in fails]),
                                          abs_lateral=mean([abs(r['end']['lateral']) for r in fails]),
                                          min_palm_z=mean([min(r['end']['palm_z']) for r in fails]),
                                          progress=mean([r['end']['progress'] for r in fails]))
        for o in ('goal', 'obstacle'):
            rs = [r['between'] for r in rows if r['outcome'] == o and r['between']['steps']]
            if rs:
                agg['between_' + o] = {k: round(sum(r[k] for r in rs) / len(rs), 3) for k in
                                      ('mean_abs_lateral', 'mean_abs_yaw_deg', 'sideways', 'both_palms_above', 'one_palm_above', 'knee_within_3cm')}
        out[name] = agg
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', nargs='+')
    p.add_argument('--episodes', type=int, default=8)
    p.add_argument('--bank', default='data/furniture/tablegap_v1')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=3000)
    p.add_argument('--max-gap', type=float, default=.545)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    results = []
    for c in a.checkpoint:
        r = diagnose(c, a.episodes, a.bank, a.seed, a.max_steps, a.max_gap)
        r['summary'] = summary(r)
        print(json.dumps(dict(checkpoint=c, summary=r['summary'])), flush=True)
        results.append(r)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
