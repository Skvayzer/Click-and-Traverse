#!/usr/bin/env python3
"""Why the robot stops in front of narrow room passages: reward terms and command while it stalls versus while it passes.

Runs the hand-contrast passages of one role (default 'narrow') with a gap below --max-width for N deterministic episodes
each, nominal, no pushes. Every step is put in a phase by the pelvis progress along the certified route and its speed:
  approach   before the first passage zone (progress < first zone start)
  stalled    progress-EMA speed < 0.1 m/s while the passage still lies ahead (same test as the training's stall term)
  inside     pelvis between the cabinets of a module (hazard_start_m .. hazard_end_m)
  other      the rest (moving between modules, after the last one)
Per phase it averages the SCALED per-step reward terms as the training applies them (the step's reward is their sum
times dt), the commanded speed (zero = the route command says stop), whether the route carrot is visible, the hand /
elbow clearances and the share of steps. Outcomes as eval_per_scene (clean / contact / refused).

  diagnose_stalling.py CKPT [CKPT ...] --bank data/furniture/tablegap_v1 --episodes 4 --out stalling.json
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
PHASES = ('approach', 'stalled', 'inside', 'other')


def diagnose(checkpoint, bank, episodes, role, max_width, max_steps, seed):
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

    def width(s):
        modules = ((s.get('source') or {}).get('hand_contrast') or {}).get('modules') or [{}]
        return float(modules[0].get('width_m', math.inf))
    ids = [i for i, s in enumerate(scenes) if (s.get('source') or {}).get('kind') == 'contrastive-hand-passage'
           and s['source']['hand_contrast'].get('role') == role and width(s) < max_width]
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
    # route frame and the cabinet spans (hazard intervals) of each env's scene
    start = torch.tensor([scenes[int(s)]['start'][:2] for s in job], device=dev)
    goal = torch.tensor([scenes[int(s)]['goal'][:2] for s in job], device=dev)
    tangent = (goal - start) / torch.linalg.vector_norm(goal - start, dim=-1, keepdim=True)
    zones = [scenes[int(s)]['source']['hand_contrast']['zones'] for s in job]
    first_zone = torch.tensor([z[0]['start_m'] for z in zones], device=dev)
    hazards = torch.tensor([[(z_['hazard_start_m'], z_['hazard_end_m']) for z_ in z][:3] for z in zones], device=dev)
    live = torch.ones(n_envs, dtype=torch.bool, device=dev)
    ema = torch.zeros(n_envs, device=dev)
    sums = {}                       # (phase, key) -> [n_envs] running sums
    counts = {p: torch.zeros(n_envs, device=dev) for p in PHASES}
    outcome = {k: torch.zeros(n_envs, dtype=torch.bool, device=dev) for k in ('done', 'goal', 'obstacle', 'fall')}
    max_progress = torch.zeros(n_envs, device=dev)
    t0 = time.time()
    for t in range(max_steps):
        with torch.no_grad():
            out = task.step(learner.act(task.obs, policy_ids=0, deterministic=True)['action'])
        i, d, m = task.info, task.data, out['metrics']
        done = out['done'].bool() & live
        root = d.qpos[:, :2] - start
        progress = (root * tangent).sum(-1)
        vx = (d.qvel[:, :2] * tangent).sum(-1)
        ema = .98 * ema + .02 * vx
        inside = ((progress[:, None] >= hazards[..., 0]) & (progress[:, None] <= hazards[..., 1])).any(-1)
        ahead_of_last = progress < hazards[:, -1, 1]
        stalled = (ema < .1) & ahead_of_last & (t > 25) & ~inside
        approach = (progress < first_zone) & ~stalled
        phase = {'stalled': stalled, 'inside': inside & ~stalled, 'approach': approach & ~inside}
        phase['other'] = ~(phase['stalled'] | phase['inside'] | phase['approach'])
        running = live & ~done                                         # a finished env is already reset
        cmd = i['command']
        values = {k[len('reward/'):]: v.float() for k, v in m.items() if k.startswith('reward/') and v.ndim == 1}
        values['command_speed'] = torch.linalg.vector_norm(cmd[:, 1:3], dim=-1)
        values['command_stop'] = (cmd[:, 0] < .5).float()
        values['route_blocked'] = task.navigation['blocked'].float() if 'blocked' in task.navigation else torch.zeros(n_envs, device=dev)
        values['hand_clearance_min'] = i['sdf'][:, 5:7, 0].amin(-1)
        values['speed'] = torch.linalg.vector_norm(d.qvel[:, :2], dim=-1)
        for p, mask in phase.items():
            w = (mask & running).float()
            counts[p] += w
            for k, v in values.items():
                key = (p, k)
                if key not in sums:
                    sums[key] = torch.zeros(n_envs, device=dev)
                sums[key] += w * torch.nan_to_num(v, nan=0., posinf=0., neginf=0.)
        max_progress = torch.where(running, torch.maximum(max_progress, progress), max_progress)
        if bool(done.any()):
            zero = torch.zeros_like(done)
            outcome['done'] |= done
            outcome['goal'] |= done & m.get('episode/goal_reached', m['successful']).bool()
            outcome['obstacle'] |= done & m.get('episode/obstacle', zero).bool()
            outcome['fall'] |= done & m.get('episode/fall', zero).bool()
            live &= ~done
        if not bool(live.any()):
            break
    total = sum(float(c.sum()) for c in counts.values())
    per_phase = {}
    for p in PHASES:
        n = float(counts[p].sum())
        if n == 0:
            continue
        keys = sorted({k for (q, k) in sums if q == p})
        per_phase[p] = dict(step_share=round(n / total, 4),
                            **{k: round(float(sums[(p, k)].sum()) / n, 4) for k in keys})
    E = n_envs
    clean = outcome['goal'] & ~outcome['obstacle']
    refused = ~outcome['done'] | (outcome['done'] & ~outcome['goal'] & ~outcome['obstacle'] & ~outcome['fall'])
    result = dict(checkpoint=str(checkpoint), role=role, max_width=max_width, scenes=len(ids), episodes=E, steps=t + 1,
                  seconds=round(time.time() - t0),
                  outcomes=dict(clean=round(float(clean.sum()) / E, 3), contact=round(float(outcome['obstacle'].sum()) / E, 3),
                                refused=round(float(refused.sum()) / E, 3), fall=round(float(outcome['fall'].sum()) / E, 3),
                                mean_max_progress_m=round(float(max_progress.mean()), 3)),
                  phases=per_phase)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', nargs='+')
    p.add_argument('--bank', default='data/furniture/tablegap_v1')
    p.add_argument('--episodes', type=int, default=4)
    p.add_argument('--role', default='narrow')
    p.add_argument('--max-width', type=float, default=.60)
    p.add_argument('--max-steps', type=int, default=3000)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    results = []
    for c in a.checkpoint:
        r = diagnose(c, a.bank, a.episodes, a.role, a.max_width, a.max_steps, a.seed)
        print(json.dumps(dict(checkpoint=c, outcomes=r['outcomes'])), flush=True)
        results.append(r)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
