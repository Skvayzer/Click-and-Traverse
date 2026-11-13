#!/usr/bin/env python3
"""How a policy passes table edges: sideways (turned body) or forward with the hands raised / tucked.

Runs the table-edge passage scenes (source kind table-edge-passage, 16 open + 16 forward_protected) for N episodes each,
deterministic actions, no pushes, nominal conditions unless --robust. Style is measured on walking steps (root speed
> 0.15 m/s) near the table (a hand within 0.30 m of an obstacle):
  sideways   the body faces 45-135 deg away from where it is moving (turned to sidle past)
  raised     a hand above the tabletop band (palm z > 0.80 m; tabletops are 0.70-0.72 m)
  tucked     neither hand raised, both palms within 0.22 m of the pelvis axis laterally (body frame)
Outcomes per episode as eval_per_scene: clean goal, obstacle contact, hand contact, fall.

  eval_table_style.py CKPT [CKPT ...] --episodes 8 --out result.json
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


def evaluate(checkpoint, episodes, robust, bank, seed, max_steps):
    from record_mjlab_rollout import load_policy
    from cat_mjlab.runner import create_task
    dev = 'cuda:0'
    contract = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)['contract']
    env = copy.deepcopy(contract['environment_config'])
    for key in ('side_gap_curriculum', 'sgs', 'reactive_stationary'):
        env.pop(key, None)
    env['push_config']['enable'] = bool(robust)
    if not robust:
        env.pop('robustness', None)
    if 'teleop' in env:
        env['teleop'].update(fraction=0., cat_fraction=0., cat_scenes=False)
    B = ROOT / bank
    scenes_manifest = json.load(open(str(B) + '_packed/manifest.json'))['scenes']
    ids = [i for i, s in enumerate(scenes_manifest) if (s.get('source') or {}).get('kind') == 'table-edge-passage']
    roles = {i: ((scenes_manifest[i]['source'].get('hand_contrast') or {}).get('role')) for i in ids}
    n_envs = len(ids) * episodes
    ns = SimpleNamespace(bank_manifest=Path(str(B) + '_packed/manifest.json'), body_collision_bank=Path(str(B) + '_collision/manifest.json'),
                         body_collision_resets=Path(str(B) + '_resets/manifest.json'), device=dev, seed=seed, num_envs=n_envs,
                         compile_task=False, nconmax=contract['nconmax'], njmax=contract['njmax'], reactive_bank=None, reactive_row=None)
    task, _, _ = create_task(ns, environment_config=env)
    job = torch.tensor(ids, device=dev).repeat(episodes)          # one episode per env, scene fixed per env
    original_reset = task.reset
    task.reset = lambda env_ids=None, scene_ids=None: original_reset(
        task.all_ids if env_ids is None else env_ids, job[(task.all_ids if env_ids is None else env_ids).long()])
    task.reset()
    learner, _ = load_policy(checkpoint, device=dev, policy_id=0)
    pelvis_site = 1
    live = torch.ones(n_envs, dtype=torch.bool, device=dev)             # first episode of each env still running
    z = lambda: torch.zeros(n_envs, device=dev)
    acc = dict(walk_near=z(), sideways=z(), raised=z(), tucked=z(), backward=z())
    out_flags = {k: torch.zeros(n_envs, dtype=torch.bool, device=dev) for k in ('done', 'goal', 'obstacle', 'hands', 'fall')}
    start = time.time()
    for t in range(max_steps):
        with torch.no_grad():
            out = task.step(learner.act(task.obs, policy_ids=0, deterministic=True)['action'])
        i, d = task.info, task.data
        q = d.qpos[:, 3:7]
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        vel = d.qvel[:, :2]; speed = torch.linalg.vector_norm(vel, dim=-1)
        moving_dir = torch.atan2(vel[:, 1], vel[:, 0])
        angle = torch.remainder(yaw - moving_dir + math.pi, 2 * math.pi) - math.pi
        angle = angle.abs()                                              # 0 forward, pi/2 sideways, pi backward
        hands = i['positions'][:, 5:7]                                   # left, right palm (world)
        near = i['sdf'][:, 5:7, 0].amin(-1) < .30
        walk_near = live & (speed > .15) & near
        raised = (hands[:, :, 2] > .80).any(-1)
        rel = hands[:, :, :2] - i['positions'][:, pelvis_site:pelvis_site + 1, :2]
        lateral = (-torch.sin(yaw)[:, None] * rel[..., 0] + torch.cos(yaw)[:, None] * rel[..., 1]).abs()
        tucked = ~raised & (lateral < .22).all(-1)
        acc['walk_near'] += walk_near
        acc['sideways'] += walk_near & (angle > math.pi / 4) & (angle < 3 * math.pi / 4)
        acc['backward'] += walk_near & (angle >= 3 * math.pi / 4)
        acc['raised'] += walk_near & raised
        acc['tucked'] += walk_near & tucked
        done = out['done'].bool() & live
        if bool(done.any()):
            m = out['metrics']; zero = torch.zeros_like(done)
            out_flags['done'] |= done
            out_flags['goal'] |= done & m.get('episode/goal_reached', m['successful']).bool()
            out_flags['obstacle'] |= done & m.get('episode/obstacle', zero).bool()
            out_flags['hands'] |= done & m.get('episode/body_collision_hands', zero).bool()
            out_flags['fall'] |= done & m.get('episode/fall', zero).bool()
            live &= ~done
        if not bool(live.any()):
            break
    def summarise(mask):
        w = float(acc['walk_near'][mask].sum())
        e = float(out_flags['done'][mask].sum())
        f = lambda k: round(float(acc[k][mask].sum()) / w, 4) if w else None
        g = lambda k: round(float((out_flags[k] & out_flags['done'])[mask].sum()) / e, 4) if e else None
        return dict(episodes=int(e), walking_near_table_steps=int(w), sideways=f('sideways'), backward=f('backward'),
                    hand_raised=f('raised'), hands_tucked=f('tucked'), goal=g('goal'), clean_goal=round(float((out_flags['goal'] & ~out_flags['obstacle'])[mask].sum()) / e, 4) if e else None,
                    obstacle_contact=g('obstacle'), hand_contact=g('hands'), fall=g('fall'))
    role = torch.tensor([0 if roles[int(s)] == 'open' else 1 for s in job], device=dev)
    return dict(checkpoint=str(checkpoint), robust=robust, episodes_per_scene=episodes, steps=t + 1, seconds=round(time.time() - start),
                all=summarise(torch.ones_like(live)), open=summarise(role == 0), forward_protected=summarise(role == 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', nargs='+')
    p.add_argument('--episodes', type=int, default=8)
    p.add_argument('--robust', action='store_true')
    p.add_argument('--bank', default='data/furniture/sidegap_v2')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=3000)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    results = []
    for c in a.checkpoint:
        r = evaluate(c, a.episodes, a.robust, a.bank, a.seed, a.max_steps)
        print(json.dumps(r), flush=True)
        results.append(r)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
