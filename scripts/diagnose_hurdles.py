#!/usr/bin/env python3
"""Hurdle-scene diagnostic (usage: diagnose_hurdles.py CHECKPOINT field|keyboard). Hurdle-scene diagnostic: keyboard vs field driving, v2 vs v5, which body part ends episodes and under which command."""
import copy, json, sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT = Path('/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from record_mjlab_rollout import load_policy
from cat_mjlab.runner import create_task
dev = 'cuda:0'
c = torch.load(sys.argv[1], map_location='cpu', weights_only=True, mmap=True)['contract']
env = copy.deepcopy(c['environment_config'])
env.pop('side_gap_curriculum', None)      # this bank has no side-gap scenes
env['push_config']['enable'] = False
mode = sys.argv[2]
STOCHASTIC = len(sys.argv) > 3 and sys.argv[3] == 'stochastic'   # sample actions as in training
if 'teleop' in env:
    env['teleop'].update(fraction=1. if mode == 'keyboard' else 0., cat_fraction=1. if mode == 'keyboard' else 0., cat_scenes=mode == 'keyboard')
B = 'data/furniture/table_edges_v1'
ns = SimpleNamespace(bank_manifest=ROOT / (B + '_packed/manifest.json'), body_collision_bank=ROOT / (B + '_collision/manifest.json'),
                     body_collision_resets=ROOT / (B + '_resets/manifest.json'), device=dev, seed=1, num_envs=1024, compile_task=False,
                     nconmax=c['nconmax'], njmax=c['njmax'], reactive_bank=None, reactive_row=None)
task, sim, _ = create_task(ns, environment_config=env)
ids = [i for i, s in enumerate(task.bank.manifest['scenes']) if s['scene_id'].startswith(('published-hurdle', 'published-multi-hurdle'))
       and 'crouch' not in s['scene_id']]
pool = torch.tensor(ids, device=dev)
orig = task.reset
def pinned(env_ids=None, scene_ids=None):
    n = task.num_envs if env_ids is None else len(env_ids)
    if scene_ids is None:
        scene_ids = pool[torch.randint(len(pool), (n,), device=dev)]
    return orig(env_ids, scene_ids)
task.reset = pinned
task.reset()
learner, _ = load_policy(sys.argv[1], device=dev, policy_id=0)
REG = ('feet', 'legs', 'trunk', 'head', 'arms', 'hands')
stats = dict(episodes=0, success=0, timeout=0, contact=0, fall=0, **{'hit_' + r: 0 for r in REG})
cmd = dict(stop=0, turn=0, through=0, other=0)
for t in range(3000):
    with torch.no_grad():
        jb = task.joystick_body.clone() if getattr(task, 'joystick_body', None) is not None else None
        out = task.step(learner.act(task.obs, policy_ids=0, deterministic=not STOCHASTIC)['action'])
    done = out['done'].bool(); m = out['metrics']
    stats['success'] += int(m['successful'].bool().sum())     # recorded at the step the goal is reached
    if not bool(done.any()): continue
    d = torch.nonzero(done).flatten()
    stats['episodes'] += len(d)
    contact = m.get('episode/obstacle', torch.zeros_like(done))[d].bool()
    stats['contact'] += int(contact.sum()); stats['fall'] += int(m.get('episode/fall', torch.zeros_like(done))[d].bool().sum())
    stats['timeout'] += int((out['truncated'].bool()[d]).sum())
    for r in REG:
        k = 'episode/body_collision_' + r
        if k in m: stats['hit_' + r] += int(m[k][d].bool().sum())
    if jb is not None and mode == 'keyboard':
        b = jb[d][contact]
        stop = (b.abs().sum(-1) < 1e-6); turn = ~stop & (b[:, :2].abs().sum(-1) < 1e-6)
        through = ~stop & ~turn & (b[:, 1].abs() < 1e-6) & (b[:, 2].abs() < 1e-6) & (b[:, 0] > 0)
        cmd['stop'] += int(stop.sum()); cmd['turn'] += int(turn.sum()); cmd['through'] += int(through.sum()); cmd['other'] += int((~stop & ~turn & ~through).sum())
E = max(stats['episodes'], 1)
print(json.dumps(dict(policy=Path(sys.argv[1]).name, mode=mode, actions='stochastic' if STOCHASTIC else 'deterministic', scenes=len(ids), episodes=stats['episodes'], refused=round(1 - stats['success'] / E - stats['contact'] / E, 3),
                      success=round(stats['success'] / E, 3), contact=round(stats['contact'] / E, 3), timeout=round(stats['timeout'] / E, 3),
                      fall=round(stats['fall'] / E, 3), hit={r: round(stats['hit_' + r] / E, 3) for r in REG},
                      **({'command_at_contact': cmd} if mode == 'keyboard' else {}))))
