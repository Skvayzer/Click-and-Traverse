#!/usr/bin/env python3
"""Side-gap diagnostic (GPU, 1024 robots, deterministic): outcome, contact part, facing and lateral alignment at the wall.

usage: diagnose_side_gaps.py CHECKPOINT [free|sideways|forward]
  free      the training reset (CAT scatter)
  sideways  placed sideways (+-90 deg), lined up with the gap, 0.45 m before the wall
  forward   placed facing +x, lined up, 0.45 m before the wall
"""
import copy, json, math, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
ROOT = Path('/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from record_mjlab_rollout import load_policy
from cat_mjlab.runner import create_task
from cat_ppo.furniture.generalist_fields import scene_directory
dev = 'cuda:0'
c = torch.load(sys.argv[1], map_location='cpu', weights_only=True, mmap=True)['contract']
env = copy.deepcopy(c['environment_config']); env['push_config']['enable'] = False
B = 'data/furniture/sidegap_v2'
ns = SimpleNamespace(bank_manifest=ROOT / (B + '_packed/manifest.json'), body_collision_bank=ROOT / (B + '_collision/manifest.json'),
                     body_collision_resets=ROOT / (B + '_resets/manifest.json'), device=dev, seed=3, num_envs=1024, compile_task=False,
                     nconmax=c['nconmax'], njmax=c['njmax'], reactive_bank=None, reactive_row=None)
task, sim, _ = create_task(ns, environment_config=env)
man = task.bank.manifest; mp = ROOT / (B + '_packed/manifest.json')
ids, wall_x, gap_y, variant, width = [], {}, {}, {}, {}
for i, s in enumerate(man['scenes']):
    if not s['scene_id'].startswith('sidegap2'): continue
    spec = json.loads((scene_directory(man, mp, s) / 'scene.json').read_text())
    xs = []
    for b in spec['boxes']:
        cth, sth = math.cos(b['yaw']), math.sin(b['yaw'])
        for sx in (-1, 1):
            for sy in (-1, 1):
                xs.append(b['center'][0] + sx * b['half_size'][0] * cth - sy * b['half_size'][1] * sth)
    ids.append(i); wall_x[i] = min(xs); gap_y[i] = spec['side_gap']['offset_m']; variant[i] = spec['side_gap']['variant']; width[i] = spec['side_gap']['gap_m']
pool = torch.tensor(ids, device=dev)
WX = torch.zeros(len(man['scenes']), device=dev); GY = torch.zeros_like(WX)
for i in ids: WX[i] = wall_x[i]; GY[i] = gap_y[i]
orig = task.reset
MODE = sys.argv[2] if len(sys.argv) > 2 else 'free'
def pinned(env_ids=None, scene_ids=None):
    n = task.num_envs if env_ids is None else len(env_ids)
    out = orig(env_ids, pool[torch.randint(len(pool), (n,), device=dev)] if scene_ids is None else scene_ids)
    if MODE in ('sideways', 'forward'):
        e = torch.arange(task.num_envs, device=dev) if env_ids is None else env_ids
        sc = task.scene_ids[e]; yaw = torch.full((len(e),), math.pi / 2 if MODE == 'sideways' else 0., device=dev)
        if MODE == 'sideways': yaw = yaw * torch.where(torch.rand(len(e), device=dev) < .5, -1., 1.)
        q = sim.data.qpos
        q[e, 0] = WX[sc] - .45; q[e, 1] = GY[sc]
        q[e, 3] = torch.cos(yaw / 2); q[e, 4] = 0.; q[e, 5] = 0.; q[e, 6] = torch.sin(yaw / 2)
        sim.data.qvel[e] = 0.; task.info['odom_delay'][e] = q[e, :7]
        sim.forward(e)
    return out
task.reset = pinned; task.reset()
learner, _ = load_policy(sys.argv[1], device=dev, policy_id=0)
N = task.num_envs; near_steps = torch.zeros(N, device=dev); dwell_steps = torch.zeros(N, device=dev); rec_yaw = torch.full((N,), float('nan'), device=dev); rec_lat = torch.full((N,), float('nan'), device=dev)
rows = []
REG = ('feet', 'legs', 'trunk', 'head', 'arms', 'hands')
for t in range(3000):
    scene = task.scene_ids.clone()
    q = sim.data.qpos; x = q[:, 0].clone(); y = q[:, 1].clone()   # clones: task.step resets done worlds in place
    yaw = torch.atan2(2 * (q[:, 3] * q[:, 6] + q[:, 4] * q[:, 5]), 1 - 2 * (q[:, 5] ** 2 + q[:, 6] ** 2))
    # Dwell: within 1 m before the wall, turned sideways (|sin yaw| > 0.7) and nearly still (< 0.1 m/s) --
    # what an unconditioned sideways bonus pays for without passing.
    before = (x >= WX[scene] - 1.0) & (x <= WX[scene])
    still = torch.linalg.vector_norm(sim.data.qvel[:, :2], dim=-1) < .1
    near_steps += before.float(); dwell_steps += (before & (torch.sin(yaw).abs() > .7) & still).float()
    near = torch.isnan(rec_yaw) & (x >= WX[scene] - .35)
    rec_yaw = torch.where(near, yaw.abs(), rec_yaw); rec_lat = torch.where(near, (y - GY[scene]).abs(), rec_lat)
    with torch.no_grad():
        out = task.step(learner.act(task.obs, policy_ids=0, deterministic=True)['action'])
    m = out['metrics']; done = out['done'].bool()
    if 'successful' in m:
        succ_step = m['successful'].bool()
    if bool(done.any()):
        d = torch.nonzero(done).flatten()
        for j in d.tolist():
            parts = [r for r in REG if bool(m.get('episode/body_collision_' + r, torch.zeros_like(done))[j])]
            hv = bool(m.get('episode/hand_violation', torch.zeros_like(done))[j]); ev = bool(m.get('episode/elbow_violation', torch.zeros_like(done))[j])
            goal = bool(m.get('episode/goal_reached', torch.zeros_like(done))[j]) if 'episode/goal_reached' in m else None
            rows.append(dict(near=float(near_steps[j]), dwell=float(dwell_steps[j]), passed=(float(x[j]) > 1.95) and not bool(m.get('episode/obstacle', torch.zeros_like(done))[j]), scene=int(scene[j]), parts=parts, hand_violation=hv, elbow_violation=ev, contact=bool(m.get('episode/obstacle', torch.zeros_like(done))[j]),
                             goal=goal, yaw_at_wall=float(rec_yaw[j]), lateral_at_wall=float(rec_lat[j]), x=float(x[j]), length=int(m['episode_length'][j])))
        rec_yaw[d] = float('nan'); rec_lat[d] = float('nan'); near_steps[d] = 0.; dwell_steps[d] = 0.
R = rows; n = len(R)
reach = [r for r in R if not math.isnan(r['yaw_at_wall'])]
print(json.dumps(dict(start=MODE, policy=Path(sys.argv[1]).name, episodes=n, reached_wall=round(len(reach) / n, 3),
    passed_clean=round(sum(r['passed'] for r in R) / n, 3), contact=round(sum(r['contact'] for r in R) / n, 3), never_reached_wall=round(sum(math.isnan(r['yaw_at_wall']) for r in R) / n, 3))))
from collections import Counter
near_total = sum(r['near'] for r in R); dwell_total = sum(r['dwell'] for r in R)
print(json.dumps(dict(dwell_share_of_time_before_wall=round(dwell_total / max(near_total, 1), 3),
                      episodes_dwelling_over_5s=round(sum(r['dwell'] > 250 for r in R) / n, 3),
                      mean_dwell_s=round(dwell_total / n * .02, 2))))
print('contact parts:', dict(Counter(p for r in R if r['contact'] for p in (r['parts'] or (['hand_envelope'] if r['hand_violation'] else ['elbow'] if r['elbow_violation'] else ['field'])))))
yaws = np.degrees([r['yaw_at_wall'] for r in reach]); lats = np.array([r['lateral_at_wall'] for r in reach])
print('facing when reaching the wall (|yaw| deg): <20:', int((yaws < 20).sum()), ' 20-60:', int(((yaws >= 20) & (yaws < 60)).sum()), ' 60-120 (sideways):', int(((yaws >= 60) & (yaws <= 120)).sum()), ' >120:', int((yaws > 120).sum()))
print('lateral offset from gap centre when reaching the wall: median %.2f m, share within gap half-width: %.2f' % (np.median(lats), np.mean([r['lateral_at_wall'] < width[r['scene']] / 2 for r in reach])))
by = {}
for r in R:
    v = variant[r['scene']]; b = by.setdefault(v, Counter()); b['n'] += 1; b['contact'] += r['contact']; b['passed'] += r['passed']; b['reached'] += not math.isnan(r['yaw_at_wall'])
    b['sideways'] += (not math.isnan(r['yaw_at_wall'])) and 60 <= math.degrees(r['yaw_at_wall']) <= 120
for v, b in sorted(by.items()):
    print(f"  {v:11s} n={b['n']:4d} passed {b['passed']/b['n']:.2f} contact {b['contact']/b['n']:.2f} reached wall {b['reached']/b['n']:.2f} sideways at wall {b['sideways']/max(b['reached'],1):.2f}")
