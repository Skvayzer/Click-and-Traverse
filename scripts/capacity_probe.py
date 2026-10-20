#!/usr/bin/env python3
"""Is the three-teacher student capacity-limited? Offline width sweep on the student's own state distribution.

collect: CPU simulator, the DAgger student drives (action mean + 0.05 noise, as in DAgger), the scene's routed teacher
         labels every state (pre-squash action mean). Episodes start in scenes sampled uniformly from the routing
         owners (so narrow / traversal / rooms all appear), up to --episode-steps, contact ends them as in training.
fit:     the same supervised regression for several MLP widths on identical data (train/val split by episode);
         per-teacher validation MSE. Wider -> clearly lower VALIDATION error = the standard width was limiting;
         all widths plateau together = the limit is elsewhere (teacher conflict on similar states, data).

  capacity_probe.py collect --out DIR --shard K --steps N      (one CPU process per shard)
  capacity_probe.py fit --data DIR
"""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
STUDENT = 'outputs/distill_three_teachers_v1_20261018/resume.pt'
TEACHERS = dict(traversal='outputs/traversal_expert_v1_20261012/snapshots/update_1764.pt',
                narrow='outputs/saved_checkpoints/narrow_expert_v2_u1400.pt',
                rooms='outputs/saved_checkpoints/cat_g1_rooms_keyboard_v9_u492_full.pt')
ROUTING = 'outputs/distill_v1_prep/routing_v2.json'
WIDTHS = {'0.1M': (256, 128, 64), '0.29M (ours)': (512, 256, 128, 64), '0.9M': (1024, 512, 256, 128), '3.6M': (2048, 1024, 512, 256)}


def collect(args):
    from cat_mjlab.runner import create_task
    from cat_mjlab.sim_cpu import CPUSimulation
    from cat_mjlab.distill import load_expert
    from cat_mjlab.learning import gaussian_parameters
    torch.set_num_threads(2)
    contract = torch.load(ROOT/STUDENT, map_location='cpu', weights_only=True, mmap=True)['contract']
    env = copy.deepcopy(contract['environment_config'])
    env['push_config']['enable'] = False
    for key in ('teleop', 'sgs'):
        env.pop(key, None)
    bank = ROOT/'data/furniture/sidegap_v2'
    ns = SimpleNamespace(bank_manifest=Path(str(bank)+'_packed/manifest.json'), body_collision_bank=Path(str(bank)+'_collision/manifest.json'),
                         body_collision_resets=Path(str(bank)+'_resets/manifest.json'), device='cpu', seed=args.shard, num_envs=1,
                         compile_task=False, nconmax=contract['nconmax'], njmax=contract['njmax'], reactive_bank=None, reactive_row=None)
    task, sim, _ = create_task(ns, environment_config=env, sim_class=CPUSimulation)
    task.collision_every_substep = False      # contact once per control step (as the CPU demo): ~5x faster on one CPU world
    owner = json.loads((ROOT/ROUTING).read_text())['owner']
    index = {s['scene_id']: i for i, s in enumerate(task.bank.manifest['scenes'])}
    names = list(TEACHERS)
    by_teacher = {n: [index[s] for s, o in owner.items() if o == n and s in index] for n in names}
    scene_owner = {index[s]: names.index(o) for s, o in owner.items() if s in index}
    student = load_expert(ROOT/STUDENT, 'cpu'); teachers = [load_expert(ROOT/TEACHERS[n], 'cpu') for n in names]
    rng = random.Random(1000+args.shard)
    original_reset = task.reset
    def pick():
        return rng.choice(by_teacher[rng.choice(names)])       # balanced over teachers
    def reset(env_ids=None, scene_ids=None):
        ids = task.all_ids if env_ids is None else env_ids
        return original_reset(ids, torch.tensor([pick()]*len(ids)))
    task.reset = reset
    task.reset()
    obs_rows, labels, owners, episodes = [], [], [], []
    episode, t, start = 0, 0, time.time()
    while len(obs_rows) < args.steps:
        obs = task.obs['state']
        k = scene_owner[int(task.scene_ids[0])]
        with torch.no_grad():
            label, _ = gaussian_parameters(teachers[k].logits(obs[:, :teachers[k].config.actor_obs], 0))
            mean, _ = gaussian_parameters(student.logits(obs, 0))
        obs_rows.append(obs[0].numpy().copy()); labels.append(label[0].numpy().copy()); owners.append(k); episodes.append(args.shard*100000+episode)
        out = task.step(torch.tanh(mean+.05*torch.randn_like(mean)))
        t += 1
        if bool(out['done'][0]) or t >= args.episode_steps:
            if not bool(out['done'][0]):
                task.reset()
            episode += 1; t = 0
        if len(obs_rows) % 2000 == 0:
            print(f'shard {args.shard}: {len(obs_rows)} samples, {episode} episodes, {time.time()-start:.0f}s', flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/f'shard{args.shard}.npz', obs=np.asarray(obs_rows, np.float32), label=np.asarray(labels, np.float32),
                        owner=np.asarray(owners, np.int8), episode=np.asarray(episodes, np.int64), teachers=np.array(names))


def fit(args):
    import torch.nn as nn
    shards = sorted(Path(args.data).glob('shard*.npz'))
    d = [np.load(s) for s in shards]
    obs = np.concatenate([x['obs'] for x in d]); label = np.concatenate([x['label'] for x in d])
    owner = np.concatenate([x['owner'] for x in d]); episode = np.concatenate([x['episode'] for x in d]); names = list(d[0]['teachers'])
    eps = np.unique(episode); rng = np.random.default_rng(0); val_eps = set(rng.choice(eps, max(1, len(eps)//10), replace=False).tolist())
    val = np.isin(episode, list(val_eps))
    device = args.device
    X, Y = torch.tensor(obs, device=device), torch.tensor(label, device=device)
    mu, sd = X[~torch.tensor(val, device=device)].mean(0), X[~torch.tensor(val, device=device)].std(0)+1e-3
    print(json.dumps(dict(samples=len(obs), train=int((~val).sum()), val=int(val.sum()), episodes=len(eps),
                          per_teacher={n: int((owner == i).sum()) for i, n in enumerate(names)})))
    tr, va = torch.tensor(np.where(~val)[0], device=device), torch.tensor(np.where(val)[0], device=device)
    results = {}
    for name, hidden in WIDTHS.items():
        torch.manual_seed(0)
        layers, width = [], obs.shape[1]
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ELU()]; width = h
        net = nn.Sequential(*layers, nn.Linear(width, Y.shape[1])).to(device)
        params = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        steps = args.epochs*max(1, len(tr)//args.batch)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        start = time.time()
        for step in range(steps):
            idx = tr[torch.randint(len(tr), (args.batch,), device=device)]
            loss = (net((X[idx]-mu)/sd)-Y[idx]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            def mse(sel):
                return ((net((X[sel]-mu)/sd)-Y[sel]).square().mean(-1))
            tr_err, va_err = mse(tr), mse(va)
            own_tr, own_va = torch.tensor(owner, device=device)[tr], torch.tensor(owner, device=device)[va]
            row = dict(params=params, seconds=round(time.time()-start),
                       train=round(float(tr_err.mean()), 5), val=round(float(va_err.mean()), 5),
                       **{f'val_{n}': round(float(va_err[own_va == i].mean()), 5) for i, n in enumerate(names)},
                       **{f'train_{n}': round(float(tr_err[own_tr == i].mean()), 5) for i, n in enumerate(names)})
        results[name] = row
        print(name, json.dumps(row), flush=True)
    Path(args.data, 'fit_results.json').write_text(json.dumps(results, indent=1)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect'); c.add_argument('--out', required=True); c.add_argument('--shard', type=int, default=0)
    c.add_argument('--steps', type=int, default=40000); c.add_argument('--episode-steps', type=int, default=300)
    f = sub.add_parser('fit'); f.add_argument('--data', required=True); f.add_argument('--epochs', type=int, default=60)
    f.add_argument('--batch', type=int, default=4096); f.add_argument('--lr', type=float, default=1e-3); f.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    collect(args) if args.cmd == 'collect' else fit(args)


if __name__ == '__main__':
    main()
