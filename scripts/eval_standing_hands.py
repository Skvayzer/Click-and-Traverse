#!/usr/bin/env python3
"""Deterministic evaluation of standing hand protection: every env replays one fixed approaching-object row.

All policies are measured the same way: standing episodes from the reactive bank (rows forced, authored object
parameters, repeated approaches re-aimed at the live hand), deterministic actions, the feet-planted rule in SHADOW
mode (measured, never ending an episode), no pushes, no robustness randomisation unless --robust.

Reports per checkpoint: approach events (success by bucket, hand-object contact, hand retreat), feet (fraction of settled
steps violating the 5 cm / 5-step rule, foot travel), mechanical joint power |tau*qdot| split into legs (0-11), waist
(12-14) and arms (15-28), overall and while an object is moving, and falls.

  eval_standing_hands.py --checkpoint A.pt [B.pt ...] --rows 0:2000 --steps 800 --device cuda:0 --out result.json
"""
import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

GROUPS = dict(legs=slice(0, 12), waist=slice(12, 15), arms=slice(15, 29))
BUCKETS = ('danger', 'anticipation', 'negative')


def build(checkpoint, rows, device, robust):
    from cat_mjlab.runner import create_task
    from record_mjlab_rollout import load_policy
    learner, saved = load_policy(checkpoint, device=device, policy_id=0)
    contract = saved['contract']
    env = copy.deepcopy(contract['environment_config'])
    for key in ('sgs', 'reactive_stationary', 'reactive_retreat_success_m', 'reactive_hand_clearance'):
        env.pop(key, None)                       # same event definition (5 cm retreat) for every policy
    env['reactive_spot_radius'] = .3
    if not robust:
        env.pop('robustness', None)
        env['push_config']['enable'] = False
    env['teleop'].update(fraction=0., cat_fraction=0., cat_scenes=False)
    bank = 'data/furniture/sidegap_v2'
    ns = SimpleNamespace(bank_manifest=Path(bank + '_packed/manifest.json'), body_collision_bank=Path(bank + '_collision/manifest.json'),
                         body_collision_resets=Path(bank + '_resets/manifest.json'), device=device, seed=0, num_envs=len(rows),
                         compile_task=False, nconmax=contract['nconmax'], njmax=contract['njmax'],
                         reactive_bank=Path('data/furniture/reactive_standing_v2_20260923/manifest.json'), reactive_row=None)
    kwargs = {}
    if device == 'cpu':
        from cat_mjlab.sim_cpu import CPUSimulation
        kwargs['sim_class'] = CPUSimulation
    task, _, _ = create_task(ns, environment_config=env, **kwargs)
    task.collision_every_substep = False
    task.reactive_objects.force_rows = torch.as_tensor(rows, device=task.device)
    return learner, task


@torch.no_grad()
def evaluate(checkpoint, rows, steps, device, robust):
    learner, task = build(checkpoint, rows, device, robust)
    obj = task.reactive_objects
    task.reset()
    n = len(rows)
    zeros = lambda: torch.zeros(n, device=task.device)
    acc = dict(events=zeros(), success=zeros(), contact_events=zeros(), retreat=zeros(), settled=zeros(), violation=zeros(),
               lifted=zeros(), feet_moved=zeros(), standing=zeros(), moving=zeros(), falls=zeros(), steps=zeros())
    by_bucket = {b: dict(events=zeros(), success=zeros()) for b in range(3)}
    far = dict(threat=zeros(), r15=zeros(), r25=zeros(), retreat=zeros())   # threatening events: retreat distance
    power = {g: dict(all=zeros(), moving=zeros()) for g in GROUPS}
    contact_open = torch.zeros(n, dtype=torch.bool, device=task.device)
    ever_violated = torch.zeros(n, dtype=torch.bool, device=task.device)
    first_violation = torch.full((n,), float(steps), device=task.device)
    step = 0
    for _ in range(steps):
        out = task.step(learner.act(task.obs, policy_ids=0, deterministic=True)['action'])
        m, t = out['metrics'], task.telemetry
        standing = t['reactive_standing'] > 0
        moving_obj = standing & ~obj.state['retreating'].all(-1) & (obj.state['clock'] > 0)
        per_joint = (task.data.ctrl * task.data.qvel[:, 6:]).abs()
        for g, sl in GROUPS.items():
            p = per_joint[:, sl].sum(-1)
            power[g]['all'] += p * standing
            power[g]['moving'] += p * moving_obj
        contact_open |= m['reactive/robot_initiated_contact'].bool()
        finished = m['reactive/event_finished'].bool() & standing
        if bool(finished.any()):
            success = m['reactive/event_success'].bool()
            bucket = m['reactive/event_bucket'].long()
            acc['events'] += finished
            acc['success'] += finished & success
            acc['contact_events'] += finished & contact_open
            acc['retreat'] += m['reactive/event_retreat_m'] * finished
            for b in range(3):
                by_bucket[b]['events'] += finished & (bucket == b)
                by_bucket[b]['success'] += finished & success & (bucket == b)
            threat = finished & (bucket < 2)
            retreat = m['reactive/event_retreat_m']
            far['threat'] += threat
            far['r15'] += threat & (retreat >= .15)
            far['r25'] += threat & (retreat >= .25)
            far['retreat'] += retreat * threat
            contact_open &= ~finished
        acc['settled'] += t['reactive_settled']
        acc['violation'] += t['reactive_violation']
        new = (t['reactive_violation'] > 0) & ~ever_violated
        first_violation = torch.where(new, float(step), first_violation)
        ever_violated |= new
        step += 1
        acc['lifted'] += t['reactive_end_lift']
        acc['feet_moved'] += t['reactive_feet_moved_m']
        acc['standing'] += standing
        acc['moving'] += moving_obj
        acc['falls'] += out['done'].bool() & m.get('episode/fall', torch.zeros_like(out['done'])).bool()
        acc['steps'] += 1
    s = {k: float(v.sum()) for k, v in acc.items()}
    ratio = lambda a, b: round(a / b, 4) if b else None
    result = dict(checkpoint=str(checkpoint), envs=n, steps=steps, robust=robust,
                  events=int(s['events']), event_success=ratio(s['success'], s['events']),
                  contact_per_event=ratio(s['contact_events'], s['events']), retreat_m=ratio(s['retreat'], s['events']),
                  success_by_bucket={BUCKETS[b]: ratio(float(v['success'].sum()), float(v['events'].sum())) for b, v in by_bucket.items()},
                  events_by_bucket={BUCKETS[b]: int(v['events'].sum()) for b, v in by_bucket.items()},
                  threat_retreat_mean_m=ratio(float(far['retreat'].sum()), float(far['threat'].sum())),
                  threat_retreat_ge_15cm=ratio(float(far['r15'].sum()), float(far['threat'].sum())),
                  threat_retreat_ge_25cm=ratio(float(far['r25'].sum()), float(far['threat'].sum())),
                  feet_violation_per_settled_step=ratio(s['violation'], s['settled']),
                  feet_lift_per_settled_step=ratio(s['lifted'], s['settled']),
                  feet_moved_mean_m=ratio(s['feet_moved'], s['settled']),
                  envs_ever_violating=round(float(ever_violated.float().mean()), 4),
                  first_violation_step_median=float(first_violation[ever_violated].median()) if bool(ever_violated.any()) else None,
                  falls=int(s['falls']),
                  power_W={g: dict(standing=ratio(float(v['all'].sum()), s['standing']), object_moving=ratio(float(v['moving'].sum()), s['moving']))
                           for g, v in power.items()})
    result['power_W']['total'] = {k: round(sum(result['power_W'][g][k] or 0. for g in GROUPS), 2) for k in ('standing', 'object_moving')}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', nargs='+', required=True)
    p.add_argument('--rows', default='0:2000', help='bank rows a:b[:stride] (one env per row)')
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--robust', action='store_true', help='keep the checkpoint\'s training randomisation and pushes')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    rows = list(range(*(int(x) for x in a.rows.split(':'))))
    results = []
    for checkpoint in a.checkpoint:
        r = evaluate(checkpoint, rows, a.steps, a.device, a.robust)
        print(json.dumps(r), flush=True)
        results.append(r)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
