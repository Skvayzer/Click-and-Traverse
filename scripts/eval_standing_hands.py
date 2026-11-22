#!/usr/bin/env python3
"""Deterministic evaluation of standing hand protection: every env replays one fixed approaching-object row.

All policies are measured the same way: standing episodes from the reactive bank (rows forced, authored object
parameters, repeated approaches re-aimed at the live hand), deterministic actions, the feet-planted rule in SHADOW
mode (measured, never ending an episode), no pushes, no robustness randomisation unless --robust.

Reports per checkpoint: approach events (success by bucket, hand-object contact, hand retreat), feet (fraction of settled
steps violating the 5 cm / 5-step rule, foot travel), falls, and substep energy by joint group.
Completed approaches include torso-relative retreat, response classification, and per-event energy.

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

from cat_mjlab.energy import GROUPS, SHAPE, print_motor_weights, report_energy
from standing_event_energy import StandingEventEnergy, matched_extra
BUCKETS = ('danger', 'anticipation', 'negative')


def build(checkpoint, rows, device, robust, hold_plan=None, bank=None, env_from=None):
    from cat_mjlab.runner import create_task
    from record_mjlab_rollout import load_policy
    archive = str(checkpoint).endswith('.npz')
    if archive:
        # A released-CAT array archive: the task comes from env_from, built on the archive's native 222/310 inputs.
        if env_from is None:
            raise ValueError('an .npz policy needs --env-from <a .pt checkpoint>')
        from cat_mjlab.learning import Learner, LearnerConfig
        from cat_mjlab.conversion import load_array_archive
        learner = Learner(LearnerConfig(algorithm='ppo'), device=device)
        load_array_archive(learner, checkpoint, restore_optimizer=False)
        _, saved = load_policy(env_from, device='cpu', policy_id=0)
    else:
        learner, saved = load_policy(checkpoint, device=device, policy_id=0)
    contract = saved['contract']
    env = copy.deepcopy(contract['environment_config'])
    if archive:
        env['sdf_rate_obs'] = False; env.pop('tracker', None)
    for key in ('sgs', 'reactive_stationary', 'reactive_retreat_success_m', 'reactive_hand_clearance'):
        env.pop(key, None)                       # same event definition (5 cm retreat) for every policy
    env['reactive_spot_radius'] = .3
    if not robust:
        env.pop('robustness', None)
        env['push_config']['enable'] = False
    env['teleop'].update(fraction=0., cat_fraction=0., cat_scenes=False)
    if hold_plan is not None:
        if not env.get('tracker'):
            raise ValueError('--hold-plan needs a checkpoint trained with the tracker')
        env['tracker']['hold_plan'] = float(hold_plan)     # 1: every standing episode on a stand-still plan, 0: none
    # The static bank only sets the standing room; a goal-seed cache (V7 and later) is tied to its own bank.
    bank = bank or ('data/furniture/tablegap_v1' if env.get('goal_seed') else 'data/furniture/sidegap_v2')
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
def evaluate(checkpoint, rows, steps, device, robust, hold_plan=None, bank=None, env_from=None):
    learner, task = build(checkpoint, rows, device, robust, hold_plan, bank, env_from)
    obj = task.reactive_objects
    task.reset()
    n = len(rows)
    zeros = lambda: torch.zeros(n, device=task.device)
    acc = dict(events=zeros(), success=zeros(), contact_events=zeros(), retreat=zeros(), settled=zeros(), violation=zeros(),
               lifted=zeros(), feet_moved=zeros(), standing=zeros(), moving=zeros(), falls=zeros(), steps=zeros())
    by_bucket = {b: dict(events=zeros(), success=zeros()) for b in range(3)}
    far = dict(threat=zeros(), r15=zeros(), r25=zeros(), retreat=zeros())   # threatening events: retreat distance
    energy_j = torch.zeros(SHAPE, device=task.device)
    moving_j = torch.zeros_like(energy_j)
    episode_j = torch.zeros_like(energy_j)
    episode_count = zeros()
    episode_steps = zeros()
    obj.energy_observer = observer = StandingEventEnergy(task)
    contact_open = torch.zeros(n, dtype=torch.bool, device=task.device)
    ever_violated = torch.zeros(n, dtype=torch.bool, device=task.device)
    first_violation = torch.full((n,), float(steps), device=task.device)
    step = 0
    for _ in range(steps):
        out = task.step(learner.act(task.obs, policy_ids=0, deterministic=True)['action'])
        m, t = out['metrics'], task.telemetry
        standing = t['reactive_standing'] > 0
        done = out['done'].bool() & standing
        moving_obj = standing & ~out['done'].bool() & ~obj.state['retreating'].all(-1) & (obj.state['clock'] > 0)
        step_power = m['energy/step_w']
        energy_j += torch.where(standing[:, None, None, None], step_power, 0.).sum(0) * task.dt
        moving_j += torch.where(moving_obj[:, None, None, None], step_power, 0.).sum(0) * task.dt
        episode_j += torch.where(done[:, None, None, None], m['energy/episode_j'], 0.).sum(0)
        episode_steps += m['energy/episode_steps'] * done
        episode_count += done
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
                  falls=int(s['falls']))
    all_group_power = energy_j.cpu()[0, 0].tolist()
    moving_group_power = moving_j.cpu()[0, 0].tolist()
    result['power_W'] = {
        g: dict(standing=ratio(all_group_power[k], s['standing'] * task.dt),
                object_moving=ratio(moving_group_power[k], s['moving'] * task.dt))
        for k, g in enumerate(GROUPS)
    }
    result['completed_episodes'] = int(episode_count.sum())
    result['energy'] = report_energy(episode_j.cpu(), result['completed_episodes'], float(episode_steps.sum()) * task.dt)
    result['energy_during_standing'] = report_energy(energy_j.cpu(), 0, s['standing'] * task.dt)
    result['energy_during_object_motion'] = report_energy(moving_j.cpu(), 0, s['moving'] * task.dt)
    result['energy_by_response'], result['approaches'] = observer.summary(rows, BUCKETS)
    result['matched_extra'] = matched_extra(result['approaches'])
    result['power_W']['total'] = {k: round(sum(result['power_W'][g][k] or 0. for g in GROUPS), 2) for k in ('standing', 'object_moving')}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', nargs='+', required=True)
    p.add_argument('--rows', default='0:2000', help='bank rows a:b[:stride] (one env per row)')
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--robust', action='store_true', help='keep the checkpoint\'s training randomisation and pushes')
    p.add_argument('--bank', help='static scene bank (default: tablegap_v1 for checkpoints with a goal-seed cache, else sidegap_v2)')
    p.add_argument('--env-from', help='a .pt checkpoint whose task configuration an .npz (released CAT) policy runs in')
    p.add_argument('--hold-plan', type=float, help='tracker checkpoints: share of episodes on a stand-still plan (1 = all)')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    rows = list(range(*(int(x) for x in a.rows.split(':'))))
    print_motor_weights()
    results = []
    for checkpoint in a.checkpoint:
        r = evaluate(checkpoint, rows, a.steps, a.device, a.robust, a.hold_plan, a.bank, a.env_from)
        print(json.dumps(r if not a.out else {k: v for k, v in r.items() if k != 'approaches'}), flush=True)
        results.append(r)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
