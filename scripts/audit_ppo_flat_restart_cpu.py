#!/usr/bin/env python3
"""Read-only CPU checkpoint/log audit; never constructs a GPU task."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import torch
from cat_mjlab.learning import Learner, LearnerConfig, prepare_sapg_rollout
from cat_mjlab.conversion import extract_native_leader

torch.set_num_threads(2)
torch.manual_seed(17)
root = Path(__file__).resolve().parents[1]
report = {'device': 'cpu', 'checkpoints': {}}
models = []
for name in ('cat_flat_balance_v3_30720', 'cat_flat_balance_v2_30720_20260920'):
    directory = root / 'outputs' / name
    snapshot = torch.load(directory / 'resume.pt', map_location='cpu', weights_only=True, mmap=True)
    state = snapshot['learner']; cfg = LearnerConfig(**state['config'])
    source = Learner(cfg, device='cpu'); source.model.load_state_dict(state['model'])
    target = Learner(replace(cfg, algorithm='ppo', num_policies=1), device='cpu')
    target.model.load_state_dict(extract_native_leader(state['model'], state['config']), strict=True)
    obs = {key: value[:1024].clone() for key, value in snapshot['task']['obs'].items()}
    with torch.no_grad():
        a = source.act(obs, deterministic=True); b = target.act(obs, deterministic=True)
    errors = {key: float((a[key]-b[key]).abs().max()) for key in ('action', 'mean', 'action_std', 'value')}
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    flat_n = 0; strict_sum = 0.; soft_sum = 0.; replay = []
    for row in rows[:100]:
        n = row['balance/leader_step_count']; flat_n += n
        strict_sum += row['balance/leader_walking_compliant_step_count']
        soft_sum += row['balance/leader_posture_bonus_mean_per_step'] / .2 * n
        retention = row.get('success/cat_goal_success_rate')
        if retention is not None and flat_n:
            replay.append(.5*strict_sum/flat_n + .25*soft_sum/flat_n + .25*retention)
    report['checkpoints'][name] = dict(durable_updates=state['updates'], durable_steps=state['env_steps'],
        model_max_abs_errors=errors, fresh_optimizer_entries=len(target.optimizer.state),
        logged_updates=len(rows), last_logged_step=rows[-1]['global_step'],
        vram_min_gib=min(r['performance/device_vram_used_gib'] for r in rows),
        vram_max_gib=max(r['performance/device_vram_used_gib'] for r in rows),
        control_steps_per_second_last=rows[-1]['performance/control_steps_per_second'],
        proposed_score_replay_range=[min(replay),max(replay)],
        proposed_score_replay_improvements=sum(v > max(replay[:i],default=-1) for i,v in enumerate(replay)))
    models.append({k:v.clone() for k,v in state['model'].items()})
report['v3_initial_weights_equal_v2'] = all(torch.equal(models[0][k],models[1][k]) for k in models[0])
# Exact tensor-byte accounting using the actual SAPG preparation on a small CPU rollout.
n,t = 6,32
shape=(n,t)
data = {key:torch.zeros(n,t,width) for key,width in [('state',cfg.actor_obs),('next_state',cfg.actor_obs),
    ('privileged_state',cfg.critic_obs),('next_privileged_state',cfg.critic_obs),('raw_action',29)]}
data.update({key:torch.zeros(shape) for key in ('log_prob','reward','discount','truncation')})
data['policy_id']=torch.arange(6)[:,None].expand(shape).clone()
data['task_bucket']=torch.zeros(shape,dtype=torch.long)
augmented=prepare_sapg_rollout(source.model,data,cfg,follower_id=1)
def size(tree): return sum(v.numel()*v.element_size() for v in tree.values())
scale=30720/n/2**30
raw=size(data)*scale
aug=size(augmented)*scale
steady=size({k:v for k,v in augmented.items() if k not in ('raw_advantage','initial_value_error')})*scale
fixed=13.5467+.06569
baseline=report['checkpoints']['cat_flat_balance_v3_30720']['vram_max_gib']
report['memory'] = dict(physical_rollout_gib=raw, augmented_with_diagnostics_gib=aug,
    extra_augmented_retained_during_update_gib=steady, follower_temporary_gib=raw/6,
    fixed_fields_collision_gib=fixed, baseline_gib=baseline,
    capacity_gib=47.492, reserve_gib=4.784, working_ceiling_gib=47.492-4.784,
    model='fixed + (baseline - fixed - savings_credit) * N/30720',
    conservative_credit_fraction=.5, extra_uncertainty_gib=1.)
report['memory']['counts']={}
for count in (30720,33792,37632,38400,39168,49152):
    expected=fixed+(baseline-fixed-steady)*count/30720
    budget=fixed+(baseline-fixed-.5*steady)*count/30720+1
    report['memory']['counts'][count]=dict(expected_gib=expected,conservative_budget_gib=budget,
        budget_free_gib=47.492-budget,num_minibatches=count//768,batch_size=768,
        transitions=count*32,leader_data_multiple=count/5120)
print(json.dumps(report, indent=2))
