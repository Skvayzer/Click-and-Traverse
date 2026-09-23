#!/usr/bin/env python3
"""Full-bank CPU construction, frozen-policy physics smoke and isolation checks. No training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import torch
from test_mjlab_task import _CPUSimulation
from cat_mjlab.scene_bank import SceneBank
from cat_mjlab.collision import CollisionChecker
from cat_mjlab.task import CATTask
from cat_mjlab.learning import Learner,LearnerConfig
from cat_mjlab.runner import native_initialization,collect_rollout
from cat_mjlab.config import wholebody_config
from cat_ppo.furniture.generalist_fields import sha256

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--checkpoint-native',type=Path,default=ROOT/'outputs/cat_hand_posture_30720_20260919/resume.pt');parser.add_argument('--flat-bonus-scale',type=float);parser.add_argument('--flat-region-scale',type=float);parser.add_argument('--output',type=Path,default=ROOT/'outputs/flat_balance_build_validation/cpu-smoke.json');args=parser.parse_args()
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 manifest=ROOT/'data/furniture/cat_flat_balance_v1_20260920/manifest.json';collision=manifest.parent.with_name(manifest.parent.name+'_collision')/'manifest.json';resets=manifest.parent.with_name(manifest.parent.name+'_resets')/'manifest.json'
 bank=SceneBank(manifest,device='cpu',collision_manifest=collision,reset_manifest=resets)
 print('Full bank loaded',flush=True)
 sums=torch.zeros(6).scatter_add_(0,bank.sampling_ids,bank.probabilities())
 torch.testing.assert_close(sums,bank.sampling_masses,atol=2e-6,rtol=2e-5)
 assert bank.flat_balance.sum()==1 and bank.rooms['obstacle_count'][-1]==0
 assert bank.episode_lengths[-1]==500 and bank.levels is None and bank.width_curriculum is None
 cfg,weights,environment=native_initialization(args.checkpoint_native)
 cfg.update(max_action_std=.15,num_minibatches=40)
 learner=Learner(LearnerConfig(**cfg),device='cpu');learner.model.load_state_dict(weights,strict=True)
 assert not learner.optimizer.state
 sim=_CPUSimulation(6);checker=CollisionChecker(sim.model,collision,field_manifest=manifest,device='cpu')
 config=wholebody_config(environment,stabilization=True,hand_protection=True,hand_contrast=True)
 from cat_mjlab.balance import reward_overrides
 config['flat_balance_reward']=reward_overrides(bank.balance_settings,config.get('flat_balance_reward'),bonus_scale=args.flat_bonus_scale,region_scale=args.flat_region_scale)
 config['randomize_initial_episode_steps']=False
 task=CATTask(sim,bank,config,collision=checker,seed=91)
 # Flat, original, procedural, ordinary room, and two narrow rungs.
 selected=torch.tensor([2350,0,64,2290,2338,2346]);task.reset(scene_ids=selected)
 before=task.navigation_counts.clone()
 ids=torch.zeros(6,dtype=torch.long);task.set_policy_ids(ids)
 rewards=[];leak_max=0.
 for step in range(12):
  acted=learner.act(task.obs,policy_ids=ids);transition=task.step(acted['action']);rewards.append(transition['reward'].tolist())
  assert torch.isfinite(transition['reward']).all()
  metrics=transition['metrics'];nonflat=~metrics['balance/flat']
  assert torch.equal(metrics['balance/bonus'][nonflat],torch.zeros_like(metrics['balance/bonus'][nonflat]))
  contacts=sim.contact_flags(task.contact_pairs)
  enabled,_=task._rewards(acted['action'],contacts)
  settings=task.balance_settings;task.balance_settings=None
  legacy,_=task._rewards(acted['action'],contacts);task.balance_settings=settings
  mask=~bank.flat_balance[task.scene_ids]
  torch.testing.assert_close(enabled[mask],legacy[mask],rtol=0,atol=0)
  leak_max=max(leak_max,float((enabled[mask]-legacy[mask]).abs().max()))
 print('CPU physical smoke passed; no reward leakage',flush=True)
 # New streak is saved/restored only for the new bank, not added to legacy snapshots.
 task.balance_streak[:]=7;state=copy.deepcopy(task.state_dict());task.balance_streak.zero_();task.load_state_dict(state);assert (task.balance_streak==7).all()
 task.reset(scene_ids=selected);assert (task.balance_streak==0).all()
 task.enable_compilation(backend='eager')
 _,collected=collect_rollout(task,learner,unroll_length=2,trajectories=6,policy_ids=ids)
 assert collected['metrics']['balance/leader_step_count']>=2
 # Flat horizons truncate without a goal or qualification gate.
 task.reset(scene_ids=selected);task.info['wrapper_steps'][0]=499
 result=task.step(torch.zeros(6,29));assert result['done'][0]
 assert result['metrics']['balance/flat'][0] and task.info['wrapper_steps'][0]==0
 assert not learner.optimizer.state
 report=dict(flat_balance_reward=config['flat_balance_reward'],checkpoint_source=str(args.checkpoint_native),bank_sha256=sha256(manifest),collision_sha256=sha256(collision),resets_sha256=sha256(resets),scene_count=bank.count,bucket_counts=torch.bincount(bank.sampling_ids).tolist(),masses=sums.tolist(),flat_obstacle_count=int(bank.rooms['obstacle_count'][-1]),flat_horizon=500,nonflat_reward_difference_max=leak_max,optimizer_state_entries=len(learner.optimizer.state),sigma_ceiling=.15,physics_control_steps=15,compile_check='CPU Torch fullgraph eager backend, new reward separately tested with CPU Inductor',metric_smoke=collected['metrics'],elapsed_seconds=time.monotonic()-start)
 out=args.output;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
