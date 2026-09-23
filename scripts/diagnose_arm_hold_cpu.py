#!/usr/bin/env python3
"""Frozen-checkpoint CPU arm hold diagnosis. No training or GPU initialization.

Uses an in-memory four-scene field subset, unchanged production CATTask, certified
raised reset poses and per-substep production collision checks. Saves terminal
states before automatic resets; later episodes are excluded. Counterfactuals
change only actions. Run from repository root with .venv-mjlab/bin/python.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse, copy, csv, hashlib, json, sys, time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import numpy as np
import torch
import mujoco
from test_mjlab_task import _CPUSimulation
from cat_mjlab.scene_bank import SceneBank
from cat_mjlab.task import CATTask
from cat_mjlab.learning import ActorCritic,LearnerConfig,gaussian_parameters
from cat_mjlab.collision import CollisionChecker
from cat_mjlab.raised_reset import build_raised_reset_pool
from cat_ppo.furniture.generalist_fields import scene_directory
from cat_ppo.furniture.room_navigation import pack_room_scenes
from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
from cat_mjlab.acceptance import pack_geometry
from cat_mjlab.passage_rewards import passage_parameters,LEGACY_SDF_KNEE
from cat_ppo.furniture.control import JOINT_NAMES
T=lambda x,dtype=None:torch.as_tensor(np.asarray(x).copy(),dtype=dtype)

def subset(path,indices):
 m=json.loads(path.read_text());records=[m['scenes'][i] for i in indices]
 dirs=[scene_directory(m,path,r) for r in records];scenes=[json.loads((d/'scene.json').read_text()) for d in dirs]
 b=SceneBank.__new__(SceneBank);b.path=path;b.manifest=m;b.device=torch.device('cpu');b.count=len(indices)
 from cat_mjlab.fields import sample_ragged_field
 b.sample_kernel=sample_ragged_field
 sizes=[int(np.prod(r['shape'])) for r in records];b.offsets=T(np.cumsum([0]+sizes[:-1]),torch.long)
 b.fields={name:T(np.concatenate([np.load(d/(name+'.npy'),mmap_mode='r').reshape(-1,ch) for d in dirs])) for name,ch in [('sdf',1),('gf',3),('bf',3)]}
 for attr,key,dtype in [('shapes','shape',torch.long),('origins','origin',torch.float32),('dxs','dx',torch.float32),('starts','start',torch.float32),('goals','goal',torch.float32),('reset_xy_scale','reset_xy_scale',torch.float32),('reset_yaws','reset_yaw',torch.float32)]:setattr(b,attr,T([r[key] for r in records],dtype))
 for attr in ['is_cat','reset_is_cat','crossed_is_plane','flat_balance']:setattr(b,attr,torch.zeros(b.count,dtype=torch.bool))
 b.rooms={k:T(v) for k,v in pack_room_scenes(scenes).items()};b.contrast={k:T(v) for k,v in pack_hand_contrast(scenes).items()};b.acceptance={k:T(v) for k,v in pack_geometry(scenes).items()}
 params=passage_parameters(scenes,bank_path=path);b.contrast.update({k:T(v) for k,v in params.items()});b.has_sdf_reward_overrides=bool(np.any(params['sdf_reward_knee']!=LEGACY_SDF_KNEE))
 b.episode_lengths=T([r['episode_length'] for r in records],torch.long);b.weights=torch.ones(b.count);b.roles=b.levels=b.groups=b.width_curriculum=None
 b.balance_settings=m['flat_balance']['settings'];b.has_contrast=True;b.navigation_groups=torch.full((b.count,),2,dtype=torch.long);b.hand_scene_kind=torch.zeros(b.count,dtype=torch.long)
 b.probabilities=lambda weights=None,stage=None:torch.ones(b.count)/b.count
 return b,records,scenes

class TraceTask(CATTask):
 capture=False
 def _rewards(self,action,contacts):
  reward,parts=super()._rewards(action,contacts)
  self.ledger={k:v.detach().clone()*self.dt for k,v in parts.items()}
  return reward,parts
 def _outcomes(self,done,truncation):
  if self.capture:self.captured=self.frame(done)
  return super()._outcomes(done,truncation)
 def frame(self,done=None):
  i=self.info;c=self.contrast;hands=i['positions'][:,5:7];delta=hands[:,:,:2]-self.data.qpos[:,None,:2];t=c['route_tangent'];n=torch.stack((-t[:,1],t[:,0]),-1)
  local=torch.stack(((delta*t[:,None]).sum(-1),(delta*n[:,None]).sum(-1),hands[:,:,2]),-1)
  distance=torch.linalg.vector_norm((c['hand_regions_min']-local[:,None]).clamp_min(0)+(local[:,None]-c['hand_regions_max']).clamp_min(0),dim=-1)
  error=torch.where(c['region_valid'],distance.amax(-1),torch.inf).amin(-1)
  rot=self.data.site_xmat.reshape(self.num_envs,-1,3,3)
  _,metrics=self.contrast_reward_terms(c,hands,self.data.qpos[:,:2],rot[:,self.pelvis_site,:,0],rot[:,self.torso_site,:,0],region_scale=self.config['hand_contrast_region_scale'])
  out=dict(qpos=self.data.qpos, qvel=self.data.qvel, target=i['motor_targets'],hand_world=hands,hand_local=local,distance=distance,error=error,command=i['command'],core=c['core_active'],done=torch.zeros(self.num_envs,dtype=torch.bool) if done is None else done,**metrics)
  out.update({k:v for k,v in self.telemetry.items() if k in ['left_arm_posture_gate','right_arm_posture_gate','reward_floor_slope']})
  out.update({'reward_'+k:v for k,v in getattr(self,'ledger',{}).items()})
  return {k:v.detach().cpu().numpy().copy() for k,v in out.items()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/arm_hold_diagnosis_cpu');p.add_argument('--steps',type=int,default=200);p.add_argument('--modes',nargs='+',default=['policy','hold_arms','hold_upper','hold_pose']);args=p.parse_args();args.output.mkdir(exist_ok=True)
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 snap=torch.load(args.output/'policy_snapshot.pt',map_location='cpu',weights_only=True)
 cfg=copy.deepcopy(snap['contract']['environment_config']);policy=ActorCritic(LearnerConfig(**snap['config'])).eval();policy.load_state_dict(snap['model'])
 manifest=ROOT/'data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json';full=json.loads(manifest.read_text());protected=[i for i,r in enumerate(full['scenes']) if r.get('source',{}).get('hand_contrast',{}).get('role')=='forward_protected'];indices=[protected[i] for i in [0,3,6,9]]
 assert hashlib.sha256(manifest.read_bytes()).hexdigest()==snap['contract']['bank_sha256']
 print('Loading four protected scenes',indices,flush=True);bank,records,scenes=subset(manifest,indices);sim=_CPUSimulation(len(indices));collision_path=ROOT/cfg['wholebody']['body_collision']['bank_manifest']
 checker=CollisionChecker(sim.model,collision_path,field_manifest=manifest,device='cpu')
 class MappedCollision:
  proposal_path=checker.proposal_path
  def __call__(self,ids,data):return checker(T(indices,torch.long)[ids],data)
 pool,eligible,certificate=build_raised_reset_pool(sim.model,full,manifest,collision_path,proposal_path=checker.proposal_path)
 bank.reset_pool=T(pool[indices]);cfg['hand_raised_reset_fraction']=0.;cfg['randomize_initial_episode_steps']=False
 task=TraceTask(sim,bank,cfg,collision=MappedCollision(),seed=71)
 task.raised_reset_pool=T(pool[indices]);task.raised_reset_eligible=T(eligible[indices]);task.raised_reset_fraction=1.
 # Match live reset pose but condition the audit on being seeded, in each scene.
 nominal=task.nominal.numpy();static=[]
 for j,idx in enumerate(protected):
  q=pool[idx,0];d=mujoco.MjData(sim.model);d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(sim.model,d)
  tau=d.qfrc_bias[6:].copy();target=q[7:]+tau/task.kps.numpy()
  static.append(dict(scene_id=full['scenes'][idx]['scene_id'],qpos=q.tolist(),pose_action=((q[7:]-nominal)/.8).tolist(),gravity_torque=tau.tolist(),gravity_compensated_action=((target-nominal)/.8).tolist(),kp=task.kps.tolist(),nominal=nominal.tolist(),physical_lower=task.lower.tolist(),physical_upper=task.upper.tolist()))
 (args.output/'static_hold.json').write_text(json.dumps(dict(joint_names=list(JOINT_NAMES),scenes=static,certificate=certificate),indent=2)+'\n')
 summary=[]
 for mode in args.modes:
  # Replay identical task RNG for each action intervention.
  task.generator.manual_seed(1729);task.capture=False;task.reset(scene_ids=torch.arange(len(indices)));task.capture=True
  init=task.data.qpos.clone();hold=(init[:,7:]-task.nominal)/.8
  task.telemetry={};task.ledger={};frames=[task.frame()];active=np.ones(len(indices),bool);ends=np.full(len(indices),args.steps);actions=[];means=[];stds=[];requested=[];prev=[];started=time.monotonic()
  for step in range(args.steps):
   logits=policy.logits(task.obs['state'],0);mu,sigma=gaussian_parameters(logits);a=mu.tanh()
   if mode=='sampled':a=(mu+sigma*torch.randn(mu.shape,generator=task.generator)).tanh()
   if mode in ('hold_arms','hold_upper','hold_pose'):
    first=15 if mode=='hold_arms' else 12;a[:,first:]=hold[:,first:]
   if mode=='hold_pose':a[:,:12]=0
   if mode=='hold_gravity':
    a[:,12:]=hold[:,12:]
    for k in range(len(indices)):
     d=mujoco.MjData(sim.model);d.qpos[:]=init[k].numpy();mujoco.mj_forward(sim.model,d)
     a[k,12:]+=T(d.qfrc_bias[18:],torch.float32)/(task.kps[12:]*task.info['kp'][k]*.8)
    a.clamp_(-1,1)
   actions.append(a.numpy().copy());means.append(mu.tanh().numpy().copy());stds.append(sigma.numpy().copy());prev.append(task.info['motor_targets'].numpy().copy());requested.append((task.nominal[12:]+.8*a[:,12:]).numpy().copy())
   result=task.step(a);frames.append(task.captured);new=active&result['done'].numpy();ends[new]=step+1;active&=~new
   if step%50==0:print(mode,step,'active',int(active.sum()),'errors',frames[-1]['error'].round(3).tolist(),'seconds',round(time.monotonic()-started,1),flush=True)
   if not active.any():break
  common=set.intersection(*(set(f) for f in frames));arrays={k:np.stack([f[k] for f in frames]) for k in common}
  for k in set(frames[-1])-common:arrays[k]=np.concatenate((np.full_like(frames[-1][k][None],np.nan),np.stack([f[k] for f in frames[1:]])))
  arrays.update(action=np.asarray(actions),policy_mean_action=np.asarray(means),policy_raw_std=np.asarray(stds),requested_upper=np.asarray(requested),previous_target=np.asarray(prev),ends=ends,initial_qpos=init.numpy(),kp=task.info['kp'].numpy(),kd=task.info['kd'].numpy())
  np.savez_compressed(args.output/(mode+'.npz'),**arrays)
  # Human-readable every-step export, one row per scene per step, all arm joints.
  with (args.output/(mode+'.csv')).open('w') as f:
   writer=None
   for j,r in enumerate(records):
    end=int(ends[j]);end=min(end,len(frames)-1)
    for step in range(end+1):
     row=dict(scene=j,scene_id=r['scene_id'],step=step,time_s=step*.02,region_error_m=float(arrays['error'][step,j]),hand_good=float(arrays['hand_contrast_hand_good'][step,j]),heading_good=float(arrays['hand_contrast_heading_good'][step,j]),region_cost=float(arrays['hand_contrast_region_cost'][step,j]),heading_cost=float(arrays['hand_contrast_heading_cost'][step,j]),core=bool(arrays['core'][step,j]),done=bool(arrays['done'][step,j]))
     for h,side in enumerate(['left','right']):
      for k,axis in enumerate('xyz'):row[f'{side}_hand_{axis}_world']=float(arrays['hand_world'][step,j,h,k]);row[f'{side}_hand_{axis}_route']=float(arrays['hand_local'][step,j,h,k])
      for reg in range(2):row[f'{side}_distance_region{reg}']=float(arrays['distance'][step,j,reg,h])
     for k in range(4):row[f'command{k}']=float(arrays['command'][step,j,k])
     for k in range(15,29):
      name=JOINT_NAMES[k];row[name+'_action']=float(arrays['action'][step-1,j,k]) if step else float(hold[j,k]);row[name+'_target']=float(arrays['target'][step,j,k]);row[name+'_angle']=float(arrays['qpos'][step,j,k+7]);row[name+'_requested']=float(nominal[k]+.8*arrays['action'][step-1,j,k]) if step else float(init[j,k+7])
     if writer is None:writer=csv.DictWriter(f,fieldnames=row);writer.writeheader()
     writer.writerow(row)
    sl=slice(1,end+1);good=arrays['hand_contrast_hand_good'][sl,j];bad=np.flatnonzero(good<.5);summary.append(dict(mode=mode,scene=j,scene_id=r['scene_id'],steps=end,first_bad_step=int(bad[0]+1) if len(bad) else None,hand_good_fraction=float(good.mean()),heading_good_fraction=float(arrays['hand_contrast_heading_good'][sl,j].mean()),final_error=float(arrays['error'][end,j]),max_command=float(np.abs(arrays['command'][:end+1,j]).max()),final_root_height=float(arrays['qpos'][end,j,2]),done=bool(arrays['done'][end,j])))
  (args.output/'summary.json').write_text(json.dumps(dict(scenes=indices,steps=args.steps,seed=1729,reset='certified seeded production reset',physics='CPU MuJoCo; ten 2 ms steps per 20 ms action; unchanged production rewards and collision checks',randomization='live noise/PD/RFI/push settings; deterministic mean except sampled mode',rows=summary),indent=2)+'\n')
  print('SUMMARY',json.dumps([r for r in summary if r['mode']==mode]),flush=True)
if __name__=='__main__':main()
