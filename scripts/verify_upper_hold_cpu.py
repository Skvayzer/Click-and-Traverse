#!/usr/bin/env python3
"""CPU-only upper-control ablations using the cached original certified poses.
No training; tiny JSON/CSV output; independent zero-command hold interventions.
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


class ZeroCommandTask(TraceTask):
 def _fields(self,*args,**kwargs):
  gf,bf,sdf,cmd=super()._fields(*args,**kwargs)
  return gf,bf,sdf,torch.zeros_like(cmd)

def main():
 static_only='--static-only' in sys.argv
 out=ROOT/'outputs/upper_hold_fix_cpu';out.mkdir(exist_ok=True)
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 snap=torch.load(ROOT/'outputs/arm_hold_diagnosis_cpu/policy_snapshot.pt',map_location='cpu',weights_only=True)
 cfg=copy.deepcopy(snap['contract']['environment_config']);policy=ActorCritic(LearnerConfig(**snap['config'])).eval();policy.load_state_dict(snap['model'])
 manifest=ROOT/'data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json';full=json.loads(manifest.read_text());protected=[i for i,r in enumerate(full['scenes']) if r.get('source',{}).get('hand_contrast',{}).get('role')=='forward_protected'];indices=[protected[i] for i in [0,3,6,9]]
 assert hashlib.sha256(manifest.read_bytes()).hexdigest()==snap['contract']['bank_sha256']
 print('Loading four protected scenes',indices,flush=True);bank,records,scenes=subset(manifest,indices);sim=_CPUSimulation(len(indices));collision_path=ROOT/cfg['wholebody']['body_collision']['bank_manifest']
 checker=CollisionChecker(sim.model,collision_path,field_manifest=manifest,device='cpu')
 class MappedCollision:
  proposal_path=checker.proposal_path
  def __call__(self,ids,data):return checker(T(indices,torch.long)[ids],data)
 cached=json.loads((ROOT/'outputs/arm_hold_diagnosis_cpu/static_hold.json').read_text())
 pool=np.tile(np.array(cached['scenes'][0]['qpos'],dtype=np.float32), (len(full['scenes']),1,1));eligible=np.zeros(len(full['scenes']),bool)
 for idx,row in zip(protected,cached['scenes']):pool[idx,0]=row['qpos'];eligible[idx]=True
 certificate=cached['certificate']
 bank.reset_pool=T(pool[indices]);cfg['hand_raised_reset_fraction']=0.;cfg['randomize_initial_episode_steps']=False

 from cat_mjlab.upper_control import UpperGravity
 gravity=UpperGravity(sim.model,'cpu')
 hold_rows=[];max_error=0.
 for j,idx in enumerate(indices):
  d=mujoco.MjData(sim.model);d.qpos[:]=pool[idx,0];mujoco.mj_forward(sim.model,d)
  ff=gravity(SimpleNamespace(xpos=T(d.xpos[None],torch.float32),xmat=T(d.xmat[None],torch.float32)))[0].numpy()
  max_error=max(max_error,float(np.max(np.abs(ff-d.qfrc_bias[18:]))))
  if j==0:
   for k in range(15,29):
    scale=1. if k in (15,22) else .8
    q=d.qpos[7+k];nom=cached['scenes'][0]['nominal'][k];kp=cached['scenes'][0]['kp'][k];tau=d.qfrc_bias[6+k]
    hold_rows.append(dict(joint=JOINT_NAMES[k],gravity_Nm=tau,before=(q-nom+tau/kp)/.8,gc_only=(q-nom)/.8,all_fixes=(q-nom+(tau-ff[k-12])/kp)/scale))
 # Also validate arbitrary orientations/configurations against MuJoCo zero-velocity bias.
 rng=np.random.default_rng(39)
 for _ in range(30):
  d.qpos[7:]=rng.uniform(sim.model.jnt_range[1:,0],sim.model.jnt_range[1:,1]);quat=rng.normal(size=4);d.qpos[3:7]=quat/np.linalg.norm(quat);d.qvel[:]=0;mujoco.mj_forward(sim.model,d)
  ff=gravity(SimpleNamespace(xpos=T(d.xpos[None],torch.float32),xmat=T(d.xmat[None],torch.float32)))[0].numpy()
  max_error=max(max_error,float(np.max(np.abs(ff-d.qfrc_bias[18:]))))
 results=[];static=[]
 for mode in ['before','gravity','authority','sdf','all']:
  c=copy.deepcopy(cfg);c['upper_gravity_compensation']=mode in ('gravity','all');c['protected_hand_sdf_margin']=mode in ('sdf','all')
  c['upper_action_scales']={JOINT_NAMES[k]:1. for k in (15,22)} if mode in ('authority','all') else {}
  task=ZeroCommandTask(sim,bank,c,collision=MappedCollision(),seed=71)
  task.raised_reset_pool=T(pool[indices]);task.raised_reset_eligible=T(eligible[indices]);task.raised_reset_fraction=1.
  task.generator.manual_seed(1729);task.reset(scene_ids=torch.arange(len(indices)))
  init=task.data.qpos.clone();hold=(init[:,19:]-task.nominal[12:])/task.upper_action_scales
  # Paired instantaneous ledger: same root/legs, zero velocity, arm posture changed only.
  for posture in ['certified','nominal_arms']:
   task.data.qpos.copy_(init)
   if posture=='nominal_arms':task.data.qpos[:,22:]=task.nominal[15:]
   task.data.qvel.zero_();task.data.ctrl.zero_();sim.forward();q=task.data.qpos[:,7:]
   act=torch.zeros((4,29));act[:,12:]=(q[:,12:]-task.nominal[12:])/task.upper_action_scales
   task.info['motor_targets']=q.clone();task.info['previous_upper']=q[:,12:].clone();task.info['previous_previous_upper']=q[:,12:].clone()
   task.info['last_act']=act.clone();task.info['last_last_act']=act.clone();task.info['last_joint_vel'].zero_()
   positions=task._poses(task.all_ids);gf,bf,sdf,cmd=task._fields(positions,task.data.qpos[:,:2],task.all_ids)
   task.info.update(positions=positions,gf=gf,bf=bf,sdf=sdf,command=cmd)
   task.info['velocities'].zero_()
   task.info['elbow_clearance']=task._elbow_fields(task.all_ids,False)[:,-2:]
   reward,parts=task._rewards(act,sim.contact_flags(task.contact_pairs))
   assert not task.collision(task.scene_ids,sim.final_collision_data()).any(), 'Static posture collides'
   static.append(dict(mode=mode,posture=posture,clearance_m=sdf[:,5:7,0].tolist(),reward=float(reward.mean()),ledger={k:float(v.mean()*task.dt) for k,v in parts.items()}))
  if static_only:
   (out/'static_ledger.json').write_text(json.dumps(static,indent=2)+'\n')
   continue
  # Reinitialize identical randomization/history after static ledger.
  task.generator.manual_seed(1729);task.capture=False;task.reset(scene_ids=torch.arange(4));task.capture=True
  task.telemetry={};task.ledger={};active=np.ones(4,bool);first_bad=[None]*4;total_good=np.zeros(4,int);ends=np.full(4,200);trace=[];ledgers=[]
  for step in range(1,201):
   logits=policy.logits(task.obs['state'],0);mu,_=gaussian_parameters(logits);a=mu.tanh();a[:,15:]=hold[:,3:]
   result=task.step(a);f=task.captured;good=f['error']<=.05
   assert np.max(np.abs(f['command']))==0
   for j in range(4):
    if active[j]:
     total_good[j]+=int(good[j])
     if not good[j] and first_bad[j] is None:first_bad[j]=step
     trace.append(dict(mode=mode,scene=j,step=step,error_m=float(f['error'][j]),root_z=float(f['qpos'][j,2]),done=bool(f['done'][j])))
   if step<=40:ledgers.append({k:float(v.mean()) for k,v in task.ledger.items()})
   new=active & result['done'].numpy();ends[new]=step;active &= ~new
   if step%50==0:print(mode,step,f['error'].round(4).tolist(),flush=True)
   if not active.any():break
  results.append(dict(mode=mode,first_bad=first_bad,continuous_compliant_steps=[(v-1 if v is not None else int(ends[j])) for j,v in enumerate(first_bad)],total_good=total_good.tolist(),ends=ends.tolist(),first40_ledger={k:float(np.mean([r[k] for r in ledgers])) for k in ledgers[0]}))
  with (out/(mode+'.csv')).open('w') as file:
   w=csv.DictWriter(file,fieldnames=trace[0]);w.writeheader();w.writerows(trace)
  (out/'measurements.json').write_text(json.dumps(dict(gravity_max_error_Nm=max_error,hold_actions=hold_rows,static=static,dynamic=results),indent=2)+'\n')
  print('RESULT',results[-1],flush=True)
if __name__=='__main__':main()
