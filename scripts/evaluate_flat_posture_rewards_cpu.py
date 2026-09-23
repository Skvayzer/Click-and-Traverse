#!/usr/bin/env python3
"""CPU-only counterfactual reward audit. FK/IK samples, NOT dynamic walking evidence.
No actor, checkpoint, bank or training-path mutation. Uses production CATTask._rewards.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import sys,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch,mujoco
from scipy.optimize import least_squares
from audit_hand_posture_rewards import Probe,T

def walking_posture_bonus(region_cost, route_velocity, *, dt=.02, scale=3., speed=.6):
 """Evaluation-only candidate; dense, positive and zero without forward progress."""
 return dt*scale*(1-region_cost).clamp(0,1)*(route_velocity/speed).clamp(0,1)

def main():
 torch.set_num_threads(2)
 cfg=json.loads((ROOT/'outputs/cat_hand_posture_30720_20260919/run.json').read_text())['contract']['environment_config']
 probe=Probe(cfg,json.loads((ROOT/'docs/assets/collision-proxy-proposal-20260916/proposal.json').read_text()))
 m=probe.model;d=mujoco.MjData(m);nom=probe.qpose().astype(np.float64);d.qpos[:]=nom;mujoco.mj_forward(m,d)
 from cat_mjlab.constants import FEET_SITES,HAND_SITES
 feet=[m.site(s).id for s in FEET_SITES];hands=[m.site(s).id for s in HAND_SITES]
 footpos=d.site_xpos[feet].copy();footrot=d.site_xmat[feet].copy()
 def fk(q):d.qpos[:]=q;mujoco.mj_forward(m,d)
 def pose(height,raised):
  q=nom.copy();q[2]=height
  def legs(x):
   q[7:19]=x;fk(q)
   return np.r_[(d.site_xpos[feet]-footpos).ravel()*10,(d.site_xmat[feet]-footrot).ravel(),.01*(x-nom[7:19])]
  lo=m.jnt_range[1:13,0]+1e-5;hi=m.jnt_range[1:13,1]-1e-5
  fit=least_squares(legs,np.clip(nom[7:19],lo,hi),bounds=(lo,hi),max_nfev=300);q[7:19]=fit.x
  if raised:
   for side,sid,y in [(0,hands[0],.125),(1,hands[1],-.125)]:
    start=22+7*side;idx=slice(start,start+7);joint=slice(start-6,start+1)
    low=np.maximum(m.jnt_range[joint,0]+1e-5,nom[idx]-.8);high=np.minimum(m.jnt_range[joint,1]-1e-5,nom[idx]+.8)
    def arm(x):
     q[idx]=x;fk(q);return np.r_[(d.site_xpos[sid]-[.429,y,.870])*100,.001*(x-nom[idx])]
    fit=least_squares(arm,np.clip(probe.qpose('raised')[idx],low,high),bounds=(low,high),max_nfev=500);q[idx]=fit.x
  fk(q);return q.copy(),dict(hands=d.site_xpos[hands].tolist(),feet_error_m=float(np.max(np.linalg.norm(d.site_xpos[feet]-footpos,axis=1))),head_z=float(d.site_xpos[m.site('head').id,2]))
 manifest=ROOT/'data/furniture/cat_hand_posture_v1_20260919/manifest.json';full=json.loads(manifest.read_text());r=next(r for r in full['scenes'] if r.get('source',{}).get('hand_protection',{}).get('level')==0);directory=manifest.parent/r['path'];scene=json.loads((directory/'scene.json').read_text())
 bank=probe.bank(r,directory,scene)
 # Analytic empty-obstacle fields: finite far distance, zero boundary force, straight guidance.
 def sample(name,pos,ids):
  if name=='sdf':return torch.full((*pos.shape[:-1],1),10.)
  value=torch.zeros_like(pos)
  if name=='gf':value[...,0]=.6
  return value
 bank.sample=sample
 rows={}
 cases=[('walk_nominal',.643,False,.6),('walk_raised',.643,True,.6),('stand_raised',.643,True,0.),('crouch_raise',.60,True,.6)]
 cases += [(f'sensitivity_{h}_{pose_name}',h,up,.6) for h in [.62,.66,.69] for pose_name,up in [('nominal',False),('raised',True)]]
 for label,height,raised,speed in cases:
  q,geom=pose(height,raised);xy=np.asarray(scene['route'][:1]);obj=probe.task(bank,xy,np.zeros(1),[q])
  obj.navigation['tangent']=T([[1,0]]);obj.info['command']=T([[1,.6,0,0]])
  obj._crossed=lambda positions,ids:torch.zeros(positions.shape[:2],dtype=torch.bool)
  c=obj.contrast;c['route_tangent']=T([[1,0]]);c['forward_weight']=torch.zeros(1);c['enabled']=torch.ones(1,dtype=torch.bool);c['phase_weight']=torch.ones(1);c['hand_active']=torch.ones((1,2),dtype=torch.bool);c['region_valid']=torch.tensor([[True,False]])
  center=T([[[.429,.125,.870],[.429,-.125,.870]]]);half=T([.018,.010,.018]);c['hand_regions_min'][:,0]=center-half;c['hand_regions_max'][:,0]=center+half
  out=probe.rewards(obj,speed=speed,double_support=(speed==0));components={k:float(v[0]) for k,v in out['components'].items()};base=float(out['pre'][0]);cost=-components['wholebody_hand_contrast_region']/.02
  rows[label]=dict(geometry=geom,region_cost=cost,pre=base,post=float(out['post'][0]),minimum_phase_pre=float(out['minimum_phase_pre'][0]),components=components,weights={str(w):dict(pre=base-.02*(w-1)*cost,minimum_phase_pre=float(out['minimum_phase_pre'][0])-.02*(w-1)*cost) for w in [1,3,5,8,12]})
  for weight,result in rows[label]['weights'].items():
   obj.config=copy.deepcopy(cfg);obj.config['reward_config']['scales']['wholebody_hand_contrast_region']=-float(weight)
   measured=probe.rewards(obj,speed=speed,double_support=(speed==0))
   result.update(post=float(measured['post'][0]),zero_fraction=float(measured['zero_fraction'][0]),clamp_possible=result['minimum_phase_pre']<=0)
  # Candidate: replace negative region cost with a dense proximity bonus,
  # multiplied by positive route speed. No sparse tolerance gate.
  bonus=float(walking_posture_bonus(torch.tensor(cost),torch.tensor(speed)))
  rows[label]['candidate_bonus3']=dict(bonus=bonus,pre=base+.02*cost+bonus,minimum_phase_pre=float(out['minimum_phase_pre'][0])+.02*cost+bonus)
  obj.config=cfg
 # Smooth 0.6-s nominal-to-raised interpolation at a fixed walking root.
 q0,_=pose(.643,False);q1,_=pose(.643,True);times=np.arange(31)*.02
 def blend(t):
  u=np.clip(t/.6,0,1);return u*u*(3-2*u)
 qs=q0[None]+blend(times)[:,None]*(q1-q0)[None]
 xy=np.repeat(np.asarray(scene['route'][:1]),31,axis=0)
 obj=probe.task(bank,xy,np.zeros(31),qs);obj.config=cfg
 obj.navigation['tangent']=T([[1,0]]).repeat(31,1);obj.info['command']=T([[1,.6,0,0]]).repeat(31,1)
 obj._crossed=lambda positions,ids:torch.zeros(positions.shape[:2],dtype=torch.bool)
 c=obj.contrast;c['route_tangent']=T([[1,0]]).repeat(31,1);c['forward_weight']=torch.zeros(31);c['enabled']=torch.ones(31,dtype=torch.bool);c['phase_weight']=torch.ones(31);c['hand_active']=torch.ones((31,2),dtype=torch.bool);c['region_valid']=torch.tensor([[True,False]]).repeat(31,1)
 c['hand_regions_min'][:,0]=center-half;c['hand_regions_max'][:,0]=center+half
 obj.info['previous_upper']=T((q0[None]+blend(times-.02)[:,None]*(q1-q0)[None])[:,19:])
 obj.info['previous_previous_upper']=T((q0[None]+blend(times-.04)[:,None]*(q1-q0)[None])[:,19:])
 vel=np.gradient(obj.info['positions'].numpy(),.02,axis=0);vel[:,:,0]+=.6
 obj.data.qvel[:,6:]=T(np.gradient(qs[:,7:],.02,axis=0));obj.info['last_joint_vel']=T(np.gradient((q0[None]+blend(times-.02)[:,None]*(q1-q0)[None])[:,7:],.02,axis=0))
 out=probe.rewards(obj,speed=.6,site_velocity=vel);cost=-out['components']['wholebody_hand_contrast_region']/.02
 candidate=out['pre']+.02*cost+walking_posture_bonus(cost,torch.full_like(cost,.6))
 transition=dict(method='Prescribed smoothstep joint interpolation, not a simulated feasible controller',original_mean=float(out['pre'].mean()),candidate_mean=float(candidate.mean()),candidate_minimum_phase_pre=float((out['minimum_phase_pre']+.02*cost+.06*(1-cost)).min()),components_mean={k:float(v.mean()) for k,v in out['components'].items()})
 report=dict(transition=transition,method='Counterfactual FK and bounded arm IK with feet planted; gait-phase averaged reward, synthetic matched velocity. Not a dynamically feasible walking trajectory; actuator force uses static gravity compensation, not measured walking torque.',config=cfg,rows=rows)
 outdir=ROOT/'outputs/flat_posture_reward_audit_cpu';outdir.mkdir(exist_ok=True);(outdir/'reward_audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
