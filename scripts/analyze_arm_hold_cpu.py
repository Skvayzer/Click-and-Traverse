#!/usr/bin/env python3
"""Postprocess measured CPU traces and finite-dimensional FK counterfactuals."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import json,sys,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
from scipy.optimize import least_squares
from cat_mjlab.model import assemble_training_xml
from cat_mjlab.raised_reset import commandable_limits
from cat_mjlab import constants
from cat_ppo.furniture.generalist_fields import scene_directory
OUT=ROOT/'outputs/arm_hold_diagnosis_cpu';p=np.load(OUT/'policy.npz');hold=np.load(OUT/'hold_arms.npz');model=mujoco.MjModel.from_xml_string(assemble_training_xml());d=mujoco.MjData(model)
manifest_path=ROOT/'data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json';manifest=json.loads(manifest_path.read_text());indices=json.loads((OUT/'summary.json').read_text())['scenes'];sites=[model.site(s).id for s in constants.HAND_SITES]
scenes=[json.loads((scene_directory(manifest,manifest_path,manifest['scenes'][i])/'scene.json').read_text()) for i in indices]
def measure(q,scene):
 d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(model,d);t=np.diff(np.asarray(scene['route'])[:2],axis=0)[0];t/=np.linalg.norm(t);n=np.array([-t[1],t[0]]);hands=d.site_xpos[sites];xy=hands[:,:2]-q[:2];local=np.stack((xy@t,xy@n,hands[:,2]),-1);z=scene['hand_contrast']['zones'][0]
 dist=np.linalg.norm(np.maximum(np.array(z['hand_regions_min'])-local,0)+np.maximum(local-np.array(z['hand_regions_max']),0),axis=-1)
 return float(np.where(z['region_valid'],dist.max(-1),np.inf).min()),local
rows=[]
for j,scene in enumerate(scenes):
 for step in [0,1,2,5,10,20,50,100,200]:
  q=p['qpos'][step,j].copy();arm_only=p['qpos'][0,j].copy();arm_only[22:]=q[22:];body_only=q.copy();body_only[22:]=p['qpos'][0,j,22:]
  rows.append(dict(scene=j,step=step,measured_error=float(p['error'][step,j]),fk_error=measure(q,scene)[0],achieved_arms_reset_body_error=measure(arm_only,scene)[0],reset_arms_achieved_body_error=measure(body_only,scene)[0],arm_tracking_rms=float(np.sqrt(np.mean((q[22:]-p['target'][step,j,15:])**2))),hold_arms_error=float(hold['error'][step,j])))
# Local bounded arm IK: non-arm pose fixed to measured standing pose. A failed
# local solve is not a global infeasibility proof. Both centre and box objectives.
ik=[]
for j,scene in enumerate(scenes):
 for scale in [.8,1.0]:
  q=p['qpos'][200,j].copy();lo,hi=commandable_limits(model,scale);z=scene['hand_contrast']['zones'][0];lower=np.asarray(z['hand_regions_min'][0]);upper=np.asarray(z['hand_regions_max'][0]);centre=(lower+upper)/2
  def fun(a):
   q[22:]=a;_,local=measure(q,scene);return (local-centre).ravel()
  best=None
  for initial in [p['qpos'][0,j,22:],p['qpos'][200,j,22:]]:
   sol=least_squares(fun,np.clip(initial,lo[15:]+1e-8,hi[15:]-1e-8),bounds=(lo[15:],hi[15:]),max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10,diff_step=1e-5)
   fun(sol.x);error,local=measure(q,scene)
   if best is None or error<best['error']:best=dict(scene=j,scale=scale,error=error,qpos=q.tolist(),hand_local=local.tolist(),max_action=float(np.max(np.abs((q[22:]-np.asarray(constants.DEFAULT_QPOS[22:]))/scale))))
  ik.append(best)
ledger=[]
for k in p.files:
 if k.startswith('reward_') and k!='reward_floor_slope':ledger.append(dict(component=k,policy=float(p[k][1:41].mean()),hold_arms=float(hold[k][1:41].mean()),delta_hold_minus_policy=float(hold[k][1:41].mean()-p[k][1:41].mean())))
report=dict(fk_decomposition=rows,standing_arm_ik=ik,first_40_steps_reward_ledger=ledger,arm_tracking_rms_rad=float(np.sqrt(np.mean((p['qpos'][1:,:,22:]-p['target'][1:,:,15:])**2))),policy_shoulder_action_first_step=p['action'][0][:,[15,22]].tolist(),policy_shoulder_action_mean=p['action'][:,:,[15,22]].mean((0,1)).tolist(),first_20_steps_shoulder_action_range=[float(p['action'][:20,:,[15,22]].min()),float(p['action'][:20,:,[15,22]].max())])
(OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'time_series_excerpt.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True,layout='constrained');t=np.arange(len(p['qpos']))*.02
 ax=axes[0]
 for k,side in [(15,'left'),(22,'right')]:ax.plot(t[1:],p['action'][:,0,k],label=side+' policy action')
 ax.axhline(-1,color='black',ls='--',label='pose target action = -1');ax.axhline(-1.09095,color='red',ls=':',label='static gravity hold at nominal Kp');ax.set_ylabel('Shoulder-pitch action');ax.legend(ncol=2,fontsize=8)
 ax=axes[1];ax.plot(t,p['target'][:,0,15],label='left motor target');ax.plot(t,p['qpos'][:,0,22],label='left achieved joint');ax.axhline(-.6,color='black',ls='--',label='certified joint angle');ax.set_ylabel('Shoulder pitch (rad)');ax.legend(fontsize=8)
 ax=axes[2]
 for j in range(4):ax.plot(t,p['error'][:,j],label=f'scene {j}')
 ax.axhline(.05,color='black',ls='--',label='strict tolerance');ax.set_ylabel('Paired region error (m)');ax.legend(ncol=5,fontsize=8)
 ax=axes[3]
 for j in range(4):ax.plot(t,p['qpos'][:,j,2],label=f'scene {j}')
 ax.axhline(.55,color='black',ls='--');ax.set_ylabel('Root height (m)');ax.set_xlabel('Time (seconds)')
 for ax in axes:ax.grid(alpha=.2)
 fig.suptitle('Update 20: seeded protected CPU rollouts, commanded velocity identically zero\nFrozen mean policy; live observation noise, PD and force randomization')
 fig.savefig(OUT/'policy_time_series.png',dpi=130);plt.close(fig)
except ImportError as e:print('Plot unavailable:',e)
print(json.dumps({k:v for k,v in report.items() if k not in ['first_40_steps_reward_ledger','fk_decomposition']},indent=2))
