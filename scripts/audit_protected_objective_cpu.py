"""CPU MuJoCo PD counterfactuals with production rewards and stored v5 fields.
No training, policy, checkpoint access, bank writes, or invented walking velocity.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
import sys, json, math, hashlib
from types import SimpleNamespace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch, mujoco
from audit_hand_posture_rewards import Probe, T
from cat_mjlab.raised_reset import raised_pose
from cat_mjlab.balance import posture_terms
from cat_ppo.furniture.generalist_fields import scene_directory
from cat_mjlab import constants


def audit(*, sample_callback=None, bank_path=None):
 torch.set_num_threads(2)
 path=bank_path or ROOT/'data/furniture/cat_flat_hand_balance_v5_20260921/manifest.json'
 manifest=json.loads(path.read_text())
 source=ROOT/'outputs/cat_hand_priority_30720_20260922/run.json'
 cfg=json.loads(source.read_text())['contract']['environment_config']
 assert cfg['reward_config']['scales']['wholebody_hand_contrast_region']==-20
 assert cfg['reward_config']['scales']['wholebody_hand_contrast_heading']==-5
 assert cfg['flat_balance_reward']['bonus_scale']==1.
 probe=Probe(cfg,json.loads((ROOT/'docs/assets/collision-proxy-proposal-20260916/proposal.json').read_text()))
 model=probe.model; results=[]
 from cat_mjlab.collision import CollisionChecker
 checker=CollisionChecker(model,path.parent.with_name(path.parent.name+'_collision')/'manifest.json',field_manifest=path,device='cpu')
 for scene_index,record in enumerate(manifest['scenes']):
  if record.get('source',{}).get('hand_contrast',{}).get('role')!='forward_protected':continue
  directory=scene_directory(manifest,path,record);scene=json.loads((directory/'scene.json').read_text());zone=scene['hand_contrast']['zones'][0]
  compliant=raised_pose(model,zone);nominal=compliant.copy();nominal[19:]=constants.DEFAULT_QPOS[19:]
  bank=probe.bank(record,directory,scene);bank.flat_balance=torch.tensor([False])
  route=np.asarray(scene['route']);tangent=(route[1]-route[0]);tangent/=np.linalg.norm(tangent);yaw=math.atan2(tangent[1],tangent[0])
  start=route[0]+tangent*((zone['start_m']+zone['end_m'])/2)
  scene_rows={}
  for label,initial in [('nominal',nominal),('commandable',compliant),('slew_raise',nominal)]:
   data=mujoco.MjData(model);data.qpos[:]=initial;data.qpos[:2]=start;data.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
   target=initial[7:].copy();frames=[];subposes=[]
   mujoco.mj_forward(model,data)
   def capture():
    velocity=[]
    for sid in range(model.nsite):
     vel=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_SITE,sid,vel,0);velocity.append(vel)
    contacts=[]
    for foot in constants.FEET_GEOMS:
     fid=model.geom(foot).id
     contacts.append(any(c.dist<0 and set(c.geom)=={fid,model.geom('floor').id} for c in data.contact))
    frames.append((data.qpos.copy(),data.qvel.copy(),data.actuator_force.copy(),np.asarray(velocity),contacts,data.site_xpos.copy(),target.copy()))
   capture()
   # Physical 0.4-second hold, 10 physics steps per control step, no pushes/noise.
   for step in range(20):
    if label=='slew_raise':target[12:]+=np.clip(compliant[19:]-target[12:],-cfg['upper_target_rate']*.02,cfg['upper_target_rate']*.02)
    for substep in range(10):
     data.ctrl[:]=np.clip(np.asarray(constants.KPs)*(target-data.qpos[7:])-np.asarray(constants.KDs)*data.qvel[6:],-constants.TORQUE_LIMIT,constants.TORQUE_LIMIT)
     mujoco.mj_step(model,data)
     mujoco.mj_forward(model,data)
     subposes.append((data.xpos.copy(),data.xmat.reshape(-1,3,3).copy()))
    capture()
   qs=np.asarray([f[0] for f in frames]);yaws=np.arctan2(2*(qs[:,3]*qs[:,6]+qs[:,4]*qs[:,5]),1-2*(qs[:,5]**2+qs[:,6]**2))
   obj=probe.task(bank,qs[:,:2],yaws,qs,preserve_root_orientation=True);n=len(qs)
   obj.balance_settings=manifest['flat_balance']['settings'];obj.balance_reward=cfg['flat_balance_reward'];obj.balance_terms=posture_terms
   centers=T(obj.balance_settings['target_centers']);half=T(obj.balance_settings['half_size']);obj.balance_lower=centers-half;obj.balance_upper=centers+half
   obj.data.qvel=T([f[1] for f in frames]);obj.data.actuator_force=T([f[2] for f in frames]);vel=T([f[3] for f in frames])
   positions=T([f[5] for f in frames])[:,obj.site_ids]
   obj.info['velocities']=torch.cat((torch.zeros_like(positions[:1]),(positions[1:]-positions[:-1])/.02))
   obj.info['velocities'][:,[1,2,7,8,9,10]]=0.
   obj.info['torso_angvel']=torch.einsum('bij,bj->bi',obj.info['navi'].transpose(-1,-2),vel[:,obj.torso_site,:3])
   obj.info['last_joint_vel']=torch.cat((torch.zeros(1,29),obj.data.qvel[:-1,6:]))
   targets=T([f[6] for f in frames])
   obj.info['motor_targets']=targets
   obj.info['previous_upper']=torch.cat((targets[:1,12:],targets[:-1,12:]))
   obj.info['previous_previous_upper']=torch.cat((targets[:1,12:].repeat(2,1),targets[:-2,12:]))
   action=torch.zeros(n,29);action[:,12:]=(targets[:,12:]-obj.nominal[12:])/.8
   if label=='slew_raise':action[1:,12:]=(T(compliant[19:])-obj.nominal[12:])/.8
   obj.info['last_act']=action.clone()
   obj.info['last_last_act']=torch.cat((action[:1],action[:-1]))
   phase=torch.arange(n)[:,None]*(2*math.pi*1.4*.02)+torch.tensor([[0.,math.pi]])
   cosine=phase.cos();obj.info['gait']=torch.where(cosine>.6,1.,torch.where(cosine<-.6,-1.,0.))
   def sensor(name,ids):
    site=obj.pelvis_site if name=='global_linvel_pelvis' else int(obj.site_ids[3 if name.startswith('left') else 4])
    return vel[ids,site,3:]
   obj._sensor=sensor
   post,components=obj._rewards(action,torch.tensor([f[4] for f in frames],dtype=torch.bool))
   components={k:v*.02 for k,v in components.items()};pre=sum(components.values())
   collision=checker(torch.full((n,),scene_index,dtype=torch.long),obj.data).any(-1)
   subobj=SimpleNamespace(bank=bank,data=SimpleNamespace(xpos=T([f[0] for f in subposes]),xmat=T([f[1] for f in subposes])))
   collision[1:] |= checker(torch.full((200,),scene_index,dtype=torch.long),subobj.data).any(-1).reshape(20,10).any(-1)
   penalty=-collision.float()*cfg['wholebody']['body_collision']['event_penalty']
   components['body_collision_event']=penalty;post+=penalty
   core=obj.contrast['core_active']&(obj.contrast['role']==1)
   core[0]=False # initial static reference is not a physical step
   # Match immediate production termination; field/self-contact grace is 50 steps.
   fall=(obj.data.site_xmat[:,obj.pelvis_site,2,2]<0)|(obj.info['positions'][:,0,2]<.7)
   terminal=collision|fall|obj.navigation['violation']|~torch.isfinite(obj.data.qpos).all(-1)
   terminal[0]=False
   active=terminal.long().cumsum(0)-terminal.long()==0
   valid_core=core&active
   if sample_callback is not None:
    sample_callback(obj, action, torch.tensor([f[4] for f in frames],dtype=torch.bool), label, core, valid_core, penalty)
   scene_rows[label]=dict(static_compliant=bool(obj.telemetry['hand_contrast_hand_good'][0]),static_pre=float(pre[0]),static_post=float(post[0]),valid_episode_core_steps=int(valid_core.sum()),valid_episode_core_clipped_steps=int(((pre<0)&valid_core).sum()),steps=n-1,protected_core_steps=int(core.sum()),core_clipped_steps=int(((pre<0)&core).sum()),
    core_clipping_fraction=float((pre[core]<0).float().mean()),compliant_fraction=float(obj.telemetry['hand_contrast_hand_good'][core].mean()),
    collision_steps=int(collision[1:].sum()),pre_mean=float(pre[1:].mean()),post_mean=float(post[1:].mean()),
    ledger={k:float(v[1:].mean()) for k,v in components.items()},
    static_ledger={k:float(v[0]) for k,v in components.items()})
  d=mujoco.MjData(model);d.qpos[:]=compliant;mujoco.mj_forward(model,d)
  hands=d.site_xpos[[model.site(name).id for name in constants.HAND_SITES]]-np.r_[compliant[:2],0]
  results.append(dict(scene_id=record['scene_id'],commandable_action_max=float(np.max(np.abs((compliant[19:]-constants.DEFAULT_QPOS[19:])/.8))),hands_local=hands.tolist(),rows=scene_rows))
 ledger={k:sum(r['rows']['commandable']['ledger'][k]-r['rows']['nominal']['ledger'][k] for r in results)/len(results) for k in results[0]['rows']['nominal']['ledger']}
 return dict(method='20 actual CPU MuJoCo PD control steps (0.4 s) per pose per scene; same crouched root/legs, nominal versus commandable arms; stored fields and production reward kernels. No learned policy and no synthetic motion. Native 1.4 Hz gait phases; measured contacts, forces and sensor velocities; production finite-difference field velocities; substep collision checks. Includes separate zero-velocity initial-pose ledger. Also measures a physical 2 rad/s slew from nominal to compliant targets. This is a short hold/raise audit, not walking or training-distribution clipping; full traces continue after termination only as counterfactuals, with valid-episode clipping counts reported separately.',transition_raised_minus_nominal_ledger={k:sum(r['rows']['slew_raise']['ledger'][k]-r['rows']['nominal']['ledger'][k] for r in results)/len(results) for k in ledger},config_source=str(source.relative_to(ROOT)),config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),config={k:v for k,v in cfg.items() if k!='hand_raised_reset_certificate'},scenes=results,mean_raised_minus_nominal_ledger=ledger,static_raised_minus_nominal_ledger={k:sum(r['rows']['commandable']['static_ledger'][k]-r['rows']['nominal']['static_ledger'][k] for r in results)/len(results) for k in ledger},net_pre_delta=sum(v for k,v in ledger.items() if k!='body_collision_event'),net_post_delta=sum(r['rows']['commandable']['post_mean']-r['rows']['nominal']['post_mean'] for r in results)/len(results),valid_episode_core_steps=sum(v['valid_episode_core_steps'] for r in results for label,v in r['rows'].items() if label!='slew_raise'),valid_episode_core_clipped_steps=sum(v['valid_episode_core_clipped_steps'] for r in results for label,v in r['rows'].items() if label!='slew_raise'),protected_core_steps=sum(v['protected_core_steps'] for r in results for label,v in r['rows'].items() if label!='slew_raise'),protected_core_clipped_steps=sum(v['core_clipped_steps'] for r in results for label,v in r['rows'].items() if label!='slew_raise'))

if __name__=='__main__':
 result=audit();out=ROOT/'outputs/protected_objective_audit_cpu';out.mkdir(exist_ok=True);(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('scenes','config')},indent=2))
