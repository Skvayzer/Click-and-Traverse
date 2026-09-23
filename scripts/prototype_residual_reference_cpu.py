#!/usr/bin/env python3
"""Isolated CPU feasibility experiment; no training or production modifications.
Run from repository root. Fixed settings deliberately avoid a new CLI contract.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
import copy, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import mujoco
from scipy.optimize import least_squares
from diagnose_arm_hold_cpu import ROOT, T, subset, TraceTask, _CPUSimulation, CollisionChecker, ActorCritic, LearnerConfig, gaussian_parameters
from cat_mjlab.raised_reset import commandable_limits
from cat_mjlab.constants import HAND_SITES

class ProbeTask(TraceTask):
    def frame(self, done=None):
        f=super().frame(done)
        f['progress']=self.navigation['progress_m'].numpy().copy()
        f['fall']=self.episode['fall'].numpy().copy()
        f['obstacle']=self.episode['obstacle'].numpy().copy()
        return f

class Reference:
    """Warm-start bounded least squares, analytic MuJoCo hand Jacobians.

    Current root, legs, and waist are fixed; only 14 arm joints are optimized.
    Target is the center of the active route-frame boxes. Maximum 12 function
    evaluations per step; actual target remains subject to production slew/PD.
    """
    def __init__(self, task):
        self.model=task.model; self.data=mujoco.MjData(self.model)
        self.sites=[self.model.site(n).id for n in HAND_SITES]
        lo,hi=commandable_limits(self.model,task.upper_action_scales.numpy())
        self.lo,self.hi=lo[15:],hi[15:];self.warm={};self.times=[];self.validation=[];self.validated=set()
    def solve(self, task, j):
        started=time.perf_counter();d=self.data
        q=task.data.qpos[j].numpy().astype(float).copy()
        c=task.contrast;t=c['route_tangent'][j].numpy();n=np.array([-t[1],t[0]])
        valid=np.flatnonzero(c['region_valid'][j].numpy());r=int(valid[0])
        lo=c['hand_regions_min'][j,r].numpy();hi=c['hand_regions_max'][j,r].numpy()
        target=(lo+hi)/2
        world=np.column_stack((q[:2]+target[:,0,None]*t+target[:,1,None]*n,target[:,2]))
        def fun(x):
            d.qpos[:]=q;d.qpos[22:]=x;mujoco.mj_kinematics(self.model,d);mujoco.mj_comPos(self.model,d)
            return (d.site_xpos[self.sites]-world).ravel()
        def jac(x):
            fun(x);rows=[]
            for site in self.sites:
                jp=np.zeros((3,self.model.nv));mujoco.mj_jacSite(self.model,d,jp,None,site);rows.append(jp[:,21:])
            return np.concatenate(rows)
        x=np.clip(self.warm.get(j,q[22:]),self.lo+1e-9,self.hi-1e-9)
        result=least_squares(fun,x,jac=jac,bounds=(self.lo,self.hi),max_nfev=12,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        self.warm[j]=result.x;fun(result.x)
        delta=d.site_xpos[self.sites,:2]-q[:2]
        local=np.column_stack((delta@t,delta@n,d.site_xpos[self.sites,2]))
        error=np.linalg.norm(np.maximum(lo-local,0)+np.maximum(local-hi,0),axis=-1).max()
        self.times.append(time.perf_counter()-started)
        if error > .05 and j not in self.validated:
            # Diagnostic only: longer solve does not replace the rollout reference.
            self.validated.add(j)
            refined=least_squares(fun,result.x,jac=jac,bounds=(self.lo,self.hi),max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            fun(refined.x)
            delta=d.site_xpos[self.sites,:2]-q[:2]
            local=np.column_stack((delta@t,delta@n,d.site_xpos[self.sites,2]))
            refined_error=float(np.linalg.norm(np.maximum(lo-local,0)+np.maximum(local-hi,0),axis=-1).max())
            analytic=jac(result.x);eps=1e-6
            numeric=np.column_stack([(fun(result.x+np.eye(14)[k]*eps)-fun(result.x-np.eye(14)[k]*eps))/(2*eps) for k in range(14)])
            self.validation.append(dict(scene=int(j),short_error_m=float(error),refined_error_m=refined_error,jacobian_max_abs_error=float(np.abs(analytic-numeric).max()),refined_nfev=int(refined.nfev)))
        return result.x,error

def main(amplitudes=(0.,.01,.03,.1,.3), output_name="residual_reference_cpu"):
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    out=ROOT/'outputs'/output_name;out.mkdir(exist_ok=True);snapshot=ROOT/'outputs/residual_reference_cpu/policy_snapshot.pt'
    snap=torch.load(snapshot,map_location='cpu',weights_only=True)
    cfg=copy.deepcopy(snap['contract']['environment_config'])
    policy=ActorCritic(LearnerConfig(**snap['config'])).eval();policy.load_state_dict(snap['model'])
    manifest=ROOT/'data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json'
    evaluation_bank_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()
    full=json.loads(manifest.read_text());protected=[i for i,r in enumerate(full['scenes']) if r.get('source',{}).get('hand_contrast',{}).get('role')=='forward_protected']
    indices=[protected[i] for i in (0,3,6,9)];bank,records,scenes=subset(manifest,indices)
    cfg['wholebody']['body_collision']['bank_manifest']='data/furniture/cat_flat_hand_balance_v6_20260922_collision/manifest.json'
    sim=_CPUSimulation(4);checker=CollisionChecker(sim.model,ROOT/cfg['wholebody']['body_collision']['bank_manifest'],field_manifest=manifest,device='cpu')
    class Mapped:
        proposal_path=checker.proposal_path
        def __call__(self,ids,data):return checker(T(indices,torch.long)[ids],data)
    cached=json.loads((ROOT/'outputs/arm_hold_diagnosis_cpu/static_hold.json').read_text())
    poses={r['scene_id']:r['qpos'] for r in cached['scenes']}
    pool=T([[poses[r['scene_id']]] for r in records],torch.float32);bank.reset_pool=pool
    cfg['hand_raised_reset_fraction']=0.;cfg['randomize_initial_episode_steps']=False
    cfg['upper_gravity_compensation']=True
    # Include shoulder authority from the stated 38--83-step baseline.
    cfg['upper_action_scales']={'left_shoulder_pitch_joint':1.,'right_shoulder_pitch_joint':1.}
    task=ProbeTask(sim,bank,cfg,collision=Mapped(),seed=71)
    task.raised_reset_pool=pool;task.raised_reset_eligible=torch.ones(4,dtype=torch.bool);task.raised_reset_fraction=1.
    report=dict(evaluation_bank_sha256=evaluation_bank_sha256,training_bank_sha256=snap['contract']['bank_sha256'],checkpoint_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),checkpoint_step=snap.get('step'),scenes=indices,config=cfg,dt=task.dt,noise='independent uniform [-amplitude,+amplitude] normalized arm residual each 20 ms, clipped to commandable limits',episodes=[])
    trace=[];start=time.monotonic()
    for amplitude in amplitudes:
        task.generator.manual_seed(1729);task.capture=False;task.reset(scene_ids=torch.arange(4));task.capture=True
        task.telemetry={};task.ledger={};reference=Reference(task);rng=np.random.default_rng(941)
        active=np.ones(4,bool);initial=task.frame();farthest=initial['progress'].copy()
        stats=[]
        for j in range(4):
            zone=scenes[j]['hand_contrast']['zones'][0]
            stats.append(dict(scene=records[j]['scene_id'],amplitude=amplitude,start_progress=float(farthest[j]),core_start=zone['start_m']+zone['fade_m'],core_end=zone['end_m']-zone['fade_m'],zone_end=zone['end_m'],continuous_steps=0,longest_steps=0,streak=0,first_bad=None,compliant_distance=0.,crossed=False))
        for step in range(1,201):
            mu,_=gaussian_parameters(policy.logits(task.obs['state'],0));action=mu.tanh();ik_errors=np.zeros(4)
            for j in np.flatnonzero(active):
                target,ik_errors[j]=reference.solve(task,j)
                residual=rng.uniform(-amplitude,amplitude,14)
                action[j,15:]=T(np.clip((target-task.nominal[15:].numpy())/task.upper_action_scales[3:].numpy()+residual,-1,1),torch.float32)
            result=task.step(action);f=task.captured
            for j in np.flatnonzero(active):
                s=stats[j];good=bool(f['error'][j]<=.05);p=float(f['progress'][j]);end=s['core_end']
                advance=max(0.,min(p,end)-max(float(farthest[j]),s['core_start']))
                if good:s['compliant_distance']+=advance
                farthest[j]=max(farthest[j],p)
                s['streak']=s['streak']+1 if good else 0;s['longest_steps']=max(s['longest_steps'],s['streak'])
                if not good and s['first_bad'] is None:s['first_bad']=step
                if s['first_bad'] is None:s['continuous_steps']=step
                s.update(steps=step,final_progress=p,fall=bool(f['fall'][j]),obstacle=bool(f['obstacle'][j]),done=bool(f['done'][j]),final_error_m=float(f['error'][j]),crossed=p>=end)
                trace.append(dict(amplitude=amplitude,scene=j,step=step,time_s=step*task.dt,error_m=float(f['error'][j]),left_distance_m=float(f['distance'][j,0,0]),right_distance_m=float(f['distance'][j,0,1]),ik_error_m=ik_errors[j],progress_m=p,root_z=float(f['qpos'][j,2]),command_speed=float(np.linalg.norm(f['command'][j,1:3])),core=bool(f['core'][j]),fall=s['fall'],done=s['done']))
                if s['done'] or s['crossed']:active[j]=False
            if step%25==0:print('noise',amplitude,'step',step,'active',active.tolist(),'error',f['error'].round(3).tolist(),'elapsed',round(time.monotonic()-start,1),flush=True)
            if not active.any():break
        for s in stats:
            s['compliant_fraction_remaining_core']=s['compliant_distance']/(s['core_end']-s['start_progress'])
            s['compliant_fraction_full_core']=s['compliant_distance']/(s['core_end']-s['core_start'])
            s['remaining_core_m']=s['core_end']-s['start_progress'];s['continuous_seconds']=s['continuous_steps']*task.dt
        report['ik_validation']=reference.validation;report['episodes'].extend(stats);report['ik_ms_mean']=float(np.mean(reference.times)*1000);report['ik_ms_p95']=float(np.quantile(reference.times,.95)*1000);report['elapsed_s']=time.monotonic()-start
        (out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
        with (out/'trace.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=trace[0]);writer.writeheader();writer.writerows(trace)
        print('RESULT',json.dumps(stats),flush=True)
    assert not torch.cuda.is_initialized()
if __name__=='__main__':main()
