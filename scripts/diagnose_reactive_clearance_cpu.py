#!/usr/bin/env python3
"""Frozen policy, paired open-loop moving-sphere diagnostic, entirely on CPU.

Sphere uses analytic obstacle SDF/normal/guidance through production observations.
Like production bank obstacles it is a virtual collision obstacle, not a contact
force. Stop analysis at first sphere contact or task termination; no pushed-hand
motion is counted. Controls receive no sphere. No bank/checkpoint mutations.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['OMP_NUM_THREADS']='2'
import sys, json, shutil, hashlib, copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import numpy as np
import torch
from test_mjlab_task import _CPUSimulation, _tiny_bank
from cat_mjlab.task import CATTask
from cat_mjlab.learning import ActorCritic,LearnerConfig,gaussian_parameters

class ProbeTask(CATTask):
    speed=0.
    def _rand(self,shape,low=0.,high=1.):
        return torch.full(shape,(low+high)/2,device=self.device)
    def _fields(self,positions,root_xy,ids):
        gf,bf,sdf,cmd=super()._fields(positions,root_xy,ids)
        cmd[:]=torch.tensor([.75 if self.speed else 0.,self.speed,0.,0.])
        return gf,bf,sdf,cmd
    def _outcomes(self,done,truncation):
        self.frame=dict(hands=self.info['positions'][:,5:7].numpy().copy(),qpos=self.data.qpos.numpy().copy(),
            body=self.data.xpos.numpy().copy(),done=done.numpy().copy(),
            ledger={k:float(v.mean()) for k,v in self.ledger.items()})
        return super()._outcomes(done,truncation)
    def _rewards(self,action,contacts):
        r,p=super()._rewards(action,contacts);self.ledger=p;return r,p

def main():
    out=ROOT/'outputs/reactive_clearance_cpu';out.mkdir(exist_ok=True)
    src=ROOT/'outputs/cat_recover_23_30720_20260922/best.pt';snap=out/'policy_snapshot.pt'
    if not snap.exists():shutil.copyfile(src,snap)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(src)==sha(snap)
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    s=torch.load(snap,map_location='cpu',weights_only=True)
    policy=ActorCritic(LearnerConfig(**s['config'])).eval();policy.load_state_dict(s['model'])
    dirs=dict(side=[0,1,0],front=[1,0,0],below=[0,0,-1],above=[0,0,1])
    conditions=[(d,v) for d in dirs for v in (.05,.10,.20)]
    n=2*len(conditions);radius=.06;rows=[];allframes={}
    for mode,speed in [('standing',0.),('walking',.6)]:
        cfg=copy.deepcopy(s['contract']['environment_config']);cfg.update(wholebody_hand_contrast=False,randomize_initial_episode_steps=False,hand_raised_reset_fraction=0.)
        cfg['noise_config']['level']=0.;cfg['push_config']['enable']=False
        cfg['dm_rand_config'].update(enable_pd=False,enable_rfi=False)
        bank=_tiny_bank(n);bank.episode_lengths[:]=2000;bank.goals=torch.tensor([[100.,0.,.7]]).repeat(n,1)
        bank.origins=torch.tensor([[-100.,-100.,-5.]]).repeat(n,1);bank.shapes=torch.tensor([[1000,1000,100]]).repeat(n,1)
        centers=torch.full((n,3),1000.);visible=torch.zeros(n,dtype=torch.bool)
        def sample(name,pos,ids):
            delta=pos-centers[ids,None];norm=torch.linalg.vector_norm(delta,dim=-1,keepdim=True)
            sdf=norm-radius;normal=delta/norm.clamp_min(1e-9)
            active=visible[ids,None,None]
            if name=='sdf':return torch.where(active,sdf,torch.ones_like(sdf))
            if name=='bf':return torch.where(active,normal,torch.zeros_like(pos))
            gf=torch.zeros_like(pos);gf[:,:,0]=speed
            inward=(gf*normal).sum(-1,keepdim=True).clamp_max(0)
            return gf-torch.where(active&(sdf<.5),inward*normal,0.)
        bank.sample=sample
        ProbeTask.speed=speed;task=ProbeTask(_CPUSimulation(n),bank,cfg,seed=71)
        task.math.compute_cmd_from_rtf=lambda pelvis,gf,bf:torch.tensor([.75 if speed else 0.,speed,0.,0.]).repeat(len(pelvis),1)
        task.reset(scene_ids=torch.arange(n));task.info['phase'][:]=task.info['phase'][0];task.info['gait'][:]=task.info['gait'][0]
        task.obs=task._observe(task.all_ids,torch.zeros((n,2),dtype=torch.bool))
        frames=[];first_done=np.full(n,10000);first_contact=np.full(n,10000);anchor=None
        for step in range(680):
            t=step*.02
            if step==100:anchor=task.info['positions'][::2,5].clone();root=task.data.qpos[::2,:3].clone()
            if step>=100:
                for j,(direction,v) in enumerate(conditions):
                    # Advect along the CONTROL root displacement, never chase the treated hand.
                    travel=task.data.qpos[2*j+1,:3]-root[j];travel[2]=0
                    gap=.55-v*(t-2.)
                    centers[2*j]=anchor[j]+travel+torch.tensor(dirs[direction])*(radius+float(task.hand_radii[0])+gap)
                    visible[2*j]=True
            mu,_=gaussian_parameters(policy.logits(task.obs['state'],0));result=task.step(mu.tanh())
            f=task.frame;f['centers']=centers.numpy().copy();frames.append(f)
            first_done=np.minimum(first_done,np.where(f['done'],step,10000))
            dist=np.linalg.norm(f['hands'][:,0]-f['centers'],axis=-1)-radius-float(task.hand_radii[0])
            first_contact=np.minimum(first_contact,np.where(dist<=0,step,10000))
            if step%100==0:print(mode,step,'terminated',int((first_done<=step).sum()),flush=True)
        hands=np.stack([f['hands'] for f in frames]);q=np.stack([f['qpos'] for f in frames]);c=np.stack([f['centers'] for f in frames])
        allframes[mode]=(frames,task.sim.model.body_parentid.copy())
        for j,(direction,v) in enumerate(conditions):
            a,b=2*j,2*j+1;end=min(680,101+int(.57/v/.02),int(first_done[a])+1,int(first_done[b])+1,int(first_contact[a])+1)
            sl=slice(100,end);d=np.asarray(dirs[direction]);distance=np.linalg.norm(hands[sl,a,0]-c[sl,a],axis=-1)-radius-float(task.hand_radii[0])
            retreat=-((hands[sl,a,0]-hands[sl,b,0])@d)
            relative=-(((hands[sl,a,0]-q[sl,a,:3])-(hands[sl,b,0]-q[sl,b,:3]))@d)
            onset=next((k for k in range(max(0,len(retreat)-4)) if np.all(retreat[k:k+5]>.01)),None)
            rows.append(dict(mode=mode,direction=direction,speed_m_s=v,duration_s=(end-100)*.02,
                retreat_peak_cm=float(retreat.max()*100),retreat_final_cm=float(retreat[-1]*100),arm_relative_peak_cm=float(relative.max()*100),
                onset_distance_cm=None if onset is None else float(distance[onset]*100),min_clearance_cm=float(distance.min()*100),
                inside_20cm_fraction=float((distance<.20).mean()),below_4cm_fraction=float((distance<.04).mean()),
                stayed_standing=bool(q[:end,a,2].min()>.5 and (1-2*(q[:end,a,4]**2+q[:end,a,5]**2)).min()>.7),
                min_root_height_m=float(q[:end,a,2].min()),
                terminated=bool(first_done[a]<end),contact=bool(first_contact[a]<end),
                root_speed_m_s=float((q[end-1,a,0]-q[99,a,0])/((end-100)*.02)),
                root_drift_cm=float(np.linalg.norm(q[end-1,a,:2]-q[99,a,:2])*100),
                pre_stimulus_pair_error=float(np.abs(hands[:100,a]-hands[:100,b]).max())))
        np.savez_compressed(out/(mode+'.npz'),hands=hands,qpos=q,centers=c,first_done=first_done,first_contact=first_contact)
    report=dict(checkpoint_sha256=sha(snap),sphere_radius_m=radius,settle_s=2.,conditions=rows,
        protocol='Left hand; deterministic paired no-obstacle controls; ideal analytic fields; sphere advected by control root XY; first contact/termination censored; 1 seed; no contact forces.',
        reward_ledger={mode:{k:float(np.mean([f['ledger'][k] for f in frames[100:]])) for k in frames[0]['ledger']} for mode,(frames,_) in allframes.items()})
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for mode,(frames,parents) in allframes.items():
        fig,axes=plt.subplots(4,4,figsize=(12,11),subplot_kw={'projection':'3d'})
        for row,direction in enumerate(dirs):
            j=row*3+1;a=2*j
            for col,k in enumerate((100,250,325,370)):
                ax=axes[row,col];f=frames[k];p=f['body'][a];root=f['qpos'][a,:3]
                for child,parent in enumerate(parents):
                    ax.plot(*p[[parent,child]].T,color='steelblue',lw=2)
                h=f['hands'][a,0];center=f['centers'][a]
                ax.scatter(*h,c='green',s=35);ax.scatter(*f['hands'][a+1,0],c='gray',marker='x',s=40)
                u,w=np.mgrid[0:2*np.pi:12j,0:np.pi:8j]
                ax.plot_surface(center[0]+radius*np.cos(u)*np.sin(w),center[1]+radius*np.sin(u)*np.sin(w),center[2]+radius*np.cos(w),color='red',alpha=.5)
                ax.set(xlim=(root[0]-.55,root[0]+.55),ylim=(root[1]-.55,root[1]+.55),zlim=(0,1.4),title=f'{direction}, t={k*.02-2:.1f}s')
                ax.view_init(15,35);ax.set_box_aspect((1,1,1.4));ax.set_axis_off()
        fig.suptitle(f'{mode}: 0.10 m/s sphere approach; green hand, gray control hand\nMuJoCo body skeleton; frames after contact are illustrative only')
        fig.tight_layout();fig.savefig(out/(mode+'_strip.png'),dpi=120);plt.close(fig)
    print(json.dumps(rows,indent=2),flush=True)
if __name__=='__main__':main()
