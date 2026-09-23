#!/usr/bin/env python3
"""CPU/CUDA frozen-policy and composite-bank checks; no learner or training."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'tests')]
import numpy as np
import torch
from build_reactive_approval import BANK,OUT,disk
from cat_mjlab.reactive import StandingObjects,validate_bank,load_standing_task
from cat_mjlab.task import CATTask
from cat_mjlab.config import wholebody_config


def simulation(n,device):
    if device!='cpu':
        from cat_mjlab.sim import CATSimulation
        return CATSimulation(n,device=device)
    from test_mjlab_task import _CPUSimulation
    import mujoco
    from types import SimpleNamespace
    class CPU(_CPUSimulation):
        def __init__(self,n):
            super().__init__(n);self.kin=[mujoco.MjData(self.model) for _ in range(n)]
        def final_collision_data(self):
            for q,d in zip(self.data.qpos,self.kin):d.qpos[:]=q.numpy();mujoco.mj_kinematics(self.model,d)
            return SimpleNamespace(xpos=torch.tensor(np.stack([d.xpos for d in self.kin]),dtype=torch.float32),xmat=torch.tensor(np.stack([d.xmat for d in self.kin]),dtype=torch.float32))
    return CPU(n)


def checkpoint_config():
    path=ROOT/'outputs/cat_flat_balance_ppo_37632_20260920/resume.pt'
    snap=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    config=wholebody_config(copy.deepcopy(snap['contract']['environment_config']),stabilization=True,hand_protection=True,hand_contrast=False,
        hand_clearance_weight=-20,arm_clearance_weight=-8,tracking_root_field_weight=1,
        hand_clearance_target=.09,hand_clearance_anticipation=.20,hand_clearance_near_weight=.8,hand_reward_soft_floor=0)
    config.update(randomize_initial_episode_steps=False,hand_raised_reset_fraction=0,upper_gravity_compensation=True)
    config['noise_config']['level']=0;config['push_config']['enable']=False;config['dm_rand_config'].update(enable_pd=False,enable_rfi=False)
    return snap,config


def probe(args):
    from cat_mjlab.learning import ActorCritic,LearnerConfig,gaussian_parameters
    from measure_reactive_signal import build_bank
    meta=validate_bank(json.loads(args.bank.read_text()));selected=[(i,r) for i,r in enumerate(meta['scenes']) if r['direction'] in args.directions.split(',')]
    snap,cfg=checkpoint_config();policy=ActorCritic(LearnerConfig(**snap['learner']['config'])).to(args.device).eval();policy.load_state_dict(snap['learner']['model'])
    n=len(selected)*args.seeds;sim=simulation(n,args.device)
    objects=StandingObjects(bank=args.bank,num_envs=n,model=sim.model,device=args.device)
    objects.flat_scene=0;objects.force_rows=torch.tensor([i for i,_ in selected for _ in range(args.seeds)],device=args.device)
    class ProbeTask(CATTask):
        def reset(self,*a,**kw):
            if getattr(self,'measuring',False):return self.obs
            return super().reset(*a,**kw)
    collision=lambda scenes,data:torch.zeros((len(scenes),6),device=args.device,dtype=torch.bool)
    task=ProbeTask(sim,build_bank(n,args.device),cfg,seed=20260922,collision=collision,analytic_objects=objects);task.measuring=True
    generators=[torch.Generator().manual_seed(31000+j) for j in range(n)]
    alive=np.ones(n,bool);falls=np.zeros(n,bool);contacts=np.zeros(n,bool);robot=np.zeros(n,bool);object_fault=np.zeros(n,bool);mins=np.full(n,np.inf);steps=np.zeros(n,int)
    horizons=np.array([int(np.ceil(r['duration_s']/.02)) for _,r in selected for _ in range(args.seeds)])
    t0=time.perf_counter()
    with torch.no_grad():
      for step in range(int(horizons.max())):
        mu,sigma=gaussian_parameters(policy.logits(task.obs['state'],0))
        eps=torch.stack([torch.randn(29,generator=g) for g in generators]).to(args.device)
        result=task.step((mu+sigma*eps).tanh());m=result['metrics']
        falls|=alive&m['episode/fall'].cpu().numpy();contacts|=alive&m['collision_regions'].any(-1).cpu().numpy()
        robot|=alive&m['reactive/robot_initiated_contact'].cpu().numpy();object_fault|=alive&m['reactive/object_initiated_contact'].cpu().numpy()
        distances=m['acceptance/hand_clearance'].cpu().numpy()
        values=np.array([min(distances[j,h] for h in selected[j//args.seeds][1]['target_hands']) for j in range(n)])
        mins=np.minimum(mins,np.where(alive,values,np.inf));steps+=alive
        alive&=~result['done'].cpu().numpy();alive&=(step+1<horizons)
        # Finished environments retain their physical terminal state as evidence;
        # further batch steps are ignored, never restored or used as samples.
        if step%25==0:print(f'probe {step}/{horizons.max()} live={alive.sum()} elapsed={time.perf_counter()-t0:.1f}s',flush=True)
        if not alive.any():break
    rows=[]
    for j,(_,scene) in enumerate(selected):
        sl=slice(j*args.seeds,(j+1)*args.seeds)
        rows.append(dict(id=scene['id'],seeds=args.seeds,falls=int(falls[sl].sum()),contacts=int(contacts[sl].sum()),robot_initiated_contacts=int(robot[sl].sum()),object_initiated_contacts=int(object_fault[sl].sum()),minimum_hand_surface_m=mins[sl].tolist(),steps=steps[sl].tolist(),horizon_s=scene['duration_s']))
    out=dict(device=args.device,checkpoint='outputs/cat_flat_balance_ppo_37632_20260920/resume.pt',policy='frozen policy with its own sampled action noise; no scripted tuck',scenes=rows,seconds=time.perf_counter()-t0,
        scope='real physics; terminated trajectories censored, not a stationary full-path certificate',total_falls=int(falls.sum()),total_trials=n)
    name='policy-probe.json' if args.device=='cpu' else 'policy-probe-cuda.json';(args.output/name).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)


def preflight(args):
    meta=validate_bank(json.loads(args.bank.read_text()));snap,cfg=checkpoint_config();sim=simulation(1,args.device)
    print('Loading pinned existing bank read-only for real composite-loader smoke',flush=True)
    task=load_standing_task(args.bank,sim,cfg,seed=18)
    task.reactive_objects.force_rows=torch.tensor([0],device=args.device);task.reset()
    assert task.reactive_objects.state['active'].all()
    assert torch.isfinite(task.obs['state']).all();assert task.info['command'].abs().max()==0
    result=task.step(torch.zeros((1,29),device=args.device))
    assert not result['metrics']['reactive/object_initiated_contact'].any()
    state=task.state_dict();task.load_state_dict(state)
    assert isinstance(task.analytic_objects,StandingObjects)
    validate_bank(meta)
    out=dict(passed=True,device=args.device,retained_scenes=meta['retained_scene_count'],new_scenes=len(meta['scenes']),retained_mass=.75,reactive_mass=.25,
        real_composite_loader=True,standing_command=task.info['command'].tolist(),actor_shape=list(task.obs['state'].shape),critic_shape=list(task.obs['privileged_state'].shape),state_roundtrip=True,
        source_manifest_pins_unchanged=True,scope='read-only load and one physical step; no learning',regression='See test-results.txt; legacy JAX-reference suite has missing optional jaxlie/mujoco_playground dependencies')
    (args.output/'preflight.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');p.add_argument('--mode',choices=['probe','preflight'],required=True);p.add_argument('--bank',type=Path,default=BANK);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--directions',default='above');p.add_argument('--seeds',type=int,default=8)
    args=p.parse_args();torch.set_num_threads(2);disk('before '+args.mode,args.output)
    (probe if args.mode=='probe' else preflight)(args);disk('after '+args.mode,args.output)
if __name__=='__main__':main()
