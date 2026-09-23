#!/usr/bin/env python3
"""Read-only CPU dynamics gate for level-0 hand acquisition. Never trains.

Uses production CATTask, fields, collision checker and frozen native policy.
Loads only eight scenes' fields; original manifest hashes remain authoritative.
Controller changes actions only. Reset variation is restricted only at reset.
Outputs aggregate evaluation records, not an imitation dataset.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import argparse, copy, hashlib, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import numpy as np
import torch
from test_mjlab_task import _CPUSimulation
from cat_mjlab.task import CATTask
from cat_mjlab.scene_bank import SceneBank
from cat_mjlab.learning import ActorCritic,LearnerConfig
from cat_mjlab.runner import native_initialization
from cat_mjlab.collision import CollisionChecker,transform_primitives,sphere_box_separation,capsule_box_separation
from cat_ppo.furniture.room_navigation import pack_room_scenes
from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def subset_bank(manifest,resets):
    """In-memory view; no new bank or source files are written."""
    full=json.loads(manifest.read_text());indices=[i for i,r in enumerate(full['scenes']) if r.get('source',{}).get('hand_protection',{}).get('level')==0]
    assert len(indices)==8
    records=[full['scenes'][i] for i in indices];scenes=[json.loads((manifest.parent/r['path']/'scene.json').read_text()) for r in records]
    bank=SceneBank.__new__(SceneBank);bank.path=manifest;bank.manifest=full;bank.device=torch.device('cpu');bank.count=8
    sizes=[int(np.prod(r['shape'])) for r in records];bank.offsets=torch.tensor(np.cumsum([0]+sizes[:-1]));bank.fields={}
    from cat_mjlab.fields import sample_ragged_field
    bank.sample_kernel=sample_ragged_field
    for name,n in [('sdf',1),('gf',3),('bf',3)]:bank.fields[name]=torch.from_numpy(np.concatenate([np.load(manifest.parent/r['path']/(name+'.npy')).reshape(-1,n) for r in records]))
    for attr,key,dtype in [('shapes','shape',torch.long),('origins','origin',torch.float32),('dxs','dx',torch.float32),('starts','start',torch.float32),('goals','goal',torch.float32),('reset_xy_scale','reset_xy_scale',torch.float32),('reset_yaws','reset_yaw',torch.float32)]:setattr(bank,attr,torch.tensor([r[key] for r in records],dtype=dtype))
    bank.rooms={k:torch.from_numpy(v.copy()) for k,v in pack_room_scenes(scenes).items()};bank.contrast={k:torch.from_numpy(v.copy()) for k,v in pack_hand_contrast(scenes).items()}
    for name in ['is_cat','reset_is_cat','crossed_is_plane']:setattr(bank,name,torch.zeros(8,dtype=torch.bool))
    bank.episode_lengths=torch.tensor([r['episode_length'] for r in records]);bank.weights=torch.ones(8);bank.roles=None;bank.levels=torch.zeros(8,dtype=torch.long);bank.groups=None;bank.width_curriculum=None;bank.has_sdf_reward_overrides=False;bank.has_contrast=True
    bank.navigation_groups=torch.full((8,),2,dtype=torch.long);bank.hand_scene_kind=torch.tensor([1 if s['hand_protection']['kind']=='hand_table_aisle' else 2 for s in scenes]);bank.probabilities=lambda weights=None,stage=None:torch.ones(8)/8
    meta=json.loads(resets.read_text());assert meta['field_manifest_sha256']==digest(manifest);assert digest(resets.parent/meta['file'])==meta['sha256'];bank.reset_pool=torch.from_numpy(np.load(resets.parent/meta['file'])[indices].copy())
    return bank,indices,records,scenes

class GateTask(CATTask):
    reset_style='distribution'
    resetting=False
    def _rand(self,shape,low=0.,high=1.):
        value=super()._rand(shape,low,high)
        if self.resetting and self.reset_style!='distribution':
            middle=(low+high)/2
            if self.reset_style=='nominal':return torch.full_like(value,middle)
            if (low,high)==(-1.,1.) and len(shape)==2 and shape[-1]==2:return value*.25
            if (low,high)==(-np.pi/2,np.pi/2):return value*.1
            if (low,high)==(.5,1.5) and len(shape)==2:return middle+(value-middle)*.1
            if (low,high)==(-.5,.5):return value*.1
        return value
    def reset(self,env_ids=None,scene_ids=None):
        self.resetting=True
        try:return super().reset(env_ids,scene_ids)
        finally:self.resetting=False
    def _outcomes(self,done,truncation):
        world=transform_primitives(self.data.xpos,self.data.xmat,self.gate_compiled) if hasattr(self,'gate_compiled') else None
        self_margin=torch.full((self.num_envs,),torch.inf)
        if world is not None:
            boxes=torch.tensor([0,15,16,17,18]);hands=torch.tensor([26,34]);arms=torch.tensor([22,23,24,25,30,31,32,33]);c=self.gate_compiled
            args=(world['centers'][:,None,boxes],world['rotations'][:,None,boxes],c['half_sizes'][None,None,boxes])
            h=sphere_box_separation(world['centers'][:,hands,None],c['radii'][None,hands,None],*args).flatten(1).amin(-1)
            a=capsule_box_separation(world['endpoints'][:,arms,None,0],world['endpoints'][:,arms,None,1],c['radii'][None,arms,None],*args).flatten(1).amin(-1)
            self_margin=torch.minimum(h,a)
        result=super()._outcomes(done,truncation)
        c=self.contrast;hands=self.info['positions'][:,5:7];root=self.data.qpos[:,:2];t=c['route_tangent'];n=torch.stack((-t[:,1],t[:,0]),1);delta=hands[:,:,:2]-root[:,None]
        local=torch.stack(((delta*t[:,None]).sum(-1),(delta*n[:,None]).sum(-1),hands[:,:,2]),-1)
        d=(c['hand_regions_min']-local[:,None]).clamp_min(0)+(local[:,None]-c['hand_regions_max']).clamp_min(0)
        raised_error=torch.linalg.vector_norm(d[:,0],dim=-1).amax(-1)
        oldlo=self.bank.original_regions_min[self.scene_ids] if hasattr(self.bank,'original_regions_min') else c['hand_regions_min']
        oldhi=self.bank.original_regions_max[self.scene_ids] if hasattr(self.bank,'original_regions_max') else c['hand_regions_max']
        od=(oldlo-local[:,None]).clamp_min(0)+(local[:,None]-oldhi).clamp_min(0)
        original_raised_error=torch.linalg.vector_norm(od[:,0],dim=-1).amax(-1)
        error=torch.linalg.vector_norm(d,dim=-1).amax(-1);error=torch.where(c['region_valid'],error,torch.inf).amin(-1)
        self.gate_probe=dict(raised_error=raised_error,original_raised_error=original_raised_error,self_margin=self_margin,progress=self.navigation['progress_m'].clone(),error=error,hand_z=hands[:,:,2].amin(-1),field=self.info['sdf'][:,5:7].flatten(1).amin(-1),core=self.contrast['core_active'].clone(),hand_good=self.telemetry['hand_contrast_hand_good'].clone(),heading_good=self.telemetry['hand_contrast_heading_good'].clone(),zone_counts=self.zone_steps[:,0].clone(),root_height=self.data.qpos[:,2].clone(),joint_error=(self.data.qpos[:,22:]-self.info['motor_targets'][:,15:]).abs().amax(-1))
        return result


def arm_target(task,step,mode):
    q=task.nominal[15:].expand(task.num_envs,-1).clone();seconds=step*.02
    if mode in ('staged','paused'):a=min(seconds/.2,1);b=max(0,min((seconds-.2)/.4,1))
    else:a=1.;b=min(seconds/.4,1)
    for k,(side,sign) in enumerate([('left',1),('right',-1)]):
        base=k*7;q[:,base]+=-.76*b;q[:,base+1]+=-.30*sign*a*(1-b)-.1425*sign*b;q[:,base+3]+=-.76*b
    return q


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['baseline','staged','direct','paused'],default='staged');parser.add_argument('--reset',choices=['nominal','near','distribution'],default='nominal');parser.add_argument('--batches',type=int,default=1);parser.add_argument('--steps',type=int,default=450);parser.add_argument('--seed',type=int,default=71);parser.add_argument('--target-height',choices=['original','safety-minimum'],default='original');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    manifest=ROOT/'data/furniture/cat_hand_posture_v1_20260919/manifest.json';resets=manifest.parent.with_name(manifest.parent.name+'_resets')/'manifest.json';collision=manifest.parent.with_name(manifest.parent.name+'_collision')/'manifest.json';run=json.loads((ROOT/'outputs/cat_hand_posture_30720_20260919/run.json').read_text());assert digest(manifest)==run['contract']['bank_sha256'] and digest(resets)==run['contract']['resets_sha256']
    checkpoint=ROOT/'outputs/cat_width_curriculum_compiled_30720/resume.pt';assert json.loads((checkpoint.parent/'status.json').read_text())['phase']=='stopped_by_request'
    cfg,weights,_=native_initialization(checkpoint);model=ActorCritic(LearnerConfig(**cfg)).eval();model.load_state_dict(weights,strict=True)
    bank,indices,records,scenes=subset_bank(manifest,resets);sim=_CPUSimulation(8);checker=CollisionChecker(sim.model,collision,field_manifest=manifest,device='cpu');mapping=torch.tensor(indices)
    bank.original_regions_min=bank.contrast['hand_regions_min'][:,0].clone()
    bank.original_regions_max=bank.contrast['hand_regions_max'][:,0].clone()
    if args.target_height=='safety-minimum':
        from cat_ppo.furniture.grippers import hand_sphere
        radius=max(hand_sphere(side)['radius'] for side in ('left','right'))
        for index,scene in enumerate(scenes):
            height=max(b['center'][2]+b['half_size'][2] for b in scene['boxes'] if b.get('name','').endswith(('_top','_edge')))
            lower=height+radius+.02+.05
            extent=bank.contrast['hand_regions_max'][index,0,0,:,2]-bank.contrast['hand_regions_min'][index,0,0,:,2]
            bank.contrast['hand_regions_min'][index,0,0,:,2]=lower
            bank.contrast['hand_regions_max'][index,0,0,:,2]=lower+extent
    def check(ids,data):return checker(mapping[ids],data)
    config=copy.deepcopy(run['contract']['environment_config']);config['randomize_initial_episode_steps']=False
    if args.reset=='nominal':
        config['noise_config']['level']=0;config['dm_rand_config']['enable_pd']=False;config['dm_rand_config']['enable_rfi']=False;config['push_config']['enable']=False
    task=GateTask(sim,bank,config,collision=check,seed=args.seed);task.reset_style=args.reset;task.gate_compiled=checker.compiled;rows=[];started=time.monotonic()
    for batch in range(args.batches):
        task.reset(scene_ids=torch.arange(8));active=torch.ones(8,dtype=torch.bool);streak=torch.zeros(8,dtype=torch.long)
        stats=[dict(scene=r['scene_id'],batch=batch,reset_replaced=bool(task.episode['reset_replaced'][i]),initial_xy=task.data.qpos[i,:2].tolist(),acquired=False,acquired_time=None,min_self=1e9,min_field=1e9,min_error=1e9,max_hand_z=0.,max_joint_error=0.,height_trace=[],core=0,hand_core=0,heading_core=0,status='budget') for i,r in enumerate(records)]
        for step in range(args.steps):
            mean=model.logits(task.obs['state'],torch.zeros(8,dtype=torch.long)).chunk(2,-1)[0];action=mean.tanh()
            if args.mode!='baseline':action[:,15:]=((arm_target(task,step,args.mode)-task.nominal[15:])/.8).clamp(-1,1)
            if args.mode=='paused':
                action[:,12:15]=0
                if step<50:action[:,:12]=0
            assert torch.isfinite(action).all() and action.abs().max()<=1
            result=task.step(action);probe=task.gate_probe
            for i in active.nonzero().flatten().tolist():
                row=stats[i];row['min_self']=min(row['min_self'],float(probe['self_margin'][i]));row['min_field']=min(row['min_field'],float(probe['field'][i]));row['min_error']=min(row['min_error'],float(probe['error'][i]));row['max_hand_z']=max(row['max_hand_z'],float(probe['hand_z'][i]));row['max_joint_error']=max(row['max_joint_error'],float(probe['joint_error'][i]));row['root_height']=float(probe['root_height'][i]);row['progress']=float(probe['progress'][i]);row['core']+=int(probe['core'][i]);row['hand_core']+=int(probe['core'][i] and probe['hand_good'][i]>.5);row['heading_core']+=int(probe['core'][i] and probe['heading_good'][i]>.5)
                row['height_trace'].append(dict(t=(step+1)*.02,z=float(probe['root_height'][i]),hand_z=float(probe['hand_z'][i]),core=bool(probe['core'][i]),raised_error=float(probe['raised_error'][i]),original_raised_error=float(probe['original_raised_error'][i]),self_margin=float(probe['self_margin'][i])))
                streak[i]=streak[i]+1 if probe['error'][i]<=.05 and probe['self_margin'][i]>=0 else 0
                if streak[i]>=5 and not row['acquired']:row.update(acquired=True,acquired_time=(step+1)*.02,acquired_progress=float(probe['progress'][i]))
                if result['metrics']['resolved'][i] or result['done'][i]:
                    row['time']=(step+1)*.02;flags=[k[8:] for k,v in result['metrics'].items() if k.startswith('episode/') and bool(v[i])];row['flags']=flags;clean=bool(result['metrics']['successful'][i]);row['status']='clean_goal' if clean else 'failure';row['qualified']=bool(clean and row['core']>0 and row['hand_core']>=.9*row['core'] and row['heading_core']>=.9*row['core']);active[i]=False
            if step%50==0:print('batch',batch,'step',step,'active',int(active.sum()),'acquired',sum(s['acquired'] for s in stats),'elapsed',round(time.monotonic()-started,1),flush=True)
            if not active.any():break
        rows.extend(stats);print('BATCH',json.dumps([{k:v for k,v in r.items() if k!='height_trace'} for r in stats]),flush=True)
        report=dict(method='Real CPU MuJoCo physics, production CATTask/fields/collision checker; only actuator actions overridden after initialization; no root freezing, no joint teleporting, no optimizer',target_height=args.target_height,controller=args.mode,reset=args.reset,seed=args.seed,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),bank_sha256=digest(manifest),resets_sha256=digest(resets),runtime_config=config,episodes=rows,elapsed_seconds=time.monotonic()-started)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('SUMMARY',json.dumps(dict(episodes=len(rows),acquired=sum(r['acquired'] for r in rows),clean=sum(r['status']=='clean_goal' for r in rows),qualified=sum(r.get('qualified',False) for r in rows))),flush=True)
if __name__=='__main__':main()
