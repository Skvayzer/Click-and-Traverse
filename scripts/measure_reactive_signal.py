#!/usr/bin/env python3
"""Frozen-policy reward measurements only; no learner, bank or checkpoint writes.

CPU uses real MuJoCo through the established test bridge; CUDA uses CATSimulation.
Synthetic empty-floor background + exact moving sphere, attached through the
production analytic hook. Outputs are capped small JSON/JSONL. No raster output.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse,copy,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import numpy as np
import torch
from cat_mjlab.task_math import hand_clearance_pressure
OUT=ROOT/'docs/assets/reactive-signal-20260922'


def disk(stage):
    free=shutil.disk_usage(ROOT).free
    print(f'{stage}: disk free {free/2**30:.6f} GiB ({free} bytes)',flush=True)
    if free<512*2**20:raise RuntimeError('STOP: less than 512 MiB reserve')
    return free


def sweep(device):
    distances=torch.arange(70,-1,-1,device=device,dtype=torch.float64)*.005
    d=distances.clone().requires_grad_()
    pressure,_=hand_clearance_pressure(d[:,None].expand(-1,2),target=.09,anticipation=.20,near_weight=.8)
    cost=20*pressure;arm=8*(.08-d).clamp_min(0).square()
    slope=torch.autograd.grad(cost.sum(),d,retain_graph=True)[0]
    arm_slope=torch.autograd.grad(arm.sum(),d)[0]
    rows=[dict(distance_m=float(x),hand_cost_per_s=float(c),hand_cost_per_step=float(c*.02),
        arm_cost_per_s=float(a),arm_cost_per_step=float(a*.02),hand_slope_per_s_per_m=float(g),arm_slope_per_s_per_m=float(h))
        for x,c,a,g,h in zip(d,cost,arm,slope,arm_slope)]
    q=torch.tensor([[.06641,.06641]],dtype=torch.float64,device=device)
    hp=20*hand_clearance_pressure(q,target=.09,anticipation=.20,near_weight=.8)[0]
    ap=8*(.08-.06641)**2
    from cat_mjlab.task_math import sdf_reward
    native=-sdf_reward(d[:,None,None].expand(-1,2,1))
    for row,v in zip(rows,native):
        row['native_handsdf_cost_per_s_at_weight_1']=float(v)
        row['combined_hand_distance_cost_per_s']=row['hand_cost_per_s']+float(v)
    result=dict(scope='both hands at distance, both elbows at matched clearance; one-side-only costs/slopes are half',
        negative_band_exactly_zero=bool((cost[d>=.2]==0).all()),
        cost_monotone=bool((cost[1:]>=cost[:-1]).all()),rows=rows,
        ratio_at_contact=20/(8*.08**2),at_66_41_mm=dict(hand_cost_per_s=float(hp),arm_cost_per_s=ap,ratio=float(hp)/ap))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    return result


def build_bank(n,device):
    from test_mjlab_task import _tiny_bank
    bank=_tiny_bank(1)
    def move(x):
        if isinstance(x,torch.Tensor):return x.to(device)
        if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
        return x
    for k,v in vars(bank).items():setattr(bank,k,move(v))
    bank.device=torch.device(device)
    bank.probabilities=lambda weights=None,stage=None:torch.ones(1,device=device)
    bank.episode_lengths[:]=100000;bank.crossed_is_plane[:]=False
    bank.origins[:]=torch.tensor([-20.,-20.,-2.],device=device)
    bank.shapes[:]=torch.tensor([200,200,100],device=device)
    bank.goals[:]=torch.tensor([100.,0.,.8],device=device)
    def sample(name,p,ids):
        return torch.full((*p.shape[:-1],1),10.,device=device) if name=='sdf' else torch.zeros_like(p)
    bank.sample=sample
    return bank


def engine(n,device):
    from cat_mjlab.analytic_objects import AnalyticObjects
    p=torch.full((n,1,3),100.,device=device)
    return AnalyticObjects(start=p,end=p.clone(),rotations=torch.eye(3,device=device).expand(n,1,3,3).clone(),
        sizes=torch.full((n,1,3),.06,device=device),kinds=torch.zeros((n,1),dtype=torch.long,device=device),
        valid=torch.ones((n,1),dtype=torch.bool,device=device),speed=torch.full((n,1),.1,device=device),hold=torch.full((n,1),100.,device=device))


def collision_checker(model,objects,device,*,trace=False):
    from cat_mjlab.collision import compile_proposal,transform_primitives,PROPOSAL,sphere_box_separation
    proposal=json.loads(PROPOSAL.read_text())
    c=compile_proposal(proposal,model,device)
    class Check:
        clock=None
        def __init__(self):self.events={}
        def __call__(self,scenes,data):
            b=len(scenes)
            # Reset calls may select subsets; all resets before the measurement
            # have distant objects. Terminal rows are censored and not recycled.
            if self.clock is None or len(scenes)!=len(objects.state['start']):
                return torch.zeros((b,6),dtype=torch.bool,device=device)
            centers=objects.centers(torch.arange(b,device=device),self.clock)[:,0]
            w=transform_primitives(data.xpos,data.xmat,c)
            hit=torch.zeros((b,len(c['radii'])),dtype=torch.bool,device=device)
            for kind,index in c['indices'].items():
                if not len(index):continue
                if kind=='box':
                    sep=sphere_box_separation(centers[:,None],.06,w['centers'][:,index],w['rotations'][:,index],c['half_sizes'][index])
                elif kind=='capsule':
                    ep=w['endpoints'][:,index];v=ep[:,:,1]-ep[:,:,0]
                    t=((centers[:,None]-ep[:,:,0])*v).sum(-1)/v.square().sum(-1).clamp_min(1e-12)
                    closest=ep[:,:,0]+t.clamp(0,1)[...,None]*v
                    sep=torch.linalg.vector_norm(centers[:,None]-closest,dim=-1)-c['radii'][index]-.06
                else:sep=torch.linalg.vector_norm(centers[:,None]-w['centers'][:,index],dim=-1)-c['radii'][index]-.06
                hit[:,index]=sep<=0
                if trace:
                    for row,col in (sep<=0).nonzero().tolist():
                        shape=int(index[col]);key=(row,shape)
                        if key not in self.events:
                            hand_index=[i for i,x in enumerate(proposal['shapes']) if x['id']=='left_hand'][0]
                            hand=w['centers'][row,hand_index]
                            self.events[key]=dict(env=row,time_s=float(self.clock[row]),
                                pair=[proposal['shapes'][shape]['body_name'],'analytic_sphere_0'],
                                proxy=proposal['shapes'][shape]['id'],region=proposal['shapes'][shape]['group'],
                                separation_m=float(sep[row,col]),object_center=centers[row].tolist(),
                                hand_center=hand.tolist(),left_hand_clearance_m=float(torch.linalg.vector_norm(hand-centers[row])-.06-c['radii'][hand_index]))
            return (hit[:,None]&c['region_mask'][None]).any(-1)
    return Check()


def task_class():
    from cat_mjlab.task import CATTask
    class MeasureTask(CATTask):
        def _rand(self,shape,low=0.,high=1.):return torch.full(shape,(low+high)/2,device=self.device)
        def reset(self,ids=None,scene_ids=None):
            # Once a measured trajectory ends it stays absorbing: don't collect
            # any autoreset observations/rewards as if they continued the episode.
            if getattr(self,'measuring',False):return self.obs
            return super().reset(ids,scene_ids)
        def _rewards(self,action,contacts):
            r,p=super()._rewards(action,contacts)
            self.ledger={k:v.clone() for k,v in p.items()}
            self.base_pre=sum(p.values())*self.dt-p['wholebody_hand_clearance']*self.dt
            self.all_pre=sum(p.values())*self.dt
            self.post=r.clone();self.floor_slope=self.telemetry['reward_floor_slope'].clone()
            # Controlled autograd partial: hold robot/action/other fields fixed.
            original=self.info['sdf'];saved_telemetry=self.telemetry
            with torch.enable_grad():
                varied=original.detach().clone().requires_grad_(True)
                self.info['sdf']=varied
                rr,pp=super()._rewards(action,contacts)
                self.post_derivative=torch.autograd.grad(rr.sum(),varied,retain_graph=True)[0][:,5,0].detach()
                self.pre_derivative=torch.autograd.grad((sum(pp.values())*self.dt).sum(),varied)[0][:,5,0].detach()
            self.info['sdf']=original;self.telemetry=saved_telemetry
            return r,p
    return MeasureTask


def tuck_target(model,nominal,scales):
    """Predeclared front-threat tuck: retract 12 cm and raise 10 cm; IK only.

    Bounded to the same upper action range, then production target slew applies.
    No action/weight optimization on measured returns.
    """
    import mujoco
    from scipy.optimize import least_squares
    from cat_mjlab.constants import DEFAULT_QPOS,HAND_SITES
    d=mujoco.MjData(model);q=np.asarray(DEFAULT_QPOS,dtype=float).copy();q[2]=.8
    d.qpos[:]=q;mujoco.mj_forward(model,d)
    sites=[model.site(x).id for x in HAND_SITES]
    goal=d.site_xpos[sites].copy();goal[:,0]-=.12;goal[:,2]+=.10
    start=q[19:].copy();scale=scales.cpu().numpy()
    lo=np.maximum(start-scale,model.jnt_range[13:,0]);hi=np.minimum(start+scale,model.jnt_range[13:,1])
    def fun(x):
        x=np.r_[start[:3],x]
        d.qpos[:]=q;d.qpos[19:]=x;mujoco.mj_forward(model,d)
        return np.r_[((d.site_xpos[sites]-goal)*10).ravel(),.1*(x-start)]
    result=least_squares(fun,start[3:],bounds=(lo[3:]+1e-7,hi[3:]-1e-7),max_nfev=120)
    fun(result.x)
    solution=np.r_[start[:3],result.x]
    return solution,dict(requested_hands=goal.tolist(),achieved_hands=d.site_xpos[sites].tolist(),nominal_hands=(goal-np.array([-.12,0,.10])).tolist(),upper_target=solution.tolist())


def quantiles(values):
    a=np.asarray(values,dtype=float)
    if not a.size:return dict(n=0)
    return dict(n=int(a.size),min=float(a.min()),p10=float(np.quantile(a,.1)),p50=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()),std=float(a.std()))


def attach_advantages(rows,config):
    from cat_mjlab.learning import compute_gae
    for key in sorted({(r['seed'],r['condition']) for r in rows}):
        trace=[r for r in rows if (r['seed'],r['condition'])==key]
        for begin in range(0,len(trace),32):
            block=trace[begin:begin+32]
            tensor=lambda values:torch.tensor([values],dtype=torch.float64)
            target,adv=compute_gae(tensor([0.]*len(block)),tensor([float(r['done']) for r in block]),
                tensor([r['reward']*config['reward_scaling'] for r in block]),tensor([r['critic_value'] for r in block]),
                torch.tensor([block[-1]['next_critic_value']],dtype=torch.float64),
                lambda_=config['gae_lambda'],discount=config['discounting'])
            for r,v in zip(block,adv[0]):r['ppo_advantage']=float(v)


def summarize(rows,seeds,dt):
    approach=[r for r in rows if r['condition']=='approach_A']
    floor={}
    for label,limit in [('overall',np.inf),('d_lt_020',.20),('d_lt_009',.09),('d_lt_005',.05)]:
        sel=[r for r in approach if r['hand_distance'][0]<limit]
        floor[label]=dict(n=len(sel),base_clipped_fraction=float(np.mean([r['base_pre']<0 for r in sel])) if sel else None,
            hypothetical_all_inside_clipped_fraction=float(np.mean([r['all_pre']<0 for r in sel])) if sel else None,
            pre_partial_slope_per_step_per_m=quantiles([r['pre_derivative'] for r in sel]),
            post_partial_slope_per_step_per_m=quantiles([r['post_derivative'] for r in sel]))
    clipped=[r for r in rows if r['base_pre']<0 and r['hand_distance'][0]<.2]
    floor['clipped_approach_or_fixed_steps']=dict(n=len(clipped),pre=quantiles([r['pre_derivative'] for r in clipped]),post=quantiles([r['post_derivative'] for r in clipped]),zero_post_slope_count=sum(abs(r['post_derivative'])<1e-9 for r in clipped))
    buckets={}
    for lo,hi in [(0,.05),(.05,.09),(.09,.20),(.20,.35)]:
        sel=[r for r in approach if lo<=r['hand_distance'][0]<hi]
        buckets[f'{lo:.2f}_{hi:.2f}']=dict(n=len(sel),pre=quantiles([r['pre_derivative'] for r in sel]),post=quantiles([r['post_derivative'] for r in sel]))
    contrast={}
    for label in ('near','negative'):
        sel=[r for r in rows if r['condition']==label]
        returns=[sum(r['reward'] for r in sel if r['seed']==seed) for seed in range(seeds)]
        # Between-seed contrast at a shared control time, not temporal drift.
        common=min((max([r['step'] for r in sel if r['seed']==seed],default=-1) for seed in range(seeds)),default=-1)
        endpoint=[r for r in sel if r['step']==common]
        contrast[label]=dict(inside_anticipation_fraction=float(np.mean([r['hand_distance'][0]<.2 for r in sel])),clearance_pooled=quantiles([r['hand_distance'][0] for r in sel]),
            reward_pooled=quantiles([r['reward'] for r in sel]),common_step=common,
            clearance_at_common_step=quantiles([r['hand_distance'][0] for r in endpoint]),
            reward_at_common_step=quantiles([r['reward'] for r in endpoint]),
            hand_cost_at_common_step=quantiles([-r['contributions']['wholebody_hand_clearance'] for r in endpoint]),
            raw_gae_at_common_step=quantiles([r['raw_gae'] for r in endpoint]),
            raw_gae_first_step=quantiles([r['raw_gae'] for r in sel if r['step']==0]),
            ppo_advantage_at_common_step=quantiles([r['ppo_advantage'] for r in endpoint]),
            ppo_advantage_first_step=quantiles([r['ppo_advantage'] for r in sel if r['step']==0]),returns=quantiles(returns),
            initial_pre_tanh_std=quantiles([x for r in sel if r['step']==0 for x in r['policy_sigma_upper']]))
    pairs=[];parts={};common_parts={};outcomes={}
    for seed in range(seeds):
        a=[r for r in rows if r['condition']=='approach_A' and r['seed']==seed]
        b=[r for r in rows if r['condition']=='approach_B' and r['seed']==seed]
        common=min(len(a),len(b));delta=sum(r['reward'] for r in b)-sum(r['reward'] for r in a)
        cd=sum(r['reward'] for r in b[:common])-sum(r['reward'] for r in a[:common])
        pairs.append(dict(seed=seed,delta_return=delta,delta_common_prefix_return=cd,A_steps=len(a),B_steps=len(b),
            A_return=sum(r['reward'] for r in a),B_return=sum(r['reward'] for r in b),
            delta_common_hand_clearance=float(np.mean([r['hand_distance'][0] for r in b[:common]])-np.mean([r['hand_distance'][0] for r in a[:common]]))))
        for k in a[0]['contributions']:
            parts.setdefault(k,[]).append(sum(r['contributions'][k] for r in b)-sum(r['contributions'][k] for r in a))
            common_parts.setdefault(k,[]).append(sum(r['contributions'][k] for r in b[:common])-sum(r['contributions'][k] for r in a[:common]))
    rng=np.random.default_rng(20260922)
    def ci(values):
        x=np.asarray(values);means=x[rng.integers(0,len(x),(4000,len(x)))].mean(1)
        return dict(mean=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist())
    for condition in ('approach_A','approach_B','near','negative'):
        sel=[r for r in rows if r['condition']==condition]
        outcomes[condition]={key:len({r['seed'] for r in sel if r[key]}) for key in ('fall','contact','done')}
    return dict(floor=floor,slope_buckets=buckets,contrast=contrast,counterfactual=dict(pairs=pairs,
        delta_return=ci([x['delta_return'] for x in pairs]),delta_common_prefix_return=ci([x['delta_common_prefix_return'] for x in pairs]),
        terms={k:ci(v) for k,v in parts.items()},common_prefix_terms={k:ci(v) for k,v in common_parts.items()}),outcomes=outcomes)


def rollout(args):
    from test_mjlab_task import _CPUSimulation
    from cat_mjlab.learning import ActorCritic,LearnerConfig,gaussian_parameters
    from cat_mjlab.config import wholebody_config
    snap=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    policy=ActorCritic(LearnerConfig(**snap['learner']['config'])).to(args.device).eval()
    policy.load_state_dict(snap['learner']['model'])
    base=copy.deepcopy(snap['contract']['environment_config'])
    cfg=wholebody_config(base,stabilization=True,hand_protection=True,hand_contrast=False,
        hand_clearance_weight=-20,arm_clearance_weight=-8,tracking_root_field_weight=1,
        hand_clearance_target=.09,hand_clearance_anticipation=.20,hand_clearance_near_weight=.8,hand_reward_soft_floor=0)
    cfg.update(randomize_initial_episode_steps=False,hand_raised_reset_fraction=0,upper_gravity_compensation=True)
    cfg['noise_config']['level']=0.;cfg['push_config']['enable']=False
    cfg['dm_rand_config'].update(enable_pd=False,enable_rfi=False)
    n=args.seeds*4
    if args.device=='cpu':sim=_CPUSimulation(n)
    else:
        from cat_mjlab.sim import CATSimulation
        sim=CATSimulation(n,device=args.device)
    objects=engine(n,args.device);checker=collision_checker(sim.model,objects,args.device)
    task=task_class()(sim,build_bank(n,args.device),cfg,seed=17,collision=checker,analytic_objects=objects)
    checker.clock=task.data.time
    # Empty standing command/guidance yields the native zero-command gait path.
    task.measuring=True
    task.info['phase'][:]=task.info['phase'][0].clone()
    task.info['gait'][:]=task.info['gait'][0].clone()
    generators=[torch.Generator(device=args.device).manual_seed(19000+k) for k in range(args.seeds)]
    target,tuck=tuck_target(sim.model,task.nominal,task.upper_action_scales)
    target=torch.tensor(target,dtype=torch.float32,device=args.device)
    tucked=((target-task.nominal[12:])/task.upper_action_scales).clamp(-1,1)
    # No warm-up: common canonical reset, zero base/joint velocity (override the
    # deterministic reset midpoint, already zero). Anchor to initial hand.
    anchor=task.info['positions'][:,5].clone();radius=float(task.hand_radii[0])+.06
    groups=torch.arange(n,device=args.device)%4
    nominal_gap=torch.where(groups<2,.35,torch.where(groups==2,.05,.30))
    objects.state['start'][:,0]=anchor+torch.stack((nominal_gap+radius,torch.zeros_like(nominal_gap),torch.zeros_like(nominal_gap)),-1)
    objects.state['end'].copy_(objects.state['start'])
    objects.state['end'][groups<2,0,0]-=.32  # stops at nominal 3 cm, no chasing
    objects.state['speed'][:]=args.speed
    # Refresh production observations after enabling analytic geometry.
    def refresh():
        ids=task.all_ids;p=task._poses(ids)
        gf,bf,sdf,cmd=task._fields(p,task.data.qpos[:,:2],ids)
        task.info.update(gf=gf,bf=bf,sdf=sdf,gf_delay=gf.clone(),bf_delay=bf.clone(),sdf_delay=sdf.clone(),command=cmd,command_delay=cmd.clone())
        task._observe(ids,sim.contact_flags(task.contact_pairs)[:,:2])
    refresh();initial_dist=task.info['sdf'][:,5,0].cpu().tolist()
    torch.testing.assert_close(task.obs['state'][0::4],task.obs['state'][1::4],rtol=0,atol=0)
    torch.testing.assert_close(task.data.qpos[0::4],task.data.qpos[1::4],rtol=0,atol=0)
    alive=np.ones(n,dtype=bool);rows=[];started=time.perf_counter()
    names=('approach_A','approach_B','near','negative')
    with torch.no_grad():
        for step in range(args.steps):
            values=policy.value(task.obs['privileged_state'],0)
            mu,sigma=gaussian_parameters(policy.logits(task.obs['state'],0))
            # Shared Gaussian innovations within a seed (paired A/B); distinct
            # seeds supply the policy's actual state-dependent action noise.
            eps=torch.stack([torch.randn(29,generator=g,device=args.device) for g in generators]).repeat_interleave(4,0)
            action=(mu+sigma*eps).tanh()
            # Counterfactual starts with object entering nominal anticipation.
            if step*.02 >= .15/args.speed:action[groups==1,12:]=tucked
            result=task.step(action)
            next_values=policy.value(result['terminal_obs']['privileged_state'],0).cpu().tolist()
            current_values=values.cpu().tolist()
            pre=task.all_pre.cpu().tolist();base_pre=task.base_pre.cpu().tolist()
            distance=result['metrics']['acceptance/hand_clearance'].cpu().tolist()
            elbow=task.info['elbow_clearance'].cpu().tolist()
            terms={k:(v*.02).cpu().tolist() for k,v in task.ledger.items()}
            reward=result['reward'].cpu().tolist();post=task.post.cpu().tolist()
            deriv=task.post_derivative.cpu().tolist();pderiv=task.pre_derivative.cpu().tolist()
            done=result['done'].cpu().tolist();fall=result['metrics']['episode/fall'].cpu().tolist()
            contact=result['metrics']['collision_regions'].any(-1).cpu().tolist()
            sig=sigma[:,12:].cpu().tolist()
            for j in np.where(alive)[0]:
                contributions={k:v[j] for k,v in terms.items()}
                contributions['floor_lift']=post[j]-pre[j]
                contributions['collision_event']=reward[j]-post[j]
                assert abs(sum(contributions.values())-reward[j])<2e-5
                rows.append(dict(seed=int(j//4),condition=names[j%4],step=step,time_s=(step+1)*.02,
                    critic_value=current_values[j],next_critic_value=next_values[j],hand_distance=distance[j],elbow_clearance=elbow[j],all_pre=pre[j],base_pre=base_pre[j],post_floor=post[j],reward=reward[j],
                    pre_derivative=pderiv[j],post_derivative=deriv[j],contributions=contributions,
                    policy_sigma_upper=sig[j] if step==0 else [],done=done[j],fall=fall[j],contact=contact[j]))
                alive[j]=not done[j]
            if step%25==0:
                disk(f'rollout step {step}');print(f'active {int(alive.sum())}/{n}; elapsed {time.perf_counter()-started:.1f}s',flush=True)
            if not alive.any():break
    for j in range(n):
        trace=[r for r in rows if r['seed']==j//4 and r['condition']==names[j%4]]
        gae=0.
        for row in reversed(trace):
            discount=0. if row['done'] else policy.config.discounting
            residual=row['reward']*policy.config.reward_scaling+discount*row['next_critic_value']-row['critic_value']
            gae=residual+discount*policy.config.gae_lambda*gae
            row['raw_gae']=gae
    return rows,dict(device=args.device,seeds=args.seeds,steps=args.steps,dt=.02,speed=args.speed,
        checkpoint=str(args.checkpoint),policy_config=snap['learner']['config'],initial_hand_distances=initial_dist,
        tuck=tuck,configuration=cfg,seconds=time.perf_counter()-started,
        caveats=['Canonical empty-floor standing reset; no retention-bank or protected-core population inference.',
                 'Physics and full-body sphere-object terminal checks every 2 ms; no contact forces from virtual objects.',
                 'Arm observations use current production elbow points, not the unfinished capsule draft.',
                 'All observations are noise-free; exploration is checkpoint Gaussian action noise, uncapped beyond checkpoint setting.',
                 'Ended trajectories are absorbing for analysis; zero future reward to the shared finite horizon.',
                 'Local distance derivatives are controlled partial derivatives, not policy gradients.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--mode',choices=['sweep','all','summarize'],default='all')
    p.add_argument('--seeds',type=int,default=8);p.add_argument('--steps',type=int,default=250);p.add_argument('--speed',type=float,default=.1)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/cat_flat_balance_ppo_37632_20260920/resume.pt')
    p.add_argument('--output',type=Path,default=OUT);args=p.parse_args()
    if args.seeds<2 or args.steps<1 or args.speed<=0:raise ValueError('Invalid probe size')
    # Text budget: no unbounded per-step dump on a tight shared pool.
    if args.seeds*4*args.steps>100000:raise ValueError('Probe exceeds 100k row safety cap')
    torch.set_num_threads(2)
    before=disk('start');args.output.mkdir(parents=True,exist_ok=True)
    result=dict(sweep=sweep(args.device),disk_before_bytes=before)
    (args.output/'sweep.json').write_text(json.dumps(result['sweep'],indent=2)+'\n')
    disk('sweep complete')
    if args.mode in ('all','summarize'):
        if args.mode=='all':rows,meta=rollout(args)
        else:
            previous=json.loads((args.output/'report.json').read_text())
            meta=previous['metadata'];args.seeds=meta['seeds']
            rows=[json.loads(line) for line in (args.output/'steps.jsonl').read_text().splitlines()]
            if args.seeds>8:raise ValueError('Summarize mode requires all seeds in saved text')
        attach_advantages(rows,meta['policy_config'])
        result.update(metadata=meta,measurements=summarize(rows,args.seeds,.02))
        disk('before text output')
        # Detailed log is small for CPU; larger CUDA runs save aggregate reports
        # plus complete per-step logs for the first eight seed pairs.
        selected=[r for r in rows if r['seed']<8]
        data=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in selected)
        if len(data)>32*2**20:raise RuntimeError('Text artifact exceeds 32 MiB cap')
        (args.output/'steps.jsonl').write_text(data)
        result['per_step_logged_seeds']=min(args.seeds,8)
    result['disk_after_bytes']=disk('finish')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result.get('measurements',{}),indent=2),flush=True)

if __name__=='__main__':main()
