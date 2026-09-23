#!/usr/bin/env python3
"""Measurement/kinematic solver only. Never creates a bank or a learner."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse,json,sys,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'tests')]
import numpy as np
import torch
import measure_reactive_signal as signal
OUT=ROOT/'docs/assets/reactive-tuck-solve-20260922'


def attribute(args):
    checkers=[];original=signal.collision_checker
    def traced(*a,**kw):
        c=original(*a,trace=True,**kw);checkers.append(c);return c
    signal.collision_checker=traced
    try:rows,meta=signal.rollout(args)
    finally:signal.collision_checker=original
    report=dict(metadata=meta,contacts=[],sign_convention='norm(hand_center-object_center)-0.10344617-0.06; positive=separated, negative=penetration',seeds=[])
    for seed in range(args.seeds):
        a=[r for r in rows if r['seed']==seed and r['condition']=='approach_A']
        b=[r for r in rows if r['seed']==seed and r['condition']=='approach_B']
        for event in checkers[0].events.values():
            if event['env']==4*seed+1 and event['time_s']<=b[-1]['time_s']+.001:
                report['contacts'].append(dict(seed=seed,**event))
        index=int(np.argmin([r['hand_distance'][0] for r in b]));common=min(len(a),len(b))
        report['seeds'].append(dict(seed=seed,A_min_hand_m=min(r['hand_distance'][0] for r in a),B_min_hand_m=b[index]['hand_distance'][0],
            B_postintervention_min_hand_m=min(r['hand_distance'][0] for r in b if r['time_s']>=1.5),
            closest_B_time_s=b[index]['time_s'],delta_B_minus_A_at_B_closest_m=b[index]['hand_distance'][0]-a[index]['hand_distance'][0],
            delta_common_mean_clearance_m=float(np.mean([r['hand_distance'][0] for r in b[:common]])-np.mean([r['hand_distance'][0] for r in a[:common]])),
            A_return=sum(r['reward'] for r in a),B_return=sum(r['reward'] for r in b),B_terminal_hand_m=b[-1]['hand_distance'][0],
            A_at_B_terminal_hand_m=a[len(b)-1]['hand_distance'][0]))
    signal.disk('attribution finished')
    (args.output/'contact-attribution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='metadata'},indent=2),flush=True)


def solve(args):
    from scipy.optimize import minimize
    from reactive_tuck_geometry import Geometry,DIRECTIONS,BUCKETS
    geo=Geometry();rng=np.random.default_rng(61);results=[]
    if args.resume_solve and (args.output/'kinematic-solutions.json').exists():
        previous=json.loads((args.output/'kinematic-solutions.json').read_text())['rows']
        for row in previous:
            if row['status']=='FEASIBLE':
                certificate=geo.certificate(np.array(row['upper_target']),np.array(row['start']),np.array(row['end']),row['arrival_s'])
                if certificate['passed']:
                    row['certificate']=certificate;results.append(row)
            elif row['status']=='INFEASIBLE':results.append(row)
    if geo.baseline_overlaps:raise RuntimeError(f'Baseline proxy overlaps: {geo.baseline_overlaps}')
    started=time.perf_counter()
    for direction in DIRECTIONS:
        if args.directions and direction not in args.directions.split(','):continue
        for bucket,(low,high) in BUCKETS.items():
            gap=(low+high)/2
            for speed in (.05,.5):
                if any(r['direction']==direction and r['bucket']==bucket and r['speed_m_s']==speed for r in results):continue
                signal.disk(f'solve {direction}/{bucket}/{speed}')
                start,end,duration=geo.trajectory(direction,gap,speed)
                geo.set(geo.nominal)
                # Any point of the authored path through an immutable lower-body
                # proxy proves no upper-body pose can repair this scene.
                fixed=np.stack([geo.object_distances(start+t*(end-start))[geo.fixed] for t in np.linspace(0,1,201)])
                initial=geo.object_distances(start)
                row=dict(direction=direction,bucket=bucket,gap_m=gap,bucket_range_m=[low,high],speed_m_s=speed,
                    start=start.tolist(),end=end.tolist(),arrival_s=duration,target_hand='left',
                    initial_fullbody_clearance_m=float(initial.min()),immutable_body_path_min_m=float(fixed.min()))
                if initial.min()<0 or fixed.min()<.001 or min(start[2],end[2])<.061:
                    row.update(status='INFEASIBLE',reason='initial object/robot overlap' if initial.min()<0 else 'authored path intersects fixed lower-body geometry' if fixed.min()<.001 else 'object intersects floor',best_clearance_m=None)
                    results.append(row);print(json.dumps(row),flush=True);continue
                lo=np.maximum(geo.lo,geo.nominal-2*duration);hi=np.minimum(geo.hi,geo.nominal+2*duration)
                previous=next((r for r in results if r['direction']==direction and r['bucket']==bucket and r['status']=='FEASIBLE' and r['speed_m_s']==.05),None)
                if previous is not None and np.allclose(lo,geo.lo) and np.allclose(hi,geo.hi):
                    row.update({k:copy.deepcopy(v) for k,v in previous.items() if k not in row})
                    row['certificate']=geo.certificate(np.array(row['upper_target']),start,end,duration)
                    row['allocated_trajectory_time_s']=duration
                    row['reused_identical_joint_bounds']=True
                    results.append(row);print(json.dumps(row),flush=True);continue
                cache={};knots=[.0001,.02,.1,.25,.5,.75,1.]
                def evaluate(x):
                    key=x.tobytes()
                    if key in cache:return cache[key]
                    constraints=[]
                    for t in sorted(knots):
                        geo.set(geo.nominal+t*(x-geo.nominal))
                        constraints.extend(geo.object_distances(start+t*(end-start))-.002)
                        constraints.extend(geo.self_distances()-geo.self_margins)
                        constraints.extend(geo.floor_distances()-.002)
                    clearance=float(np.linalg.norm(geo.hand()-end)-geo.hand_radius-.06)
                    value=(-clearance+1e-5*np.square(x-geo.nominal).sum(),np.array(constraints),clearance)
                    cache.clear();cache[key]=value;return value
                candidates=[];attempts=[]
                nominal_cert=geo.certificate(geo.nominal,start,end,duration)
                if nominal_cert['passed']:candidates.append((gap,geo.nominal.copy(),nominal_cert))
                # Multistart bounded numerical search: best found, NOT a claim of
                # global optimality in this nonconvex articulated workspace.
                for attempt in range(2):
                    # First search the smaller left-arm subspace; then allow all
                    # upper joints. Both are candidate searches, no global claim.
                    dims=np.arange(3,10) if attempt==0 else np.arange(17)
                    reference=geo.nominal.copy()
                    if candidates:reference=max(candidates,key=lambda c:c[0])[1].copy()
                    def expand(y):
                        x=reference.copy();x[dims]=y;return x
                    x0=reference[dims]
                    for refinement in range(2):
                        cache.clear()
                        opt=minimize(lambda y:evaluate(expand(y))[0],x0,method='SLSQP',bounds=list(zip(lo[dims],hi[dims])),
                            constraints=[dict(type='ineq',fun=lambda y:evaluate(expand(y))[1])],options=dict(maxiter=60,ftol=2e-7))
                        x=expand(opt.x);value=evaluate(x)
                        attempts.append(dict(subspace=len(dims),refinement=refinement,success=bool(opt.success),message=str(opt.message),clearance=value[2],constraint_min=float(value[1].min())))
                        if value[1].min() < -1e-6:break
                        cert=geo.certificate(x,start,end,duration)
                        attempts[-1]['certificate']=cert
                        if cert['passed']:
                            candidates.append((value[2],x.copy(),cert));break
                        knots.extend(cert['failed_fractions']);x0=opt.x
                row['attempts']=attempts
                if candidates:
                    clearance,x,cert=max(candidates,key=lambda c:c[0]);geo.set(x)
                    row.update(status='FEASIBLE',best_clearance_m=clearance,upper_target=x.tolist(),certificate=cert,
                        time_to_reach_s=float(np.max(np.abs(x-geo.nominal))/2),allocated_trajectory_time_s=duration,
                        endpoint_arm_clearance_m=float(geo.object_distances(end)[geo.arm].min()),
                        endpoint_self_clearance_m=float(geo.self_distances().min()),endpoint_floor_clearance_m=float(geo.floor_distances().min()))
                else:
                    row.update(status='UNRESOLVED',reason='two numerical starts found no continuously certified straight-joint path; not a proof of infeasibility',best_clearance_m=None)
                results.append(row);print(json.dumps(row),flush=True)
                (args.output/'kinematic-solutions.json').write_text(json.dumps(dict(rows=results,complete=False),indent=2)+'\n')
    report=dict(rows=results,complete=True,seconds=time.perf_counter()-started,nominal_upper=geo.nominal.tolist(),
        base_qpos=geo.q.tolist(),joint_lower=geo.lo.tolist(),joint_upper=geo.hi.tolist(),self_pair_count=len(geo.self_pairs),
        excluded_adjacent_pairs=[(geo.shapes[i]['id'],geo.shapes[j]['id']) for i,j in geo.excluded],
        limitations=['Bucket midpoints only, not all distances in a bucket.',
        'Speed endpoints .05 and .5 m/s; not a certificate for every intermediate speed.',
        'Best of two local constrained optimizations; no global maximum proof.',
        'Continuous certificates cover kinematic linear target paths with fixed base, not PD tracking or policy base response.',
        'Self-contact uses conservative approved proxies, excluding graph-neighbors <=2 edges and fixed lower-body pairs.'])
    (args.output/'kinematic-solutions.json').write_text(json.dumps(report,indent=2)+'\n')
    signal.disk('kinematic solve complete')


def paired(args):
    from test_mjlab_task import _CPUSimulation
    from cat_mjlab.learning import ActorCritic,LearnerConfig,gaussian_parameters
    from cat_mjlab.config import wholebody_config
    from cat_mjlab.task import CATTask
    from reactive_tuck_geometry import Geometry
    source=json.loads((args.output/'kinematic-solutions.json').read_text())
    rows=[r for r in source['rows'] if r['speed_m_s']==.5 and r['bucket']==args.bucket]
    if args.directions:rows=[r for r in rows if r['direction'] in args.directions.split(',')]
    geo=Geometry()
    snap=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    policy=ActorCritic(LearnerConfig(**snap['learner']['config'])).to(args.device).eval();policy.load_state_dict(snap['learner']['model'])
    config=wholebody_config(copy.deepcopy(snap['contract']['environment_config']),stabilization=True,hand_protection=True,hand_contrast=False,
        hand_clearance_weight=-20,arm_clearance_weight=-8,tracking_root_field_weight=1,
        hand_clearance_target=.09,hand_clearance_anticipation=.20,hand_clearance_near_weight=.8,hand_reward_soft_floor=0)
    config.update(randomize_initial_episode_steps=False,hand_raised_reset_fraction=0,upper_gravity_compensation=True)
    config['noise_config']['level']=0;config['push_config']['enable']=False;config['dm_rand_config'].update(enable_pd=False,enable_rfi=False)
    class Task(signal.task_class()):
        def _rewards(self,action,contacts):
            reward,terms=CATTask._rewards(self,action,contacts)
            self.ledger={k:v.clone()*.02 for k,v in terms.items()}
            self.ledger['floor_lift']=reward-sum(self.ledger.values())
            self.before_collision=reward.clone()
            return reward,terms
    destination=args.output/f'paired-{args.bucket}-{args.control_mode}{args.run_tag}.json'
    results=[]
    if args.resume_paired and destination.exists():results=json.loads(destination.read_text())['results']
    rng=np.random.default_rng(20260922)
    def ci(x):
        x=np.array(x);sample=x[rng.integers(0,len(x),(4000,len(x)))].mean(1)
        return dict(mean=float(x.mean()),ci95=np.quantile(sample,[.025,.975]).tolist())
    for scene in rows:
        if scene['status']!='FEASIBLE':
            results.append(dict(direction=scene['direction'],bucket=args.bucket,status='NOT_TESTED',reason=scene.get('reason','no certificate')));continue
        start=np.array(scene['start']);end=np.array(scene['end']);arrival=float(np.linalg.norm(end-start)/args.speed)
        # Optional shorter tuck search: every candidate is independently
        # certified against the same authored object trajectory before physics.
        for fraction in [float(v) for v in args.fractions.split(',')]:
            if any(r['direction']==scene['direction'] and r.get('fraction')==fraction for r in results):continue
            target=geo.nominal+fraction*(np.array(scene['upper_target'])-geo.nominal)
            certificate=geo.certificate(target,start,end,arrival)
            if not certificate['passed']:
                results.append(dict(direction=scene['direction'],bucket=args.bucket,fraction=fraction,status='NOT_TESTED',reason='fractional trajectory not certified',certificate=certificate));continue
            signal.disk(f'paired {scene["direction"]} {args.bucket} fraction={fraction}')
            n=args.seeds*2
            if args.device=='cpu':sim=_CPUSimulation(n)
            else:
                from cat_mjlab.sim import CATSimulation
                sim=CATSimulation(n,device=args.device)
            objects=signal.engine(n,args.device);checker=signal.collision_checker(sim.model,objects,args.device,trace=True)
            task=Task(sim,signal.build_bank(n,args.device),copy.deepcopy(config),seed=17,collision=checker,analytic_objects=objects)
            checker.clock=task.data.time;task.measuring=True
            task.info['phase'][:]=task.info['phase'][0].clone();task.info['gait'][:]=task.info['gait'][0].clone()
            objects.state['start'][:,0]=torch.tensor(start,device=args.device);objects.state['end'][:,0]=torch.tensor(end,device=args.device)
            objects.state['speed'][:]=args.speed
            ids=task.all_ids;pos=task._poses(ids);gf,bf,sdf,cmd=task._fields(pos,task.data.qpos[:,:2],ids)
            task.info.update(gf=gf,bf=bf,sdf=sdf,gf_delay=gf.clone(),bf_delay=bf.clone(),sdf_delay=sdf.clone(),command=cmd,command_delay=cmd.clone())
            task._observe(ids,sim.contact_flags(task.contact_pairs)[:,:2])
            torch.testing.assert_close(task.obs['state'][::2],task.obs['state'][1::2],atol=0,rtol=0)
            generators=[torch.Generator(device='cpu').manual_seed(args.seed_offset+k) for k in range(args.seeds)]
            target_tensor=torch.tensor(target,dtype=torch.float32,device=args.device)
            control=torch.ones(17,dtype=torch.bool,device=args.device) if args.control_mode=='all' else (target_tensor-task.nominal[12:]).abs()>1e-5
            alive=np.ones(n,bool);returns=np.zeros(n);fall=np.zeros(n,bool);contact=np.zeros(n,bool);minimum=np.full(n,np.inf)
            steps=np.zeros(n,int);ledgers={};trajectories=[[] for _ in range(n)];root_delta=np.zeros((n,3));first_touch=[]
            q0=task.data.qpos[:,:3].clone();t0=time.perf_counter();self_hit=np.zeros(n,bool);self_events=[];self_seen=set()
            with torch.no_grad():
                for step in range(args.steps):
                    mu,sigma=gaussian_parameters(policy.logits(task.obs['state'],0))
                    eps=torch.stack([torch.randn(29,generator=g,device='cpu') for g in generators]).to(args.device).repeat_interleave(2,0)
                    action=(mu+sigma*eps).tanh()
                    progress=min((step+1)*.02/arrival,1.)
                    requested=task.nominal[12:]+progress*(target_tensor-task.nominal[12:])
                    override=((requested-task.nominal[12:])/task.upper_action_scales).clamp(-1,1)
                    action[1::2,12:]=torch.where(control,override,action[1::2,12:])
                    transition=task.step(action);metrics=transition['metrics']
                    reward=transition['reward'].cpu().numpy();done=transition['done'].cpu().numpy()
                    hit=metrics['collision_regions'].any(-1).cpu().numpy();f=metrics['episode/fall'].cpu().numpy()
                    hand=metrics['acceptance/hand_clearance'][:,0].cpu().numpy()
                    terms={k:v.cpu().numpy() for k,v in task.ledger.items()}
                    terms['collision_event']=(transition['reward']-task.before_collision).cpu().numpy()
                    if args.audit_self:
                        import mujoco
                        qs=task.data.qpos.cpu().numpy()
                        for j in np.where(alive)[0]:
                            geo.data.qpos[:]=qs[j];mujoco.mj_kinematics(geo.model,geo.data)
                            gaps=geo.self_distances()
                            for k in np.where(gaps<=0)[0]:
                                self_hit[j]=True
                                if (j,int(k)) not in self_seen:
                                    self_seen.add((j,int(k)));i1,i2=geo.self_pairs[k]
                                    self_events.append(dict(seed=int(j//2),branch='B' if j%2 else 'A',time_s=(step+1)*.02,pair=[geo.shapes[i1]['id'],geo.shapes[i2]['id']],separation_m=float(gaps[k])))
                    returns+=np.where(alive,reward,0);steps+=alive;fall|=alive&f;contact|=alive&hit;minimum=np.minimum(minimum,np.where(alive,hand,np.inf))
                    displacement=(task.data.qpos[:,:3]-q0).cpu().numpy();root_delta[alive]=displacement[alive]
                    for key,v in terms.items():ledgers[key]=ledgers.get(key,np.zeros(n))+np.where(alive,v,0)
                    for j in np.where(alive)[0]:
                        trajectories[j].append(dict(reward=float(reward[j]),hand=float(hand[j]),terms={k:float(v[j]) for k,v in terms.items()}))
                    alive&=~done
                    if step%50==0:print(f'{scene["direction"]} fraction {fraction}: step {step}, live={alive.sum()}/{n}, {time.perf_counter()-t0:.1f}s',flush=True)
                    if not alive.any():break
            pairs=[];common_terms={}
            for seed in range(args.seeds):
                a,b=trajectories[2*seed:2*seed+2];length=min(len(a),len(b))
                pairs.append(dict(seed=seed,A_return=returns[2*seed],B_return=returns[2*seed+1],delta_return=returns[2*seed+1]-returns[2*seed],
                    A_steps=int(steps[2*seed]),B_steps=int(steps[2*seed+1]),A_contact=bool(contact[2*seed]),B_contact=bool(contact[2*seed+1]),
                    A_fall=bool(fall[2*seed]),B_fall=bool(fall[2*seed+1]),A_min_hand_m=minimum[2*seed],B_min_hand_m=minimum[2*seed+1],
                    delta_common_return=sum(x['reward'] for x in b[:length])-sum(x['reward'] for x in a[:length])))
                for key in ledgers:
                    common_terms.setdefault(key,[]).append(sum(x['terms'][key] for x in b[:length])-sum(x['terms'][key] for x in a[:length]))
            delta=ci(returns[1::2]-returns[::2]);contacts_B=int(contact[1::2].sum());contacts_A=int(contact[::2].sum())
            events=[dict(seed=e['env']//2,branch='B' if e['env']%2 else 'A',**e) for e in checker.events.values() if e['time_s']<=steps[e['env']]*.02+.001]
            added_self=[(j//2,k) for j,k in self_seen if j%2 and (j-1,k) not in self_seen]
            result=dict(direction=scene['direction'],bucket=args.bucket,gap_m=scene['gap_m'],speed_m_s=args.speed,fraction=fraction,
                seeds=args.seeds,device=args.device,steps=args.steps,allocated_motion_s=arrival,kinematic_certificate=certificate,
                delta_return=delta,delta_common_return=ci([x['delta_common_return'] for x in pairs]),
                terms={k:ci(v[1::2]-v[::2]) for k,v in ledgers.items()},common_terms={k:ci(v) for k,v in common_terms.items()},
                contacts_A=contacts_A,contacts_B=contacts_B,falls_A=int(fall[::2].sum()),falls_B=int(fall[1::2].sum()),pairs=pairs,contact_events=events,
                root_delta_m=root_delta.tolist(),seconds=time.perf_counter()-t0,control_mode=args.control_mode,
                self_proxy_audit=args.audit_self,self_distance_backend='analytic-primitives-v3-all-steps',seed_offset=args.seed_offset,added_self_proxy_pairs=[dict(seed=int(j),pair=[geo.shapes[i]['id'] for i in geo.self_pairs[k]]) for j,k in added_self],self_proxy_contacts_A=int(self_hit[::2].sum()),self_proxy_contacts_B=int(self_hit[1::2].sum()),self_proxy_events=self_events,
                status='PASS' if delta['ci95'][0]>0 and contacts_B<=contacts_A and not added_self and int(fall[1::2].sum())<=int(fall[::2].sum()) else 'FAIL',
                absolute_contact_free=bool(contacts_B==0 and not self_hit[1::2].any()),
                point_estimate_rule_pass=bool(delta['mean']>0 and contacts_B<=contacts_A),
                upper_target=target.tolist())
            results.append(result);print(json.dumps({k:v for k,v in result.items() if k in ('direction','fraction','status','delta_return','delta_common_return','contacts_A','contacts_B','falls_A','falls_B')}),flush=True)
            destination.write_text(json.dumps(dict(results=results,complete=False),indent=2)+'\n')
            del task,sim,objects,checker
    destination.write_text(json.dumps(dict(results=results,complete=True,
        note='Fractions searched on the same eight seeds; selected successes require independent holdout confirmation. No learning performed.'),indent=2)+'\n')


def residual(args):
    from cat_mjlab.task_math import sdf_reward,hand_clearance_pressure
    distances=[.20,.25,.30,.35,.09,.05,.03,0.]
    rows=[]
    for d in distances:
        x=torch.full((1,2),d,dtype=torch.float64,device=args.device)
        native=float(-sdf_reward(x[:,:,None])[0])
        explicit=float(20*hand_clearance_pressure(x,target=.09,anticipation=.20,near_weight=.8)[0][0])
        rows.append(dict(distance_m=d,native_cost_per_s_both=native,explicit_cost_per_s_both=explicit,
            episodes=[dict(seconds=t,native_both=native*t,native_single=native*t/2,explicit_both=explicit*t,explicit_single=explicit*t/2) for t in (5,10,20,80)]))
    result=dict(rows=rows,derived=True,assumption='constant matched surface distance, no term changes/floor clipping; these are upper bounds on recoverable native reward when the base floor clips',
        horizons='5 s probe; 10 s flat v2 (500 steps); 20 s 1000-step scenes; 80 s 4000-step scenes',
        negative_vs_5cm_explicit_ratio=rows[0]['native_cost_per_s_both']/rows[5]['explicit_cost_per_s_both'])
    (args.output/'native-residual.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['attribute','solve','paired','residual'],default='attribute')
    p.add_argument('--device',default='cpu');p.add_argument('--seeds',type=int,default=8);p.add_argument('--steps',type=int,default=250)
    p.add_argument('--resume-solve',action='store_true');p.add_argument('--bucket',choices=['danger','anticipation','negative'],default='danger')
    p.add_argument('--seed-offset',type=int,default=19000);p.add_argument('--run-tag',default='')
    p.add_argument('--control-mode',choices=['all','changed'],default='changed');p.add_argument('--audit-self',action='store_true');p.add_argument('--resume-paired',action='store_true')
    p.add_argument('--fractions',default='1');p.add_argument('--directions',default='');p.add_argument('--speed',type=float,default=.1);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/cat_flat_balance_ppo_37632_20260920/resume.pt')
    args=p.parse_args();torch.set_num_threads(2);signal.disk('start');args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='attribute':attribute(args)
    elif args.mode=='solve':solve(args)
    elif args.mode=='residual':residual(args)
    else:paired(args)
    signal.disk('finish')

if __name__=='__main__':main()
