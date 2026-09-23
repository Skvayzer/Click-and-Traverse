"""Small frozen-policy CPU intervention audit; no training or bank/checkpoint writes.

A/B/C are candidate controllers, not assertions of successful isolated responses.
Scores are over a common horizon censored at the first termination in any branch.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['OMP_NUM_THREADS']='2'
import sys, json, copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'scripts')]
import numpy as np
import torch
from diagnose_reactive_clearance_cpu import ProbeTask
from test_mjlab_task import _CPUSimulation, _tiny_bank
from cat_mjlab.learning import ActorCritic, LearnerConfig, gaussian_parameters

class AuditTask(ProbeTask):
    def _rewards(self, action, contacts):
        r, p=super()._rewards(action,contacts)
        self.parts={k:v.clone() for k,v in p.items()}
        self.pressure=self.telemetry['hand_clearance_pressure'].clone()
        self.actual_reward=r.clone()
        return r,p


def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    source=ROOT/'outputs/cat_flat_balance_ppo_37632_20260920/resume.pt'
    snapshot=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    policy=ActorCritic(LearnerConfig(**snapshot['learner']['config'])).eval()
    policy.load_state_dict(snapshot['learner']['model'])
    cfg=copy.deepcopy(json.loads((ROOT/'outputs/cat_recover_23_30720_20260922/run.json').read_text())['contract']['environment_config'])
    cfg.update(num_obs=222,num_pri=310,wholebody_hand_contrast=False,disable_hand_contrast=True,randomize_initial_episode_steps=False,hand_raised_reset_fraction=0.)
    cfg['reward_config']['scales'].update(wholebody_hand_contrast_region=0.,wholebody_hand_contrast_heading=0.)
    cfg['noise_config']['level']=0.;cfg['push_config']['enable']=False
    cfg['dm_rand_config'].update(enable_pd=False,enable_rfi=False)
    report=dict(checkpoint=str(source.relative_to(ROOT)),dt=.02,near_weight=cfg['hand_protection_near_weight'],encounters={})
    for kind in ('poke_front','narrow'):
        bank=_tiny_bank(3);bank.episode_lengths[:]=2000;bank.crossed_is_plane[:]=False
        bank.goals=torch.tensor([[100.,0.,.7]]).repeat(3,1)
        bank.origins=torch.tensor([[-100.,-100.,-5.]]).repeat(3,1)
        bank.shapes=torch.tensor([[1000,1000,100]]).repeat(3,1)
        active=False;center=torch.ones(3)*1000;gate_x=0.;radius=.06;halfwidth=.37
        def sample(name,pos,ids):
            if kind=='poke_front':
                delta=pos-center;length=delta.norm(dim=-1,keepdim=True)
                sdf=length-radius;normal=delta/length.clamp_min(1e-9)
            else:
                # Two finite axis-aligned cabinet boxes, 0.6 m long, 1.4 m tall.
                centers=pos.new_tensor([[gate_x+.3,-halfwidth-.15,.7],[gate_x+.3,halfwidth+.15,.7]])
                local=pos[:,:,None]-centers
                q=local.abs()-pos.new_tensor([.3,.15,.7])
                outside=q.clamp_min(0);dist=outside.norm(dim=-1)+q.amax(-1).clamp_max(0)
                nearest=dist.argmin(-1);sdf=dist.gather(-1,nearest[...,None])
                chosen=local.gather(2,nearest[:,:,None,None].expand(-1,-1,1,3)).squeeze(2)
                qc=chosen.abs()-pos.new_tensor([.3,.15,.7]);out=qc.clamp_min(0)
                normal=out*chosen.sign()/out.norm(dim=-1,keepdim=True).clamp_min(1e-9)
                inside=out.norm(dim=-1)==0
                axis=qc.argmax(-1)
                inward=torch.nn.functional.one_hot(axis,3)*chosen.sign()
                normal=torch.where(inside[...,None],inward,normal)
            if name=='sdf':return sdf if active else torch.ones_like(sdf)
            if name=='bf':return normal if active else torch.zeros_like(pos)
            gf=torch.zeros_like(pos);gf[:,:,0]=.6
            inward=(gf*normal).sum(-1,keepdim=True).clamp_max(0)
            return gf-torch.where(active & (sdf<.5),inward*normal,0.)
        bank.sample=sample;AuditTask.speed=.6
        task=AuditTask(_CPUSimulation(3),bank,copy.deepcopy(cfg),seed=71)
        task.math.compute_cmd_from_rtf=lambda *args:torch.tensor([.75,.6,0.,0.]).repeat(len(args[0]),1)
        task.reset(scene_ids=torch.arange(3));task.info['phase'][:]=task.info['phase'][0];task.info['gait'][:]=task.info['gait'][0]
        task.obs=task._observe(task.all_ids,torch.zeros(3,2,dtype=torch.bool))
        frames=[];root0=None;hand0=None;first_end=[None]*3;crossed=[False]*3;first_cross=[None]*3;common_end=None
        for step in range(450):
            if step==100:
                root0=task.data.qpos[:,:3].clone();hand0=task.info['positions'][:,5:7].clone()
                center=hand0[2,0]+torch.tensor([radius+float(task.hand_radii[0])+.12,0.,0.])
                gate_x=float(root0[2,0])+.80
                active=True
                held=task.info['motor_targets'][:,12:].clone()
                goal=held.clone()
                # Same bounded tucked posture delta as the existing FK probe.
                for side,sign in [('left',1),('right',-1)]:
                    for joint,delta in [('shoulder_pitch',-.3),('shoulder_roll',-.25*sign),('elbow',.7)]:
                        index=int(task.sim.model.joint(f'{side}_{joint}_joint').qposadr[0])-19
                        goal[0,index]+=delta
            if step>=100 and kind=='poke_front':
                travel=task.data.qpos[2,:3]-root0[2];travel[2]=0
                center=hand0[2,0]+travel+torch.tensor([radius+float(task.hand_radii[0])+.12-.1*(step-100)*.02,0.,0.])
            obs={k:v.clone() for k,v in task.obs.items()}
            if step>=100:
                # Standing-policy action generator for B: synthetic command/phase/guidance
                # inputs only. Restore ALL true task state before physical step/reward.
                original=task.info['command'][1].clone();delayed=task.info['command_delay'][1].clone()
                phase=task.info['phase'][1].clone();guidance=task.info['gf_delay'][1].clone()
                task.info['phase'][1]=0.;task.info['gf_delay'][1]=0.
                task.info['command'][1]=0.;task.info['command_delay'][1]=0.
                obs={k:v.clone() for k,v in task._observe(task.all_ids,torch.zeros(3,2,dtype=torch.bool)).items()}
                task.info['command'][1]=original;task.info['command_delay'][1]=delayed
                task.info['phase'][1]=phase;task.info['gf_delay'][1]=guidance
            mu,_=gaussian_parameters(policy.logits(obs['state'],0));action=mu.tanh()
            if step>=100:action[:,12:]=((goal-task.nominal[12:])/.8).clamp(-1,1)
            result=task.step(action)
            if step>=100:
                frames.append(dict(parts={k:v.tolist() for k,v in task.parts.items()},pressure=task.pressure.tolist(),
                    reward=task.actual_reward.tolist(),root=task.frame['qpos'][:,:3].tolist(),hands=task.frame['hands'].tolist(),
                    done=result['done'].tolist()))
                for j in range(3):
                    if first_end[j] is None:
                        crossed[j] |= bool(kind=='narrow' and task.frame['qpos'][j,0] >= gate_x+.6)
                        if crossed[j] and first_cross[j] is None:first_cross[j]=len(frames)
                        if result['done'][j]:first_end[j]=len(frames)
                if result['done'].any() and common_end is None:common_end=len(frames)
                if all(v is not None or crossed[j] for j,v in enumerate(first_end)):break
            elif result['done'].any():
                raise RuntimeError(f'{kind}: baseline terminated before obstacle onset at {step}')
        full_frames=frames
        frames=frames[:common_end] if common_end is not None else frames
        terms={k:np.asarray([f['parts'][k] for f in frames])*.02 for k in frames[0]['parts']}
        pressure=np.asarray([f['pressure'] for f in frames]);pre=sum(v for k,v in terms.items() if k!='wholebody_hand_clearance')
        base=np.clip(pre,0,10000);rows={}
        for j,label in enumerate(('A_arm_candidate','B_stop_candidate','C_hold_arms')):
            ledger={k:float(v[:,j].mean()) for k,v in terms.items()}
            ledger['reward_floor_lift']=float((base[:,j]-pre[:,j]).mean())
            ledger['body_collision_event']=0. # Virtual analytic SDF, production field termination; no collision bank.
            delta_root=np.asarray(frames[-1]['root'])[j]-root0[j].numpy()
            delta_hand=np.asarray(frames[-1]['hands'])[j]-hand0[j].numpy()
            rows[label]=dict(ledger_current=ledger,nonhand_floored=float(base[:,j].mean()),pressure=float(pressure[:,j].mean()),
                tracking_raw=float(terms['tracking_root_field'][:,j].mean()/.02),
                current_total=float((base[:,j]-.5*.02*pressure[:,j]).mean()),
                minus20_total=float((base[:,j]-20*.02*pressure[:,j]).mean()),
                root_delta_m=delta_root.tolist(),hand_relative_delta_m=(delta_hand-delta_root).tolist(),terminated=frames[-1]['done'][j])
        def threshold(a,b):
            x,y=rows[a],rows[b];den=.02*(x['pressure']-y['pressure'])
            return None if abs(den)<1e-12 else (x['nonhand_floored']-y['nonhand_floored'])/den
        extended_end=min(v for v in (first_cross[0],first_end[0],first_end[1],len(full_frames)) if v is not None)
        extended_terms={k:(np.asarray([f['parts'][k] for f in full_frames[:extended_end]])*.02).tolist() for k in full_frames[0]['parts']}
        report['encounters'][kind]=dict(first_terminal_step=first_end,root_crossed_far_gate=crossed,
            first_crossing_step=first_cross,AB_comparison_steps=extended_end,AB_comparison_terms=extended_terms,
            AB_root_delta_m=(np.asarray(full_frames[extended_end-1]['root'])-root0.numpy()).tolist(),
            AB_before_first_outcome=bool(all(v is None or v>extended_end for v in first_end[:2])),
            AB_body_max_forward_m=float(max(f['root'][1][0]-root0[1,0] for f in full_frames[:extended_end])),
            full_horizon_steps=len(full_frames),per_step_weighted_terms={k:v.tolist() for k,v in terms.items()},
            steps=len(frames),seconds=len(frames)*.02,rows=rows,
            hand_magnitude_A_B_equal=threshold('A_arm_candidate','B_stop_candidate'),
            hand_magnitude_C_B_equal=threshold('C_hold_arms','B_stop_candidate'),
            caveat='Physical CPU rollout; interventions do not guarantee isolated arm/base motion. Equalities conditional on these fixed traces, not policy optima. No full-body collision bank; no success-rate inference.')
        print(kind,json.dumps(report['encounters'][kind]),flush=True)
    out=ROOT/'outputs/response_weight_audit_cpu';out.mkdir(exist_ok=True)
    (out/'measurement.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
