#!/usr/bin/env python3
"""CPU acceptance quadrature and arm-acquisition probes for the posture bank."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','');os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.audit_hand_posture_rewards import Probe,T
from cat_ppo.furniture.generalist_fields import load_generalist_manifest

def verify(manifest):
    torch.set_num_threads(2);manifest=Path(manifest).resolve();m=load_generalist_manifest(manifest,verify_files=False)
    budget=json.loads((manifest.parent/'reward-audit.json').read_text())
    probe=Probe(budget['configuration'],json.loads((ROOT/'docs/assets/collision-proxy-proposal-20260916/proposal.json').read_text()))
    rows=[]
    for rec in m['scenes'][2338:]:
        directory=manifest.parent/rec['path'];scene=json.loads((directory/'scene.json').read_text());bank=probe.bank(rec,directory,scene)
        progress,xy,yaw=probe.route(scene,spacing=.005)
        record=dict(scene=rec['scene_id'],kind=scene['hand_protection']['kind'],transitions=[])
        for path,start,speed in [('linear',0.,.6),('linear',.15,.6),('linear',0.,0.),
                                 ('roll_in_then_raise',0.,0.),('roll_in_then_raise',0.,.15)]:
            count=22 if path=='linear' else 32
            f=np.clip((np.arange(count)-(0 if path=='linear' else 10))/20,0,1)*.95
            qs=np.array([probe.qpose('raised',float(v)) for v in f])
            if path!='linear':
                nominal=probe.qpose('nominal');target=probe.qpose('raised')
                for side,sign in [('left',1),('right',-1)]:
                    index=probe.model.joint(f'{side}_shoulder_roll_joint').qposadr[0]
                    for j in range(count):
                        a=min(j/10,1);b=np.clip((j-10)/20,0,1)
                        qs[j,index]=nominal[index]-.30*sign*a*(1-b)+(target[index]-nominal[index])*b
            distance=start+np.arange(count)*.02*speed
            points=np.stack([np.interp(distance,progress,xy[:,i]) for i in range(2)],-1);angles=np.interp(distance,progress,yaw)
            o=probe.task(bank,points,angles,qs)
            prev=np.vstack([qs[:1],qs[:-1]]);prev2=np.vstack([qs[:1],prev[:-1]])
            o.info['previous_upper']=T(prev[:,19:]);o.info['previous_previous_upper']=T(prev2[:,19:])
            o.data.qvel[:,6:]=T((qs[:,7:]-prev[:,7:])/.02);o.info['last_joint_vel']=T((prev[:,7:]-prev2[:,7:])/.02)
            positions=o.info['positions'].numpy();velocity=np.diff(positions,axis=0,prepend=positions[:1])/.02
            result=probe.rewards(o,speed,site_velocity=velocity,double_support=speed==0)
            baseline=probe.task(bank,points,angles,np.repeat(probe.qpose('nominal')[None],count,axis=0));vbase=probe.rewards(baseline,speed,double_support=speed==0)
            record['transitions'].append(dict(path=path,start_m=start,speed=speed,mean_pre=float(result['pre'].mean()),mean_post=float(result['post'].mean()),
                min_phase_pre=float(result['minimum_phase_pre'].min()),clamped_phase_fraction=float(result['zero_fraction'].mean()),
                mean_post_advantage=float((result['post']-vbase['post']).mean()),body_min=float(probe.clearance(o).min()),
                hand_field_min=float(o.info['sdf'][:,5:7].min()),max_upper_joint_speed=float(o.data.qvel[:,18:].abs().max())))
        # Isolate *added* shaping on approaches: legacy and new reward at identical FK samples.
        approach=(progress<=.5)
        for name,mode,offset in [('raised','raised',0.),('sideways','nominal',np.pi/2)]:
            obj=probe.task(bank,xy[approach],yaw[approach]+offset,np.repeat(probe.qpose(mode)[None],int(approach.sum()),axis=0))
            new=probe.rewards(obj);obj.hand_contrast=False;old=probe.rewards(obj)
            record[name+'_approach']=dict(min_phase_pre=float(new['minimum_phase_pre'].min()),clamped_fraction=float(new['zero_fraction'].mean()),
                legacy_clamped_fraction=float(old['zero_fraction'].mean()))
        rows.append(record);print(rec['scene_id'],record['transitions'][0],flush=True)
    out=dict(method='Actual FK, stored fields and CATTask._rewards; linear 0.4s and staged 0.6s raises, actual finite-difference hand velocities and target histories; double-support contacts when paused, leg phase quadrature when moving; no dynamics rollout',scenes=rows)
    (manifest.parent/'acquisition-audit.json').write_text(json.dumps(out,indent=2)+'\n')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'data/furniture/cat_hand_posture_v1_20260919/manifest.json');verify(**vars(p.parse_args()))
