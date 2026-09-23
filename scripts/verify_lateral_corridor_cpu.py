#!/usr/bin/env python3
"""Combined corridor/heading/arm/progress rescore; CPU saved FK only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import sys, json, hashlib, shutil
from pathlib import Path
import numpy as np
import mujoco
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cat_mjlab.model import assemble_training_xml
from cat_ppo.furniture.grippers import hand_sphere
from cat_mjlab.collision import capsule_box_separation, sphere_box_separation
OUT=ROOT/'outputs/lateral_corridor_combined_cpu'
MARGIN=.07899
RADIUS=max(hand_sphere(s)['radius'] for s in ('left','right'))
PCTS=[0,5,25,50,75,95,100]

def quant(a): return np.percentile(a,PCTS,axis=0).tolist()
def stats(good,p,start,end,p0):
    good=np.asarray(good); core=(p>=start)&(p<=end); streak=longest=prefix=0; broken=False
    distance=0.;far=p0
    for g,x in zip(good,p):
        advance=max(0.,min(float(x),end)-max(far,start))
        if g:distance+=advance
        far=max(far,float(x))
        streak=streak+1 if g else 0;longest=max(longest,streak)
        if not g:broken=True
        if not broken:prefix+=1
    return dict(recorded_steps=len(good),core_steps=int(core.sum()),core_good_steps=int(good[core].sum()),
                core_step_fraction=float(good[core].mean()) if core.any() else None,
                initial_steps=prefix,longest_steps=longest,initial_seconds=prefix*.02,longest_seconds=longest*.02,
                compliant_full_core_distance_fraction=distance/(end-start),reached_core_end=bool(p[-1]>=end),
                full_compliant_crossing=bool(p[-1]>=end and good.all()),
                minimum_progress=float(p.min()),final_progress=float(p[-1]))

def main():
    assert shutil.disk_usage(ROOT).free>2*1024**3
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    model=mujoco.MjModel.from_xml_string(assemble_training_xml());data=mujoco.MjData(model)
    sites=[model.site(s+'_palm').id for s in ('left','right')]
    proposal_path=ROOT/'docs/assets/collision-proxy-proposal-20260916/proposal.json'
    shapes=[s for s in json.loads(proposal_path.read_text())['shapes'] if s['group']=='arms']
    bids=[model.body(s['body_name']).id for s in shapes]
    elbow=np.array(['elbow' in s['id'] for s in shapes])
    forearm=np.array([any(x in s['id'] for x in ('elbow','wrist')) for s in shapes])
    radii=np.array([s['radius'] for s in shapes]);ends=np.array([s['endpoints'] for s in shapes])
    def fk(q):
        hh=[];ee=[]
        for pose in q:
            data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
            hh.append(data.site_xpos[sites].copy())
            ee.append(data.xpos[bids,None]+np.einsum('sij,skj->ski',data.xmat[bids].reshape(-1,3,3),ends))
        return np.array(hh),np.array(ee)
    manifest_path=ROOT/'data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    old=json.loads((ROOT/'outputs/body_region_cpu/measurements.json').read_text())
    nominal=np.array(json.loads((ROOT/'outputs/arm_hold_diagnosis_cpu/static_hold.json').read_text())['scenes'][0]['nominal'])
    scenes={};geometry=[]
    for idx,record in enumerate(manifest['scenes']):
        if record.get('source',{}).get('hand_contrast',{}).get('role')!='forward_protected':continue
        path=manifest_path.parent/record['path']/'scene.json';s=json.loads(path.read_text())
        origin=np.r_[s['route'][0],0.];t=np.diff(s['route'],axis=0)[0];t/=np.linalg.norm(t);n=np.r_[-t[1],t[0],0.]
        boxes=[b for b in s['boxes'] if b['category']=='cabinet']
        centers=np.array([b['center'] for b in boxes]);half=np.array([b['half_size'] for b in boxes]);yaw=np.array([b['yaw'] for b in boxes])
        c=np.cos(yaw);v=np.sin(yaw);rot=np.array([[c,-v,np.zeros_like(c)],[v,c,np.zeros_like(c)],[np.zeros_like(c),np.zeros_like(c),np.ones_like(c)]]).transpose(2,0,1)
        yc=(centers-origin)@n; support=np.einsum('bi,bi->b',half,np.abs(np.einsum('bji,j->bi',rot,n)))
        lower=float(np.max((yc+support)[yc<0]));upper=float(np.min((yc-support)[yc>0]))
        assert np.max(np.abs(rot[:,:2,0]@n[:2]))<1e-7 # Cabinets parallel to route.
        bound=min(upper,-lower)-RADIUS-MARGIN
        z=s['hand_contrast']['zones'][0];lo=np.array(z['hand_regions_min'][0]);hi=np.array(z['hand_regions_max'][0])
        box_extent=float(np.max(np.abs(np.stack((lo[:,1],hi[:,1]))))+.05)
        oldmargin=min(upper,-lower)-RADIUS-box_extent
        g=dict(index=idx,scene=record['scene_id'],scene_file=str(path.relative_to(ROOT)),scene_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               lower_face_m=lower,upper_face_m=upper,gap_m=upper-lower,margin_m=MARGIN,symmetric_bound_m=bound,
               asymmetric_center_interval_m=[lower+RADIUS+MARGIN,upper-RADIUS-MARGIN],
               original_centered_box_margin_m=oldmargin,exact_same_scene_margin_bound_m=box_extent,
               core_start_m=z['start_m']+z['fade_m'],core_end_m=z['end_m']-z['fade_m'],
               cabinet_faces=[dict(name=b['name'],lateral_center_m=float(y),lateral_support_m=float(h)) for b,y,h in zip(boxes,yc,support)])
        geometry.append(g);scenes[idx]=(s,g,origin,t,n,centers,half,rot)
    def evaluate(idx,q,h,p,p0,mode,baseline=None):
        s,g,o,t,n,centers,half,rot=scenes[idx];bound=g['symmetric_bound_m']
        actual,ep=fk(q);lateral=(h-o)@n;actual_y=(actual-o)@n;root=(q[:,:3]-o)@n
        good=(np.abs(lateral)<=bound).all(-1);fk_good=(np.abs(actual_y)<=bound).all(-1)
        local=lateral-root[:,None]
        if baseline is None:
            z=s['hand_contrast']['zones'][0];lo=np.array(z['hand_regions_min'][0]);hi=np.array(z['hand_regions_max'][0]);
            delta=h-q[:,None,:3];xyz=np.stack((delta[:,:,:2]@t,local,h[:,:,2]),-1)
            baseline=np.linalg.norm(np.maximum(lo-xyz,0)+np.maximum(xyz-hi,0),axis=-1).max(-1)<=.05
        core=(p>=g['core_start_m'])&(p<=g['core_end_m']);select=core&fk_good
        ey=(ep-o)@n
        lateral_clear=np.minimum(ey.min(-1)-g['lower_face_m'],g['upper_face_m']-ey.max(-1))-radii
        tensor=lambda a:torch.as_tensor(a,dtype=torch.float64)
        finite=capsule_box_separation(tensor(ep[:,:,None,0]),tensor(ep[:,:,None,1]),tensor(radii[None,:,None]),tensor(centers[None,None]),tensor(rot[None,None]),tensor(half[None,None])).numpy().min(-1)
        hand_finite=sphere_box_separation(tensor(actual[:,:,None]),RADIUS,tensor(centers[None,None]),tensor(rot[None,None]),tensor(half[None,None])).numpy().min(-1)
        if select.any():
            assert hand_finite[select].min()>=MARGIN-1e-8
        from cat_mjlab.lateral_corridor import combined_pose
        headings=[]
        for pose in q:
            data.qpos[:]=pose; mujoco.mj_kinematics(model,data)
            f=data.site_xmat[[model.site('imu_in_pelvis').id,model.site('imu_in_torso').id]].reshape(2,3,3)[:,:2,0]
            headings.append(bool(((f@t)/np.maximum(np.linalg.norm(f,axis=-1),1e-8)>=np.cos(np.deg2rad(15))).all()))
        pose_good=combined_pose(tensor(actual_y),tensor(np.full(len(q),bound)),torch.tensor(headings),tensor(lateral_clear.min(-1))).numpy()
        speed=np.diff(np.r_[p0,p])/.02
        combined=pose_good & (speed>=.2)
        # S uses average crossing speed, not a minimum at every gait sample.
        # Preserve that episode-level budget; per-step >=.2 is an extra diagnostic.
        pose_stats=stats(pose_good,p,g['core_start_m'],g['core_end_m'],p0)
        crossed=np.flatnonzero(p>=g['core_end_m'])
        budget_pass=bool(len(crossed) and (crossed[0]+1)*.02 <= (g['core_end_m']-p0)/.2)
        pose_stats['full_compliant_crossing'] &= budget_pass
        arms={}
        for name,mask in [('elbow',elbow),('forearm_including_elbow_wrist',forearm),('all_arm',np.ones(len(shapes),bool))]:
            arms[name]=dict(min_lateral_clearance_all_core_m=float(lateral_clear[core][:,mask].min()),
                min_lateral_clearance_hand_compliant_core_m=float(lateral_clear[select][:,mask].min()) if select.any() else None,
                min_finite_cabinet_clearance_hand_compliant_core_m=float(finite[select][:,mask].min()) if select.any() else None,
                hand_compliant_core_steps_with_lateral_violation=int((lateral_clear[select][:,mask].min(-1)<0).sum()),
                hand_compliant_core_steps_with_finite_proxy_collision=int((finite[select][:,mask].min(-1)<=0).sum()),
                hand_compliant_core_steps_below_requested_margin=int((lateral_clear[select][:,mask].min(-1)<MARGIN).sum()))
        row=dict(index=idx,mode=mode,crossing_speed_budget_pass=budget_pass,combined_gate=pose_stats,combined=stats(combined,p,g['core_start_m'],g['core_end_m'],p0),combined_pose=stats(pose_good,p,g['core_start_m'],g['core_end_m'],p0),box=stats(baseline,p,g['core_start_m'],g['core_end_m'],p0),
             corridor=stats(good,p,g['core_start_m'],g['core_end_m'],p0),
             exact_original_margin_corridor=stats((np.abs(lateral)<=g['exact_same_scene_margin_bound_m']).all(-1),p,g['core_start_m'],g['core_end_m'],p0),
             zero_margin_corridor=stats((np.abs(lateral)<=bound+MARGIN).all(-1),p,g['core_start_m'],g['core_end_m'],p0),
             fk_corridor=stats(fk_good,p,g['core_start_m'],g['core_end_m'],p0),
             legacy_vs_fk_classification_disagreements=int((good!=fk_good).sum()),
             root_relative_only_false_safe_steps=int((((np.abs(local)<=bound).all(-1))&~good).sum()),
             lateral_route_signed_quantiles_m=quant(lateral[core]),lateral_root_signed_quantiles_m=quant(local[core]),
             worst_hand_abs_lateral_quantiles_m=quant(np.abs(lateral[core]).max(-1)),
             worst_hand_excess_quantiles_m=quant(np.maximum(np.abs(lateral[core]).max(-1)-bound,0)),
             root_cross_track_quantiles_m=quant(root[core]),hand_compliant_fk_core_steps=int(select.sum()),arms=arms,
             minimum_swept_hand_clearance_m=float(min(g['upper_face_m'],-g['lower_face_m'])-RADIUS-np.abs(lateral).max()),
             min_hand_finite_clearance_on_compliant_core_m=float(hand_finite[select].min()) if select.any() else None)
        return row,(lateral[core],local[core],np.maximum(np.abs(lateral[core]).max(-1)-bound,0))
    from cat_mjlab.lateral_corridor import combined_pose
    results=[];pooled={};counter=[];static_nominal=[]
    for path in sorted((ROOT/'outputs/body_region_cpu').glob('*.npz')):
        a=np.load(path);mode=path.stem.split('_')[0]
        for j,idx in enumerate(a['indices']):
            idx=int(idx);mask=a['index']==idx;q=a['q'][mask];p=a['progress'][mask]
            oldrow=next(r for r in old['episodes'] if r['index']==idx and r['mode']==mode)
            row,dist=evaluate(idx,q,a['legacy_hands'][mask],p,oldrow['start_progress'],mode,a['error'][mask]<=.05 if mode=='absolute' else None)
            if mode=='absolute':
                np.testing.assert_allclose(row['box']['compliant_full_core_distance_fraction'],oldrow['compliant_fraction_full_core'],atol=1e-9)
                np.testing.assert_equal(row['box']['initial_steps'],oldrow['continuous_steps'])
                h,_=fk(q);np.testing.assert_allclose(h,a['actual'][mask],atol=1e-10)
                # Kinematic gameability witnesses; keep saved torso/legs, replace only arms.
                s,g,o,t,n,*_=scenes[idx]
                for heading_deg in (0,90):
                    pose=a['initial_q'][j].copy();pose[:2]=o[:2]+t*(g['core_start_m']+.1)
                    pose[19:]=nominal[12:]
                    yaw=np.arctan2(t[1],t[0])+np.deg2rad(heading_deg)
                    pose[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
                    hh,ee=fk(pose[None]);yy=(hh-o)@n;ey=(ee-o)@n
                    ac=np.minimum(ey.min(-1)-g['lower_face_m'],g['upper_face_m']-ey.max(-1))-radii
                    static_nominal.append(dict(index=idx,heading_deg=heading_deg,lateral_hand_m=yy[0].tolist(),
                         hand_compliant=bool((np.abs(yy)<=g['symmetric_bound_m']).all()),
                         min_arm_lateral_clearance_m=float(ac.min()),combined_pose_pass=bool(combined_pose(torch.tensor(yy),torch.tensor([g['symmetric_bound_m']]),torch.tensor([heading_deg==0]),torch.tensor([ac.min()]))[0])))
                for variant in ('nominal_arms_actual_root','nominal_arms_centered_root','nominal_arms_centered_route_heading'):
                    poses=q.copy();poses[:,22:]=nominal[15:]
                    if variant!='nominal_arms_actual_root':poses[:,:3]-=((poses[:,:3]-o)@n)[:,None]*n
                    if variant=='nominal_arms_centered_route_heading':
                        yaw=np.arctan2(t[1],t[0]);poses[:,3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
                    hh,ee=fk(poses);yy=(hh-o)@n;eg=(ee-o)@n
                    clear=np.minimum(eg.min(-1)-g['lower_face_m'],g['upper_face_m']-eg.max(-1))-radii
                    counter.append(dict(index=idx,variant=variant,frames=len(q),compliant_steps=int((np.abs(yy)<=g['symmetric_bound_m']).all(-1).sum()),
                         max_abs_hand_lateral_range_m=[float(np.abs(yy).max(-1).min()),float(np.abs(yy).max(-1).max())],
                         min_arm_lateral_clearance_m=float(clear.min())))
            results.append(row);pooled.setdefault(mode,[]).append(dist)
    a=np.load(ROOT/'outputs/arm_hold_diagnosis_cpu/policy.npz')
    indices=json.loads((ROOT/'outputs/arm_hold_diagnosis_cpu/summary.json').read_text())['scenes']
    for j,idx in enumerate(indices):
        end=int(a['ends'][j]);q=a['qpos'][1:end+1,j].astype(float);s,g,o,t,*_=scenes[idx]
        p=(q[:,:2]-o[:2])@t;p0=float((a['initial_qpos'][j,:2]-o[:2])@t)
        row,dist=evaluate(idx,q,a['hand_world'][1:end+1,j],p,p0,'unmodified_policy',a['error'][1:end+1,j]<=.05)
        yy=(a['hand_world'][51:end+1,j]-o)@scenes[idx][4]
        row['after_first_second']=dict(steps=len(yy),compliant_steps=int((np.abs(yy)<=g['symmetric_bound_m']).all(-1).sum()),
            worst_hand_abs_lateral_quantiles_m=quant(np.abs(yy).max(-1)))
        results.append(row);pooled.setdefault('unmodified_policy',[]).append(dist)
    aggregate={}
    for mode,arrays in pooled.items():
        rs=[r for r in results if r['mode']==mode];y=np.concatenate([x[0] for x in arrays]);local=np.concatenate([x[1] for x in arrays]);excess=np.concatenate([x[2] for x in arrays])
        aggregate[mode]=dict(scenes=len(rs),lateral_route_signed_quantiles_m=quant(y),lateral_root_signed_quantiles_m=quant(local),
            worst_hand_abs_lateral_quantiles_m=quant(np.abs(y).max(-1)),worst_hand_excess_quantiles_m=quant(excess),
            metrics={k:dict(core_steps=sum(r[k]['core_steps'] for r in rs),core_good_steps=sum(r[k]['core_good_steps'] for r in rs),
                full_compliant_crossings=sum(r[k]['full_compliant_crossing'] for r in rs),
                mean_full_core_distance_fraction=float(np.mean([r[k]['compliant_full_core_distance_fraction'] for r in rs]))) for k in ('box','corridor','fk_corridor','combined_pose','combined','combined_gate')})
    assert all(not r['combined_pose_pass'] for r in static_nominal if r['heading_deg']==90)
    assert not torch.cuda.is_initialized()
    report=dict(criteria=dict(heading='Both pelvis and torso within 15 degrees of route forward', arm_margin_m=MARGIN, combined='Combined pose AND endpoint forward difference / .02 >= .2 m/s (stricter dense diagnostic)', combined_gate='Combined pose with S crossing-average >= .2 m/s; seeded first-core diagnostic only, not a complete upstream multi-zone S evaluation'),scope='Saved first-core trajectories, 20-ms endpoints only; no resimulation. Baseline four unmodified-policy scenes are seeded and from an older frozen snapshot, not the current live policy. Nominal-arm witnesses are kinematic, not dynamic rollouts.',
        radius_m=RADIUS,margin_m=MARGIN,quantile_percentiles=PCTS,geometry=geometry,episodes=results,aggregate=aggregate,nominal_arm_counterfactuals=counter,static_nominal_heading_witnesses=static_nominal,
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [manifest_path,proposal_path,ROOT/'outputs/arm_hold_diagnosis_cpu/policy.npz',*sorted((ROOT/'outputs/body_region_cpu').glob('*.npz'))]},
        limitations=['No between-endpoint arm poses saved; continuous means consecutive sampled 20-ms steps.', 'Finite capsule separation is a proxy certificate; negative proxy distance is not proof of mesh collision.', 'Crossing score covers first protected core only; no claims about later cores.', 'No dynamics after original termination; changing objective might change trajectory, which is unmeasured.'])
    OUT.mkdir(exist_ok=True);(OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))
    for r in results:
        if r['mode']=='body':continue
        print(r['index'],r['mode'],'box',r['box']['core_good_steps'],r['box']['core_steps'],'corr',r['corridor']['core_good_steps'],r['corridor']['initial_steps'],r['corridor']['longest_steps'],round(r['corridor']['compliant_full_core_distance_fraction']*100,2),'elbow',r['arms']['elbow'])
if __name__=='__main__':main()
