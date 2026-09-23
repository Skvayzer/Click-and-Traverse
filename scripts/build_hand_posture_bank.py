#!/usr/bin/env python3
"""CPU-only metadata/certificate upgrade; reuse all fields, geometry and resets."""
from __future__ import annotations
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse,copy,json,math,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.audit_hand_posture_rewards import Probe,T
from cat_mjlab.config import wholebody_config
from cat_mjlab import constants as const
from cat_ppo.furniture.generalist_fields import load_generalist_manifest,sha256,_json_hash
from cat_ppo.furniture.scenes import _digest
from cat_ppo.furniture.contrastive_passages import _field_region_lower_bound
from cat_ppo.furniture.hand_posture_upgrade import SCHEMA
from cat_ppo.furniture.body_collision_bank import load_body_collision_bank

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.pending.json')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def publish(path,value):
    value.pop('manifest_sha256',None);value['manifest_sha256']=_json_hash(value);write(path,value)
def reuse(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha256(source)!=sha256(target):raise ValueError(f'Existing bytes differ: {target}')
        return
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)
def regions(probe):
    import mujoco
    centers=[]
    for mode in ('raised','tucked'):
        d=mujoco.MjData(probe.model);d.qpos[:]=probe.qpose(mode);mujoco.mj_forward(probe.model,d)
        centers.append(d.site_xpos[[probe.model.site(s).id for s in const.HAND_SITES]].copy())
    centers=np.array(centers);half=np.array([[.018,.010,.018],[.012,.008,.015]])[:,None,:]
    return centers-half,centers+half

def exact_field_box_bound(sdf,origin,centers,halves,radii):
    """Multiaffine extrema at clipped cell corners, including both cell limits.

    Preserves CAT's x-fast values / z-fast weights, including boundary jumps.
    Bounds an enclosing AABB, so it is conservative for rotated target boxes.
    """
    import itertools
    corners=np.array(list(itertools.product([0,1],repeat=3)))
    value_corners=corners[:,::-1]
    minimum=float('inf')
    for center,half,radius in zip(centers.reshape(-1,3),halves.reshape(-1,3),radii.reshape(-1)):
        low=(center-half-origin)/.04;high=(center+half-origin)/.04
        if np.any(low<0) or np.any(high>np.array(sdf.shape)-2):return -float('inf')
        bases=np.array(list(itertools.product(*(range(int(math.floor(a)),int(math.floor(b))+1) for a,b in zip(low,high)))))
        lo=np.maximum(low-bases,0);hi=np.minimum(high-bases,1)
        fraction=np.where(corners[None],hi[:,None],lo[:,None])
        weights=np.prod(np.where(corners[None,None],fraction[:,:,None],1-fraction[:,:,None]),axis=-1)
        indices=bases[:,None]+value_corners
        values=sdf[indices[...,0],indices[...,1],indices[...,2]]
        minimum=min(minimum,float((weights*values[:,None]).sum(-1).min())-float(radius))
    return minimum

def region_bounds(probe,scene,record,directory,xy,yaw,lo,hi,tolerance=0):
    boxes=scene['boxes'];bc=np.array([b['center'] for b in boxes]);bh=np.array([b['half_size'] for b in boxes]);by=np.array([b['yaw'] for b in boxes]);sdf=np.load(directory/'sdf.npy',mmap_mode='r');result=[]
    rot=np.zeros((len(xy),3,3));rot[:,0,0]=rot[:,1,1]=np.cos(yaw);rot[:,0,1]=-np.sin(yaw);rot[:,1,0]=np.sin(yaw);rot[:,2,2]=1
    for low,high in zip(lo,hi):
        centers=np.einsum('nij,hj->nhi',rot,(low+high)/2)+np.c_[xy,np.zeros(len(xy))][:,None]
        half=(high-low)/2+np.array([.01,0,0])+tolerance
        worldhalf=np.einsum('nij,hj->nhi',abs(rot),half)
        delta=centers[:,:,None]-bc
        local=np.stack([delta[...,0]*np.cos(by)+delta[...,1]*np.sin(by),-delta[...,0]*np.sin(by)+delta[...,1]*np.cos(by),delta[...,2]],-1)
        # Enclose the rotated target in each obstacle's frame. Minkowski box
        # expansion gives a tighter conservative bound than an isotropic ball.
        relative=yaw[:,None]-by[None,:]
        transformed=np.stack([abs(np.cos(relative))[:,None]*half[None,:,None,0]+abs(np.sin(relative))[:,None]*half[None,:,None,1],
            abs(np.sin(relative))[:,None]*half[None,:,None,0]+abs(np.cos(relative))[:,None]*half[None,:,None,1],
            np.broadcast_to(half[None,:,None,2],(len(xy),2,len(boxes)))],-1)
        off=abs(local)-bh-transformed
        signed=np.linalg.norm(np.maximum(off,0),axis=-1)+np.minimum(off.max(-1),0)-probe.radii[None,:,None]
        analytic=float(signed.min())
        field=_field_region_lower_bound(sdf,np.array(record['origin']),centers,worldhalf,np.broadcast_to(probe.radii,centers.shape[:-1]))
        if tolerance==0 and field<=.025:
            field=exact_field_box_bound(sdf,np.array(record['origin']),centers,worldhalf,np.broadcast_to(probe.radii,centers.shape[:-1]))
        result.append(dict(analytic_min=analytic,field_min=field))
    return result

def upgrade(probe,record,directory):
    scene=json.loads((directory/'scene.json').read_text());original=copy.deepcopy(scene)
    b=probe.bank(record,directory,scene);progress,xy,yaws=probe.route(scene)
    low,high=regions(probe);certs={}
    for mode in ('raised','tucked'):
        qs=np.repeat(probe.qpose(mode)[None],len(xy),0);o=probe.task(b,xy,yaws,qs)
        certs[mode]=dict(body=float(probe.clearance(o).min()),hand_field=float(o.info['sdf'][:,5:7].min()),elbow_field=float(o.info['elbow_clearance'].min()))
    # Sample all 35 primitives and conservative hand fields during endpoint raises.
    q=np.array([probe.qpose('raised',float(f)) for f in np.linspace(0,.95,20)])
    trans=probe.task(b,np.repeat(xy[[0,-1]],20,axis=0),np.repeat(yaws[[0,-1]],20),np.tile(q,(2,1)))
    transition=dict(body=float(probe.clearance(trans).min()),hand_field=float(trans.info['sdf'][:,5:7].min()))
    bounds=region_bounds(probe,scene,record,directory,xy,yaws,low,high)
    tolerance=region_bounds(probe,scene,record,directory,xy,yaws,low,high,tolerance=.05)
    raised=certs['raised']
    if min(raised['body'],transition['body'])<=.008 or min(raised['hand_field'],transition['hand_field'])<=.025 or bounds[0]['field_min']<=.025 or tolerance[0]['analytic_min']<=0:
        raise ValueError(f'Raised certificate failed: {record["scene_id"]}: {certs}, {transition}, {bounds}, {tolerance}')
    shelf=scene['hand_protection']['kind']=='hand_shelf_passage'
    tucked=shelf and certs['tucked']['body']>.008 and certs['tucked']['hand_field']>.005 and certs['tucked']['elbow_field']>0 and bounds[1]['analytic_min']>.005 and bounds[1]['field_min']>.005 and tolerance[1]['analytic_min']>0
    zone=dict(start_m=0.,end_m=float(progress[-1]),fade_m=.15,forward_weight=1.,hand_active=[1,1],hand_regions_min=low.tolist(),hand_regions_max=high.tolist(),region_valid=[True,bool(tucked)])
    cert=dict(schema='hand-contrast-certificate-v1',route_transition_validated=True,navigation_radius_m=.23,primitive_count=35,
        geometry_hash=_digest(dict(boxes=scene['boxes'],room_dimensions=scene['room_dimensions'])),route_hash=_digest(scene['route']),
        body_proxy_sha256=probe.compiled_hash,route_sample_count=len(xy),transition_sample_count=40,
        body_min_separation_m=min(raised['body'],transition['body']),hand_field_min_clearance_m=min(raised['hand_field'],transition['hand_field']),
        pose_checks=certs,transition_checks=transition,target_region_bounds=bounds,target_tolerance_bounds=tolerance,
        sampled_kinematic_stances_validated=False,dynamic_feasibility_validated=False,target_region_full_arm_ik_certified=False,
        method='Actual CPU MuJoCo FK, all 35 approved primitives, stored CAT fields, swept target-box lower bounds; endpoint raises')
    scene['hand_contrast']=dict(schema='hand-contrast-v1',group_id='hand-posture-'+scene['scene_id'],role='forward_protected',navigation_radius_m=.23,
        zones=[zone],modules=[dict(role='forward_protected')],certificate=cert,
        region_frame='xy:route tangent/left normal relative to root xy; z:absolute world metres',mode_names=['raised','tucked'],
        geometry_family='existing_hand_table' if not shelf else 'existing_hand_shelf',dynamic_feasibility_validated=False)
    # Explicit static/gait-phase quadrature using the actual runtime reward entry point.
    bank=probe.bank(record,directory,scene);audit={}
    for name,mode,angle in [('raised_forward','raised',0),('nominal_forward','nominal',0),('nominal_sideways','nominal',math.pi/2),('tucked_forward','tucked',0)]:
        o=probe.task(bank,xy,yaws+angle,np.repeat(probe.qpose(mode)[None],len(xy),0));v=probe.rewards(o)
        core=o.contrast['core_active'];margin=probe.clearance(o)
        audit[name]=dict(mean_pre=float(v['pre'][core].mean()),mean_post=float(v['post'][core].mean()),min_phase_pre=float(v['minimum_phase_pre'][core].min()),zero_fraction=float(v['zero_fraction'][core].mean()),
            body_min=float(margin.min()),collision_fraction=float((margin[core]<=0).float().mean()),
            components={k:float(t[core].mean()) for k,t in v['components'].items()},
            pre=v['pre'].tolist(),post=v['post'].tolist(),minimum_phase_pre=v['minimum_phase_pre'].tolist())
    report=dict(scene=scene['scene_id'],kind=scene['hand_protection']['kind'],level=scene['hand_protection']['level'],tucked_allowed=bool(tucked),certificate=cert,progress=progress.tolist(),reward=audit,
        raised_minus_sideways=audit['raised_forward']['mean_post']-audit['nominal_sideways']['mean_post'])
    assert {k:v for k,v in scene.items() if k!='hand_contrast'}==original
    return scene,report

def build(base,collision,resets,output,config_run):
    torch.set_num_threads(2)
    base=Path(base).resolve();collision=Path(collision).resolve();resets=Path(resets).resolve();output=Path(output).resolve()
    if output==base.parent or output.is_relative_to(base.parent):
        raise ValueError('Posture output must be separate from the source bank')
    if (output/'manifest.json').exists():
        raise FileExistsError('Published posture banks are immutable; choose a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    parent=load_generalist_manifest(base,verify_files=True)
    # Preserve the parent's historical report/plan links, not only its fields.
    for file in base.parent.glob('*.json'):
        if file!=base:reuse(file,output/file.name)
    archive=base.parent/'parent_bank'
    if archive.exists():
        for file in archive.rglob('*'):
            if file.is_file():reuse(file,output/'parent_bank'/file.relative_to(archive))
    config=json.loads(Path(config_run).read_text())['contract']['environment_config'];config=wholebody_config(config,hand_contrast=True)
    proposal_path=ROOT/'docs/assets/collision-proxy-proposal-20260916/proposal.json';probe=Probe(config,json.loads(proposal_path.read_text()));probe.compiled_hash=sha256(proposal_path)
    reports=[];manifest=copy.deepcopy(parent);new=[]
    for i,record in enumerate(parent['scenes']):
        if not record['source'].get('hand_protection'):new.append(record);continue
        directory=base.parent/record['path'];target=output/record['path'];target.mkdir(parents=True,exist_ok=True)
        scene,report=upgrade(probe,record,directory);reports.append(report)
        for file in directory.iterdir():
            if file.is_file() and file.name not in ('scene.json','source.json'):reuse(file,target/file.name)
        source=json.loads((directory/'source.json').read_text());source['hand_contrast']=scene['hand_contrast']
        write(target/'scene.json',scene);write(target/'source.json',source)
        updated=copy.deepcopy(record);updated['scene_sha256']=sha256(target/'scene.json');updated['source']['hand_contrast']=scene['hand_contrast'];updated['source']['metadata_sha256']=sha256(target/'source.json');new.append(updated)
        print(f'{len(reports)}/24 {record["scene_id"]}: tuck={report["tucked_allowed"]} raised-side={report["raised_minus_sideways"]:.6f}',flush=True)
    manifest['scenes']=new;manifest['external_retention_bank']=dict(manifest=str(base),sha256=sha256(base),storage='immutable 2338-scene external prefix')
    manifest['hand_posture_upgrade']=dict(schema=SCHEMA,retained_scene_count=2338,retained_scene_records_sha256=_json_hash(parent['scenes'][:2338]),selection_roles=[1],base_manifest_sha256=sha256(base),objectives='forward heading plus certified paired hand regions; raised-only tables',reward_coefficients_changed=False)
    manifest['hand_protection_curriculum']['clean_goal_success_threshold']=.35
    manifest['storage']='Retention by hash-pinned reference; hand fields byte-identical reuse; new posture metadata'
    publish(output/'manifest.json',manifest);load_generalist_manifest(output/'manifest.json',verify_files=True)
    write(output/'reward-audit.json',dict(method='CPU FK and uniform gait-phase quadrature; prescribed velocities/contacts, static bias torque, no physics rollout',configuration=config,scenes=reports))
    # Collision/reset state is reusable ONLY because geometry, fields, order, reset law and robot are identical.
    _,cm=load_body_collision_bank(collision,expected_field_manifest=base,expected_proxy_sha256=probe.compiled_hash)
    cout=output.with_name(output.name+'_collision');cout.mkdir(exist_ok=True)
    for entry in cm['arrays'].values():reuse(collision.parent/entry['file'],cout/entry['file'])
    for old,newrec in zip(cm['scenes'],new):
        reuse(collision.parent/old['geometry_file'],cout/old['geometry_file'])
        if old['provenance']['kind']=='canonical-room-OBBs':old['provenance']['source_sha256']=newrec['scene_sha256']
    cm.update(field_manifest_sha256=sha256(output/'manifest.json'),field_manifest_content_sha256=manifest['manifest_sha256'],metadata_upgrade=dict(source_manifest=str(collision),sha256=sha256(collision),geometry_arrays_unchanged=True))
    publish(cout/'manifest.json',cm);load_body_collision_bank(cout/'manifest.json',expected_field_manifest=output/'manifest.json')
    rm=json.loads(resets.read_text())
    if rm['field_manifest_sha256']!=sha256(base) or rm['collision_bank_sha256']!=sha256(collision) or rm['proxy_sha256']!=probe.compiled_hash or sha256(resets.parent/rm['file'])!=rm['sha256']:
        raise ValueError('Parent reset hash chain differs')
    rout=output.with_name(output.name+'_resets');rout.mkdir(exist_ok=True);reuse(resets.parent/rm['file'],rout/rm['file'])
    rm.update(field_manifest=str(output/'manifest.json'),field_manifest_sha256=sha256(output/'manifest.json'),collision_bank=str(cout/'manifest.json'),collision_bank_sha256=sha256(cout/'manifest.json'),metadata_upgrade=dict(source_manifest=str(resets),sha256=sha256(resets),reset_poses_unchanged=True,geometry_unchanged=True))
    write(rout/'manifest.json',rm)
    print('Published',output/'manifest.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ROOT/'data/furniture/cat_hand_protection_v1_20260917/manifest.json')
    p.add_argument('--collision',type=Path,default=ROOT/'data/furniture/body_collision_hand_v1_20260917/manifest.json')
    p.add_argument('--resets',type=Path,default=ROOT/'data/furniture/body_collision_hand_resets_v1_20260917/manifest.json')
    p.add_argument('--output',type=Path,default=ROOT/'data/furniture/cat_hand_posture_v1_20260919')
    p.add_argument('--config-run',type=Path,default=ROOT/'outputs/cat_width_curriculum_compiled_30720/run.json')
    build(**vars(p.parse_args()))
