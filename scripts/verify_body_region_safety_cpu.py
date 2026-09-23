#!/usr/bin/env python3
"""Geometric witnesses for the observed-pose body box; no dynamics or writes to banks."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1',
                  OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import itertools
import json
import numpy as np
import mujoco
from prototype_body_region_cpu import ROOT, OUT, Region, rotation, _CPUSimulation
from cat_ppo.furniture.generalist_fields import scene_directory
from cat_ppo.furniture.grippers import hand_mesh_vertices, hand_sphere


def distances(points, boxes):
    centers=np.array([b['center'] for b in boxes]);half=np.array([b['half_size'] for b in boxes])
    yaw=np.array([b['yaw'] for b in boxes]);c=np.cos(yaw);s=np.sin(yaw)
    R=np.array([[c,-s,np.zeros_like(c)],[s,c,np.zeros_like(c)],[np.zeros_like(c),np.zeros_like(c),np.ones_like(c)]]).transpose(2,0,1)
    local=np.einsum('kji,pkj->pki',R,points[:,None]-centers)
    d=np.abs(local)-half
    return np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)


def main():
    manifest=ROOT/'data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json'
    full=json.loads(manifest.read_text());summary=json.loads((OUT/'measurements.json').read_text())
    model=_CPUSimulation(1).model;data=mujoco.MjData(model)
    signs=np.array(list(itertools.product((-1,1),repeat=3)))
    result=[]
    for e in summary['episodes']:
        if e['mode']!='body':continue
        a=next(np.load(f) for f in sorted(OUT.glob('body_*.npz')) if e['index'] in np.load(f)['indices'])
        j=list(a['indices']).index(e['index']);idx=e['index']
        scene=json.loads((scene_directory(full,manifest,full['scenes'][idx])/'scene.json').read_text())
        cabinets=[b for b in scene['boxes'] if b['category']=='cabinet']
        t=np.diff(np.array(scene['route']),axis=0)[0];t/=np.linalg.norm(t);n=np.r_[-t[1],t[0],0]
        route=np.array([[t[0],-t[1],0],[t[1],t[0],0],[0,0,1]])
        z=scene['hand_contrast']['zones'][0]
        lo=np.array(z['hand_regions_min'][0]);hi=np.array(z['hand_regions_max'][0])
        reg=Region.__new__(Region);reg.q0=a['initial_q'][j];reg.R0=rotation(reg.q0)
        reg.center=(hi+lo)/2;reg.half=(hi-lo)/2;reg.route=route
        reg.local_center=(reg.center@route.T-[0,0,reg.q0[2]])@reg.R0;reg.local_axes=reg.R0.T@route
        q=a['q'][a['index']==idx][-1];data.qpos[:]=q;mujoco.mj_kinematics(model,data)
        root=np.array(e['safety']['worst_root']);center,axes=reg.box(root,'body')
        palm=[];mesh=[];witnesses=[]
        for h,side in enumerate(('left','right')):
            # All 8 corners, expanded in either route-normal direction by 50 mm.
            points=center[h]+(signs*reg.half[h])@axes.T
            expected=reg.half[h]@np.abs(axes.T@n)
            np.testing.assert_allclose(np.max((points-center[h])@n),expected,atol=1e-10)
            candidates=np.concatenate((points+.05*n,points-.05*n))
            d=distances(candidates,cabinets);k,b=np.unravel_index(d.argmin(),d.shape)
            point=candidates[k]
            local=(point-center[h])@axes
            metric=np.linalg.norm(np.maximum(np.abs(local)-reg.half[h],0))
            assert metric<=.050001
            vertices=np.concatenate(list(hand_mesh_vertices(side).values()))-np.array(hand_sphere(side)['center'])
            wrist=data.xmat[model.body(side+'_wrist_yaw_link').id].reshape(3,3)
            # Same wrist orientation as observed endpoint, translated into allowed box.
            world=vertices@wrist.T+point
            md=distances(world,cabinets)
            palm.append(float(d[k,b]));mesh.append(float(md.min()))
            witnesses.append(dict(hand=side,allowed_site=point.tolist(),box_distance_m=float(metric),
                                  cabinet=cabinets[b]['name'],site_signed_distance_m=float(d[k,b]),
                                  translated_mesh_min_vertex_signed_distance_m=float(md.min())))
        # Independent actual-pose mesh vertex collision evidence at termination.
        actual=[]
        for side in ('left','right'):
            vertices=np.concatenate(list(hand_mesh_vertices(side).values()))
            bid=model.body(side+'_wrist_yaw_link').id
            world=vertices@data.xmat[bid].reshape(3,3).T+data.xpos[bid]
            actual.append(float(distances(world,cabinets).min()))
        result.append(dict(index=idx,worst_root_time_s=e['safety']['worst_time_s'],
                           allowed_site_min_signed_distance_m=min(palm),
                           allowed_translated_mesh_min_vertex_signed_distance_m=min(mesh),
                           actual_terminal_mesh_min_vertex_signed_distance_m=min(actual),witnesses=witnesses))
    (OUT/'safety_witnesses.json').write_text(json.dumps(dict(
        scope='Allowed-region geometric counterexamples, not IK reachability certificates. Negative vertex SDF is an actual mesh vertex inside a cabinet; positive vertex SDF alone is not a mesh clearance certificate.',
        scenes=result),indent=2)+'\n')
    for r in result:print(r['index'],*[round(r[k]*1000,3) for k in ('allowed_site_min_signed_distance_m','allowed_translated_mesh_min_vertex_signed_distance_m','actual_terminal_mesh_min_vertex_signed_distance_m')])


if __name__=='__main__':main()
