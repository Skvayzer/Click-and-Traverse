#!/usr/bin/env python3
"""Reduce the measurement-only body-region experiment; no simulation or training."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1',
                  OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import json
import numpy as np
from scipy.spatial.transform import Rotation
from prototype_body_region_cpu import ROOT, OUT, Region, rotation
from cat_ppo.furniture.generalist_fields import scene_directory


def main():
    manifest=ROOT/'data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json'
    full=json.loads(manifest.read_text())
    summary=json.loads((OUT/'measurements.json').read_text())
    reduced=[]; all_required=[]; total_xy_bad=0; total_frames=0
    for file in sorted(OUT.glob('*.npz')):
        a=np.load(file); mode=file.stem.split('_')[0]
        for j,idx in enumerate(a['indices']):
            scene=json.loads((scene_directory(full,manifest,full['scenes'][idx])/'scene.json').read_text())
            zone=scene['hand_contrast']['zones'][0]
            lo=np.array(zone['hand_regions_min'][0]);hi=np.array(zone['hand_regions_max'][0])
            center=(lo+hi)/2;half=(hi-lo)/2
            t=np.diff(np.array(scene['route']),axis=0)[0];t/=np.linalg.norm(t)
            route=np.array([[t[0],-t[1],0],[t[1],t[0],0],[0,0,1]])
            reg=Region.__new__(Region);reg.q0=a['initial_q'][j];reg.R0=rotation(reg.q0)
            reg.route=route;reg.center=center;reg.half=half
            reg.local_center=(center@route.T-[0,0,reg.q0[2]])@reg.R0
            reg.local_axes=reg.R0.T@route
            mask=a['index']==idx
            rows={k:a[k][mask] for k in ('step','q','actual','reference','no_pelvis_actual','parts','height','orientation','error','legacy_hands','progress')}
            q=rows['q']; hands=rows['actual']
            abscent=center@route.T+np.column_stack((q[:,:2],np.zeros(len(q))))[:,None,:]
            local=(hands-abscent)@route
            excess=np.maximum(np.abs(local)-half,0)
            errors=np.linalg.norm(excess,axis=-1)
            xyerr=np.linalg.norm(excess[:,:,:2],axis=-1)
            frozenerr=np.linalg.norm(np.maximum(np.abs((rows['no_pelvis_actual']-abscent)@route)-half,0),axis=-1)
            referr=np.linalg.norm(np.maximum(np.abs((rows['reference']-abscent)@route)-half,0),axis=-1)
            bodyerr=np.array([reg.distance(h,p,'body')[0] for h,p in zip(hands,q)])
            # Required symmetric absolute Z half-extent at unchanged center and XY.
            # If XY already consumes >50 mm, NO finite Z extent can fix that hand.
            budget=np.sqrt(np.maximum(.05**2-xyerr**2,0))
            required=np.maximum(np.abs(hands[:,:,2]-center[:,2])-budget,0)
            valid=(xyerr<=.05).all(-1)
            # Exact signed decomposition of distance-to-box for the worst hand.
            # The last component includes the clipped nearest-box-point offset.
            nearest=abscent+np.clip(local,-half,half)@route.T
            vector=hands-nearest
            unit=vector/np.maximum(errors[:,:,None],1e-12)
            parts=rows['parts'].copy()
            parts[:,3]-=nearest-abscent
            projections=np.einsum('nphc,nhc->nph',parts,unit)
            np.testing.assert_allclose(projections.sum(1),errors,atol=2e-6)
            roots=a['roots'][:int(a['ends'][j])*10,j]
            relative=np.array([route.T@rotation(r) for r in roots])
            rpy=Rotation.from_matrix(relative).as_euler('xyz',degrees=True)
            row=dict(index=int(idx),mode=mode,frames=len(q),xy_irreparable_frames=int((~valid).sum()),
                     absolute_fk_good_frames=int((errors.max(-1)<=.05).sum()),
                     body_rescore_fk_good_frames=int((bodyerr.max(-1)<=.05).sum()),
                     no_pelvis_fk_good_frames=int((frozenerr.max(-1)<=.05).sum()),
                     abs_error_rms_m=float(np.sqrt(np.mean(errors**2))),
                     no_pelvis_error_rms_m=float(np.sqrt(np.mean(frozenerr**2))),
                     parts_rms_m=np.sqrt(np.mean(np.sum(rows['parts']**2,axis=-1),axis=(0,2))).tolist(),
                     root_rpy_min_deg=rpy.min(0).tolist(),root_rpy_max_deg=rpy.max(0).tolist(),
                     z_half_needed_on_xy_feasible_frames_m=float(required[valid].max()) if valid.any() else None,
                     z_half_for_vertical_only_m=float(max(0,np.abs(hands[:,:,2]-center[:,2]).max()-.05)),
                     hand_z_range_m=[float(hands[:,:,2].min()),float(hands[:,:,2].max())],
                     max_legacy_fk_hand_difference_m=float(np.linalg.norm(hands-rows['legacy_hands'],axis=-1).max()),
                     samples=[])
            for step in sorted(set(s for s in (10,25,50,len(q)) if s<=len(q))):
                k=step-1;h=int(errors[k].argmax())
                # Split pelvis orientation: tilt about fixed initial yaw, then yaw.
                R=rotation(q[k]);R0=reg.R0
                yaw=float(np.arctan2(R[1,0],R[0,0]));yaw0=float(np.arctan2(R0[1,0],R0[0,0]))
                rz=lambda v:np.array([[np.cos(v),-np.sin(v),0],[np.sin(v),np.cos(v),0],[0,0,1]])
                root_local=(rows['reference'][k]-q[k,:3])@R
                tilt=root_local@(rz(yaw0)@rz(-yaw)@R-R0).T
                yawpart=rows['orientation'][k]-tilt
                actual_local=(hands[k]-q[k,:3])@R
                no_height_tilt=actual_local@(rz(yaw)@rz(-yaw0)@R0).T+np.r_[q[k,:2],reg.q0[2]]
                no_height_tilt_error=reg.distance(no_height_tilt,q[k],'absolute')[0].max()
                row['samples'].append(dict(step=int(step),worst_hand=h,legacy_error_m=float(rows['error'][k]),
                    abs_fk_error_m=float(errors[k].max()),body_fk_error_m=float(bodyerr[k].max()),
                    no_pelvis_actual_error_m=float(frozenerr[k].max()),reference_endpoint_error_m=float(referr[k].max()),
                    body_displacement_m=float(np.linalg.norm(rows['parts'][k,1,h])),
                    height_change_m=float(rows['height'][k,h,2]),tilt_displacement_m=float(np.linalg.norm(tilt[h])),
                    yaw_displacement_m=float(np.linalg.norm(yawpart[h])),
                    height_tilt_displacement_m=float(np.linalg.norm(rows['height'][k,h]+tilt[h])),
                    signed_height_tilt_error_m=float((rows['height'][k,h]+tilt[h])@unit[k,h]),
                    signed_yaw_error_m=float(yawpart[h]@unit[k,h]),
                    no_height_tilt_actual_error_m=float(no_height_tilt_error),
                    tracking_displacement_m=float(np.linalg.norm(rows['parts'][k,0,h])),
                    signed_error_parts_m=projections[k,:,h].tolist(),
                    xyz_excess_m=excess[k,h].tolist(),xy_error_m=float(xyerr[k].max())))
            reduced.append(row)
            if mode=='absolute':
                row['z_rescores']=[]
                legacy_delta=(rows['legacy_hands']-abscent)@route
                for half_z in (.05, float('inf')):
                    enlarged=half.copy();enlarged[:,2]=half_z
                    e=np.linalg.norm(np.maximum(np.abs(legacy_delta)-enlarged,0),axis=-1).max(-1)
                    good=e<=.05;bad=np.flatnonzero(~good)
                    farthest=float((reg.q0[:2]-np.array(scene['route'][0]))@t)
                    distance=0.;start=zone['start_m']+zone['fade_m'];end=zone['end_m']-zone['fade_m']
                    for p,g in zip(rows['progress'],good):
                        advance=max(0.,min(float(p),end)-max(farthest,start))
                        if g:distance+=advance
                        farthest=max(farthest,float(p))
                    row['z_rescores'].append(dict(half_z_m=None if np.isinf(half_z) else half_z,
                        continuous_steps=int(bad[0]) if len(bad) else len(good),
                        compliant_fraction_full_core=distance/(end-start),
                        full_compliant_crossing=bool(rows['progress'][-1]>=end and good.all())))
                total_xy_bad+=int((~valid).sum());total_frames+=len(q)
                all_required.extend(required[valid].flatten().tolist())
    report=dict(part_order=['arm_tracking_vs_requested_IK','pelvis_height_and_orientation',
                            'waist_motion_at_initial_pelvis','requested_reference_at_initial_pelvis_and_waist_minus_nearest_box_point'],
                absolute_frames=total_frames,absolute_xy_irreparable_frames=total_xy_bad,
                absolute_z_half_needed_on_xy_feasible_frames_m=max(all_required),scenes=reduced)
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='scenes'},indent=2))
    for r in reduced:
        s=next(s for s in summary['episodes'] if s['index']==r['index'] and s['mode']==r['mode'])
        print(r['mode'],r['index'],'hold',s['continuous_steps'],'fraction',round(s['compliant_fraction_full_core'],4),
              'safety mm',round(s['safety']['body_lateral_actual_m']*1000,2),
              'xybad',r['xy_irreparable_frames'],'last',r['samples'][-1])


if __name__=='__main__':main()
