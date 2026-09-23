#!/usr/bin/env python3
"""Build compact tables/plots from the actual frozen-policy CPU probe traces."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cat_ppo.furniture.grippers import hand_sphere
out=ROOT/'outputs/reactive_clearance_cpu';report=json.loads((out/'results.json').read_text())
radius=hand_sphere('left')['radius'];sphere=report['sphere_radius_m'];directions=['side','front','below','above']
fig,axes=plt.subplots(4,2,figsize=(12,12),sharex=True)
valid_old=[];valid_new=[]
for mode in ['standing','walking']:
    data=np.load(out/(mode+'.npz'));hands=data['hands'];centers=data['centers'];q=data['qpos']
    for j,c in enumerate(c for c in report['conditions'] if c['mode']==mode):
        a,b=2*j,2*j+1;end=100+round(c['duration_s']/.02);sl=slice(100,end)
        d=np.linalg.norm(hands[sl,a]-centers[sl,a,None],axis=-1)-radius-sphere
        early=np.clip((.20-d)/.20,0,1)**2;near=np.clip((.04-d)/.04,0,1)**2
        old=-.5*(.8*near+.2*early).mean(-1)*.02
        new=-20*(.5*near+.5*early).mean(-1)*.02
        c['original_clearance_reward_mean_per_step']=float(old.mean())
        c['recommended_clearance_reward_mean_per_step']=float(new.mean())
        valid_old.extend(old);valid_new.extend(new)
        vec=np.array({'side':[0,1,0],'front':[1,0,0],'below':[0,0,-1],'above':[0,0,1]}[c['direction']])
        retreat=-((hands[sl,a,0]-hands[sl,b,0])@vec)*100
        root_relative=-(((hands[sl,a,0]-q[sl,a,:3])-(hands[sl,b,0]-q[sl,b,:3]))@vec)*100
        c['arm_relative_final_cm']=float(root_relative[-1])
        ax=axes[directions.index(c['direction']),int(mode=='walking')]
        times=np.arange(len(d))*.02
        line=ax.plot(times,root_relative,label=f"{c['speed_m_s']*100:g} cm/s")[0]
        ax.plot(times,d[:,0]*100,':',color=line.get_color(),alpha=.7)
        ax.axhline(0,color='black',lw=.5);ax.axhline(4,color='red',lw=.5,alpha=.5)
        ax.set_title(mode+' / '+c['direction']);ax.set_ylabel('cm');ax.grid(alpha=.15)
        ax.legend(fontsize=7)
fig.suptitle('Measured baseline: solid = root-relative hand retreat vs matched control\nDotted = hand–sphere clearance; positive retreat is away; stop at contact/termination')
for ax in axes[-1]:ax.set_xlabel('Seconds after approach starts')
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/'response_traces.png',dpi=120);plt.close(fig)
report['initial_surface_gap_m']=.55
report['valid_treated_clearance_reward_mean_per_step']=dict(original=float(np.mean(valid_old)),recommended=float(np.mean(valid_new)))
report['reward_ledger_scope']='Raw weighted reward/second averaged across all worlds including controls and post-terminal resets. Use valid_treated_clearance_reward_mean_per_step for censored clearance comparisons.'
report['onset_definition']='First of five consecutive 20 ms samples with >1 cm world hand retreat versus matched control; not necessarily hand-channel-specific, since the sphere enters all body query fields.'
report['upright_definition']='Root height > .50 m and pelvis local up dot world up > .70 throughout measured trial. Root-relative retreat subtracts translation, not torso rotation.'
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['| Mode | From | Speed cm/s | Peak retreat cm | Root-relative peak cm | Onset gap cm | Min gap cm | <20 cm % | <4 cm % | Upright | Contact |',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
for c in report['conditions']:
    onset='none' if c['onset_distance_cm'] is None else f"{c['onset_distance_cm']:.1f}"
    lines.append(f"| {c['mode']} | {c['direction']} | {c['speed_m_s']*100:g} | {c['retreat_peak_cm']:.1f} | {c['arm_relative_peak_cm']:.1f} | {onset} | {c['min_clearance_cm']:.2f} | {c['inside_20cm_fraction']*100:.1f} | {c['below_4cm_fraction']*100:.1f} | {'yes' if c['stayed_standing'] else 'NO'} | {'yes' if c['contact'] else 'no'} |")
(out/'table.md').write_text('\n'.join(lines)+'\n')
print(report['valid_treated_clearance_reward_mean_per_step'])

# Render measured configurations with a CPU skeletal view; no OpenGL/GPU needed.
import mujoco
from cat_mjlab.model import assemble_training_xml
model=mujoco.MjModel.from_xml_string(assemble_training_xml());mjdata=mujoco.MjData(model)
for mode in ['standing','walking']:
    data=np.load(out/(mode+'.npz'));fig,axes=plt.subplots(4,4,figsize=(12,10),subplot_kw={'projection':'3d'})
    cases=[c for c in report['conditions'] if c['mode']==mode and c['speed_m_s']==.1]
    for row,c in enumerate(cases):
        a=2*(row*3+1);last=99+round(c['duration_s']/.02)
        for col,k in enumerate(np.linspace(100,last,4).astype(int)):
            ax=axes[row,col];mjdata.qpos[:]=data['qpos'][k,a];mujoco.mj_forward(model,mjdata)
            p=mjdata.xpos;root=data['qpos'][k,a,:3]
            for child,parent in enumerate(model.body_parentid):
                if parent:ax.plot(*p[[parent,child]].T,color='steelblue',lw=2)
            ax.scatter(*mjdata.site_xpos[model.site('head').id],c='steelblue',s=100)
            hand=data['hands'][k,a,0];center=data['centers'][k,a]
            ax.scatter(*hand,c='green',s=40);ax.scatter(*data['hands'][k,a+1,0],c='gray',marker='x',s=40)
            u,w=np.mgrid[0:2*np.pi:12j,0:np.pi:8j]
            ax.plot_surface(center[0]+sphere*np.cos(u)*np.sin(w),center[1]+sphere*np.sin(u)*np.sin(w),center[2]+sphere*np.cos(w),color='red',alpha=.5)
            ax.set(xlim=(root[0]-.7,root[0]+.7),ylim=(root[1]-.7,root[1]+.7),zlim=(0,1.45),title=f"{c['direction']}  {(k-100)*.02:.2f}s")
            ax.view_init(15,125);ax.set_box_aspect((1,1,1.1));ax.set_axis_off()
    fig.suptitle(f'{mode}: measured 10 cm/s sphere approach\nCPU skeletal reconstruction; green = hand, gray cross = paired control; last frame = trial endpoint')
    fig.tight_layout();fig.savefig(out/(mode+'_strip.png'),dpi=120);plt.close(fig)
