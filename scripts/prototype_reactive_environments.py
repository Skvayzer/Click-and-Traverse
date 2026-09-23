#!/usr/bin/env python3
"""Isolated geometry/kinematic review only. No policy, bank writes or simulation steps."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['MUJOCO_GL']='egl'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import sys,json,math,time,shutil,tempfile,atexit,xml.etree.ElementTree as ET
_cache=tempfile.mkdtemp(prefix='reactive-mpl-');os.environ['MPLCONFIGDIR']=_cache
atexit.register(shutil.rmtree,_cache,ignore_errors=True)
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
from PIL import Image,ImageDraw,ImageFont
from cat_mjlab.model import assemble_training_xml
from cat_ppo.furniture.hand_passages import arm_pose
OUT=ROOT/'docs/assets/reactive-prototypes-20260922'
RNG=np.random.default_rng(20260922)
DIRECTIONS=['front','front-left','left','back-left','back','back-right','right','front-right','above','below']

def guard():
    if shutil.disk_usage(ROOT).free<512*1024**2:raise RuntimeError('Less than 512 MiB free; stop')
    if OUT.exists() and sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>55*1024**2:raise RuntimeError('Output budget reached')

def obstacle(shape,center,size,**kw):return dict(shape=shape,center=list(center),size=list(size),**kw)

def position(o,t):
    if o.get('motion')=='swing':
        theta=o['amplitude']*math.sin(o['omega']*t+o['phase'])
        return np.array(o['pivot'])+np.array([0,o['length']*math.sin(theta),-o['length']*math.cos(theta)])
    return np.array(o['center'])+np.array(o.get('velocity',[0,0,0]))*t

def sdf(p,o,t=0):
    """Exact point SDF: axis-aligned box, sphere, capped vertical cylinder/rod."""
    q=np.asarray(p)-position(o,t);s=o['size']
    if o['shape']=='sphere':return np.linalg.norm(q,axis=-1)-s[0]
    if o['shape']=='box':
        d=np.abs(q)-s;return np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(np.max(d,axis=-1),0)
    d=np.stack([np.linalg.norm(q[...,:2],axis=-1)-s[0],np.abs(q[...,2])-s[1]],-1)
    return np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(np.max(d,axis=-1),0)

def normal(p,o,t=0):
    e=np.eye(3)*1e-5
    g=np.stack([(sdf(p+a,o,t)-sdf(p-a,o,t))/2e-5 for a in e],-1)
    return g/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-9)

def main():
    guard();OUT.mkdir(parents=True,exist_ok=True)
    model=mujoco.MjModel.from_xml_string(assemble_training_xml());data=mujoco.MjData(model)
    q=arm_pose();data.qpos[:]=q;mujoco.mj_forward(model,data)
    hands=np.array([data.site_xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f'{s}_palm')] for s in ['left','right']])
    radii=np.array([model.geom_size[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'furniture_{s}_hand_sphere'),0] for s in ['left','right']])
    skeleton=data.xpos.copy();parents=model.body_parentid.copy();scenes=[]
    for cls in ['E1','E2']:
        miss_ids=set(RNG.choice(120,36,replace=False).tolist())
        for i in range(120):
            di=i%10;d=np.array([math.cos(di*math.pi/4),math.sin(di*math.pi/4),0]) if di<8 else np.array([0,0,1 if di==8 else -1])
            shape=['sphere','box','cylinder','rod'][(i//10)%4];size={'sphere':[.075], 'box':[.10,.07,.09], 'cylinder':[.065,.18], 'rod':[.008,.22]}[shape]
            speed=float(RNG.uniform(.05,.5));distance=float(RNG.uniform(1,1.5));near=i in miss_ids
            target=['left','right','both'][(i//4)%3];ids=[0,1] if target=='both' else [0 if target=='left' else 1]
            mode='standing' if cls=='E1' else ['walking','turning','side-stepping'][(i//10)%3]
            # Solve a finite, open-loop intercept at fixed world speed from a 1–1.5 m spawn.
            from scipy.optimize import brentq
            vel=np.array([.4*speed,0,0]) if mode=='walking' else np.array([0,.4*speed,0]) if mode=='side-stepping' else np.zeros(3)
            yaw_rate=.10 if mode=='turning' else 0.
            def nominal_hand(h,t):
                yaw=yaw_rate*t
                rot=np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
                return rot@hands[h]+vel*t
            obs=[];times=[]
            for h in ids:
                b=np.array([0,0,1]) if di<8 else np.array([0,1 if h==0 else -1,0])
                gap=float(RNG.uniform(.25,.4)) if near else 0.
                support=(size[2] if di<8 else size[1]) if shape=='box' else size[1] if shape in ['rod','cylinder'] and di<8 else size[0]
                offset=b*(gap+radii[h]+support) if near else np.zeros(3)
                spawn=hands[h]+offset+d*math.sqrt(distance**2-float(offset@offset))
                hit_time=brentq(lambda t:np.linalg.norm(nominal_hand(h,t)+offset-spawn)-speed*t,.0001,100.)
                aim=nominal_hand(h,hit_time)
                velocity=(aim+offset-spawn)/hit_time
                o=obstacle(shape,spawn,size,velocity=velocity.tolist(),target_hand=h,intercept_time=hit_time)
                o['nominal_surface_clearance_m']=float(sdf(aim,o,hit_time)-radii[h])
                # Check both hands throughout the pass: a miss must miss the untargeted hand too.
                ts=np.linspace(0,hit_time+1,301)
                o['sampled_both_hand_min_clearance_m']=float(min(np.min(sdf(np.array([nominal_hand(k,t) for t in ts])-np.array([position(o,t)-position(o,0) for t in ts]),o)-radii[k]) for k in range(2)))
                obs.append(o);times.append(hit_time)
            hit_time=max(times)
            # Mark physically impossible underground approaches; retain requested coverage in proposal.
            underground=any(min(position(o,0)[2],position(o,hit_time)[2])-(size[2] if shape=='box' else size[1] if len(size)>1 else size[0])<0 for o in obs)
            scenes.append(dict(id=f'{cls}-{i:03}',cls=cls,direction=DIRECTIONS[di],speed=speed,spawn_distance=distance,near_miss=near,target=target,robot_mode=mode,robot_velocity=vel.tolist(),yaw_rate=yaw_rate,closest_time=hit_time,requires_below_floor=underground,objects=obs))
    for i in range(3):
        obs=[obstacle('box',[2.5,y,1],[3.5,.05,1],role='wall') for y in [-.85,.85]]
        x=-.1
        for j in range(10):
            x+=float(RNG.uniform(.35,.68));side=1 if j%2 else -1
            tip=[.20,.35,.65][(j+i)%3];depth=.85-tip;z=[.75,float(hands[0,2]),1.1][j%3]
            shape=['box','box','cylinder'][j%3]
            size=[.12,depth/2,.025] if shape=='box' else [.04,.18]
            center=[x,side*(.85-depth/2),z]
            # Horizontal wall pipe represented as a box-bound in schematic; actual cylinders rotate.
            o=obstacle(shape,center,size,role=['shelf edge','bracket','pipe'][j%3],tip_y_m=tip)
            if shape=='cylinder':o.update(horizontal_pipe=True,size=[.04,depth/2])
            obs.append(o)
        scenes.append(dict(id=f'E3-{i:03}',cls='E3',robot_mode='walking',robot_velocity=[.25,0,0],objects=obs))
    for i in range(3):
        o=obstacle(['box','sphere','rod'][i],[1.1,-1.1,float(hands[0,2])],[[.10,.1,.25],[.10],[.008,.35]][i],velocity=[0,.25+.10*i,0])
        if i==1:o.update(motion='swing',pivot=[1.1,0,2.1],length=1.25,amplitude=.75,omega=.45,phase=-math.pi/2)
        scenes.append(dict(id=f'E4-{i:03}',cls='E4',robot_mode='walking',robot_velocity=[.25,0,0],objects=[o]))
    validation={}
    for cls in ['E1','E2']:
        batch=[s for s in scenes if s['cls']==cls]
        obs=[o for s in batch for o in s['objects']]
        speeds=[float(np.linalg.norm(o['velocity'])) for o in obs]
        distances=[float(np.linalg.norm(np.array(o['center'])-hands[o['target_hand']])) for o in obs]
        misses=[o['sampled_both_hand_min_clearance_m'] for s in batch if s['near_miss'] for o in s['objects']]
        assert sum(s['near_miss'] for s in batch)==36
        assert min(speeds)>=.05-1e-8 and max(speeds)<=.5+1e-8
        assert min(distances)>=1-1e-8 and max(distances)<=1.5+1e-8
        assert min(misses)>=.25-1e-5 and max(misses)<=.4+1e-4
        validation[cls]=dict(objects=len(obs),speed_range_m_s=[min(speeds),max(speeds)],initial_distance_range_m=[min(distances),max(distances)],sampled_negative_both_hand_clearance_range_m=[min(misses),max(misses)],near_miss_episodes=36)
    (OUT/'geometry-validation.json').write_text(json.dumps(validation,indent=2))
    (OUT/'scenes.json').write_text(json.dumps(dict(schema='reactive-review-v1-NOT-A-BANK',seed=20260922,robot_qpos=q.tolist(),hand_centers=hands.tolist(),hand_radii=radii.tolist(),scenes=scenes),indent=2))
    def plot_scene(ax,s,side=False):
        axes=[0,2] if side else [0,1];t=s.get('closest_time',2)*.65
        for j,p in enumerate(parents):ax.plot(*skeleton[[p,j]][:,axes].T,c='#36576d',lw=2)
        ax.scatter(*hands[:,axes].T,c='#0c9c76',s=25,zorder=4)
        for o in s['objects']:
            c=position(o,0);end=position(o,t);sz=o['size'];color='#dd783c' if not s.get('near_miss') else '#24a486'
            half=np.array(sz if o['shape']=='box' else [sz[0],sz[0],sz[1] if len(sz)>1 else sz[0]])
            if o.get('horizontal_pipe'):half=half[[0,2,1]]
            if o.get('role')=='wall':color='#c5ccd3'
            for center,alpha in [(c,.8),(end,.25)] if 'velocity' in o or 'motion' in o else [(c,.8)]:
                ax.add_patch(Rectangle(center[axes]-half[axes],*(2*half[axes]),color=color,alpha=alpha))
            if 'velocity' in o or 'motion' in o:
                tt=np.linspace(0,s.get('closest_time',6),50);path=np.array([position(o,k) for k in tt]);ax.plot(*path[:,axes].T,'--',color=color,lw=1)
                ax.annotate('',end[axes],c[axes],arrowprops=dict(arrowstyle='->',color=color,lw=2))
        ax.annotate('start / facing +X',xy=(0,.05 if side else 0),xytext=(-.8,-.4),fontsize=8,arrowprops=dict(arrowstyle='->'))
        if s['cls'] in ['E2','E3','E4']:
            v=np.array(s['robot_velocity'])[axes];v=v/max(np.linalg.norm(v),1e-9)*.6
            ax.arrow(0,.05 if side else 0,*v,color='#267db5',head_width=.06)
            if s.get('yaw_rate'):ax.text(.2,.3,'turn CCW 0.10 rad/s',fontsize=8,color='#267db5')
        ax.set_aspect('equal');ax.grid(alpha=.15);ax.set_xlabel('X (m)');ax.set_ylabel('Z (m)' if side else 'Y (m)')
        ax.set_xlim((-1.6,2) if s['cls']!='E3' else (-.6,6.1));ax.set_ylim((-.5,2.7) if side else (-1.8,1.8))
        title=s['id']+' | '+s['robot_mode']
        if s['robot_mode']!='standing':title+=f" {np.linalg.norm(s['robot_velocity'])*100:.1f} cm/s"
        if 'speed'in s:title+=f" | {s['direction']} {s['speed']*100:.1f} cm/s\n{s['target']} | {'MISS' if s['near_miss'] else 'THREAT'} | {s['objects'][0]['shape']} size={s['objects'][0]['size']} m"
        elif s['cls']=='E3':title+=' | tips 0.20/0.35/0.65 m from center\nhip / hand / chest; irregular gaps; static'
        else:title+=' | '+('swing, peak 42 cm/s' if s['objects'][0].get('motion') else f"crossing {s['objects'][0]['velocity'][1]*100:.0f} cm/s")+'\nsize='+str(s['objects'][0]['size'])+' m'
        if s.get('requires_below_floor'):title+='\nINFEASIBLE BELOW-FLOOR SPAWN: revise distance'
        ax.set_title(title,fontsize=9)
    examples={c:[s for s in scenes if s['cls']==c] for c in ['E1','E2','E3','E4']}
    examples['E1']=[scenes[k] for k in [30,64,99]];examples['E2']=[scenes[120+k] for k in [37,48,52]]
    for cls,ss in examples.items():
        fig,axs=plt.subplots(3,2,figsize=(13,12))
        for row,s in enumerate(ss):
            for col in range(2):plot_scene(axs[row,col],s,bool(col))
        fig.suptitle(cls+' geometry review: TOP (left), SIDE (right)\nActual G1 start skeleton; scripted obstacles; not policy behaviour. Sizes: half-extents or radius/half-length.',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/f'{cls}-views.png',dpi=135);plt.close(fig)
    batch=scenes[:240];fig,axs=plt.subplots(2,2,figsize=(12,8))
    for j,cls in enumerate(['E1','E2']):
        ss=[s for s in batch if s['cls']==cls]
        for near,c in [(False,'#d47736'),(True,'#209e85')]:
            subset=[s for s in ss if s['near_miss']==near];axs[0,j].scatter([DIRECTIONS.index(s['direction']) for s in subset],[100*s['speed'] for s in subset],c=c,label='near miss' if near else 'threat',alpha=.7)
        axs[0,j].set(xticks=range(10),xticklabels=DIRECTIONS,ylabel='cm/s',title=f'{cls}: 120 proposals; {sum(s["near_miss"] for s in ss)} misses')
        axs[0,j].tick_params(axis='x',rotation=55);axs[0,j].legend()
        axs[1,j].hist([s['spawn_distance'] for s in ss],bins=10,color='#36576d');axs[1,j].set(xlabel='initial target-hand center distance (m)',ylabel='count',title='1.0–1.5 m initial distance, including misses')
    fig.suptitle('Seed 20260922 | 10 directions, 4 shapes, left/right/both\nBelow-floor cases are proposals requiring revised spawn rules, NOT feasible resets')
    fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(OUT/'coverage.png',dpi=130);plt.close(fig)
    # Render actual meshes, using noncontact visual geoms and kinematic robot poses.
    def render_sequence(s,video=False):
        if os.environ.get('PROTO_STILLS_ONLY'):video=False
        xml=ET.fromstring(assemble_training_xml());world=xml.find('worldbody')
        floor=world.find("geom[@name='floor']");floor.attrib.pop('material',None);floor.set('rgba','.90 .93 .95 1')
        for j,o in enumerate(s['objects']):
            attrs=dict(name=f'proto_{j}',type='cylinder' if o['shape']=='rod' else o['shape'],size=' '.join(map(str,o['size'])),pos=' '.join(map(str,position(o,0))),rgba='.87 .34 .12 1',contype='0',conaffinity='0')
            if o.get('role')=='wall':attrs['rgba']='.45 .55 .65 .18'
            if o.get('horizontal_pipe'):attrs['quat']='.7071068 .7071068 0 0'
            ET.SubElement(world,'geom',**attrs)
        m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));d=mujoco.MjData(m);m.vis.global_.offwidth=800;m.vis.global_.offheight=600
        cam=mujoco.MjvCamera();cam.lookat[:]=[.6 if s['cls']!='E3' else 2,0,.8];cam.distance=4.3 if s['cls']!='E3' else 7;cam.azimuth=125;cam.elevation=-24
        opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0
        writer=None
        if video:
            import imageio.v2 as imageio
            writer=imageio.get_writer(OUT/f"{s['cls']}-scripted.mp4",fps=10,codec='libx264',quality=6,macro_block_size=1)
        duration=min(s.get('closest_time',6)+1,8) if video else 0
        try:
            with mujoco.Renderer(m,width=800,height=600) as r:
                for frame,t in enumerate(np.linspace(0,duration,max(1,int(duration*10)))):
                    d.qpos[:]=q;d.qpos[:3]+=np.array(s.get('robot_velocity',[0,0,0]))*t
                    yaw=s.get('yaw_rate',0)*t;d.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
                    for j,o in enumerate(s['objects']):m.geom_pos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f'proto_{j}')]=position(o,t)
                    mujoco.mj_forward(m,d);r.update_scene(d,camera=cam,scene_option=opt);im=Image.fromarray(r.render())
                    draw=ImageDraw.Draw(im);draw.rectangle((0,0,800,74),fill='white');draw.text((15,10),f"{s['id']} | t={t:.2f}s | {s['robot_mode']} | SCRIPTED, NO POLICY",fill='black')
                    desc=(f"{s['direction']} | object {s['speed']:.2f} m/s" if 'direction' in s else 'STATIC obstacles | robot 0.25 m/s' if s['cls']=='E3' else 'CROSSING obstacle 0.25 m/s | robot 0.25 m/s')+' | start (0,0), facing +X'
                    draw.text((15,31),desc,fill='black');draw.text((15,51),'Illustrative geometry; ghost intersections are not learned avoidance or physical contacts.',fill='black')
                    if frame==0:im.save(OUT/f"{s['cls']}-mesh.png")
                    if writer:writer.append_data(np.asarray(im))
                    if frame%30==0:guard()
        finally:
            if writer:writer.close()
    for cls in examples:
        s=examples[cls][0]
        if cls=='E1':s=next(s for s in scenes if s['cls']=='E1' and s['direction']=='left' and not s['near_miss'] and s['speed']>.3)
        if not os.environ.get('PROTO_SKIP_MESH'):render_sequence(s,cls in ['E1','E4'])
    # CPU throughput, exact analytic sphere union; no GPU initialization.
    import torch
    torch.set_num_threads(2);B,Q,M=30000,11,4
    p=torch.rand(B,Q,3);c=torch.rand(B,M,3);rad=torch.full((B,M),.075)
    def query():
        delta=p[:,:,None]-c[:,None];length=torch.linalg.vector_norm(delta,dim=-1);dist=length-rad[:,None];v,idx=dist.min(-1);n=delta/length.clamp_min(1e-9)[...,None];return v,torch.gather(n,2,idx[:,:,None,None].expand(-1,-1,1,3)).squeeze(2)
    for _ in range(3):query()
    timings=[]
    for _ in range(10):
        start=time.perf_counter();query();timings.append((time.perf_counter()-start)*1000)
    # Analytic moving-object visibility and thin rod checks, isolated from production.
    rod=obstacle('rod',[0,0,1],[.008,.22],velocity=[0,.2,0]);probe=np.array([[0,.3,1]])
    checks=dict(rod_clearance_t0=float(sdf(probe,rod,0)[0]),rod_clearance_t1=float(sdf(probe,rod,1)[0]),rod_normal_t0=normal(probe,rod,0).tolist())
    assert np.isclose(checks['rod_clearance_t0'],.292) and np.isclose(checks['rod_clearance_t1'],.092)
    for o,pt,expected in [(obstacle('box',[0,0,0],[.1,.2,.3]),[.4,0,0],.3),(obstacle('cylinder',[0,0,0],[.1,.2]),[0,0,.5],.3),(obstacle('sphere',[0,0,0],[.1]),[.4,0,0],.3)]:assert np.isclose(sdf(pt,o),expected)
    report=dict(scene_count=len(scenes),hand_centers_m=hands.tolist(),hand_radii_m=radii.tolist(),below_floor_proposals=sum(s.get('requires_below_floor',False) for s in batch),near_misses=sum(s['near_miss'] for s in batch),cpu_benchmark=dict(B=B,Q=Q,M=M,threads=2,repetitions=10,median_ms=float(np.median(timings)),min_ms=min(timings),max_ms=max(timings),scope='Torch eager CPU sphere distances plus nearest normals; excludes static fields, physics, GPU and compilation'),analytic_checks=checks,output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()))
    (OUT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));guard()
if __name__=='__main__':main()
