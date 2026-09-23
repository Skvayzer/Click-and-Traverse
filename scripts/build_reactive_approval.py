#!/usr/bin/env python3
"""Build/certify/render opt-in standing scenes. Never starts a learner."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('EGL_PLATFORM','surfaceless');os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
os.environ.setdefault('LP_NUM_THREADS','2')
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,json,sys,shutil,time,math,copy,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'tests')]
import numpy as np
import torch
import mujoco
from types import SimpleNamespace
from cat_mjlab.reactive import BodyEnvelope,StandingObjects,digest,validate_bank
from cat_mjlab.analytic_objects import primitive_distance_normal
from cat_mjlab.collision import PROPOSAL
from reactive_tuck_geometry import Geometry,DIRECTIONS,BUCKETS
OUT=ROOT/'docs/assets/reactive-approval-20260922'
BANK=ROOT/'data/furniture/reactive_standing_approval_20260922/manifest.json'

def disk(stage,out=OUT):
    free=shutil.disk_usage(ROOT).free
    total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) if out.exists() else 0
    print(f'{stage}: free={free/2**30:.3f} GiB; review output={total/2**20:.3f} MiB',flush=True)
    if free<2*2**30 or total>480*2**20:raise RuntimeError('Disk reserve/output ceiling reached; stop')
    return dict(stage=stage,free_bytes=free,review_bytes=total)

def tensor_data(data,device):
    return SimpleNamespace(xpos=torch.tensor(data.xpos[None],device=device,dtype=torch.float32),xmat=torch.tensor(data.xmat[None],device=device,dtype=torch.float32))

def point_sdf(points,center,rotation,size,kind,device):
    t=lambda x:torch.as_tensor(np.asarray(x),device=device,dtype=torch.float32)
    return primitive_distance_normal(t(points)[None],t(center)[None,None],t(rotation)[None,None],t(size)[None,None],torch.tensor([[kind]],device=device))[0][0,:,0].cpu().numpy()

def geometry_arrays(geo,envelope,device):
    data=tensor_data(geo.data,device)
    centers=envelope.spheres(data,torch.tensor([0],device=device))[0].cpu().numpy()
    radii=envelope.radii.cpu().numpy()
    hand_ids=[geo.model.geom(f'furniture_{side}_hand_sphere').id for side in ('left','right')]
    hands=geo.data.geom_xpos[hand_ids].copy()
    return centers,radii,hands

def sampled_path(start,end,centers,radii,rotation,size,kind,device,n=161):
    t=lambda x:torch.as_tensor(np.asarray(x),device=device,dtype=torch.float32)
    positions=np.linspace(start,end,n);min_gap=float('inf')
    for chunk in np.array_split(positions,max(1,math.ceil(n/64))):
        b=len(chunk);d,_=primitive_distance_normal(t(centers)[None].expand(b,-1,-1),t(chunk)[:,None],t(rotation)[None,None].expand(b,1,3,3),t(size)[None,None].expand(b,1,3),torch.full((b,1),kind,device=device,dtype=torch.long))
        min_gap=min(min_gap,float((d[:,:,0]-t(radii)[None]).min()))
    # SDF is 1-Lipschitz under translation. Every point on the full path is
    # <= half a sample interval from a checked point. Retreat retraces it.
    error=float(np.linalg.norm(end-start)/(n-1)/2)
    return dict(sampled_min_body_gap_m=min_gap,continuous_body_lower_bound_m=min_gap-error,sample_spacing_bound_m=error,samples=n)

def build(args):
    from scipy.optimize import brentq
    disk('before build',args.output);args.output.mkdir(parents=True,exist_ok=True)
    geo=Geometry();env=BodyEnvelope(geo.model,args.device);rng=np.random.default_rng(20260922)
    rows=[];certs=[]
    nominal=geo.q.copy()
    # Put the foot soles on the floor, without changing joint angles.
    foot_geom=[i for i in range(geo.model.ngeom) if 'foot' in (geo.model.geom(i).name or '') and geo.model.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX]
    zmin=min(geo.data.geom_xpos[i,2]-np.sum(np.abs(geo.data.geom_xmat[i].reshape(3,3)[2])*geo.model.geom_size[i]) for i in foot_geom)
    nominal[2]-=zmin
    shape_sizes={'sphere':(.045,.045,.045),'box':(.045,.03,.025),'cylinder':(.035,.065,.035),'rod':(.008,.10,.008)}
    for di,(direction,vector) in enumerate(DIRECTIONS.items()):
      for bi,(bucket,(lo,hi)) in enumerate(BUCKETS.items()):
        gap=float(rng.uniform(lo+.003,hi-.003));shape=list(shape_sizes)[(di+bi)%4];kind={'sphere':0,'box':1,'cylinder':2,'rod':2}[shape]
        size=np.array(shape_sizes[shape]);rotation=np.eye(3)
        if direction in ('above','below') and kind==2:rotation=np.array([[0,0,1],[0,1,0],[-1,0,0.]])
        support=float(size[0] if kind==0 else np.abs(rotation[2])@size if kind==1 else size[0]*np.linalg.norm(rotation[2,:2])+size[1]*abs(rotation[2,2]))
        # Lateral approaches always target the near hand, never cross the torso.
        target=0 if vector[1]>0 else 1 if vector[1]<0 else bi%2
        both=direction in ('front','back','below') and bi==2
        target_ids=[0,1] if both else [target]
        found=None
        for attempt in range(161):
            q=nominal.copy()
            if attempt:
                # Author a feasible standing start posture, not an evasive controller.
                # Base/legs/waist stay fixed; arms remain within real +/-0.8 action reach.
                for side,sign,offset in [('left',1,22),('right',-1,29)]:
                    change=rng.uniform(-.65,.65,7);change[1]=sign*rng.uniform(.05,.65)
                    q[offset:offset+7]=np.clip(q[offset:offset+7]+change,geo.lo[offset-19:offset-12],geo.hi[offset-19:offset-12])
            geo.q=q;geo.set(q[19:]);self_gap=float(geo.self_distances().min())
            if self_gap<1e-6:continue
            centers,radii,hands=geometry_arrays(geo,env,args.device)
            starts=[];ends=[];checks=[];ok=True
            for h in target_ids:
                ray=vector.copy()
                # Front diagonals approach outside the near arm, not through the hips.
                if direction in ('front_right','front_left'):
                    ray=np.array([.45,1. if h==0 else -1.,0.]);ray/=np.linalg.norm(ray)
                def f(distance):return float(point_sdf(hands[h:h+1],hands[h]+ray*distance,rotation,size,kind,args.device)[0]-.10344617-gap)
                distance=brentq(f,.001,2.,xtol=1e-7)
                end=hands[h]+ray*distance;start=end+ray*.30
                if direction=='below':
                    start=end.copy();start[2]=support+.003
                if min(start[2],end[2])-support<.002 or np.linalg.norm(end-start)<.025:ok=False;break
                check=sampled_path(start,end,centers,radii,rotation,size,kind,args.device,n=81)
                if check['continuous_body_lower_bound_m']<.004:ok=False;break
                starts.append(start);ends.append(end);checks.append(check)
            if not ok:continue
            if both:
                # Circumsphere separation is a conservative object/object certificate.
                bound=np.linalg.norm(size) if kind==1 else size[0] if kind==0 else np.linalg.norm(size[:2])
                relative=np.array(starts[0])-starts[1];delta=(np.array(ends[0])-ends[1])-relative
                u=np.clip(-relative@delta/max(delta@delta,1e-20),0,1)
                if np.linalg.norm(relative+u*delta)<2*bound+.005:continue
            found=(q,centers,radii,hands,starts,ends,self_gap,attempt);break
        if found is None:raise RuntimeError(f'No certified authoring for {direction}/{bucket}; keep completed videos, do not admit scene')
        q,centers,radii,hands,starts,ends,self_gap,attempt=found
        speed=float(rng.uniform(.05,.5))
        if di==0 and bi==0:speed=.05
        if di==9 and bi==2:speed=.5
        duration=max(np.linalg.norm(b-a)/speed for a,b in zip(starts,ends));hold=.65
        valid=[True]*len(starts)+[False]*(2-len(starts))
        row=dict(id=f'{direction}-{bucket}',direction=direction,bucket=bucket,target='both' if both else ('left' if target==0 else 'right'),target_hands=target_ids,
            shape=shape,qpos=q.tolist(),start=[x.tolist() for x in starts]+[[10.,10.,10.]]*(2-len(starts)),end=[x.tolist() for x in ends]+[[10.,10.,10.]]*(2-len(ends)),
            rotations=[rotation.tolist()]*2,sizes=[size.tolist()]*2,kinds=[kind]*2,valid=valid,speed=[speed]*2,hold=[hold]*2,clearance=[gap]*2,
            duration_s=2*duration+hold+.25,arrival_s=duration,sampling_weight=.1 if direction=='above' else 1.,flag='elevated fall risk; review policy probe' if direction=='above' else '',
            path_note='near-hand outward horizontal diagonal' if direction in ('front_right','front_left') else 'floor-level rise' if direction=='below' else 'near-hand radial approach')
        certificate=[sampled_path(a,b,centers,radii,rotation,size,kind,args.device,n=1001) for a,b in zip(starts,ends)]
        achieved=[float(point_sdf(hands[h:h+1],end,rotation,size,kind,args.device)[0]-.10344617) for h,end in zip(target_ids,ends)]
        cert=dict(id=row['id'],target_surface_m=gap,achieved_surface_m=achieved,object_paths=certificate,min_floor_clearance_m=float(min(min(a[2],b[2])-support for a,b in zip(starts,ends))),
            initial_self_proxy_gap_m=self_gap,robot_nonfoot_floor_gap_m=float(geo.floor_distances().min()),robot_foot_sole_floor_m=0.,joint_action_limits_passed=bool(np.all(q[19:]>=geo.lo-1e-7) and np.all(q[19:]<=geo.hi+1e-7)),authoring_attempts=attempt,passed=all(x['continuous_body_lower_bound_m']>0 for x in certificate),
            scope='full approach/hold/retreat against stationary authored robot and all conservative covering proxies; not a no-contact promise for arbitrary robot motion')
        rows.append(row);certs.append(cert)
        print(f"{row['id']}: {row['target']} {shape}, target={gap:.5f}, achieved={achieved}, fullbody_bound={min(x['continuous_body_lower_bound_m'] for x in certificate):.5f}, pose_attempt={attempt}",flush=True)
    base=ROOT/'data/furniture/cat_flat_hand_balance_v2_20260921/manifest.json'
    pin=lambda p:dict(path=str(p.resolve()),sha256=digest(p))
    base_meta=json.loads(base.read_text());background=next(i for i,r in enumerate(base_meta['scenes']) if r.get('source',{}).get('flat_balance'))
    manifest=dict(schema='cat-reactive-standing-v1',reactive_mass=.25,retained_mass=.75,base_bank=pin(base),collision_bank=pin(base.parent.with_name(base.parent.name+'_collision')/'manifest.json'),reset_bank=pin(base.parent.with_name(base.parent.name+'_resets')/'manifest.json'),proxy=pin(PROPOSAL),
        background_scene_index=background,episode_steps=math.ceil(max(r['duration_s'] for r in rows)/.02)+50,scenes=rows,
        retained_scene_count=len(base_meta['scenes']),sampling='75% unchanged base sampler; 25% weighted reactive sampler; above weight 0.1 versus 1 for other directions',
        above_share_of_reactive=.3/27.3,above_share_total=.25*.3/27.3,approval_status='AWAITING OWNER VIDEO REVIEW; NO TRAINING AUTHORIZED')
    args.bank.parent.mkdir(parents=True,exist_ok=True);args.bank.write_text(json.dumps(manifest,indent=2)+'\n');validate_bank(manifest)
    (args.output/'certification.json').write_text(json.dumps(dict(scenes=certs,all_passed=all(r['passed'] for r in certs)),indent=2)+'\n')
    disk('bank and full-path certification complete',args.output)


def font(size):
    from PIL import ImageFont
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def render(args):
    from PIL import Image,ImageDraw
    import imageio.v2 as imageio
    from cat_mjlab.model import assemble_training_xml
    disk('before rendering',args.output)
    meta=validate_bank(json.loads(args.bank.read_text()));certs={r['id']:r for r in json.loads((args.output/'certification.json').read_text())['scenes']}
    xml=ET.fromstring(assemble_training_xml());world=xml.find('worldbody')
    floor=world.find("geom[@name='floor']");floor.attrib.pop('material',None);floor.set('rgba','.88 .90 .92 1')
    model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));model.vis.global_.offwidth=720;model.vis.global_.offheight=540
    model.vis.quality.shadowsize=0;model.vis.quality.offsamples=0
    model.vis.headlight.ambient[:]=.5;model.vis.headlight.diffuse[:]=.7
    data=mujoco.MjData(model);opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0
    cam=mujoco.MjvCamera();summaries=[]
    if args.only and (args.output/'render-audit.json').exists():
        summaries=[r for r in json.loads((args.output/'render-audit.json').read_text()) if r['id'] not in args.only.split(',')]
    rows=meta['scenes']
    if args.only:rows=[r for r in rows if r['id'] in args.only.split(',')]
    with mujoco.Renderer(model,width=720,height=540) as renderer:
      for row in rows:
        data.qpos[:]=row['qpos'];mujoco.mj_forward(model,data)
        hands=np.array([data.site_xpos[model.site(f'{side}_palm').id] for side in ('left','right')]);h=row['target_hands'][0]
        # Close framing follows the target/object midpoint. Ground remains visible below.
        side=1 if h==0 else -1;v=DIRECTIONS[row['direction']]
        cam.azimuth=(65 if side>0 else -65) if v[0]>=0 else (120 if side>0 else -120)
        cam.elevation=-12 if row['direction']=='below' else -20
        cam.distance=1.55 if row['target']!='both' else 2.25
        objects=StandingObjects(bank=args.bank,num_envs=1,model=model,device=args.device)
        objects.force_rows=torch.tensor([meta['scenes'].index(row)],device=args.device)
        ids=torch.tensor([0],device=args.device);objects.choose(ids,torch.zeros((1,2),device=args.device),torch.tensor([meta['background_scene_index']],device=args.device))
        static=tensor_data(data,args.device)
        # Run the actual closed-loop object controller at 2 ms, record at 10 fps.
        # No robot physics steps or robot state rewinds occur in this geometry preview.
        frames=[];minimum=float('inf');closest=0;minbody=float('inf');guardlower=float('inf');prev=objects.state['position'].clone()
        steps=math.ceil(row['duration_s']/.002)
        for step in range(steps+1):
            if step:
                _,_,gaps=objects.advance(static,.002)
                motion=torch.linalg.vector_norm(objects.state['position']-prev,dim=-1).max().item();prev=objects.state['position'].clone()
                minbody=min(minbody,float(gaps.min()));guardlower=min(guardlower,float(gaps.min())-motion)
            if step%1000==0:print(f"{row['id']}: guard {step}/{steps}",flush=True)
            if step%50 and step!=steps:continue
            pos=objects.state['position'][0].cpu().numpy().copy();ds=[]
            for j,hand_id in enumerate(row['target_hands']):ds.append(float(point_sdf(hands[hand_id:hand_id+1],pos[j],np.array(row['rotations'][j]),np.array(row['sizes'][j]),row['kinds'][j],args.device)[0]-.10344617))
            value=min(ds)
            if value<minimum-1e-7:minimum=value;closest=len(frames)
            frames.append(dict(t=step*.002,position=pos,distances=ds))
        if abs(minimum-row['clearance'][0])>.0015:raise RuntimeError(f"Guard preview missed target {row['id']}: {minimum} vs {row['clearance'][0]}")
        if guardlower<=0:raise RuntimeError(f'Object guard segment not certified: {row["id"]}')
        filename=row['id']+'.mp4';writer=imageio.get_writer(args.output/filename,fps=10,codec='libx264',ffmpeg_params=['-crf','27','-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart'],macro_block_size=1,ffmpeg_log_level='error',output_params=['-threads','1'])
        try:
          for fi,frame in enumerate(frames):
            focus=(hands[row['target_hands']].mean(0)+frame['position'][:len(row['target_hands'])].mean(0))/2
            cam.lookat[:]=focus;renderer.update_scene(data,camera=cam,scene_option=opt)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
            for j in range(len(row['target_hands'])):
                geom=renderer.scene.geoms[renderer.scene.ngeom];kind=row['kinds'][j];size=np.array(row['sizes'][j]);mjkind=[mujoco.mjtGeom.mjGEOM_SPHERE,mujoco.mjtGeom.mjGEOM_BOX,mujoco.mjtGeom.mjGEOM_CYLINDER][kind]
                mujoco.mjv_initGeom(geom,mjkind,size,frame['position'][j],np.array(row['rotations'][j]).ravel(),np.array([.94,.28,.06,1.]));renderer.scene.ngeom+=1
                # Outline the actual enclosing sphere as three fine wire rings.
                center=hands[row['target_hands'][j]]
                for plane in (0,1,2):
                  ring=[]
                  for theta in np.linspace(0,2*np.pi,33):
                    xyz=center.copy();axes=[k for k in range(3) if k!=plane];xyz[axes]+=.10344617*np.array([np.cos(theta),np.sin(theta)]);ring.append(xyz)
                  for a,b in zip(ring,ring[1:]):
                    geom=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.02,.55,.63,.5]));mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.0009,a,b);renderer.scene.ngeom+=1
            im=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,720,102),fill='#122330');draw.text((12,7),f"{row['direction'].upper()} / {row['bucket']} / {row['shape']} / {row['target']}",font=font(19),fill='white')
            draw.text((12,35),'SURFACE GAP  '+ ' / '.join(f'{d*1000:.1f} mm' for d in frame['distances']),font=font(25),fill='#72f0ce')
            draw.text((12,72),f"Target {row['clearance'][0]*1000:.1f} mm | min {minimum*1000:.1f} mm | {row['speed'][0]*100:.1f} cm/s",font=font(16),fill='white')
            draw.rectangle((0,482,720,540),fill='#122330');draw.text((12,486),f"t={frame['t']:.2f}s | STATIC POSE / OBJECT GUARD | no policy",font=font(16),fill='white')
            draw.text((12,512),'Cyan: hand enclosing sphere (103.45 mm radius)',font=font(15),fill='#72dbea')
            if fi==closest:
                draw.rectangle((340,439,708,477),fill='#f7ca4c');draw.text((351,445),'CLOSEST APPROACH',font=font(22),fill='black');im.save(args.output/(row['id']+'.jpg'),quality=85)
                # Brief visual hold; physical timestamp is unchanged and explicit.
                for _ in range(5):writer.append_data(np.asarray(im))
            writer.append_data(np.asarray(im))
            if fi%50==0:print(f"{row['id']}: frame {fi}/{len(frames)}",flush=True)
        finally:writer.close()
        summaries.append(dict(id=row['id'],video=filename,closest_frame=closest,minimum_surface_m=minimum,target_m=row['clearance'][0],fullbody_min_sampled_m=minbody,guard_segment_lower_bound_m=guardlower,frames=len(frames),duration_s=row['duration_s']))
        print(f"rendered {filename}: min={minimum:.6f}, certified guard lower={guardlower:.6f}",flush=True);disk('render '+row['id'],args.output)
    (args.output/'render-audit.json').write_text(json.dumps(summaries,indent=2)+'\n')
    make_index(args,meta,certs,summaries)
    disk('rendering complete',args.output)


def make_index(args,meta,certs,summaries):
    import html
    out=['<!doctype html><html><head><meta charset="utf-8"><title>Standing approach approval</title><style>body{font:16px system-ui;background:#101d28;color:#eaf3f7;margin:24px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:22px}video{width:100%}article{background:#1e3240;padding:14px;border-radius:10px}a{color:#72dbea}p{line-height:1.5}strong{color:#72f0ce}</style></head><body><h1>Standing approaches — owner review</h1><p>30 actual-mesh geometry previews, not learned avoidance. Cyan wire = 103.45 mm hand sphere. Numeric overlay is exact primitive-to-hand-sphere surface clearance. Closest frame is marked and held for 0.5 s. Object controller runs at 2 ms; robot pose is static. No training launched.</p><p>Full approach/hold/retreat certified against all conservative body envelopes and floor; this does not promise no contact if a robot moves into the object. The object-side guard never rewinds or rejects robot physics. Above is flagged: prior unguarded scripted tuck fell 4/8; see policy-probe.json for the new guarded frozen-policy audit when present.</p><p>75% pinned existing tasks / 25% reactive. Equal bucket masses; above only 1.10% of reactive resets (0.275% overall). Below rises from floor level. Right-side directions target right hand; front diagonals re-path outside the near hip.</p><p><a href="below-danger.mp4">Corrected below approach</a> · <a href="certification.json">Full-path certificates</a> · <a href="render-audit.json">Measured minima</a> · <a href="policy-probe.json">Frozen-policy probe</a> · <a href="preflight.json">Preflight</a></p><div class="grid">']
    for r in summaries:
        c=certs[r['id']];bound=min(x['continuous_body_lower_bound_m'] for x in c['object_paths'])
        out.append(f'<article><h2>{html.escape(r["id"])}</h2><video controls preload="none" poster="{r["id"]}.jpg" src="{r["video"]}"></video><p><strong>Target {r["target_m"]*1000:.1f} mm → measured {r["minimum_surface_m"]*1000:.1f} mm</strong><br>Full-path body lower bound {bound*1000:.2f} mm; floor minimum {c["min_floor_clearance_m"]*1000:.2f} mm. Certificate PASS.</p></article>')
    out.append('</div></body></html>');(args.output/'index.html').write_text('\n'.join(out))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');p.add_argument('--mode',choices=['build','render','all'],default='all');p.add_argument('--output',type=Path,default=OUT);p.add_argument('--bank',type=Path,default=BANK);p.add_argument('--only',default='')
    args=p.parse_args();torch.set_num_threads(2)
    if args.mode in ('build','all'):build(args)
    if args.mode in ('render','all'):render(args)
if __name__=='__main__':main()
