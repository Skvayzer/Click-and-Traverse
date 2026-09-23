"""CPU FK, stored-field and actual CATTask reward instrumentation; no training."""
from __future__ import annotations
import math
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from cat_mjlab.model import assemble_training_xml
from cat_mjlab import constants as const, task_math as tm
from cat_mjlab.task import CATTask
from cat_mjlab.fields import sample_ragged_field
from cat_mjlab.navigation import route_context,swept_root_clearance,hand_contrast_context,contrast_reward_terms
from cat_mjlab.collision import compile_proposal,transform_primitives,box_box_separation,capsule_box_separation,sphere_box_separation
from cat_ppo.furniture.room_navigation import pack_room_scenes
from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
from cat_ppo.furniture.grippers import hand_sphere

T=lambda x:torch.as_tensor(np.asarray(x).copy(),dtype=torch.float32)

class Probe:
    def __init__(self,config,proposal):
        self.config=config;self.model=mujoco.MjModel.from_xml_string(assemble_training_xml())
        self.compiled=compile_proposal(proposal,self.model,'cpu')
        self.radii=np.array([hand_sphere(s)['radius'] for s in ('left','right')])
    def qpose(self,mode='nominal',fraction=.95):
        q=np.array(const.DEFAULT_QPOS).copy();q[:2]=0
        if mode!='nominal':
            for side,sign in [('left',1),('right',-1)]:
                changes={'shoulder_pitch':-.8,'shoulder_roll':-.15*sign,'elbow':-.8} if mode=='raised' else {'shoulder_pitch':-.3,'shoulder_roll':-.25*sign,'elbow':.7}
                for joint,delta in changes.items():q[self.model.joint(f'{side}_{joint}_joint').qposadr[0]]+=fraction*delta
        return q
    def bank(self,record,directory,scene):
        b=SimpleNamespace(record=record,scene=scene)
        b.rooms={k:torch.as_tensor(v.copy()) for k,v in pack_room_scenes([scene]).items()}
        b.contrast={k:torch.as_tensor(v.copy()) for k,v in pack_hand_contrast([scene]).items()}
        b.crossed_is_plane=torch.tensor([False]);b.goals=T([record['goal']])
        b.fields={k:T(np.load(directory/(k+'.npy')).reshape(-1,1 if k=='sdf' else 3)) for k in ('gf','bf','sdf')}
        def sample(name,pos,ids):
            n=len(pos)
            return sample_ragged_field(b.fields[name],pos,origin=T(record['origin']).repeat(n,1),dx=torch.full((n,),record['dx']),shape=torch.tensor(record['shape']).repeat(n,1),offset=torch.zeros(n,dtype=torch.long))
        b.sample=sample
        return b
    def route(self,scene,spacing=.02):
        route=np.array(scene['route']);lengths=np.linalg.norm(np.diff(route,axis=0),axis=1);starts=np.r_[0,np.cumsum(lengths)]
        progress=np.linspace(0,starts[-1],math.ceil(starts[-1]/spacing)+1)
        segment=np.minimum(np.searchsorted(starts[1:],progress,side='right'),len(lengths)-1)
        tangents=(route[segment+1]-route[segment])/lengths[segment,None]
        xy=route[segment]+(progress-starts[segment])[:,None]*tangents
        return progress,xy,np.arctan2(tangents[:,1],tangents[:,0])
    def task(self,bank,xy,yaws,qs,*,preserve_root_orientation=False):
        m=self.model;n=len(xy);obj=object.__new__(CATTask)
        obj.config=self.config;obj.bank=bank;obj.model=m;obj.num_envs=n;obj.device=torch.device('cpu');obj.dt=.02
        obj.hand_contrast=bool(bank.scene.get('hand_contrast'));obj.hand_protection=obj.stabilization=True
        obj.math=SimpleNamespace(**{name:getattr(tm,name) for name in ('native_rewards','upper_stability_terms','compute_cmd_from_rtf')})
        for name in ('swept_root_clearance','route_context','hand_contrast_context','contrast_reward_terms'):setattr(obj,name,globals()[name])
        obj.scene_ids=torch.zeros(n,dtype=torch.long);obj.all_ids=torch.arange(n);obj.navigation={};obj.contrast={};obj.info={}
        names=['head','imu_in_pelvis','imu_in_torso',*const.FEET_SITES,*const.HAND_SITES,*const.KNEE_SITES,*const.SHOULDER_SITES]
        obj.site_ids=torch.tensor([m.site(s).id for s in names]);obj.elbow_ids=torch.tensor([m.site(s+'_elbow_probe').id for s in ('left','right')])
        obj.pelvis_site=m.site('imu_in_pelvis').id;obj.torso_site=m.site('imu_in_torso').id;obj.pelvis_body=m.body('pelvis').id
        obj.leg_ids=torch.tensor([m.body(s).id for s in ('left_knee_link','left_ankle_roll_link','right_knee_link','right_ankle_roll_link')])
        obj.hand_radii=T(self.radii);obj.nominal=T(const.DEFAULT_QPOS[7:])
        limits=T(m.jnt_range[1:]);ctr=limits.mean(-1);span=limits[:,1]-limits[:,0];obj.lower=ctr-.5*.95*span;obj.upper=ctr+.5*.95*span
        rows={k:[] for k in ('qpos','site_xpos','site_xmat','xpos','xmat','subtree_com','actuator_force')};d=mujoco.MjData(m)
        for point,yaw,q in zip(xy,yaws,qs):
            d.qpos[:]=q;d.qpos[:2]=point
            if not preserve_root_orientation:d.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
            mujoco.mj_forward(m,d)
            for k in rows:rows[k].append((d.qfrc_bias[6:] if k=='actuator_force' else getattr(d,k)).copy())
        obj.data=SimpleNamespace(**{k:T(v) for k,v in rows.items()},qvel=torch.zeros(n,35));d=obj.data
        d.site_xmat=d.site_xmat.reshape(n,-1,3,3);d.xmat=d.xmat.reshape(n,-1,3,3)
        pelvis=d.site_xmat[:,obj.pelvis_site];torso=d.site_xmat[:,obj.torso_site];navi=tm.navi_rotation(pelvis);pos=d.site_xpos[:,obj.site_ids]
        obj.episode={'outside_bounds':torch.zeros(n,dtype=torch.bool)}
        # Offline samples must carry the segment already reached by a traversal.
        # Resetting every interior sample to segment zero points the carrot backwards.
        route=T(bank.scene['route']);delta=route[1:]-route[:-1]
        u=((d.qpos[:,:2,None].transpose(1,2)-route[:-1])*delta).sum(-1)/delta.square().sum(-1)
        projected=route[:-1]+u.clamp(0,1)[:,:,None]*delta
        segment=(projected-d.qpos[:,None,:2]).square().sum(-1).argmin(-1)
        obj.navigation=dict(root_xy=d.qpos[:,:2].clone(),segment=segment,violation=torch.zeros(n,dtype=torch.bool))
        obj._navigation(obj.all_ids,reset=False)
        gf,bf,sdf,command=obj._fields(pos,d.qpos[:,:2],obj.all_ids);gf=gf/(torch.linalg.vector_norm(gf,dim=-1,keepdim=True)+1e-6)
        obj.info=dict(positions=pos,velocities=torch.zeros_like(pos),sdf=sdf,gf=gf,command=command,navi=navi,
            pelvis_rpy=tm.matrix_rpy(navi.transpose(-1,-2)@pelvis),torso_rpy=tm.matrix_rpy(navi.transpose(-1,-2)@torso),torso_angvel=torch.zeros(n,3),
            last_act=torch.zeros(n,29),last_last_act=torch.zeros(n,29),last_joint_vel=torch.zeros(n,29),motor_targets=d.qpos[:,7:].clone(),
            previous_upper=d.qpos[:,19:].clone(),previous_previous_upper=d.qpos[:,19:].clone(),
            elbow_clearance=bank.sample('sdf',d.site_xpos[:,obj.elbow_ids],obj.scene_ids)[:,:,0]-.05,foot_height=torch.full((n,),.07))
        return obj
    def clearance(self,obj):
        world=transform_primitives(obj.data.xpos,obj.data.xmat,self.compiled);boxes=obj.bank.scene['boxes']
        obstacle=(T([b['center'] for b in boxes])[None,None],T([[[math.cos(b['yaw']),-math.sin(b['yaw']),0],[math.sin(b['yaw']),math.cos(b['yaw']),0],[0,0,1]] for b in boxes])[None,None],T([b['half_size'] for b in boxes])[None,None]);rows=[]
        for kind,ids in self.compiled['indices'].items():
            centers=world['centers'][:,ids,None]
            if kind=='box':v=box_box_separation(centers,world['rotations'][:,ids,None],self.compiled['half_sizes'][ids][None,:,None],*obstacle)
            elif kind=='capsule':
                ends=world['endpoints'][:,ids];v=capsule_box_separation(ends[:,:,None,0],ends[:,:,None,1],self.compiled['radii'][ids][None,:,None],*obstacle)
            else:v=sphere_box_separation(centers,self.compiled['radii'][ids][None,:,None],*obstacle)
            rows.append(v.flatten(1).amin(-1))
        return torch.stack(rows).amin(0)
    def rewards(self,obj,speed=.6,site_velocity=None,double_support=False):
        n=obj.num_envs;direction=obj.navigation['tangent'];root_velocity=torch.cat((direction*speed,torch.zeros(n,1)),1)
        action=torch.zeros(n,29);action[:,12:]=(obj.info['motor_targets'][:,12:]-obj.nominal[12:])/.8
        obj.info['last_act']=action.clone();obj.info['last_last_act']=action.clone();obj.info['command'][:,0]=1
        obj.info['last_last_act'][:,12:]=(obj.info['previous_upper']-obj.nominal[12:])/.8
        rows=[];phase_fraction=math.acos(.6)/math.pi
        for phase,w in enumerate([phase_fraction,phase_fraction,1-2*phase_fraction]):
            gait=torch.tensor([[1.,-1.] if phase==0 else [-1.,1.] if phase==1 else [0.,0.]]).repeat(n,1)
            contacts=(gait==1) if phase<2 else torch.ones(n,2,dtype=torch.bool)
            if double_support:contacts=torch.ones(n,2,dtype=torch.bool)
            vel=root_velocity[:,None].repeat(1,11,1) if site_velocity is None else T(site_velocity).clone()
            vel[:,3:5]=torch.where(contacts[:,:,None],0.,2*root_velocity[:,None])
            obj.info.update(gait=gait,velocities=vel)
            def sensor(name,ids):return root_velocity[ids] if name=='global_linvel_pelvis' else vel[ids,3 if name.startswith('left') else 4]
            obj._sensor=sensor
            post,comp=obj._rewards(action,contacts);comp={k:v*.02 for k,v in comp.items()};pre=sum(comp.values())
            rows.append((w,post,pre,comp))
        return dict(pre=sum(w*p for w,_,p,_ in rows),post=sum(w*p for w,p,_,_ in rows),
            minimum_phase_pre=torch.stack([p for _,_,p,_ in rows]).amin(0),
            zero_fraction=sum(w*(p==0).float() for w,p,_,_ in rows),
            components={k:sum(w*c[k] for w,_,_,c in rows) for k in rows[0][3]})
