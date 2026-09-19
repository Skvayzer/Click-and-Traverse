"""The whole-body CAT task over mjlab Simulation and batched Torch tensors.

The custom step order intentionally preserves CAT's pre-final-integration
derived data, per-substep collision latching, delayed fields, and reward clip.
This task is consumed by our learner, rather than mjlab's example G1 PPO task.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from . import task_math as tm
from .navigation import route_context,swept_root_clearance,hand_contrast_context,contrast_reward_terms


def _get(config,path,default=None):
    value=config
    for key in path.split('.'):
        if isinstance(value,dict):value=value.get(key,default)
        else:value=getattr(value,key,default)
        if value is None:return default
    return value


def _selected_data(data,ids):
    return SimpleNamespace(xpos=data.xpos[ids],xmat=data.xmat[ids])


class CATTask:
    """Stateful batched task. reset/step have no policy-learning side effects.

    ``step`` returns terminal observations before autoreset separately from
    ``obs``. A goal does not reset physics; first arrival is counted once.
    """
    observation_size=222
    privileged_observation_size=310
    action_size=29

    def __init__(self,sim,bank,config,*,collision=None,num_envs=None,device=None,seed=0):
        from cat_mjlab import constants as const
        from cat_ppo.furniture.control import JOINT_NAMES,wholebody_observation_contract
        from cat_ppo.furniture.grippers import hand_sphere
        self.sim,self.bank,self.config,self.collision=sim,bank,config,collision
        self.math=SimpleNamespace(**{name:getattr(tm,name) for name in (
            'quat_mul','delay_body_pos','navi_rotation','world_to_navi','matrix_rpy',
            'motor_targets','pd_torque','compute_cmd_from_rtf','update_phase','observations',
            'upper_stability_terms','native_rewards')})
        self.route_context=route_context
        self.swept_root_clearance=swept_root_clearance
        self.hand_contrast_context=hand_contrast_context
        self.contrast_reward_terms=contrast_reward_terms
        self.compiled=False
        self.data,self.model=sim.data,sim.model
        self.device=self.data.qpos.device if device is None else torch.device(device)
        self.num_envs=len(self.data.qpos) if num_envs is None else num_envs
        if self.num_envs!=len(self.data.qpos):raise ValueError('Simulation world count differs')
        self.dt=float(_get(config,'ctrl_dt',.02));self.sim_dt=float(_get(config,'sim_dt',.002))
        if abs(self.dt-.02)>1e-9 or abs(self.sim_dt-.002)>1e-9:raise ValueError('Migration requires CAT 20ms/2ms timing')
        self.n_substeps=10
        if self.model.nq!=36 or self.model.nv!=35 or self.model.nu!=29:raise ValueError('Expected fixed-Dex3 G1 with 29 actuators')
        for index,name in enumerate(JOINT_NAMES):
            if self.model.actuator(name).id!=index or self.model.joint(name).qposadr[0]!=7+index:
                raise ValueError(f'CAT coordinate ordering changed: {name}')
        if collision is None and bank.has_contrast:raise ValueError('Contrastive passages require body collision checks')
        if collision is not None and bank.reset_pool is None:raise ValueError('Body collision checks need certified reset pool')
        if bank.reset_pool is not None and bank.reset_pool.shape[-1]!=self.model.nq:raise ValueError('Reset model coordinate mismatch')
        self.hand_contrast=bool(_get(config,'wholebody_hand_contrast',False))
        if self.hand_contrast!=bank.has_contrast:raise ValueError('Contrastive metadata and task flag must agree')
        self.hand_protection=bool(_get(config,'wholebody_hand_protection',False))
        self.stabilization=bool(_get(config,'wholebody_stabilization',False))
        self.contract=wholebody_observation_contract()
        self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.all_ids=torch.arange(self.num_envs,device=self.device)
        tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=self.device)
        self.init_q=tensor(const.DEFAULT_QPOS);self.nominal=self.init_q[7:]
        self.kps,self.kds,self.torque_limit=map(tensor,(const.KPs,const.KDs,const.TORQUE_LIMIT))
        limits=tensor(self.model.jnt_range[1:]);center=limits.mean(-1);span=limits[:,1]-limits[:,0]
        factor=float(_get(config,'soft_joint_pos_limit_factor',.95))
        self.lower,self.upper=center-.5*span*factor,center+.5*span*factor
        self.site_names=['head','imu_in_pelvis','imu_in_torso',*const.FEET_SITES,*const.HAND_SITES,
                         *const.KNEE_SITES,*const.SHOULDER_SITES]
        self.site_ids=torch.tensor([self.model.site(n).id for n in self.site_names],device=self.device)
        self.elbow_ids=torch.tensor([self.model.site(n).id for n in ('left_elbow_probe','right_elbow_probe')],device=self.device)
        self.pelvis_site,self.torso_site=self.model.site('imu_in_pelvis').id,self.model.site('imu_in_torso').id
        self.pelvis_body=self.model.body('pelvis').id
        self.leg_ids=torch.tensor([self.model.body(n).id for n in ('left_knee_link','left_ankle_roll_link','right_knee_link','right_ankle_roll_link')],device=self.device)
        self.hand_radii=tensor([hand_sphere(s)['radius'] for s in ('left','right')])
        sensor_names=[f'{prefix}_{frame}' for frame in ('pelvis','torso') for prefix in ('upvector','global_linvel','global_angvel','local_linvel','gyro')]
        sensor_names+=['left_foot_global_linvel','right_foot_global_linvel']
        self.sensor_indices={}
        for name in sensor_names:
            sensor=self.model.sensor(name);a=int(sensor.adr[0]);n=int(sensor.dim[0])
            self.sensor_indices[name]=slice(a,a+n)
        pairs=[('left_foot','floor'),('right_foot','floor'),('right_foot','left_foot'),('left_foot','right_shin'),('right_foot','left_shin')]
        self.contact_pairs=torch.tensor([[self.model.geom(a).id,self.model.geom(b).id] for a,b in pairs],device=self.device)
        self.info={};self.navigation={};self.contrast={};self.episode={};self.telemetry={}
        self.obs={'state':torch.zeros((self.num_envs,222),device=self.device),'privileged_state':torch.zeros((self.num_envs,310),device=self.device)}
        self.scene_ids=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.scene_episode_ema=torch.zeros(bank.count,device=self.device)
        self.scene_success_ema=torch.zeros_like(self.scene_episode_ema)
        self.curriculum_stage=torch.zeros((),dtype=torch.long,device=self.device)
        self.curriculum_completed=torch.zeros(3,dtype=torch.long,device=self.device)
        self.curriculum_goals=torch.zeros_like(self.curriculum_completed)
        self.probabilities=bank.probabilities(stage=self.curriculum_stage)
        self.navigation_counts=torch.zeros((4,2),dtype=torch.long,device=self.device)
        self.contrast_counts=torch.zeros((3,2),dtype=torch.long,device=self.device)
        self.zone_steps=torch.zeros((self.num_envs,6,3),dtype=torch.long,device=self.device)
        self.outcome_counted=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.episode_reward=torch.zeros(self.num_envs,device=self.device)
        self.reset()
        # The released learner offsets EpisodeWrapper's horizon only. Native
        # task age remains zero for contact grace and held-odometry cadence.
        if bool(_get(config,'randomize_initial_episode_steps',True)):
            self.info['wrapper_steps']=torch.randint(int(_get(config,'episode_length',1000)),
                (self.num_envs,),generator=self.generator,device=self.device)

    def enable_compilation(self,*,backend='inductor'):
        """Fuse pure task kernels without capturing mutable state or reset logic.

        No reduce-overhead/CUDA-graph pool is requested: fields and rollout
        storage already occupy substantial VRAM. Dynamic world batches let
        selective resets share kernels when Torch supports the operation.
        Compilation is lazy; the first control transitions perform compilation.
        """
        if self.compiled:return
        options=dict(backend=backend,fullgraph=True,dynamic=True)
        if backend=='inductor':options['mode']='default'
        for name,function in vars(self.math).items():
            setattr(self.math,name,torch.compile(function,**options))
        for name in ('route_context','swept_root_clearance','hand_contrast_context','contrast_reward_terms'):
            setattr(self,name,torch.compile(getattr(self,name),**options))
        if hasattr(self.bank,'enable_compilation'):self.bank.enable_compilation(backend=backend)
        if hasattr(self.collision,'enable_compilation'):self.collision.enable_compilation(backend=backend)
        self.compiled=True

    def _rand(self,shape,low=0.,high=1.):
        return low+(high-low)*torch.rand(shape,device=self.device,generator=self.generator)

    def _put(self,destination,key,ids,value):
        if key not in destination:
            destination[key]=torch.zeros((self.num_envs,*value.shape[1:]),device=value.device,dtype=value.dtype)
        destination[key][ids]=value

    def _sensor(self,name,ids):return self.data.sensordata[ids,self.sensor_indices[name]]

    def _poses(self,ids):return self.data.site_xpos[ids][:,self.site_ids]

    def _room_meta(self,ids):return {k:v[self.scene_ids[ids]] for k,v in self.bank.rooms.items()}

    def _navigation(self,ids,reset=False):
        root=self.data.qpos[ids,:2]
        meta=self._room_meta(ids)
        previous=root if reset else self.navigation['root_xy'][ids]
        segment=torch.zeros(len(ids),dtype=torch.long,device=self.device) if reset else self.navigation['segment'][ids]
        violation=torch.zeros(len(ids),dtype=torch.bool,device=self.device) if reset else self.navigation['violation'][ids]
        values=self.route_context(root,previous,segment,violation,meta['route'],meta['route_count'],
            meta['obstacles'],meta['obstacle_count'],radius=meta['navigation_radius'])
        values['enabled']=meta['enabled'];values['violation']&=meta['enabled']
        for key in ('root_clearance','swept_clearance'):values[key]=torch.where(meta['enabled'],values[key],0.)
        for k,v in values.items():self._put(self.navigation,k,ids,v)
        metadata={k:v[self.scene_ids[ids]] for k,v in self.bank.contrast.items()}
        values=self.hand_contrast_context(metadata,values['progress_m'],values['tangent'])
        for k,v in values.items():self._put(self.contrast,k,ids,v)

    def _fields(self,positions,root_xy,ids):
        scenes=self.scene_ids[ids]
        gf,bf,sdf=(self.bank.sample(k,positions,scenes) for k in ('gf','bf','sdf'))
        sdf=sdf.clone();sdf[:,5:7,0]-=self.hand_radii
        nav={k:v[ids] for k,v in self.navigation.items()};meta=self._room_meta(ids)
        delta=nav['target']-root_xy;length=torch.linalg.vector_norm(delta,dim=-1)
        direction=delta/length.clamp_min(1e-6)[:,None]
        visible=self.swept_root_clearance(root_xy,nav['target'],meta['obstacles'],meta['obstacle_count'],radius=meta['navigation_radius'])>0
        active=visible&~nav['blocked']&~nav['violation']
        speed=torch.linalg.vector_norm(nav['guidance'][:,:2],dim=-1)
        velocity=direction*(speed*active)[:,None]
        room_command=torch.cat((torch.where(torch.linalg.vector_norm(velocity,dim=-1)>.01,.75,0.)[:,None],velocity,torch.zeros_like(velocity[:,:1])),-1)
        replacement=gf.clone();replacement[:,:,:2]=direction[:,None]*.6
        normal=bf/(torch.linalg.vector_norm(bf,dim=-1,keepdim=True)+1e-9)
        inward=(replacement*normal).sum(-1,keepdim=True).clamp_max(0)
        replacement-=inward*normal*(sdf<.5);replacement*=active[:,None,None]
        replacement[:,1]=torch.cat((velocity,torch.zeros_like(velocity[:,:1])),-1)
        gf=torch.where(nav['enabled'][:,None,None],replacement,gf)
        cmd=self.math.compute_cmd_from_rtf(gf[:,1],gf[:,[0,3,4,5,6]],bf[:,[0,3,4,5,6]])
        cmd=torch.where(nav['enabled'][:,None],room_command,cmd)
        return gf,bf,sdf,cmd

    def _elbow_fields(self,ids,actor):
        positions=self.data.site_xpos[ids][:,self.elbow_ids]
        if actor:positions=self.math.delay_body_pos(self.data.qpos[ids],self.info['odom_delay'][ids],positions)
        scenes=self.scene_ids[ids]
        gf,bf,sdf=(self.bank.sample(k,positions,scenes) for k in ('gf','bf','sdf'))
        sdf=sdf-.05
        command=self.info['command_delay' if actor else 'command'][ids]
        xy=.6*command[:,1:3]/torch.linalg.vector_norm(command[:,1:3],dim=-1,keepdim=True).clamp_min(1e-6)
        replacement=gf.clone();replacement[:,:,:2]=xy[:,None]
        normal=bf/(torch.linalg.vector_norm(bf,dim=-1,keepdim=True)+1e-9)
        replacement-=(replacement*normal).sum(-1,keepdim=True).clamp_max(0)*normal*(sdf<.5)
        replacement*=(command[:,0]>.5)[:,None,None]
        gf=torch.where(self.navigation['enabled'][ids,None,None],replacement,gf)
        move=(self.info['step'][ids]==0)|(self.info['command'][ids,0]>.5)
        gf=gf*move[:,None,None]/(torch.linalg.vector_norm(gf,dim=-1,keepdim=True)+1e-6)
        bf=bf/(torch.linalg.vector_norm(bf,dim=-1,keepdim=True)+1e-6)
        if actor:
            gf=self.math.world_to_navi(self.info['navi'][ids],gf)
            bf=self.math.world_to_navi(self.info['navi'][ids],bf)*(sdf<.5);sdf=sdf.clamp(-1,.5)
        return torch.cat((gf.flatten(1),bf.flatten(1),sdf.flatten(1)),-1)

    def _observe(self,ids,contacts):
        info={k:v[ids] for k,v in self.info.items()}
        elbow_true=self._elbow_fields(ids,False);elbow_actor=self._elbow_fields(ids,True)
        self._put(self.info,'elbow_clearance',ids,elbow_true[:,-2:])
        level=float(_get(self.config,'noise_config.level',1.))
        noise={key:self._rand((len(ids),size),-1,1)*level*float(_get(self.config,'noise_config.scales.'+key,scale))
               for key,size,scale in (('gyro',3,.2),('gravity',3,.05),('joint_pos',29,.03),('joint_vel',29,1.5))}
        site_rotation=self.data.site_xmat.reshape(self.num_envs,-1,3,3)[ids,self.pelvis_site]
        gravity=-site_rotation[:,2,:]
        result=self.math.observations(joint_pos=self.data.qpos[ids,7:],joint_vel=self.data.qvel[ids,6:],nominal=self.nominal,
            gyro=self._sensor('gyro_pelvis',ids),gravity=gravity,linear_velocity=self._sensor('local_linvel_pelvis',ids),
            noise=noise,last_action=info['last_act'],targets=info['motor_targets'],command=info['command'],
            command_delay=info['command_delay'],foot_height=info['foot_height'],phase=info['phase'],navi=info['navi'],
            gf=info['gf'],bf=info['bf'],sdf=info['sdf'],gf_delay=info['gf_delay'],bf_delay=info['bf_delay'],sdf_delay=info['sdf_delay'],
            positions=info['positions'],velocities=info['velocities'],torso_rpy=info['torso_rpy'],gait=info['gait'],contacts=contacts,
            kp=info['kp'],kd=info['kd'],rfi=info['rfi'],elbow_true=elbow_true,elbow_actor=torch.nan_to_num(elbow_actor))
        if result['state'].shape[-1]!=222 or result['privileged_state'].shape[-1]!=310:
            raise RuntimeError('Observation feature contract changed')
        for k,v in result.items():self.obs[k][ids]=v
        return result

    @torch.no_grad()
    def reset(self,env_ids=None,scene_ids=None):
        ids=self.all_ids if env_ids is None else env_ids.long();n=len(ids)
        if n==0:return self.obs
        if scene_ids is None:
            cdf=self.probabilities.cumsum(0);cdf=cdf/cdf[-1]
            scene_ids=torch.searchsorted(cdf,self._rand((n,))).clamp_max(self.bank.count-1)
        self.scene_ids[ids]=scene_ids
        qpos=self.init_q.expand(n,-1).clone();qpos[:,:2]+=self._rand((n,2),-1,1);qpos[:,2]=.8
        yaw=self._rand((n,),-np.pi/2,np.pi/2)
        quat=torch.stack(((yaw/2).cos(),torch.zeros_like(yaw),torch.zeros_like(yaw),(yaw/2).sin()),-1)
        qpos[:,3:7]=self.math.quat_mul(qpos[:,3:7],quat)
        qpos[:,7:]=torch.maximum(torch.minimum(qpos[:,7:]*self._rand((n,29),.5,1.5),self.upper),self.lower)
        scenes=self.scene_ids[ids];room=~self.bank.reset_is_cat[scenes]
        position=self.bank.starts[scenes,:2]+qpos[:,:2]*self.bank.reset_xy_scale[scenes]
        yaw=self.bank.reset_yaws[scenes];q=torch.stack(((yaw/2).cos(),torch.zeros_like(yaw),torch.zeros_like(yaw),(yaw/2).sin()),-1)
        qpos[:,:2]=torch.where(room[:,None],position,qpos[:,:2])
        qpos[:,3:7]=torch.where(room[:,None],self.math.quat_mul(q,qpos[:,3:7]),qpos[:,3:7])
        self.sim.reset_data(ids)
        self.data.qpos[ids]=qpos;self.data.qvel[ids]=0.;self.data.qvel[ids,:6]=self._rand((n,6),-.5,.5)
        self.data.ctrl[ids]=qpos[:,7:];self.sim.forward(ids)
        replaced=torch.zeros(n,dtype=torch.bool,device=self.device)
        if self.collision is not None:
            replaced=self.collision(scenes,_selected_data(self.data,ids)).any(-1)
            pick=torch.randint(self.bank.reset_pool.shape[1],(n,),generator=self.generator,device=self.device)
            qpos=torch.where(replaced[:,None],self.bank.reset_pool[scenes,pick],qpos)
            self.data.qpos[ids]=qpos;self.data.ctrl[ids]=qpos[:,7:];self.sim.forward(ids)
        defaults={'step':torch.zeros(n,dtype=torch.long,device=self.device),
            'wrapper_steps':torch.zeros(n,dtype=torch.long,device=self.device),'push_step':torch.zeros(n,dtype=torch.long,device=self.device),
            'motor_targets':self.nominal.expand(n,-1).clone(),'last_act':torch.zeros((n,29),device=self.device),
            'last_last_act':torch.zeros((n,29),device=self.device),'last_joint_vel':torch.zeros((n,29),device=self.device),
            'navi':torch.eye(3,device=self.device).expand(n,-1,-1).clone(),
            'pelvis_rpy':torch.zeros((n,3),device=self.device),'torso_rpy':torch.zeros((n,3),device=self.device),
            'torso_angvel':torch.zeros((n,3),device=self.device),'stop_timestep':torch.full((n,),100,dtype=torch.long,device=self.device),
            'gait':torch.zeros((n,2),device=self.device),'odom_delay':qpos[:,:7].clone(),
            'previous_upper':self.nominal[12:].expand(n,-1).clone(),'previous_previous_upper':self.nominal[12:].expand(n,-1).clone()}
        defaults['push_interval']=torch.round(self._rand((n,),*_get(self.config,'push_config.interval_range',[5.,10.]))/self.dt).long()
        frequency=self._rand((n,),*_get(self.config,'gait_config.freq_range',[1.3,1.5]))
        defaults['phase_dt']=2*torch.pi*self.dt*frequency
        left=torch.rand((n,),device=self.device,generator=self.generator)<.5
        defaults['phase']=torch.where(left[:,None],qpos.new_tensor([0.,np.pi]),qpos.new_tensor([np.pi,0.]))
        defaults['foot_height']=self._rand((n,),*_get(self.config,'gait_config.foot_height_range',[.07,.07]))
        defaults['kp']=self._rand((n,),*_get(self.config,'dm_rand_config.kp_range',[.75,1.25])) if _get(self.config,'dm_rand_config.enable_pd',True) else torch.ones(n,device=self.device)
        defaults['kd']=self._rand((n,),*_get(self.config,'dm_rand_config.kd_range',[.75,1.25])) if _get(self.config,'dm_rand_config.enable_pd',True) else torch.ones(n,device=self.device)
        defaults['rfi']=self._rand((n,29),*_get(self.config,'dm_rand_config.rfi_lim_range',[.5,1.5]))*float(_get(self.config,'dm_rand_config.rfi_lim',.1))*self.torque_limit
        if not _get(self.config,'dm_rand_config.enable_rfi',True):defaults['rfi'].zero_()
        for k,v in defaults.items():self._put(self.info,k,ids,v)
        self._navigation(ids,reset=True)
        positions=self._poses(ids);gf,bf,sdf,command=self._fields(positions,qpos[:,:2],ids)
        # Original reset normalizes before deriving command; rooms override it.
        gf=gf/(torch.linalg.vector_norm(gf,dim=-1,keepdim=True)+1e-6)
        bf=bf/(torch.linalg.vector_norm(bf,dim=-1,keepdim=True)+1e-6)
        normalized_command=self.math.compute_cmd_from_rtf(gf[:,1],gf[:,[0,3,4,5,6]],bf[:,[0,3,4,5,6]])
        command=torch.where(self.navigation['enabled'][ids,None],command,normalized_command)
        for k,v in dict(gf=gf,bf=bf,sdf=sdf,gf_delay=gf,bf_delay=bf,sdf_delay=sdf,positions=positions,
                        velocities=torch.zeros_like(positions),command=command,command_delay=command,last_command=torch.zeros_like(command)).items():
            self._put(self.info,k,ids,v)
        keys=('goal_reached','raw_goal','outside_bounds','fall','obstacle','self_contact','numerical','hand_violation','elbow_violation','body_collision','reset_replaced',
              'body_collision_feet','body_collision_legs','body_collision_trunk','body_collision_head','body_collision_arms','body_collision_hands')
        for k in keys:self._put(self.episode,k,ids,replaced if k=='reset_replaced' else torch.zeros(n,dtype=torch.bool,device=self.device))
        self.episode['outside_bounds'][ids]=self._goal_status(ids)[2]
        self.outcome_counted[ids]=False;self.zone_steps[ids]=0;self.episode_reward[ids]=0
        self._observe(ids,self.sim.contact_flags(self.contact_pairs)[ids,:2])
        return self.obs

    def _crossed(self,positions,ids):
        scenes=self.scene_ids[ids]
        plane=positions[...,0]>1.5
        goal=torch.linalg.vector_norm(positions[...,:2]-self.bank.goals[scenes,None,:2],dim=-1)<=.5
        crossed=torch.where(self.bank.crossed_is_plane[scenes,None],plane,goal)
        navigation=~self.navigation['enabled'][ids]|(self.navigation['route_complete'][ids]&~self.navigation['violation'][ids])
        return crossed&navigation[:,None]

    def _goal_status(self,ids):
        scenes=self.scene_ids[ids];root=self.data.qpos[ids,:3];feet=self.info['positions'][ids,3:5]
        origin=self.bank.origins[scenes];extent=origin+(self.bank.shapes[scenes]-1)*self.bank.dxs[scenes,None]
        points=torch.cat((root[:,None],feet),1)
        lateral=((points[:,:,1]<origin[:,None,1])|(points[:,:,1]>extent[:,None,1])).any(-1)
        xy=((root[:,:2]<origin[:,:2])|(root[:,:2]>extent[:,:2])).any(-1)
        outside=self.episode['outside_bounds'][ids]|torch.where(self.bank.crossed_is_plane[scenes],lateral,xy)
        raw=self._crossed(points,ids).all(-1)&~xy
        return raw&~outside,raw,outside

    def _termination(self,regions,contacts):
        i=self.info;threshold=-float(_get(self.config,'term_collision_threshold',.04));grace=i['step']>=50
        fields=(i['sdf']<threshold).flatten(1).any(-1)&grace
        elbows=(i['elbow_clearance']<threshold).any(-1)&grace&bool(_get(self.config,'terminate_on_elbow_collision',True))
        self_contact=contacts[:,2:].any(-1)&grace
        fall=(self._sensor('upvector_pelvis',self.all_ids)[:,2]<0)|(i['positions'][:,0,2]<.7)
        numerical=torch.isnan(self.data.qpos).any(-1)|torch.isnan(self.data.qvel).any(-1)
        root=self.navigation['enabled']&self.navigation['violation'];body=regions.any(-1)
        obstacle=fields|elbows|root|body;done=fall|self_contact|numerical|obstacle
        flags=dict(fall=fall,obstacle=obstacle,self_contact=self_contact,numerical=numerical,body_collision=body,
                   hand_violation=(i['sdf'][:,5:7]<threshold).flatten(1).any(-1)&grace,elbow_violation=elbows)
        for k,v in flags.items():self.episode[k]|=v
        for index,region in enumerate(('feet','legs','trunk','head','arms','hands')):
            self.episode['body_collision_'+region]|=regions[:,index]
        clean,raw,outside=self._goal_status(self.all_ids);self.episode['outside_bounds']=outside
        self.episode['goal_reached']|=clean&~done;self.episode['raw_goal']|=raw&~done
        if self.collision is not None:self.episode['goal_reached']&=~done
        return done,flags

    def _rewards(self,action,contacts):
        i=self.info;d=self.data
        rewards=self.math.native_rewards(action=action,last_action=i['last_act'],last_last_action=i['last_last_act'],
            joint_pos=d.qpos[:,7:],joint_vel=d.qvel[:,6:],last_joint_vel=i['last_joint_vel'],lower=self.lower,upper=self.upper,
            actuator_force=d.actuator_force,command=i['command'],pelvis_rpy=i['pelvis_rpy'],torso_rpy=i['torso_rpy'],
            head_z=i['positions'][:,0,2],torso_height=float(_get(self.config,'torso_height',[.5,1.])[1]),
            global_velocity=self._sensor('global_linvel_pelvis',self.all_ids),torso_angvel=i['torso_angvel'],navi=i['navi'],
            leg_rotations=d.xmat.reshape(self.num_envs,-1,3,3)[:,self.leg_ids],feet_pos=i['positions'][:,3:5],
            feet_sensor_velocity=torch.stack((self._sensor('left_foot_global_linvel',self.all_ids),self._sensor('right_foot_global_linvel',self.all_ids)),1),
            subtree_com=d.subtree_com[:,self.pelvis_body],feet_contact=contacts[:,:2],gait=i['gait'],foot_height=i['foot_height'],
            foot_height_stance=float(_get(self.config,'reward_config.foot_height_stance',0.)),gf=i['gf'],positions=i['positions'],velocities=i['velocities'],
            sdf=i['sdf'],crossed=self._crossed(i['positions'],self.all_ids),dt=self.dt,max_yaw=abs(float(_get(self.config,'ang_vel_yaw',[-.5,.5])[1])))
        costs,telemetry=self.math.upper_stability_terms(i['motor_targets'][:,12:],i['previous_upper'],i['previous_previous_upper'],self.nominal[12:],
            i['sdf'][:,5:7],i['elbow_clearance'],dt=self.dt,velocity_scale=float(_get(self.config,'upper_velocity_cost_scale',2.)),
            acceleration_scale=float(_get(self.config,'upper_acceleration_cost_scale',20.)),posture_scale=float(_get(self.config,'upper_action_scale',.8)),
            hand_margin=float(_get(self.config,'hand_clearance_margin',.12)),elbow_margin=float(_get(self.config,'arm_clearance_margin',.08)),
            clearance_taper=float(_get(self.config,'upper_posture_clearance_taper',.12)),waist_weight=float(_get(self.config,'upper_cost_waist_weight',4.)),
            hand_protection=self.hand_protection,arm_velocity_weight=float(_get(self.config,'upper_arm_velocity_weight',.5)),
            arm_acceleration_weight=float(_get(self.config,'upper_arm_acceleration_weight',.1)),
            protection_target_clearance=float(_get(self.config,'hand_protection_target_clearance',.04)),
            protection_anticipation_distance=float(_get(self.config,'hand_protection_anticipation_distance',.20)),
            protection_near_weight=float(_get(self.config,'hand_protection_near_weight',.8)),
            protection_posture_taper=float(_get(self.config,'hand_protection_posture_taper',.10)),
            contrast_arm_active=self.contrast['hand_active'] if self.hand_contrast else None)
        enabled=bool(_get(self.config,'hand_protection_enabled',True))
        rewards['wholebody_hand_clearance']=(telemetry['hand_clearance_pressure'] if self.hand_protection else
            (float(_get(self.config,'hand_clearance_margin',.12))-i['sdf'][:,5:7]).clamp_min(0).square().flatten(1).mean(-1))*enabled
        rewards['wholebody_arm_clearance']=(float(_get(self.config,'arm_clearance_margin',.08))-i['elbow_clearance']).clamp_min(0).square().mean(-1)*enabled
        if self.stabilization:rewards.update(costs)
        if self.hand_contrast:
            rotations=d.site_xmat.reshape(self.num_envs,-1,3,3)
            contrast,report=self.contrast_reward_terms(self.contrast,i['positions'][:,5:7],d.qpos[:,:2],
                rotations[:,self.pelvis_site,:,0],rotations[:,self.torso_site,:,0],
                region_scale=float(_get(self.config,'hand_contrast_region_scale',.15)),
                hand_good_distance=float(_get(self.config,'hand_contrast_metric_tolerance',.05)))
            rewards.update(contrast);telemetry.update(report)
        self.telemetry=telemetry
        scales=_get(self.config,'reward_config.scales',{})
        missing=set(rewards)-set(scales)
        if missing:raise ValueError(f'Missing preserved reward scales: {sorted(missing)}')
        scaled={k:v*float(scales[k]) for k,v in rewards.items()}
        return (sum(scaled.values())*self.dt).clamp(0,10000),scaled

    def _outcomes(self,done,truncation):
        failure=done&~truncation
        for key in ('fall','obstacle','self_contact','numerical','body_collision','hand_violation','elbow_violation','outside_bounds'):
            failure|=self.episode[key]
        successful=self.episode['goal_reached']&~failure
        resolved=~self.outcome_counted&(successful|failure|done)
        successful&=resolved
        groups=self.bank.navigation_groups[self.scene_ids]
        for column,mask in enumerate((resolved,successful)):
            increments=torch.zeros(3,dtype=torch.long,device=self.device).scatter_add_(0,groups,mask.long())
            self.navigation_counts[:3,column]+=increments;self.navigation_counts[3,column]+=increments.sum()
        if self.hand_contrast:
            c=self.contrast;t=self.telemetry
            zone=torch.arange(6,device=self.device)[None]==c['zone_index'][:,None]
            core=c['core_active'][:,None]&zone
            values=torch.stack((core,core&(t['hand_contrast_heading_good']>.5)[:,None],core&(t['hand_contrast_hand_good']>.5)[:,None]),-1)
            self.zone_steps+=(values&(~self.outcome_counted)[:,None,None]).long()
            seen=self.zone_steps[:,:,0]>0
            heading=seen&(self.zone_steps[:,:,1]>=.9*self.zone_steps[:,:,0])
            hands=seen&(self.zone_steps[:,:,2]>=.9*self.zone_steps[:,:,0])
            qualified=(~c['required_forward_zones']|heading).all(-1)&(~c['required_hand_zones']|hands).all(-1)
            valid=(c['role']>=1)&(c['role']<=3);roles=(c['role']-1).clamp(0,2).long()
            for column,mask in enumerate((resolved&valid,successful&valid&qualified)):
                self.contrast_counts[:,column]+=torch.zeros(3,dtype=torch.long,device=self.device).scatter_add_(0,roles,mask.long())
        self.outcome_counted|=resolved
        # Original survival-based adaptation remains for old scenes. Only hand
        # tasks use clean arrival; contrastive roles use first-outcome events.
        adapted_done=resolved if self.bank.roles is not None else done
        adapted_success=successful if self.bank.roles is not None else truncation
        if self.bank.levels is not None:
            levels=self.bank.levels[self.scene_ids];hand=levels>=0
            clean=self.episode['goal_reached']&~failure
            adapted_success=torch.where(hand,clean,truncation)
            indices=levels.clamp_min(0)
            self.curriculum_completed.scatter_add_(0,indices,(hand&resolved).long())
            self.curriculum_goals.scatter_add_(0,indices,(hand&successful).long())
            stage=self.curriculum_stage
            rate=self.curriculum_goals[stage]/self.curriculum_completed[stage].clamp_min(1)
            self.curriculum_stage=stage+((stage<2)&(self.curriculum_completed[stage]>=64)&(rate>=.6)).long()
        counts=torch.zeros_like(self.scene_episode_ema).scatter_add_(0,self.scene_ids,adapted_done.float())
        goals=torch.zeros_like(self.scene_success_ema).scatter_add_(0,self.scene_ids,(adapted_done&adapted_success).float())
        decay=float(_get(self.config,'pf_config.sampling_ema_decay',.95));alpha=float(_get(self.config,'pf_config.sampling_alpha',1.))
        self.scene_episode_ema=decay*self.scene_episode_ema+counts;self.scene_success_ema=decay*self.scene_success_ema+goals
        rates=self.scene_success_ema/(self.scene_episode_ema+1e-6)
        weights=((1-rates)**alpha).clamp_min(1e-3)
        self.probabilities=self.bank.probabilities(weights,stage=self.curriculum_stage)
        return resolved,successful

    @torch.no_grad()
    def step(self,action):
        if action.shape!=(self.num_envs,29):raise ValueError('Expected [num_envs,29] actions')
        ids=self.all_ids;i=self.info;d=self.data
        theta=self._rand((self.num_envs,),0.,2*torch.pi)
        magnitude=self._rand((self.num_envs,),*_get(self.config,'push_config.magnitude_range',[.1,1.]))
        signal=((i['push_step']+1)%i['push_interval']==0)&bool(_get(self.config,'push_config.enable',True))
        push=torch.stack((theta.cos(),theta.sin()),-1)*(signal*magnitude)[:,None]
        d.qvel[:,:2]+=push
        targets=self.math.motor_targets(action,i['motor_targets'],self.nominal,self.lower,self.upper,
            action_scale=float(_get(self.config,'action_scale',.5)),upper_action_scale=float(_get(self.config,'upper_action_scale',.8)),
            upper_target_rate=float(_get(self.config,'upper_target_rate',2.)),dt=self.dt)
        regions=torch.zeros((self.num_envs,6),dtype=torch.bool,device=self.device)
        for _ in range(self.n_substeps):
            torque=self.math.pd_torque(d.qpos[:,7:],d.qvel[:,6:],targets,self.kps,self.kds,i['kp'],i['kd'],
                i['rfi'],self._rand((self.num_envs,29),-1.,1.),self.torque_limit)
            d.ctrl.copy_(torque)
            self.sim.step()
            if self.collision is not None:regions|=self.collision(self.scene_ids,d)
        if self.collision is not None:regions|=self.collision(self.scene_ids,self.sim.final_collision_data())
        contacts=self.sim.contact_flags(self.contact_pairs)
        i['motor_targets']=targets
        rotations=d.site_xmat.reshape(self.num_envs,-1,3,3)
        pelvis,torso=rotations[:,self.pelvis_site],rotations[:,self.torso_site]
        navi=self.math.navi_rotation(pelvis);i['navi']=navi
        i['pelvis_rpy']=self.math.matrix_rpy(navi.transpose(-1,-2)@pelvis)
        i['torso_rpy']=self.math.matrix_rpy(navi.transpose(-1,-2)@torso)
        i['torso_angvel']=torch.einsum('bij,bj->bi',navi.transpose(-1,-2)@torso,self._sensor('gyro_torso',ids))
        i['last_command']=i['command'].clone()
        positions=self._poses(ids);velocities=(positions-i['positions'])/self.dt
        # The original stores finite-difference velocities only for head/feet/hands.
        velocities[:,[1,2,7,8,9,10]]=0.
        self._navigation(ids)
        gf,bf,sdf,command=self._fields(positions,d.qpos[:,:2],ids)
        update=i['step']%5==0
        odom=torch.where(update[:,None],d.qpos[:,:7],i['odom_delay'])
        delayed=self.math.delay_body_pos(d.qpos,odom,positions)
        gfd,bfd,sdfd,command_delay=self._fields(delayed,odom[:,:2],ids)
        command,stop,phase,gait=self.math.update_phase(command,i['last_command'],i['stop_timestep'],i['phase'],i['phase_dt'],
            float(_get(self.config,'gait_config.gait_bound',.6)),self.navigation['enabled'])
        move=(command[:,0]>.5)[:,None,None]
        normalize=lambda field:field/(torch.linalg.vector_norm(field,dim=-1,keepdim=True)+1e-6)
        normalized_gfd,normalized_bfd=normalize(gfd)*move,normalize(bfd)
        # CAT computes the true command before normalizing fields, but computes
        # the delayed command AFTER normalization and the current move gate.
        # Room guidance is an explicit override independent of this projection.
        native_delayed=self.math.compute_cmd_from_rtf(normalized_gfd[:,1],normalized_gfd[:,[0,3,4,5,6]],normalized_bfd[:,[0,3,4,5,6]])
        command_delay=torch.where(self.navigation['enabled'][:,None],command_delay,native_delayed)
        i.update(gf=normalize(gf)*move,bf=normalize(bf),sdf=sdf,gf_delay=normalized_gfd,bf_delay=normalized_bfd,sdf_delay=sdfd,
                 command=command,command_delay=command_delay,odom_delay=odom,stop_timestep=stop,phase=phase,gait=gait,
                 positions=positions,velocities=velocities)
        i['push_step']+=1;i['step']+=1
        i['last_last_act']=i['last_act'].clone();i['last_act']=action.clone()
        self._observe(ids,contacts[:,:2])
        terminated,faults=self._termination(regions,contacts)
        reward,components=self._rewards(action,contacts)
        penalty=-float(_get(self.config,'wholebody.body_collision.event_penalty',1.))*regions.any(-1)
        if self.collision is not None:reward+=penalty;components['body_collision_event']=penalty
        i['wrapper_steps']+=1
        timeout=i['wrapper_steps']>=self.bank.episode_lengths[self.scene_ids]
        truncated=timeout&~terminated;done=terminated|timeout
        resolved,successful=self._outcomes(done,truncated)
        self.episode_reward+=reward
        terminal_obs={k:v.clone() for k,v in self.obs.items()}
        metrics={'reward/'+k:v for k,v in components.items()}
        metrics.update(navigation_counts=self.navigation_counts.clone(),contrast_counts=self.contrast_counts.clone(),
            episode_return=self.episode_reward.clone(),episode_length=i['step'].clone(),resolved=resolved,successful=successful,
            scene_ids=self.scene_ids.clone(),collision_regions=regions)
        for k,v in self.episode.items():metrics['episode/'+k]=v.clone()
        i['last_joint_vel']=d.qvel[:,6:].clone()
        i['previous_previous_upper']=i['previous_upper'].clone();i['previous_upper']=targets[:,12:].clone()
        # Only physically completed worlds reset; cached terminal observations
        # and terminal rewards retain the transition before reset.
        self.reset(torch.nonzero(done,as_tuple=False).flatten())
        return dict(obs=self.obs,reward=reward,terminated=terminated,truncated=truncated,done=done,
                    terminal_obs=terminal_obs,metrics=metrics)

    def state_dict(self):
        """Torch task state; simulation and learner states are saved separately."""
        names=('info','navigation','contrast','episode','telemetry','obs','scene_ids','scene_episode_ema','scene_success_ema',
               'curriculum_stage','curriculum_completed','curriculum_goals','probabilities','navigation_counts','contrast_counts',
               'zone_steps','outcome_counted','episode_reward')
        return {**{name:getattr(self,name) for name in names},'rng':self.generator.get_state()}

    def load_state_dict(self,state):
        expected=self.state_dict()
        if set(state)!=set(expected):raise ValueError('Incomplete or unknown task checkpoint fields')
        def validate(actual,template,name):
            if isinstance(template,dict):
                if not isinstance(actual,dict):raise ValueError(f'Invalid task state mapping: {name}')
                # Telemetry is observational, first populated on the first step.
                if name!='telemetry' and set(actual)!=set(template):raise ValueError(f'Changed task state keys: {name}')
                return {k:validate(v,template.get(k,v),name+'.'+k) for k,v in actual.items()}
            if not isinstance(actual,torch.Tensor) or actual.shape!=template.shape or actual.dtype!=template.dtype:
                raise ValueError(f'Task checkpoint shape/dtype mismatch: {name}')
            return actual.to(self.device).clone()
        restored={k:validate(v,expected[k],k) for k,v in state.items() if k!='rng'}
        if not isinstance(state['rng'],torch.Tensor):raise ValueError('Missing task RNG state')
        # Validate everything before mutating the live task.
        self.generator.set_state(state['rng'].cpu())
        for name,value in restored.items():setattr(self,name,value)
