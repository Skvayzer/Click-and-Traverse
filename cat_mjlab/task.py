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
from .config import hand_curriculum_threshold
from .observation_contract import ACTOR_SIZE, CRITIC_SIZE
from .navigation import route_context,swept_root_clearance,hand_contrast_context,contrast_reward_terms

HAND_BODY_REGIONS=("trunk","head","arm","hand")


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
    observation_size=ACTOR_SIZE
    privileged_observation_size=CRITIC_SIZE
    action_size=29

    def __init__(self,sim,bank,config,*,collision=None,num_envs=None,device=None,seed=0,analytic_objects=None):
        from cat_mjlab import constants as const
        from cat_ppo.furniture.control import JOINT_NAMES,mjlab_observation_contract
        from cat_ppo.furniture.grippers import hand_sphere
        sizes=(_get(config,'num_obs',self.observation_size),_get(config,'num_pri',self.privileged_observation_size))
        if sizes!=(self.observation_size,self.privileged_observation_size):
            raise ValueError('Task requires native actor 222 / critic 310; obsolete observation contracts are unsupported')
        config['disable_hand_contrast']=True
        self.sim,self.bank,self.config,self.collision=sim,bank,config,collision
        self.math=SimpleNamespace(**{name:getattr(tm,name) for name in (
            'quat_mul','delay_body_pos','navi_rotation','world_to_navi','matrix_rpy',
            'motor_targets','pd_torque','compute_cmd_from_rtf','update_phase','observations',
            'upper_stability_terms','native_rewards','heading_probe_points','yaw_tracking',
            'facing_probe_points','clearance_gate','safe_velocity_target','hysteresis')})
        self.route_context=route_context
        self.swept_root_clearance=swept_root_clearance
        self.hand_contrast_context=hand_contrast_context
        self.contrast_reward_terms=contrast_reward_terms
        self.compiled=False
        self.data,self.model=sim.data,sim.model
        self.device=self.data.qpos.device if device is None else torch.device(device)
        self.num_envs=len(self.data.qpos) if num_envs is None else num_envs
        self.analytic_objects=analytic_objects
        self.reactive_objects=analytic_objects if hasattr(analytic_objects,'advance') else None
        if analytic_objects is not None:
            if analytic_objects.state['start'].shape[0]!=self.num_envs or analytic_objects.state['start'].device!=self.device:
                raise ValueError('Analytic objects must match task worlds and device')
            self.analytic_object_state=analytic_objects.state
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
        self.balance_settings=getattr(bank,'balance_settings',None)
        if self.balance_settings is not None:
            from .balance import reward_overrides
            self.balance_reward=reward_overrides(self.balance_settings,config.get('flat_balance_reward'))
        self.hand_contrast=bool(_get(config,'wholebody_hand_contrast',False))
        self.checkpoint_selection_roles=tuple(getattr(bank, 'manifest', {}).get(
            'hand_posture_upgrade', {}).get('selection_roles', (0,1,2,3)))
        if self.hand_contrast!=bank.has_contrast:raise ValueError('Contrastive metadata and task flag must agree')
        self.hand_protection=bool(_get(config,'wholebody_hand_protection',False))
        hand_curriculum_threshold(config, getattr(bank, 'manifest', None))
        self.stabilization=bool(_get(config,'wholebody_stabilization',False))
        self.sdf_rate=bool(config.get('sdf_rate_obs',False))
        self.contract=mjlab_observation_contract(self.sdf_rate)
        self.observation_size=len(self.contract['actor_features'])
        elbow_block=[n for n in self.contract['actor_features'] if n.startswith('pf.elbows.') and '.dfrate.' not in n]
        self.elbow_df_index=torch.tensor([i for i,n in enumerate(elbow_block) if n.startswith('pf.elbows.df.')],device=self.device)
        self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.all_ids=torch.arange(self.num_envs,device=self.device)
        tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=self.device)
        self.init_q=tensor(const.DEFAULT_QPOS);self.nominal=self.init_q[7:]
        # Home pose for the upper-body posture cost only (observations and the action
        # nominal are untouched, so the warm start is unaffected). The default hangs the
        # hands 3 cm from the thigh capsules; shoulder pitch -0.3 puts them 24 cm in
        # front of the hips with 13 cm of thigh clearance (FK-checked).
        self.upper_home=self.nominal[12:].clone()
        home_pitch=config.get('upper_home_shoulder_pitch')
        if home_pitch is not None:
            self.upper_home[3]=float(home_pitch);self.upper_home[10]=float(home_pitch)
        self.kps,self.kds,self.torque_limit=map(tensor,(const.KPs,const.KDs,const.TORQUE_LIMIT))
        from .upper_control import action_scales, UpperGravity
        self.upper_action_scales=tensor(action_scales(config))
        from .lateral_corridor import ArmGeometry
        self.corridor_arms = ArmGeometry(self.model,self.device) if self.hand_contrast else None
        self.upper_gravity=UpperGravity(self.model,self.device) if config.get('upper_gravity_compensation',False) else None
        limits=tensor(self.model.jnt_range[1:]);center=limits.mean(-1);span=limits[:,1]-limits[:,0]
        factor=float(_get(config,'soft_joint_pos_limit_factor',.95))
        self.lower,self.upper=center-.5*span*factor,center+.5*span*factor
        self.site_names=['head','imu_in_pelvis','imu_in_torso',*const.FEET_SITES,*const.HAND_SITES,
                         *const.KNEE_SITES,*const.SHOULDER_SITES]
        self.site_ids=torch.tensor([self.model.site(n).id for n in self.site_names],device=self.device)
        self.elbow_ids=torch.tensor([self.model.site(n).id for n in ('left_elbow_probe','right_elbow_probe')],device=self.device)
        self.pelvis_site,self.torso_site=self.model.site('imu_in_pelvis').id,self.model.site('imu_in_torso').id
        self.pelvis_body=self.model.body('pelvis').id
        # Leg capsules for hand self-clearance: thigh hip->knee (MJCF thigh capsule r=.06),
        # shin knee->ankle (MJCF shin capsule r=.08). Hands are the .103 m envelopes.
        body=lambda n:self.model.body(n).id
        self.capsule_a_ids=torch.tensor([body('left_hip_pitch_link'),body('left_knee_link'),body('right_hip_pitch_link'),body('right_knee_link')],device=self.device)
        self.capsule_b_ids=torch.tensor([body('left_knee_link'),body('left_ankle_pitch_link'),body('right_knee_link'),body('right_ankle_pitch_link')],device=self.device)
        self.capsule_radii=torch.tensor([.06,.08,.06,.08],device=self.device)
        # Where each standing (reactive) episode started; leaving it is not avoidance.
        self.reactive_spot=torch.zeros((self.num_envs,2),device=self.device)
        self.leg_ids=torch.tensor([self.model.body(n).id for n in ('left_knee_link','left_ankle_roll_link','right_knee_link','right_ankle_roll_link')],device=self.device)
        self.hand_radii=tensor([hand_sphere(s)['radius'] for s in ('left','right')])
        sensor_names=[f'{prefix}_{frame}' for frame in ('pelvis','torso') for prefix in ('upvector','global_linvel','global_angvel','local_linvel','gyro')]
        sensor_names+=['left_foot_global_linvel','right_foot_global_linvel']
        self.sensor_indices={}
        for name in sensor_names:
            sensor=self.model.sensor(name);a=int(sensor.adr[0]);n=int(sensor.dim[0])
            self.sensor_indices[name]=slice(a,a+n)
        pairs=[('left_foot','floor'),('right_foot','floor'),('right_foot','left_foot'),('left_foot','right_shin'),('right_foot','left_shin')]
        # Hand sphere x leg capsule pairs (added by model.assemble_training_xml); columns 5:
        # feed the hand_self_contact flag. Guarded so older models without them still load.
        hand_pairs=[(f'furniture_{h}_hand_sphere',leg) for h in ('left','right') for leg in ('left_thigh','right_thigh','left_shin','right_shin')]
        try:
            for a,b in hand_pairs:self.model.geom(a);self.model.geom(b)
            pairs=pairs+hand_pairs
        except KeyError:
            pass
        self.hand_pair_count=len(pairs)-5
        # Opt-in hand vs torso/head/other arm/other hand pairs (model.hand_body_pairs), after the leg pairs.
        self.hand_body_start=len(pairs);self.hand_body_region=None
        if self.config.get('hand_body_contact'):
            from .model import hand_body_pairs
            extra=hand_body_pairs();pairs=pairs+[(a,b) for a,b,_ in extra]
            self.hand_body_region=torch.tensor([HAND_BODY_REGIONS.index(r) for _,_,r in extra],device=self.device)
        self.contact_pairs=torch.tensor([[self.model.geom(a).id,self.model.geom(b).id] for a,b in pairs],device=self.device)
        self.info={};self.navigation={};self.contrast={};self.episode={};self.telemetry={}
        self.obs={'state':torch.zeros((self.num_envs,self.observation_size),device=self.device),'privileged_state':torch.zeros((self.num_envs,self.privileged_observation_size),device=self.device)}
        self.scene_ids=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.scene_episode_ema=torch.zeros(bank.count,device=self.device)
        self.scene_success_ema=torch.zeros_like(self.scene_episode_ema)
        self.curriculum_stage=torch.zeros((),dtype=torch.long,device=self.device)
        width=getattr(bank,'width_curriculum',None)
        self.curriculum_completed=torch.zeros(len(width['rung_widths_m']) if width else 3,dtype=torch.long,device=self.device)
        self.curriculum_goals=torch.zeros_like(self.curriculum_completed)
        self.probabilities=bank.probabilities(stage=self.curriculum_stage)
        self.navigation_counts=torch.zeros((4,2),dtype=torch.long,device=self.device)
        self.contrast_counts=torch.zeros((3,2),dtype=torch.long,device=self.device)
        # Public success counters describe the deployed SAPG leader. Sampling
        # continues to learn from the physical outcomes of every policy.
        self.policy_ids=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.role_counts=torch.zeros((4,2),dtype=torch.long,device=self.device)
        self.zone_steps=torch.zeros((self.num_envs,6,3),dtype=torch.long,device=self.device)
        if config.get('hand_speed_curriculum'):
            if config.get('hand_tolerance_curriculum'):
                raise ValueError('Run speed and tolerance curricula separately')
            from .speed_curriculum import initialize
            self.speed_state=initialize(config['hand_speed_curriculum'],self.num_envs,self.device)
        if config.get('hand_tolerance_curriculum'):
            from .tolerance_curriculum import initialize, RUNGS
            self.tolerance_state=initialize(config['hand_tolerance_curriculum'], self.device)
            self.tolerance_rungs=torch.tensor(RUNGS, device=self.device)
            self.tolerance_episode_rung=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
            self.tolerance_zone_steps=torch.zeros((self.num_envs,6),dtype=torch.long,device=self.device)
        self.outcome_counted=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.episode_reward=torch.zeros(self.num_envs,device=self.device)
        if getattr(self,'balance_settings',None) is not None:
            from .balance import posture_terms
            self.balance_terms=posture_terms
            centers=tensor(self.balance_settings['target_centers']);half=tensor(self.balance_settings['half_size'])
            self.balance_lower=centers-half;self.balance_upper=centers+half
            self.balance_streak=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.raised_reset_fraction=float(_get(config,'hand_raised_reset_fraction',0.))
        if self.raised_reset_fraction > 0:
            from .raised_reset import build_raised_reset_pool
            pool,eligible,certificate=build_raised_reset_pool(self.model,bank.manifest,bank.path,
                _get(config,'wholebody.body_collision.bank_manifest'),proposal_path=collision.proposal_path,
                region_scale=float(_get(config,'hand_contrast_region_scale',.15)),
                tolerance=float(_get(config,'hand_contrast_metric_tolerance',.05)),
                upper_action_scale=action_scales(config) if config.get('upper_action_scales') else float(_get(config,'upper_action_scale',.8)))
            self.raised_reset_pool=tensor(pool)
            self.raised_reset_eligible=torch.as_tensor(eligible,device=self.device)
            config['hand_raised_reset_certificate']=certificate
        else:
            config.pop('hand_raised_reset_certificate',None)
        self.reset()
        # The released learner offsets EpisodeWrapper's horizon only. Native
        # task age remains zero for contact grace and held-odometry cadence.
        if bool(_get(config,'randomize_initial_episode_steps',True)):
            self.info['wrapper_steps']=torch.randint(int(_get(config,'episode_length',1000)),
                (self.num_envs,),generator=self.generator,device=self.device)
            if getattr(self,'balance_settings',None) is not None:
                self.info['wrapper_steps'][self.bank.flat_balance[self.scene_ids]]=0

    def set_policy_ids(self,policy_ids):
        """Assign fixed world policies; policy zero supplies success metrics.

        PPO and single-policy recordings default to all leader worlds. A
        runner must install SAPG assignments before collecting any outcomes;
        changing an already-observed population would mix metric definitions.
        """
        ids=torch.as_tensor(policy_ids,device=self.device)
        if (ids.shape!=(self.num_envs,) or ids.dtype==torch.bool or ids.is_complex()
                or ids.is_floating_point() or (ids<0).any()):
            raise ValueError('Expected one nonnegative integer policy ID per world')
        ids=ids.long()
        if not torch.equal(ids,self.policy_ids) and (self.navigation_counts!=0).any():
            raise ValueError('Cannot change policy assignment after recording outcomes')
        self.policy_ids=ids.clone()

    def enable_compilation(self,*,backend='inductor'):
        """Fuse pure task kernels without capturing mutable state or reset logic.

        No reduce-overhead/CUDA-graph pool is requested: fields and rollout
        storage already occupy substantial VRAM. Dynamic world batches let
        selective resets share kernels when Torch supports the operation.
        Compilation is lazy; the first control transitions perform compilation.
        """
        if self.compiled:return
        from .compilation import compile_batched_kernel
        for name,function in vars(self.math).items():
            setattr(self.math,name,compile_batched_kernel(function,backend=backend))
        for name in ('route_context','swept_root_clearance','hand_contrast_context','contrast_reward_terms'):
            setattr(self,name,compile_batched_kernel(getattr(self,name),backend=backend))
        if hasattr(self.bank,'enable_compilation'):self.bank.enable_compilation(backend=backend)
        if hasattr(self.collision,'enable_compilation'):self.collision.enable_compilation(backend=backend)
        if getattr(self,'balance_settings',None) is not None:self.balance_terms=compile_batched_kernel(self.balance_terms,backend=backend)
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
        if self.config.get('route_recovery'):
            # Consecutive steps without a visible route point; drives the lost-route termination.
            previous_run=torch.zeros(len(ids),dtype=torch.long,device=self.device) if reset else self.navigation['blocked_run'][ids]
            values['blocked_run']=torch.where(values['blocked']&meta['enabled'],previous_run+1,torch.zeros_like(previous_run))
        for key in ('root_clearance','swept_clearance'):values[key]=torch.where(meta['enabled'],values[key],0.)
        for k,v in values.items():self._put(self.navigation,k,ids,v)
        metadata={k:v[self.scene_ids[ids]] for k,v in self.bank.contrast.items()}
        values=self.hand_contrast_context(metadata,values['progress_m'],values['tangent'],
            approach_distance=float(_get(self.config,'hand_contrast_approach_distance',0.)))
        for k,v in values.items():self._put(self.contrast,k,ids,v)

    def _blocking(self):
        """Lazily built 2-D blocking footprint of every scene (cat_mjlab/blocking.py)."""
        if getattr(self,'_blocking_map',None) is None:
            from .blocking import BlockingMap
            self._blocking_map=BlockingMap(self.bank)
        return self._blocking_map

    def _heading_command(self,ids,positions,root_xy,direction,velocity,active,teleop,store=False):
        """Gated heading error for command[:,3] (config 'heading_command').

        Reference heading psi_ref: the joystick's (integrated from its turn rate) in teleop worlds, the
        direction of travel in route worlds (CAT-style point navigation), the current facing when
        standing. Gate: would a body facing psi_ref fit here? Shoulder probes (+-0.16 m) and hand
        probes (+-0.22 m, minus the hand radius) beside the root and 0.3 m ahead along the travel
        direction; 0 at <= 5 cm clearance, 1 at >= 15 cm. Command = gate * wrap(psi_ref - facing): in
        the open the orientation is demanded (strafe, walk backward, turn in place); in a gap too
        narrow to face psi_ref it is not, and the policy may turn sideways to pass.
        """
        q=self.data.qpos[ids,3:7]
        facing_yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
        psi=torch.where(active,torch.atan2(direction[:,1],direction[:,0]),facing_yaw)
        reference=getattr(self,'psi_ref',None)
        if teleop is not None and reference is not None:
            psi=torch.where(teleop,reference[ids],psi)
        facing=torch.stack((psi.cos(),psi.sin()),-1)
        speed=torch.linalg.vector_norm(velocity,dim=-1)
        travel=torch.where((speed>.01)[:,None],velocity/speed.clamp_min(1e-6)[:,None],torch.zeros_like(velocity))
        scenes=self.scene_ids[ids]
        shoulders=self.math.facing_probe_points(root_xy,facing,travel,positions[:,9:11,2].mean(-1),half_width=.16)
        hands=self.math.facing_probe_points(root_xy,facing,travel,positions[:,5:7,2].mean(-1),half_width=.22)
        if self.config.get('blocking_map'):
            # 2-D blocking footprint (cat_mjlab/blocking.py): a beam above the shoulders no longer reads as
            # "a body facing psi_ref does not fit", which released the heading and turned the robot sideways.
            blocking=self._blocking()
            clearance=torch.cat((blocking.sample(shoulders[...,:2],scenes)[0],blocking.sample(hands[...,:2],scenes)[0]-.10),-1)
        else:
            clearance=torch.cat((self.bank.sample('sdf',shoulders,scenes).reshape(len(ids),-1),
                                 self.bank.sample('sdf',hands,scenes).reshape(len(ids),-1)-.10),-1)
        gate=self.math.clearance_gate(clearance)
        error=torch.remainder(psi-facing_yaw+torch.pi,2*torch.pi)-torch.pi
        commanded=self._commanded(ids)
        if store:
            self._heading_gate,self._heading_error,self._heading_commanded=gate,error,commanded
        return torch.where(commanded,gate*error,torch.zeros_like(error))

    def _teleop_resample_body(self,due):
        """Joystick commands in the body frame: (forward, sideways, turn rate).

        Stop with p_stop; turn in place with p_turn (|rate| in [0.4, yaw_max]); THROUGH with p_through
        (p_through_cat in CAT scenes): forward along the scene's stored goal field at the root -- it routes
        under beams, over hurdles and through gaps -- with the reference heading turned to it, as an
        operator steers at an opening; toward the nearest obstacle with p_toward (translation only);
        otherwise forward in [vx_min, vx_max], sideways within +-vy_max and a turn rate within +-yaw_max.
        A stop also sets the reference heading to the current facing: stop means stand where you are.
        """
        t=self.config['teleop'];n=self.num_envs;i=self.info
        r=self._rand((n,));p_stop,p_turn=float(t['p_stop']),float(t['p_turn'])
        stop=r<p_stop;turn=(r>=p_stop)&(r<p_stop+p_turn)
        toward=~stop&~turn&(self._rand((n,))<float(t['p_toward']))
        vx=self._rand((n,),float(t['vx_min']),float(t['vx_max']));vy=self._rand((n,),-float(t['vy_max']),float(t['vy_max']))
        wz=self._rand((n,),-float(t['yaw_max']),float(t['yaw_max']))
        spin=self._rand((n,),.4,float(t['yaw_max']))*torch.where(self._rand((n,))<.5,-1.,1.)
        away=i['bf'][:,1,:2];norm=torch.linalg.vector_norm(away,dim=-1,keepdim=True)
        world=torch.where(norm>1e-6,-away/norm.clamp_min(1e-6),torch.zeros_like(away))
        toward=toward&(norm[:,0]>1e-6)          # no obstacle in range (open floor): keep the random command
        if t.get('mode')=='heading':            # commands are relative to the reference heading
            axis=torch.stack((self.psi_ref.cos(),self.psi_ref.sin()),-1)
            bx=(world*axis).sum(-1);by=(world*torch.stack((-axis[:,1],axis[:,0]),-1)).sum(-1)
        else:
            bx=(world*i['navi'][:,:2,0]).sum(-1);by=(world*i['navi'][:,:2,1]).sum(-1)
        speed=self._rand((n,),float(t['speed_min']),float(t['speed_max']))
        vx=torch.where(toward,speed*bx,vx);vy=torch.where(toward,speed*by,vy);wz=torch.where(toward,0.,wz)
        vx=torch.where(turn,0.,vx);vy=torch.where(turn,0.,vy);wz=torch.where(turn,spin,wz)
        q=self.data.qpos[:,3:7]
        facing=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
        if t.get('mode')=='heading':
            p_through=torch.where(self.bank.is_cat[self.scene_ids],float(t.get('p_through_cat',0.)),float(t.get('p_through',0.)))
            through=due&~stop&~turn&(self._rand((n,))<p_through)
            if bool(through.any()):
                goal=self.bank.sample('gf',self.data.qpos[:,None,:3],self.scene_ids)[:,0,:2]
                norm=torch.linalg.vector_norm(goal,dim=-1)
                through=through&(norm>1e-3)
                self.psi_ref=torch.where(through,torch.atan2(goal[:,1],goal[:,0]),self.psi_ref)
                fast=self._rand((n,),max(.2,float(t['vx_min'])),float(t['vx_max']))
                vx=torch.where(through,fast,vx);vy=torch.where(through,0.,vy);wz=torch.where(through,0.,wz)
            self.psi_ref=torch.where(due&stop,facing,self.psi_ref)
        command=torch.stack((vx,vy,wz),-1)*(~stop)[:,None]
        self.joystick_body=torch.where(due[:,None],command,self.joystick_body)
        self._teleop_timer=torch.where(due,self._rand((n,),float(t['hold_min']),float(t['hold_max'])),self._teleop_timer)

    def _joystick_body_to_world(self):
        """Body-frame (forward, sideways, turn rate) -> world-frame velocity + turn rate, every step."""
        body=self.joystick_body;navi=self.info['navi']
        if self.config.get('heading_command') and getattr(self,'psi_ref',None) is not None:
            # (forward, sideways) relative to the reference heading, which the turn rate rotates; the
            # walking direction stays fixed in the world while the body turns sideways in a gap.
            mask=getattr(self,'joystick_mask',None)
            spin=body[:,2]*self.dt if mask is None else torch.where(mask,body[:,2]*self.dt,torch.zeros_like(body[:,2]))
            self.psi_ref=torch.remainder(self.psi_ref+spin+torch.pi,2*torch.pi)-torch.pi
            axis=torch.stack((self.psi_ref.cos(),self.psi_ref.sin()),-1)
            world=body[:,:1]*axis+body[:,1:2]*torch.stack((-axis[:,1],axis[:,0]),-1)
        else:
            world=body[:,:1]*navi[:,:2,0]+body[:,1:2]*navi[:,:2,1]
        if getattr(self,'joystick',None) is None or self.joystick.shape!=world.shape:
            self.joystick=world.clone()
        else:
            mask=getattr(self,'joystick_mask',None)
            self.joystick=world if mask is None else torch.where(mask[:,None],world,self.joystick)
        self.joystick_yaw=body[:,2].clone()

    def _commanded(self,ids):
        """Worlds whose command is room-style (route or joystick velocity) rather than CAT's field command.

        Rooms always; with a joystick, also any other scene it drives (e.g. the flat scene for
        command-following tests) -- all worlds when no per-world mask exists (recorder / demo).
        """
        enabled=self.navigation['enabled'][ids]
        if getattr(self,'joystick',None) is None:return enabled
        mask=getattr(self,'joystick_mask',None)
        return enabled|(torch.ones_like(enabled) if mask is None else mask[ids])

    def _teleop_resample(self):
        """Joystick-style commands for teleoperated room episodes (config 'teleop').

        Every hold period a new command: a stop with probability p_stop, otherwise a speed in
        [speed_min, speed_max] toward the nearest obstacle (minus the root's distance-field gradient)
        with probability p_toward, else in a uniformly random direction. Nothing steers the command
        away from obstacles: contact ends the episode, so refusing a colliding command is learned.
        """
        t=self.config['teleop'];n=self.num_envs
        self._teleop_timer-=self.dt
        due=self.joystick_mask&(self._teleop_timer<=0)
        if not bool(due.any()):return
        if t.get('mode') in ('body','heading'):
            self._teleop_resample_body(due);return
        angle=self._rand((n,),-torch.pi,torch.pi);random_dir=torch.stack((angle.cos(),angle.sin()),-1)
        away=self.info['bf'][:,1,:2];norm=torch.linalg.vector_norm(away,dim=-1,keepdim=True)
        toward=torch.where(norm>1e-6,-away/norm.clamp_min(1e-6),random_dir)
        direction=torch.where((self._rand((n,))<float(t['p_toward']))[:,None],toward,random_dir)
        speed=self._rand((n,),float(t['speed_min']),float(t['speed_max']))*(self._rand((n,))>=float(t['p_stop']))
        self.joystick=torch.where(due[:,None],direction*speed[:,None],self.joystick)
        self._teleop_timer=torch.where(due,self._rand((n,),float(t['hold_min']),float(t['hold_max'])),self._teleop_timer)

    def _gate_head_pull(self,replacement,sdf):
        """Rooms: keep the head field's vertical component only with geometry near the head.

        Room fields were generated toward a goal at z=0.75 m, so within ~1.2 m of every goal the
        head-height field points steeply down (z -0.4..-0.9) with nothing overhead. That opened the
        crouch gate (posture rewards off) and headgf paid for diving the head toward the goal:
        measured torso pitch >15 deg in 54-60% of frames within 1 m of the goal, 0% beyond 2 m.
        """
        near=self.config.get('head_guidance_near_sdf')
        if near:
            replacement[:,0,2]=torch.where(sdf[:,0,0]<float(near),replacement[:,0,2],torch.zeros_like(replacement[:,0,2]))
        return replacement

    def _fields(self,positions,root_xy,ids,store=False):
        scenes=self.scene_ids[ids]
        gf,bf,sdf=(self.bank.sample(k,positions,scenes) for k in ('gf','bf','sdf'))
        sdf=sdf.clone();sdf[:,5:7,0]-=self.hand_radii
        nav={k:v[ids] for k,v in self.navigation.items()};meta=self._room_meta(ids)
        delta=nav['target']-root_xy;length=torch.linalg.vector_norm(delta,dim=-1)
        direction=delta/length.clamp_min(1e-6)[:,None]
        visible=self.swept_root_clearance(root_xy,nav['target'],meta['obstacles'],meta['obstacle_count'],radius=meta['navigation_radius'])>0
        active=visible&~nav['blocked']&~nav['violation']
        speed=torch.linalg.vector_norm(nav['guidance'][:,:2],dim=-1)
        recovery=self.config.get('route_recovery')
        if recovery:
            # Lost the route (neither the carrot nor the route projection is in straight-line view):
            # the command used to drop to zero and the guidance to vanish, so the robot was told to
            # stand with no direction until the 80 s timeout. Follow the scene's stored geodesic
            # (FMM) goal field instead -- it routes around the obstacles -- until the route is visible.
            lost=nav['enabled']&nav['blocked']&~nav['violation']
            horizontal=gf[:,1,:2];magnitude=torch.linalg.vector_norm(horizontal,dim=-1)
            lost=lost&(magnitude>1e-3)
            direction=torch.where(lost[:,None],horizontal/magnitude.clamp_min(1e-6)[:,None],direction)
            speed=torch.where(lost,torch.full_like(speed,float(recovery['speed'])),speed)
            active=active|lost
        joystick=getattr(self,'joystick',None);teleop=None
        if joystick is not None:
            # Teleoperation (recordings / keyboard demo): a commanded world-frame velocity [N,2]
            # replaces the route. Everything downstream is unchanged: the root guidance is the raw
            # command, other body points get the usual obstacle-projected guidance, and the policy
            # sees the same fields -- nothing here steers the command away from obstacles.
            joy=joystick[ids];joy_speed=torch.linalg.vector_norm(joy,dim=-1)
            mask=getattr(self,'joystick_mask',None)
            teleop=torch.ones_like(joy_speed,dtype=torch.bool) if mask is None else mask[ids]
            direction=torch.where((teleop&(joy_speed>1e-3))[:,None],joy/joy_speed.clamp_min(1e-6)[:,None],direction)
            speed=torch.where(teleop,joy_speed,speed);active=torch.where(teleop,joy_speed>1e-3,active)
        if hasattr(self, 'speed_state'):
            from .speed_curriculum import speed_limit
            scoped, cap, hold=speed_limit(self.speed_state,{k:v[ids] for k,v in self.contrast.items()},ids,self.dt)
            speed=torch.where(scoped,torch.minimum(speed,cap),speed)
        velocity=direction*(speed*active)[:,None]
        yaw_command=torch.zeros_like(velocity[:,0])
        joystick_yaw=getattr(self,'joystick_yaw',None)
        if teleop is not None and joystick_yaw is not None:
            yaw_command=torch.where(teleop,joystick_yaw[ids],yaw_command)
        if self.config.get('heading_command'):
            yaw_command=self._heading_command(ids,positions,root_xy,direction,velocity,active,teleop,store=store)
        deadband=self.config.get('heading_deadband')
        if deadband is None:
            turning=yaw_command.abs()>.05
        else:
            # A standing robot never holds its heading to 0.05 rad, so that threshold kept the gait on (and the
            # robot marking time, drifting ~1.2 m in 20 s) with no command at all. Hysteresis: start turning in
            # place above `on`, keep turning until below `off`; a held turn key (joystick yaw rate) turns at once.
            on,off=(float(v) for v in deadband)
            if getattr(self,'_turning',None) is None:
                self._turning=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
            error=yaw_command.abs()
            turning=self.math.hysteresis(error,self._turning[ids],on=on,off=off)
            if teleop is not None and joystick_yaw is not None:
                turning=turning|(teleop&(joystick_yaw[ids].abs()>.05))
            if store:
                self._turning[ids]=turning
        moving=(torch.linalg.vector_norm(velocity,dim=-1)>.01)|turning   # turning in place moves the feet
        room_command=torch.cat((torch.where(moving,.75,0.)[:,None],velocity,yaw_command[:,None]),-1)
        replacement=gf.clone();replacement[:,:,:2]=direction[:,None]*.6
        replacement=self._gate_head_pull(replacement,sdf)
        if hasattr(self, 'speed_state'):
            replacement=torch.where(hold[:,None,None],0.,replacement)
        normal=bf/(torch.linalg.vector_norm(bf,dim=-1,keepdim=True)+1e-9)
        inward=(replacement*normal).sum(-1,keepdim=True).clamp_max(0)
        replacement-=inward*normal*(sdf<.5);replacement*=active[:,None,None]
        replacement[:,1]=torch.cat((velocity,torch.zeros_like(velocity[:,:1])),-1)
        commanded=self._commanded(ids)
        gf=torch.where(commanded[:,None,None],replacement,gf)
        cmd=self.math.compute_cmd_from_rtf(gf[:,1],gf[:,[0,3,4,5,6]],bf[:,[0,3,4,5,6]])
        cmd=torch.where(commanded[:,None],room_command,cmd)
        if self.reactive_objects is not None:
            active=self.reactive_objects.state['active'][ids]
            # Standing episodes get a zero command; the walking variant keeps the route command.
            hold=active&~self.reactive_objects.state['walking'][ids]
            gf=torch.where(hold[:,None,None],0.,gf)
            bf=torch.where(active[:,None,None],0.,bf)
            sdf=torch.where(active[:,None,None],10.,sdf)
            cmd=torch.where(active[:,None],0.,cmd)
        if self.analytic_objects is not None:
            # Dynamic perception has zero added latency. Use actual body/world
            # geometry for both actor and critic, independent of held odometry
            # used by the legacy static field. Navigation commands stay static.
            radii=sdf.new_zeros((1,positions.shape[1],1));radii[:,5:7,0]=self.hand_radii
            gf,bf,sdf=self.analytic_objects.merge(self._poses(ids),ids,self.data.time[ids],
                gf,bf,sdf,radii)
        if self.reactive_objects is not None and bool(_get(self.config,'reactive_hand_guidance',False)):
            # gf is zeroed above before the object is merged, and merge computes
            # guidance = gf - inward*normal, which stays zero when gf is zero. That left
            # handsgf -- the only reward term anywhere that depends on hand VELOCITY
            # relative to an obstacle -- identically zero in exactly the scenes built to
            # train hand retreat. Point the hand guidance along the outward normal so
            # moving away from the object is what earns the alignment reward.
            active=self.reactive_objects.state['active'][ids]
            outward=bf[:,5:7]/(torch.linalg.vector_norm(bf[:,5:7],dim=-1,keepdim=True)+1e-6)
            gf=gf.clone();gf[:,5:7]=torch.where(active[:,None,None],outward,gf[:,5:7])
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
        if self.reactive_objects is not None:
            active=self.reactive_objects.state['active'][ids]
            gf=torch.where(active[:,None,None],0.,gf);bf=torch.where(active[:,None,None],0.,bf)
            sdf=torch.where(active[:,None,None],10.,sdf)
            gf,bf,sdf=self.reactive_objects.arm_fields(self.data,ids,gf,bf,sdf)
        elif self.analytic_objects is not None:
            gf,bf,sdf=self.analytic_objects.merge(self.data.site_xpos[ids][:,self.elbow_ids],ids,
                self.data.time[ids],gf,bf,sdf,.05)
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
        sdf_rate=None
        if self.sdf_rate:
            # Rate of change of what the actor already sees (delayed hand distances, elbow distances).
            current=torch.cat((info['sdf_delay'][:,5:7].reshape(len(ids),2),torch.nan_to_num(elbow_actor)[:,self.elbow_df_index]),-1)
            if 'sdf_rate_prev' not in self.info:self.info['sdf_rate_prev']=torch.zeros((self.num_envs,4),device=self.device)
            fresh=(info['step']==0)[:,None]
            sdf_rate=torch.where(fresh,0.,(current-self.info['sdf_rate_prev'][ids])/self.dt).clamp(-3.,3.)
            self.info['sdf_rate_prev'][ids]=current
        result=self.math.observations(keep_yaw=bool(self.config.get('yaw_command') or self.config.get('heading_command')),sdf_rate=sdf_rate,joint_pos=self.data.qpos[ids,7:],joint_vel=self.data.qvel[ids,6:],nominal=self.nominal,
            gyro=self._sensor('gyro_pelvis',ids),gravity=gravity,linear_velocity=self._sensor('local_linvel_pelvis',ids),
            noise=noise,last_action=info['last_act'],targets=info['motor_targets'],command=info['command'],
            command_delay=info['command_delay'],foot_height=info['foot_height'],phase=info['phase'],navi=info['navi'],
            gf=info['gf'],bf=info['bf'],sdf=info['sdf'],gf_delay=info['gf_delay'],bf_delay=info['bf_delay'],sdf_delay=info['sdf_delay'],
            positions=info['positions'],velocities=info['velocities'],torso_rpy=info['torso_rpy'],gait=info['gait'],contacts=contacts,
            kp=info['kp'],kd=info['kd'],rfi=info['rfi'],elbow_true=elbow_true,elbow_actor=torch.nan_to_num(elbow_actor))
        if result['state'].shape[-1]!=self.observation_size or result['privileged_state'].shape[-1]!=self.privileged_observation_size:
            raise RuntimeError('Observation feature contract changed')
        for k,v in result.items():self.obs[k][ids]=v
        return result

    @torch.no_grad()
    def reset(self,env_ids=None,scene_ids=None):
        ids=self.all_ids if env_ids is None else env_ids.long();n=len(ids)
        if n==0:return self.obs
        if getattr(self,'_style_age',None) is not None:self._style_age[ids]=0     # strided style pairs restart
        if scene_ids is None:
            cdf=self.probabilities.cumsum(0);cdf=cdf/cdf[-1]
            scene_ids=torch.searchsorted(cdf,self._rand((n,))).clamp_max(self.bank.count-1)
        if self.reactive_objects is not None:
            scene_ids,reactive_active,reactive_qpos=self.reactive_objects.sample(ids,scene_ids)
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
        seeded=torch.zeros(n,dtype=torch.bool,device=self.device)
        if self.raised_reset_fraction:
            seeded=self.raised_reset_eligible[scenes]&(self._rand((n,))<self.raised_reset_fraction)
            pick=torch.randint(self.raised_reset_pool.shape[1],(n,),generator=self.generator,device=self.device)
            qpos=torch.where(seeded[:,None],self.raised_reset_pool[scenes,pick],qpos)
        if self.reactive_objects is not None and not self.reactive_objects.nominal_reset:
            qpos=torch.where(reactive_active[:,None],reactive_qpos,qpos)
        if self.reactive_objects is not None:
            self.reactive_spot[ids]=qpos[:,:2]   # the standing spot; leaving it is measured, not rewarded
        self.sim.reset_data(ids)
        self.data.qpos[ids]=qpos;self.data.qvel[ids]=0.;self.data.qvel[ids,:6]=self._rand((n,6),-.5,.5)
        if self.raised_reset_fraction:
            self.data.qvel[ids]=torch.where(seeded[:,None],0.,self.data.qvel[ids])
        if self.reactive_objects is not None:
            self.data.qvel[ids]=torch.where(reactive_active[:,None],0.,self.data.qvel[ids])
        self.data.ctrl[ids]=qpos[:,7:];self.sim.forward(ids)
        if self.reactive_objects is not None and self.reactive_objects.nominal_reset and bool(reactive_active.any()):
            # Aim the first object at where the hand IS in the ordinary reset pose.
            self.reactive_objects.rearm(self.data,ids[reactive_active],self.reactive_objects.sampling_generator)
        replaced=torch.zeros(n,dtype=torch.bool,device=self.device)
        if self.collision is not None:
            reset_regions=self.collision(scenes,_selected_data(self.data,ids))
            replaced=reset_regions.any(-1)
            if self.raised_reset_fraction and bool((replaced&seeded).any()):
                raise RuntimeError('Certified raised reset failed runtime collision check')
            pick=torch.randint(self.bank.reset_pool.shape[1],(n,),generator=self.generator,device=self.device)
            qpos=torch.where(replaced[:,None],self.bank.reset_pool[scenes,pick],qpos)
            self.data.qpos[ids]=qpos;self.data.ctrl[ids]=qpos[:,7:];self.sim.forward(ids)
        # Logging only: observe the actual post-replacement reset with the
        # existing detector. Never reuse the rejected pose as trial evidence.
        if not hasattr(self, 'acceptance_reset_regions'):
            self.acceptance_reset_regions=torch.zeros((self.num_envs,6),dtype=torch.bool,device=self.device)
        self.acceptance_reset_regions[ids]=(self.collision(scenes,_selected_data(self.data,ids))
            if self.collision is not None else False)
        defaults={'step':torch.zeros(n,dtype=torch.long,device=self.device),
            **({'goal_steps':torch.zeros(n,dtype=torch.long,device=self.device)} if self.config.get('goal_hold_seconds') else {}),
            'wrapper_steps':torch.zeros(n,dtype=torch.long,device=self.device),'push_step':torch.zeros(n,dtype=torch.long,device=self.device),
            'motor_targets':self.nominal.expand(n,-1).clone(),'last_act':torch.zeros((n,29),device=self.device),
            'last_last_act':torch.zeros((n,29),device=self.device),'last_joint_vel':torch.zeros((n,29),device=self.device),
            'navi':torch.eye(3,device=self.device).expand(n,-1,-1).clone(),
            'pelvis_rpy':torch.zeros((n,3),device=self.device),'torso_rpy':torch.zeros((n,3),device=self.device),
            'torso_angvel':torch.zeros((n,3),device=self.device),'stop_timestep':torch.full((n,),100,dtype=torch.long,device=self.device),
            'gait':torch.zeros((n,2),device=self.device),'odom_delay':qpos[:,:7].clone(),
            'previous_upper':self.nominal[12:].expand(n,-1).clone(),'previous_previous_upper':self.nominal[12:].expand(n,-1).clone()}
        if self.raised_reset_fraction:
            defaults['hand_raised_seeded']=seeded
            # Match controller memory to the physical reset; no fictitious
            # acceleration impulse or slew-limited return from nominal memory.
            defaults['motor_targets']=torch.where(seeded[:,None],qpos[:,7:],defaults['motor_targets'])
            for key in ('previous_upper','previous_previous_upper'):
                defaults[key]=torch.where(seeded[:,None],qpos[:,19:],defaults[key])
            action=torch.zeros_like(defaults['last_act'])
            action[:,12:]=(qpos[:,19:]-self.nominal[12:])/self.upper_action_scales
            if bool((action[seeded,12:].abs()>1.+1e-6).any()):
                raise ValueError('Raised reset is outside the commandable action set')
            action.clamp_(-1.,1.)
            for key in ('last_act','last_last_act'):
                defaults[key]=torch.where(seeded[:,None],action,defaults[key])
        if self.reactive_objects is not None and not self.reactive_objects.nominal_reset:
            defaults['motor_targets']=torch.where(reactive_active[:,None],qpos[:,7:],defaults['motor_targets'])
            for key in ('previous_upper','previous_previous_upper'):
                defaults[key]=torch.where(reactive_active[:,None],qpos[:,19:],defaults[key])
            action=torch.zeros_like(defaults['last_act'])
            action[:,12:]=(qpos[:,19:]-self.nominal[12:])/self.upper_action_scales
            for key in ('last_act','last_last_act'):
                defaults[key]=torch.where(reactive_active[:,None],action,defaults[key])
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
        if self.config.get('heading_command'):
            if getattr(self,'psi_ref',None) is None:
                self.psi_ref=torch.zeros(self.num_envs,device=self.device)
            q=self.data.qpos[ids,3:7]
            self.psi_ref[ids]=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
        if getattr(self,'_turning',None) is not None:
            self._turning[ids]=False
        teleop=self.config.get('teleop')
        if teleop:
            if getattr(self,'joystick_mask',None) is None:
                self.joystick=torch.zeros((self.num_envs,2),device=self.device)
                self.joystick_mask=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
                self._teleop_timer=torch.zeros(self.num_envs,device=self.device)
            eligible=self.navigation['enabled'][ids]
            fraction=torch.full((n,),float(teleop['fraction']),device=self.device)
            if teleop.get('cat_scenes'):
                # Keyboard practice around CAT obstacles too (before: rooms and the flat scene only, so a CAT
                # expert never drove a beam, hurdle or gap by joystick).
                cat=self.bank.is_cat[self.scene_ids[ids]]
                eligible=eligible|cat
                if 'cat_fraction' in teleop:   # keep most CAT practice field-driven (the traversal skill)
                    fraction=torch.where(cat,float(teleop['cat_fraction']),fraction)
            self.joystick_mask[ids]=(self._rand((n,))<fraction)&eligible
            self.joystick[ids]=0.;self._teleop_timer[ids]=0.
            if teleop.get('mode') in ('body','heading'):
                if getattr(self,'joystick_body',None) is None:
                    self.joystick_body=torch.zeros((self.num_envs,3),device=self.device)
                    self.joystick_yaw=torch.zeros(self.num_envs,device=self.device)
                self.joystick_body[ids]=0.;self.joystick_yaw[ids]=0.
        if hasattr(self, 'speed_state'):
            from .speed_curriculum import reset as reset_speed
            reset_speed(self.speed_state,ids,self.navigation['progress_m'][ids])
        positions=self._poses(ids);gf,bf,sdf,command=self._fields(positions,qpos[:,:2],ids)
        # Original reset normalizes before deriving command; rooms override it.
        gf=gf/(torch.linalg.vector_norm(gf,dim=-1,keepdim=True)+1e-6)
        bf=bf/(torch.linalg.vector_norm(bf,dim=-1,keepdim=True)+1e-6)
        normalized_command=self.math.compute_cmd_from_rtf(gf[:,1],gf[:,[0,3,4,5,6]],bf[:,[0,3,4,5,6]])
        command=torch.where(self.navigation['enabled'][ids,None],command,normalized_command)
        for k,v in dict(gf=gf,bf=bf,sdf=sdf,gf_delay=gf,bf_delay=bf,sdf_delay=sdf,positions=positions,
                        velocities=torch.zeros_like(positions),command=command,command_delay=command,last_command=torch.zeros_like(command)).items():
            self._put(self.info,k,ids,v)
        if hasattr(self, 'speed_state'):
            from .speed_curriculum import speed_limit, standing_phase
            _,_,hold=speed_limit(self.speed_state,{k:v[ids] for k,v in self.contrast.items()},ids,self.dt)
            values=standing_phase(self.info['command'][ids],self.info['command_delay'][ids],
                self.info['stop_timestep'][ids],self.info['phase'][ids],self.info['gait'][ids],hold)
            for key,value in zip(('command','command_delay','stop_timestep','phase','gait'),values):
                self.info[key][ids]=value
        if self.reactive_objects is not None:
            from .speed_curriculum import standing_phase as _stance
            standing=self.reactive_objects.state['active'][ids]&~self.reactive_objects.state['walking'][ids]
            values=_stance(self.info['command'][ids],self.info['command_delay'][ids],self.info['stop_timestep'][ids],self.info['phase'][ids],self.info['gait'][ids],standing)
            for key,value in zip(('command','command_delay','stop_timestep','phase','gait'),values):
                self.info[key][ids]=value
        keys=('goal_reached','raw_goal','outside_bounds','fall','obstacle','self_contact','hand_self_contact','reactive_left_spot','numerical','hand_violation','elbow_violation','body_collision','reset_replaced',
              'body_collision_feet','body_collision_legs','body_collision_trunk','body_collision_head','body_collision_arms','body_collision_hands')
        if self.config.get('route_recovery'):keys+=('route_lost',)
        if self.config.get('hand_body_contact'):keys+=('hand_body_contact',)
        for k in keys:self._put(self.episode,k,ids,replaced if k=='reset_replaced' else torch.zeros(n,dtype=torch.bool,device=self.device))
        self._put(self.info,'minimum_episode_hand_clearance',ids,sdf[:,5:7,0].amin(-1))
        from .response_split import initial, advance, PREFIX
        response_state = initial(self.data.qpos[ids,:3])
        advance(response_state, positions[:,5:7], self.data.qpos[ids,:3], sdf[:,5:7,0],
                bf[:,5:7], torch.zeros(n,dtype=torch.bool,device=self.device))
        for key, value in response_state.items():
            self._put(self.info, PREFIX+key, ids, value)
        self.episode['outside_bounds'][ids]=self._goal_status(ids)[2]
        if hasattr(self, 'tolerance_state'):
            self.tolerance_episode_rung[ids]=self.tolerance_state['stage']
            self.tolerance_zone_steps[ids]=0
        self.outcome_counted[ids]=False;self.zone_steps[ids]=0;self.episode_reward[ids]=0
        if getattr(self,'balance_settings',None) is not None:self.balance_streak[ids]=0
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
        self_contact=contacts[:,2:5].any(-1)&grace
        hand_self_contact=(contacts[:,5:5+self.hand_pair_count].any(-1) if self.hand_pair_count else torch.zeros_like(grace))&grace
        fall=(self._sensor('upvector_pelvis',self.all_ids)[:,2]<0)|(i['positions'][:,0,2]<.7)
        numerical=torch.isnan(self.data.qpos).any(-1)|torch.isnan(self.data.qvel).any(-1)
        root=self.navigation['enabled']&self.navigation['violation'];body=regions.any(-1)
        if getattr(self,'joystick',None) is not None:   # teleop leaves the route on purpose
            mask=getattr(self,'joystick_mask',None)
            root=torch.zeros_like(root) if mask is None else root&~mask
        obstacle=fields|elbows|root|body;done=fall|self_contact|numerical|obstacle
        if _get(self.config,'terminate_on_hand_self_contact',False):done=done|hand_self_contact
        recovery=self.config.get('route_recovery')
        route_lost=None
        if recovery:
            route_lost=self.navigation['enabled']&(self.navigation['blocked_run']*self.dt>float(recovery['lost_seconds']))
            done=done|route_lost
        active=self.reactive_objects.state['active'] if self.reactive_objects is not None else torch.zeros_like(grace)
        self.reactive_root_displacement=torch.linalg.vector_norm(i['positions'][:,1,:2]-self.reactive_spot,dim=-1)*active
        walking=self.reactive_objects.state['walking'] if self.reactive_objects is not None else torch.zeros_like(grace)
        reactive_left_spot=active&~walking&(self.reactive_root_displacement>float(_get(self.config,'reactive_spot_radius',.3)))
        flags=dict(fall=fall,obstacle=obstacle,self_contact=self_contact,hand_self_contact=hand_self_contact,reactive_left_spot=reactive_left_spot,numerical=numerical,body_collision=body,
                   hand_violation=(i['sdf'][:,5:7]<threshold).flatten(1).any(-1)&grace,elbow_violation=elbows)
        if route_lost is not None:flags['route_lost']=route_lost
        if self.hand_body_region is not None:flags['hand_body_contact']=contacts[:,self.hand_body_start:].any(-1)&grace
        for k,v in flags.items():self.episode[k]|=v
        for index,region in enumerate(('feet','legs','trunk','head','arms','hands')):
            self.episode['body_collision_'+region]|=regions[:,index]
        clean,raw,outside=self._goal_status(self.all_ids);self.episode['outside_bounds']=outside
        self.episode['goal_reached']|=clean&~done;self.episode['raw_goal']|=raw&~done
        if self.collision is not None:self.episode['goal_reached']&=~done
        return done,flags

    def _style_features(self):
        """89-D motion features for the style discriminators (layout of the motion library)."""
        from .style_prior import robot_features, KEY_SITES, KEY_BODIES
        if not hasattr(self,'_style_ids'):
            self._style_ids=(torch.tensor([self.model.site(n).id for n in KEY_SITES],device=self.device),
                             torch.tensor([self.model.body(n).id for n in KEY_BODIES],device=self.device))
        s,b=self._style_ids;d=self.data
        xmat=d.xmat.reshape(self.num_envs,-1,3,3)[:,self.pelvis_body]
        return robot_features(d.qpos[:,:36],d.qvel[:,:35],d.site_xpos[:,s],d.xpos[:,b],xmat)

    def _rewards(self,action,contacts):
        i=self.info;d=self.data
        from .passage_rewards import blended_sdf_knee
        sdf_knee = (blended_sdf_knee(self.bank.contrast, self.scene_ids, self.contrast)
                    if getattr(self.bank, 'has_sdf_reward_overrides', False) else None)
        heading=self.config.get('heading_align');heading_sdf=None;heading_margins=(.05,.15)
        if heading:
            # Counterfactual shoulder probes at shoulder height, sampled from the static bank only:
            # moving (analytic/reactive) objects live in standing scenes where move=0 zeroes the
            # term anyway. Reward-only -- these never enter observations, so obs dims are unchanged.
            cmd=i['command'][:,1:3];norm=torch.linalg.vector_norm(cmd,dim=-1,keepdim=True)
            direction=torch.where(norm>0,cmd/norm.clamp_min(1e-30),torch.zeros_like(cmd))
            probes=self.math.heading_probe_points(i['positions'][:,1,:2],direction,i['positions'][:,9:11,2].mean(-1),
                half_width=float(heading['half_width']),lookahead=float(heading['lookahead']))
            heading_sdf=self.bank.sample('sdf',probes,self.scene_ids).reshape(probes.shape[0],-1)
            hand=heading.get('hand_probe')
            if hand:
                hands=self.math.heading_probe_points(i['positions'][:,1,:2],direction,i['positions'][:,5:7,2].mean(-1),
                    half_width=float(hand['half_width']),lookahead=float(heading['lookahead']))
                hand_sdf=self.bank.sample('sdf',hands,self.scene_ids).reshape(hands.shape[0],-1)-float(hand['radius'])
                heading_sdf=torch.minimum(heading_sdf,hand_sdf)
            self._heading_sdf=heading_sdf     # reused by the style-prior gates
            heading_margins=tuple(float(m) for m in heading['margins'])
        facing_yaw=torch.atan2(i['navi'][:,1,0],i['navi'][:,0,0])
        reward_command=i['command']
        if self.config.get('safe_tracking'):
            # Reward the command minus its component into nearby obstacles (pelvis, hands, knees,
            # shoulders): driven at a table, stopping or sliding along it is what earns the reward.
            points=[1,5,6,7,8,9,10]
            if self.config.get('blocking_map'):
                # Only what blocks the body band repels: the 3-D field's horizontal part, rescaled to unit
                # length, read a beam overhead / a hurdle underfoot as a wall ahead (target: stop).
                distance,away=self._blocking().sample(i['positions'][:,points,:2],self.scene_ids)
            else:
                away=i['bf'][:,points,:2];away=away/(torch.linalg.vector_norm(away,dim=-1,keepdim=True)+1e-9)
                distance=i['sdf'][:,points,0]
            target=self.math.safe_velocity_target(reward_command[:,1:3],away,distance)
            reward_command=reward_command.clone()
            scope=self._commanded(self.all_ids)
            mask=getattr(self,'joystick_mask',None)
            if self.config.get('safe_tracking_teleop_only') and mask is not None:
                scope=scope&mask           # route worlds follow a certified route; repulsion is for the joystick
            reward_command[:,1:3]=torch.where(scope[:,None],target,reward_command[:,1:3])
            self._safe_target=target
        rewards=self.math.native_rewards(facing_yaw=facing_yaw,stand_still='stand_still' in _get(self.config,'reward_config.scales',{}),
            stillness_speed=_get(self.config,'standing_requires_stillness_speed',None),heading_sdf=heading_sdf,heading_margins=heading_margins,standing_gf=float(_get(self.config,'standing_gf_bonus',4.)),sdf_knee=sdf_knee,action=action,last_action=i['last_act'],last_last_action=i['last_last_act'],
            joint_pos=d.qpos[:,7:],joint_vel=d.qvel[:,6:],last_joint_vel=i['last_joint_vel'],lower=self.lower,upper=self.upper,
            actuator_force=d.actuator_force,command=reward_command,pelvis_rpy=i['pelvis_rpy'],torso_rpy=i['torso_rpy'],
            head_z=i['positions'][:,0,2],torso_height=float(_get(self.config,'torso_height',[.5,1.])[1]),
            global_velocity=self._sensor('global_linvel_pelvis',self.all_ids),torso_angvel=i['torso_angvel'],navi=i['navi'],
            leg_rotations=d.xmat.reshape(self.num_envs,-1,3,3)[:,self.leg_ids],feet_pos=i['positions'][:,3:5],
            feet_sensor_velocity=torch.stack((self._sensor('left_foot_global_linvel',self.all_ids),self._sensor('right_foot_global_linvel',self.all_ids)),1),
            subtree_com=d.subtree_com[:,self.pelvis_body],feet_contact=contacts[:,:2],gait=i['gait'],foot_height=i['foot_height'],
            foot_height_stance=float(_get(self.config,'reward_config.foot_height_stance',0.)),gf=i['gf'],positions=i['positions'],velocities=i['velocities'],
            sdf=i['sdf'],crossed=self._crossed(i['positions'],self.all_ids),dt=self.dt,max_yaw=abs(float(_get(self.config,'ang_vel_yaw',[-.5,.5])[1])))
        if self.config.get('protected_hand_sdf_margin',False) and self.hand_contrast:
            rewards['handsdf']=tm.protected_hand_sdf_reward(i['sdf'][:,5:7],self.contrast,sdf_knee)
        balance=getattr(self,'balance_settings',None) is not None
        if balance:
            flat=self.bank.flat_balance[self.scene_ids]
            if self.reactive_objects is not None:flat=flat&~self.reactive_objects.state['active']
        protection_target=float(_get(self.config,'hand_protection_target_clearance',.04))
        protection_anticipation=float(_get(self.config,'hand_protection_anticipation_distance',.20))
        arm_margin=float(_get(self.config,'arm_clearance_margin',.08))
        tight=self.config.get('hand_clearance_tight')
        room_scale=None
        if tight:
            # Hand/elbow clearance targets shrink with the room the body has (pelvis distance to the blocking
            # footprint): 9 cm / 20 cm cannot be kept on both sides of a 0.34-0.44 m gap even sideways, so the
            # whole passage was one long penalty and standing in front of it cost nothing. Contact still ends
            # the episode; only the margin asked for shrinks.
            room,_=self._blocking().sample(i['positions'][:,1:2,:2],self.scene_ids)
            room_scale=((room[:,0]-float(tight['low']))/(float(tight['high'])-float(tight['low']))).clamp(float(tight['floor']),1.)[:,None]
            protection_target=protection_target*room_scale;protection_anticipation=protection_anticipation*room_scale
            arm_margin=arm_margin*room_scale
        costs,telemetry=self.math.upper_stability_terms(i['motor_targets'][:,12:],i['previous_upper'],i['previous_previous_upper'],self.upper_home,
            i['sdf'][:,5:7],i['elbow_clearance'],dt=self.dt,velocity_scale=float(_get(self.config,'upper_velocity_cost_scale',2.)),
            acceleration_scale=float(_get(self.config,'upper_acceleration_cost_scale',20.)),posture_scale=float(_get(self.config,'upper_action_scale',.8)),
            hand_margin=float(_get(self.config,'hand_clearance_margin',.12)),elbow_margin=float(_get(self.config,'arm_clearance_margin',.08)),
            clearance_taper=float(_get(self.config,'upper_posture_clearance_taper',.12)),waist_weight=float(_get(self.config,'upper_cost_waist_weight',4.)),
            hand_protection=self.hand_protection,arm_velocity_weight=float(_get(self.config,'upper_arm_velocity_weight',.5)),
            arm_acceleration_weight=float(_get(self.config,'upper_arm_acceleration_weight',.1)),
            protection_target_clearance=protection_target,
            protection_anticipation_distance=protection_anticipation,
            protection_near_weight=float(_get(self.config,'hand_protection_near_weight',.8)),
            protection_posture_taper=float(_get(self.config,'hand_protection_posture_taper',.10)),
            contrast_arm_active=None)
        enabled=bool(_get(self.config,'hand_protection_enabled',True))
        rewards['wholebody_hand_clearance']=(telemetry['hand_clearance_pressure'] if self.hand_protection else
            (float(_get(self.config,'hand_clearance_margin',.12))-i['sdf'][:,5:7]).clamp_min(0).square().flatten(1).mean(-1))*enabled
        rewards['wholebody_arm_clearance']=(arm_margin-i['elbow_clearance']).clamp_min(0).square().mean(-1)*enabled
        if room_scale is not None:
            telemetry['hand_room_scale']=room_scale[:,0]
        if self.stabilization:rewards.update(costs)
        scales=_get(self.config,'reward_config.scales',{})
        if 'action_rate' in scales:rewards['action_rate']=self._action_rate
        if ('teleop_progress' in scales or 'teleop_stall' in scales) and getattr(self,'_safe_target',None) is not None:
            # Keyboard worlds: following a PASSABLE command must pay. Being upright and alive earns ~12.5 per step,
            # velocity tracking ~0.7, and any contact ends the episode -- so v3 learned to stop in front of
            # 10-20 cm hurdles (keyboard and even field-driven). The target is the blocking-footprint safe
            # velocity: walls and tables zero it (stopping there stays free), hurdles and beams do not.
            mask=getattr(self,'joystick_mask',None)
            target=self._safe_target;wanted=torch.linalg.vector_norm(target,dim=-1)
            active=(wanted>.15)&(torch.ones_like(wanted,dtype=torch.bool) if mask is None else mask)
            velocity=self._sensor('global_linvel_pelvis',self.all_ids)[:,:2]
            along=((velocity*target).sum(-1)/wanted.clamp_min(1e-6)).clamp_min(0.)
            if 'teleop_progress' in scales:
                rewards['teleop_progress']=(along/wanted.clamp_min(1e-6)).clamp_max(1.)*active
            if 'teleop_stall' in scales:
                rewards['teleop_stall']=((along<.05)&active).float()
        if self.config.get('heading_command') and getattr(self,'_heading_gate',None) is not None:
            # Face the reference heading where a body facing it fits; no demand (and no reward) in a
            # gap too narrow for it. Replaces the forward-facing bonus for route/joystick worlds.
            track=self._heading_gate*.5*(1.+torch.cos(self._heading_error))
            if 'heading_align' in rewards:
                rewards['heading_align']=torch.where(self._heading_commanded,track,rewards['heading_align'])
            telemetry['heading_gate']=torch.where(self._heading_commanded,self._heading_gate,torch.ones_like(track))
            telemetry['heading_error_abs']=(self._heading_error.abs()*self._heading_gate)*self._heading_commanded
        if self.config.get('yaw_command') and getattr(self,'joystick_yaw',None) is not None:
            mask=getattr(self,'joystick_mask',None)
            commanded=torch.ones(self.num_envs,dtype=torch.bool,device=self.device) if mask is None else mask
            if 'tracking_yaw' in scales:
                rate=self._sensor('global_angvel_pelvis',self.all_ids)[:,2]
                rewards['tracking_yaw']=self.math.yaw_tracking(self.joystick_yaw,rate)*commanded
            if 'heading_align' in rewards:   # joystick decides the facing: sideways/backward must not turn around
                rewards['heading_align']=rewards['heading_align']*~commanded
        if self.hand_body_region is not None:
            # Contact-only (penetration, no distance margin): tucking a hand close to the body is free.
            touching=contacts[:,self.hand_body_start:]
            if 'hand_body_contact' in scales:rewards['hand_body_contact']=touching.float().sum(-1)
            for k,region in enumerate(HAND_BODY_REGIONS):
                telemetry['hand_contact_'+region]=touching[:,self.hand_body_region==k].any(-1).float()
        if 'self_clearance' in scales:
            penalty,gap=tm.self_clearance_terms(i['positions'][:,5:7],self.hand_radii,d.xpos[:,self.capsule_a_ids],d.xpos[:,self.capsule_b_ids],
                self.capsule_radii,margin=float(_get(self.config,'self_clearance_margin',.04)))
            rewards['self_clearance']=penalty
            telemetry['self_clearance_min_m']=gap;telemetry['self_clearance_violation']=(gap<0).float()
        if 'spot_hold' in scales and self.reactive_objects is not None:
            # Positional: the cost of being AWAY from the standing spot, saturating at the 0.3 m
            # left-spot radius. stand_still prices speed; this prices the drift it accumulates.
            radius=float(_get(self.config,'reactive_spot_radius',.3))
            rewards['spot_hold']=(self.reactive_root_displacement/radius).clamp(0.,1.).square()
        if 'reactive_event' in scales and self.reactive_objects is not None:
            # Sparse, paid once per finished approach: +1 handled, -1 unhandled for a threatening
            # object (buckets 0/1), 0 for an unneeded flinch. Dense shaping (hand_clearance,
            # handsgf) carries the gradient; this makes the OUTCOME of an approach itself count.
            s=self.reactive_objects.state;finished=s['event_finished'].float()
            threatening=(s['event_bucket']!=2).float()
            rewards['reactive_event']=finished*(s['event_success'].float()-(1.-s['event_success'].float())*threatening)
        if any(k in scales for k in ('upright','stand_tall','torso_rate')):
            near=self.config.get('head_guidance_near_sdf')
            terms,crouch=tm.posture_terms(i['torso_rpy'][:,1],i['positions'][:,0,2],i['gf'][:,0],i['sdf'][:,0].reshape(-1),i['torso_angvel'],
                head_target=float(_get(self.config,'posture_head_target',1.20)),crouch_near=float(near) if near else None,
                torso_roll=i['torso_rpy'][:,0] if self.config.get('upright_roll') else None)
            rewards.update({k:v for k,v in terms.items() if k in scales})
            telemetry['crouch_required']=crouch.float();telemetry['torso_pitch_abs']=i['torso_rpy'][:,1].abs()
        if self.hand_contrast:
            rotations=d.site_xmat.reshape(self.num_envs,-1,3,3)
            contrast,report=self.contrast_reward_terms(self.contrast,i['positions'][:,5:7],d.qpos[:,:2],
                rotations[:,self.pelvis_site,:,0],rotations[:,self.torso_site,:,0],
                region_scale=float(_get(self.config,'hand_contrast_region_scale',.15)),
                hand_good_distance=float(_get(self.config,'hand_contrast_metric_tolerance',.05)))
            if hasattr(self, 'tolerance_state'):
                _, loose = self.contrast_reward_terms(self.contrast,i['positions'][:,5:7],d.qpos[:,:2],
                    rotations[:,self.pelvis_site,:,0],rotations[:,self.torso_site,:,0],
                    region_scale=float(_get(self.config,'hand_contrast_region_scale',.15)),
                    hand_good_distance=self.tolerance_rungs[self.tolerance_episode_rung][:,None,None])
                report['hand_tolerance_good']=loose['hand_contrast_hand_good']
            from .lateral_corridor import lateral, corridor_cost, combined_pose, MARGIN, SCALE
            geometry = {k:v[self.scene_ids] for k,v in self.bank.acceptance.items()}
            y = lateral(i['positions'][:,5:7], geometry['origin'], geometry['tangent'])
            clearance = self.corridor_arms.clearance(d, geometry)
            telemetry['corridor_arm_clearance'] = clearance
            telemetry['corridor_good'] = combined_pose(y, geometry['bound'], report['hand_contrast_heading_good'], clearance)
            if self.config.get('lateral_corridor',False):
                active = geometry['protection']
                phase = self.contrast.get('reward_phase_weight',self.contrast['phase_weight'])*active
                rewards['wholebody_lateral_corridor'] = corridor_cost(y,geometry['bound'])*phase
                rewards['wholebody_corridor_arm'] = ((MARGIN-clearance).clamp_min(0)/SCALE).square()*phase
                contrast['wholebody_hand_contrast_region'] = torch.where(active,0.,contrast['wholebody_hand_contrast_region'])
            rewards.update({k:torch.zeros_like(v) for k,v in contrast.items()});telemetry.update(report)
        if balance:
            bonus,report=self.balance_terms(flat,i['positions'][:,5:7],d.qpos[:,:2],self.navigation['tangent'],
                self._sensor('global_linvel_pelvis',self.all_ids),self.balance_lower,self.balance_upper,**self.balance_reward)
            telemetry.update({'balance_'+key:value for key,value in report.items()})
        self.telemetry=telemetry
        scales=_get(self.config,'reward_config.scales',{})
        missing=set(rewards)-set(scales)
        if missing:raise ValueError(f'Missing preserved reward scales: {sorted(missing)}')
        scaled={k:v*float(scales[k]) for k,v in rewards.items()}
        if balance:scaled['flat_balance_posture_bonus']=bonus
        for name in ('wholebody_hand_contrast_region','wholebody_hand_contrast_heading','flat_balance_posture_bonus'):
            if name in scaled:scaled[name]=torch.zeros_like(scaled[name])
        clearance=scaled['wholebody_hand_clearance']*self.dt
        pre = sum(scaled.values())*self.dt-clearance
        from .reward_floor import hand_reward_floor
        reward, clipped, lift, slope = hand_reward_floor(
            pre, self.contrast, float(_get(self.config, 'hand_reward_soft_floor', 0.)))
        self.telemetry.update(reward_floor_clipped=clipped, reward_pre_floor_negative=pre < 0,
                              reward_soft_floor_lift=lift, reward_floor_slope=slope)
        return reward+clearance,scaled

    def _outcomes(self,done,truncation):
        self.info['minimum_episode_hand_clearance']=torch.minimum(self.info['minimum_episode_hand_clearance'],self.info['sdf'][:,5:7,0].amin(-1))
        failure=done&~truncation
        for key in ('fall','obstacle','self_contact','numerical','body_collision','hand_violation','elbow_violation','outside_bounds','route_lost'):
            if key in self.episode:failure|=self.episode[key]
        successful=self.episode['goal_reached']&~failure
        resolved=~self.outcome_counted&(successful|failure|done)
        successful&=resolved
        leader=self.policy_ids==0
        navigation_leader=leader
        if getattr(self,'balance_settings',None) is not None:
            navigation_leader=leader&~self.bank.flat_balance[self.scene_ids]
        # Joystick episodes have no goal: they end by contact or by the clock, never by success. Counting
        # them halved every success rate (v8, 50% joystick: clutter 0.80 -> 0.41) and the per-scene
        # rates that drive adaptive scene sampling. They keep their own 'teleop' bucket metrics.
        teleop=getattr(self,'joystick_mask',None)
        if teleop is not None and self.config.get('teleop'):
            navigation_leader=navigation_leader&~teleop
        groups=self.bank.navigation_groups[self.scene_ids]
        for column,mask in enumerate((resolved,successful)):
            increments=torch.zeros(3,dtype=torch.long,device=self.device).scatter_add_(0,groups,(mask&navigation_leader).long())
            self.navigation_counts[:3,column]+=increments;self.navigation_counts[3,column]+=increments.sum()
        posture_success=successful
        if self.hand_contrast:
            c=self.contrast;t=self.telemetry
            zone=torch.arange(6,device=self.device)[None]==c['zone_index'][:,None]
            core=c['core_active'][:,None]&zone
            hand_good=t['hand_contrast_hand_good']>.5
            if self.config.get('lateral_corridor',False):
                hand_good=torch.where(c['role']==1,t['corridor_good'],hand_good)
            values=torch.stack((core,core&(t['hand_contrast_heading_good']>.5)[:,None],core&hand_good[:,None]),-1)
            self.zone_steps+=(values&(~self.outcome_counted)[:,None,None]).long()
            seen=self.zone_steps[:,:,0]>0
            heading=seen&(self.zone_steps[:,:,1]>=.9*self.zone_steps[:,:,0])
            hands=seen&(self.zone_steps[:,:,2]>=.9*self.zone_steps[:,:,0])
            requires_posture=(c['role']==1)|(c['role']==3)
            qualified=self.info['minimum_episode_hand_clearance']>=.04
            posture_success=successful&(~requires_posture|qualified)
            valid=(c['role']>=1)&(c['role']<=3);roles=(c['role']-1).clamp(0,2).long()
            for column,mask in enumerate((resolved&valid&leader,posture_success&valid&leader)):
                self.contrast_counts[:,column]+=torch.zeros(3,dtype=torch.long,device=self.device).scatter_add_(0,roles,mask.long())
            for column,mask in enumerate((resolved&leader,posture_success&leader)):
                # `enabled` means inside a zone NOW; goals lie beyond zones.
                # Scene role remains valid throughout the whole episode.
                valid_role=c['role']>=0
                self.role_counts[:,column]+=torch.zeros(4,dtype=torch.long,device=self.device).scatter_add_(0,c['role'].clamp_min(0).long(),(mask&valid_role).long())
        curriculum_success=posture_success
        if hasattr(self, 'tolerance_state'):
            from .tolerance_curriculum import advance
            self.tolerance_zone_steps+=(core & (t['hand_tolerance_good']>.5)[:,None]
                & (~self.outcome_counted)[:,None]).long()
            loose_hands=seen & (self.tolerance_zone_steps>=.9*self.zone_steps[:,:,0])
            loose_qualified=(~c['required_forward_zones']|heading).all(-1)&(~c['required_hand_zones']|loose_hands).all(-1)
            curriculum_success=successful & (~requires_posture|loose_qualified)
            advance(self.tolerance_state,self.config['hand_tolerance_curriculum'],roles=c['role'],
                episode_rungs=self.tolerance_episode_rung,
                eligible=resolved & leader & ~self.info.get('hand_raised_seeded',torch.zeros_like(resolved)),
                success=curriculum_success)
        if hasattr(self, 'speed_state'):
            from .speed_curriculum import observe
            observe(self.speed_state,self.config['hand_speed_curriculum'],context=c,
                previous_progress=self.speed_state['previous_progress'],progress=self.navigation['progress_m'],
                strict_good=t['hand_contrast_hand_good']>.5,heading_good=t['hand_contrast_heading_good']>.5,
                failure=failure,resolved=resolved,leader=leader,
                seeded=self.info.get('hand_raised_seeded',torch.zeros_like(resolved)),dt=self.dt)
        self.outcome_counted|=resolved
        # Original survival-based adaptation remains for old scenes. Only hand
        # tasks use clean arrival. Protected/transition scenes additionally
        # require minimum hand clearance; box compliance is diagnostic only.
        contrast_scene=(self.bank.roles[self.scene_ids]>=0) if self.bank.roles is not None else torch.zeros_like(done)
        adapted_done=torch.where(contrast_scene,resolved,done)
        if teleop is not None and self.config.get('teleop'):
            adapted_done=adapted_done&~teleop       # sampling statistics: route episodes only
        adapted_success=torch.where(contrast_scene,curriculum_success,truncation)
        width=getattr(self.bank,'width_curriculum',None)
        if width is not None:
            levels=self.bank.width_levels[self.scene_ids]
            eligible=(self.bank.roles[self.scene_ids]==2)&leader&resolved
            self.curriculum_completed.scatter_add_(0,levels.clamp_min(0),(eligible).long())
            self.curriculum_goals.scatter_add_(0,levels.clamp_min(0),(eligible&successful).long())
            stage=self.curriculum_stage
            rate=self.curriculum_goals[stage]/self.curriculum_completed[stage].clamp_min(1)
            self.curriculum_stage=stage+((stage<len(width['rung_widths_m'])-1)&
                (self.curriculum_completed[stage]>=width['min_completed'])&(rate>=width['success_threshold'])).long()
        if self.bank.levels is not None:
            levels=self.bank.levels[self.scene_ids];hand=levels>=0
            clean=self.episode['goal_reached']&~failure
            adapted_success=torch.where(hand,clean,truncation)
            indices=levels.clamp_min(0)
            self.curriculum_completed.scatter_add_(0,indices,(hand&resolved).long())
            self.curriculum_goals.scatter_add_(0,indices,(hand&successful).long())
            stage=self.curriculum_stage
            rate=self.curriculum_goals[stage]/self.curriculum_completed[stage].clamp_min(1)
            threshold=hand_curriculum_threshold(self.config, getattr(self.bank, 'manifest', None))
            self.curriculum_stage=stage+((stage<2)&(self.curriculum_completed[stage]>=64)&(rate>=threshold)).long()
        counts=torch.zeros_like(self.scene_episode_ema).scatter_add_(0,self.scene_ids,adapted_done.float())
        goals=torch.zeros_like(self.scene_success_ema).scatter_add_(0,self.scene_ids,(adapted_done&adapted_success).float())
        decay=float(_get(self.config,'pf_config.sampling_ema_decay',.95));alpha=float(_get(self.config,'pf_config.sampling_alpha',1.))
        self.scene_episode_ema=decay*self.scene_episode_ema+counts;self.scene_success_ema=decay*self.scene_success_ema+goals
        rates=self.scene_success_ema/(self.scene_episode_ema+1e-6)
        weights=((1-rates)**alpha).clamp_min(1e-3)
        self.probabilities=self.bank.probabilities(weights,stage=self.curriculum_stage)
        return resolved,successful

    def _zone_progress_counts(self):
        """Visited core zones / valid passage zones, including unshaped narrow zones."""
        valid = self.bank.contrast['zone_valid'][self.scene_ids]
        valid = valid & self.bank.contrast['enabled'][self.scene_ids, None]
        return ((self.zone_steps[:, :, 0] > 0) & valid).sum(-1), valid.sum(-1)

    @torch.no_grad()
    def step(self,action):
        if action.shape!=(self.num_envs,29):raise ValueError('Expected [num_envs,29] actions')
        ids=self.all_ids;i=self.info;d=self.data
        style_now=self._style_features() if getattr(self,'style_prior',None) is not None else None
        if self.config.get('teleop') and getattr(self,'joystick_mask',None) is not None:
            self._teleop_resample()
        if getattr(self,'joystick_body',None) is not None:
            self._joystick_body_to_world()
        theta=self._rand((self.num_envs,),0.,2*torch.pi)
        magnitude=self._rand((self.num_envs,),*_get(self.config,'push_config.magnitude_range',[.1,1.]))
        signal=((i['push_step']+1)%i['push_interval']==0)&bool(_get(self.config,'push_config.enable',True))
        push=torch.stack((theta.cos(),theta.sin()),-1)*(signal*magnitude)[:,None]
        d.qvel[:,:2]+=push
        targets=self.math.motor_targets(action,i['motor_targets'],self.nominal,self.lower,self.upper,
            action_scale=float(_get(self.config,'action_scale',.5)),upper_action_scale=self.upper_action_scales,
            upper_target_rate=float(_get(self.config,'upper_target_rate',2.)),dt=self.dt)
        regions=torch.zeros((self.num_envs,6),dtype=torch.bool,device=self.device)
        robot_initiated=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device) if self.reactive_objects is not None else None
        object_initiated=robot_initiated.clone() if robot_initiated is not None else None
        if self.reactive_objects is not None:
            self.reactive_objects.begin_step(self.info['velocities'][:,5:7])
        for _ in range(self.n_substeps):
            if self.reactive_objects is not None:
                reactive_before=self.reactive_objects.advance(self.sim.final_collision_data(),self.dt/self.n_substeps)
            feedforward=None
            if self.upper_gravity is not None:
                # Derived poses after mj_step are pre-integration; refresh kinematics.
                current=self.sim.final_collision_data()
                feedforward=torch.zeros_like(d.ctrl)
                feedforward[:,12:]=self.upper_gravity(current)
            torque=self.math.pd_torque(d.qpos[:,7:],d.qvel[:,6:],targets,self.kps,self.kds,i['kp'],i['kd'],
                i['rfi'],self._rand((self.num_envs,29),-1.,1.),self.torque_limit,feedforward=feedforward)
            d.ctrl.copy_(torque)
            self.sim.step()
            if self.reactive_objects is not None:
                dynamic_regions,robot_hit=self.reactive_objects.contacts(self.sim.final_collision_data(),*reactive_before)
                regions|=dynamic_regions;robot_initiated|=robot_hit
                object_initiated|=self.reactive_objects.last_object_contacts
            # Interactive demo (sim_cpu) checks obstacle contact once per control step: per substep it
            # is 80% of a single-world CPU step. Training keeps the per-substep check.
            if self.collision is not None and getattr(self,'collision_every_substep',True):regions|=self.collision(self.scene_ids,d)
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
        gf,bf,sdf,command=self._fields(positions,d.qpos[:,:2],ids,store=True)
        update=i['step']%5==0
        odom=torch.where(update[:,None],d.qpos[:,:7],i['odom_delay'])
        delayed=self.math.delay_body_pos(d.qpos,odom,positions)
        gfd,bfd,sdfd,command_delay=self._fields(delayed,odom[:,:2],ids)
        # Standing sets both foot phases to 0; walking resumed from there commanded a synchronized
        # hop (gait [1,1]/[-1,-1]) and the policy stayed in place: measured on the flat scene, a
        # 0.5 m/s command after a 2 s stand gave ~0.01 m/s. Restore anti-phase stepping on every
        # route/joystick restart, not only inside the old speed curriculum.
        from .speed_curriculum import restart_phase
        eligible=self._commanded(ids)
        if hasattr(self, 'speed_state'):
            eligible=eligible|(self.contrast['role']==1)|(self.contrast['role']==3)
        phase_input=restart_phase(i['phase'],i['last_command'],command,eligible)
        command,stop,phase,gait=self.math.update_phase(command,i['last_command'],i['stop_timestep'],phase_input,i['phase_dt'],
            float(_get(self.config,'gait_config.gait_bound',.6)),self._commanded(ids))
        if hasattr(self, 'speed_state'):
            from .speed_curriculum import speed_limit, standing_phase
            _,_,hold=speed_limit(self.speed_state,self.contrast,ids,self.dt)
            command,command_delay,stop,phase,gait=standing_phase(command,command_delay,stop,phase,gait,hold)
        if self.reactive_objects is not None:
            # update_phase rewrites move=1 at zero speed, so standing-object episodes were being
            # commanded to walk at zero velocity (and the standing bonus never paid). Force a
            # real stance for them; the walking variant keeps its route command.
            from .speed_curriculum import standing_phase as _stance
            standing=self.reactive_objects.state['active'][ids]&~self.reactive_objects.state['walking'][ids]
            command,command_delay,stop,phase,gait=_stance(command,command_delay,stop,phase,gait,standing)
        move=(command[:,0]>.5)[:,None,None]
        normalize=lambda field:field/(torch.linalg.vector_norm(field,dim=-1,keepdim=True)+1e-6)
        normalized_gfd,normalized_bfd=normalize(gfd)*move,normalize(bfd)
        # CAT computes the true command before normalizing fields, but computes
        # the delayed command AFTER normalization and the current move gate.
        # Room guidance is an explicit override independent of this projection.
        native_delayed=self.math.compute_cmd_from_rtf(normalized_gfd[:,1],normalized_gfd[:,[0,3,4,5,6]],normalized_bfd[:,[0,3,4,5,6]])
        command_delay=torch.where(self._commanded(ids)[:,None],command_delay,native_delayed)
        i.update(gf=normalize(gf)*move,bf=normalize(bf),sdf=sdf,gf_delay=normalized_gfd,bf_delay=normalized_bfd,sdf_delay=sdfd,
                 command=command,command_delay=command_delay,odom_delay=odom,stop_timestep=stop,phase=phase,gait=gait,
                 positions=positions,velocities=velocities)
        i['push_step']+=1;i['step']+=1
        if 'action_rate' in _get(self.config,'reward_config.scales',{}):
            # First and second differences of the policy action. The legacy smoothness_action term
            # reads last_act AFTER it is overwritten below, so its rate parts are (0, first difference).
            self._action_rate=((action-i['last_act']).square()+(action-2*i['last_act']+i['last_last_act']).square()).sum(-1)
        i['last_last_act']=i['last_act'].clone();i['last_act']=action.clone()
        self._observe(ids,contacts[:,:2])
        terminated,faults=self._termination(regions,contacts)
        reward,components=self._rewards(action,contacts)
        penalty=-float(_get(self.config,'wholebody.body_collision.event_penalty',1.))*regions.any(-1)
        if self.collision is not None:reward+=penalty;components['body_collision_event']=penalty
        i['wrapper_steps']+=1
        lengths=self.bank.episode_lengths[self.scene_ids]
        if self.reactive_objects is not None:
            lengths=torch.where(self.reactive_objects.state['active'],int(self.reactive_objects.episode_length),lengths)
        if self.config.get('teleop') and getattr(self,'joystick_mask',None) is not None:
            lengths=torch.where(self.joystick_mask,int(round(float(self.config['teleop']['episode_seconds'])/self.dt)),lengths)
        timeout=i['wrapper_steps']>=lengths
        hold=self.config.get('goal_hold_seconds')
        if hold:
            # Reaching the goal never ended a room episode: a robot that arrived at step 500 stood
            # parked there until the 4000-step horizon, so ~2/3 of room experience was standing at
            # goals and fresh navigation attempts starved. End it after a short stand instead. This is
            # a TRUNCATION (value bootstrapped), not a terminal state, so arriving early is not taxed
            # by losing the future standing reward.
            i['goal_steps']=torch.where(self.episode['goal_reached']&self.navigation['enabled'],i['goal_steps']+1,0)
            timeout=timeout|(i['goal_steps']*self.dt>=float(hold))
        truncated=timeout&~terminated;done=terminated|timeout
        resolved,successful=self._outcomes(done,truncated)
        self.episode_reward+=reward
        terminal_obs={k:v.clone() for k,v in self.obs.items()}
        metrics={'reward/'+k:v for k,v in components.items()}
        metrics.update(navigation_counts=self.navigation_counts.clone(),contrast_counts=self.contrast_counts.clone(),role_counts=self.role_counts.clone(),
            episode_return=self.episode_reward.clone(),episode_length=i['step'].clone(),resolved=resolved,successful=successful,
            scene_ids=self.scene_ids.clone(),collision_regions=regions,
            # True once this episode has reached a verdict. At `done`, still False means the
            # episode ended having decided nothing -- the honest failure mode that
            # successes/resolved silently drops from its denominator.
            outcome_counted=self.outcome_counted.clone())
        if getattr(self,'joystick_mask',None) is not None:metrics['teleop/active']=self.joystick_mask.clone()
        if self.reactive_objects is not None:
            metrics.update({'reactive/active':self.reactive_objects.state['active'].clone(),
                # The distance the scene AUTHORED the object to stop at. Achieved gap minus
                # this is the only unambiguous read on priority #1: a policy that does nothing
                # sits at 0, one that retreats goes positive, one that reaches in goes negative.
                # The raw "hand within 0.20 m" fraction cannot distinguish successful avoidance
                # from an object that never arrived, which is exactly how the inert-task bug hid.
                'reactive/target_clearance':self.reactive_objects.state['clearance'].amin(-1).clone(),
                'reactive/scene':self.reactive_objects.state['scene'].clone(),
                'reactive/robot_initiated_contact':robot_initiated,
                'reactive/object_initiated_contact':object_initiated,
                'reactive/root_displacement_m':self.reactive_root_displacement.clone(),
                'reactive/walking':self.reactive_objects.state['walking'].clone(),
                'reactive/root_speed_m_s':torch.linalg.vector_norm(self._sensor('global_linvel_pelvis',self.all_ids)[:,:2],dim=-1),
                'reactive/yaw_rate_rad_s':self.info['torso_angvel'][:,2].abs(),
                'reactive/event_finished':self.reactive_objects.state['event_finished'].clone(),
                'reactive/event_success':self.reactive_objects.state['event_success'].clone(),
                'reactive/event_bucket':self.reactive_objects.state['event_bucket'].clone(),
                'reactive/event_retreat_m':self.reactive_objects.state['event_retreat'].clone()})
        metrics['acceptance/hand_clearance']=i['sdf'][:,5:7,0].clone()
        metrics['acceptance/root_xy']=d.qpos[:,:2].clone()
        from .response_split import initial, advance, PREFIX
        if PREFIX+'active' not in i:  # Legacy checkpoint.
            i.update({PREFIX+k:v for k,v in initial(d.qpos[:,:3]).items()})
        response_state = {k:i[PREFIX+k] for k in ('active','hand','root','relative','normal')}
        response = advance(response_state, i['positions'][:,5:7], d.qpos[:,:3],
                           i['sdf'][:,5:7,0], i['bf'][:,5:7], done | resolved)
        i.update({PREFIX+k:v for k,v in response_state.items()})
        metrics.update({'response_split/'+k:v for k,v in response.items()})
        metrics['response_split/active_count'] = response_state['active'].float().clone()
        if self.hand_contrast:
            metrics['acceptance/protection_good']=self.telemetry['corridor_good'].clone()
            metrics['corridor_pose_good']=self.telemetry['corridor_good'].float().clone()
            if self.config.get('lateral_corridor',False):
                speed=(self._sensor('global_linvel_pelvis',self.all_ids)[:,:2]*self.navigation['tangent']).sum(-1)
                metrics['corridor_selection_good']=(self.telemetry['corridor_good'] & (speed>=.2)).float()
            metrics['corridor_arm_clearance']=self.telemetry['corridor_arm_clearance'].clone()
            # Capture physical transition telemetry before autoreset clears zones.
            if self.raised_reset_fraction:
                metrics['hand_raised_seeded']=i['hand_raised_seeded'].clone()
            if hasattr(self.bank, 'hand_scene_kind'):
                metrics['hand_scene_kind']=self.bank.hand_scene_kind[self.scene_ids]
            metrics['protected_core'] = (self.contrast['role']==1)&self.contrast['core_active']
            metrics['reward_floor_clipped'] = self.telemetry['reward_floor_clipped'].clone()
            for key in ('reward_pre_floor_negative', 'reward_soft_floor_lift', 'reward_floor_slope'):
                metrics[key] = self.telemetry[key].clone()
            for key in ('self_clearance_min_m', 'self_clearance_violation', 'crouch_required', 'torso_pitch_abs',
                        'hand_contact_trunk', 'hand_contact_head', 'hand_contact_arm', 'hand_contact_hand',
                        'heading_gate', 'heading_error_abs', 'hand_room_scale'):
                if key in self.telemetry: metrics[key] = self.telemetry[key].clone()
            metrics['contrast_role'] = self.contrast['role'].clone()
            metrics['contrast_unresolved'] = (~self.outcome_counted | resolved).clone()
            for key in ('heading_cost', 'region_cost', 'heading_good', 'hand_good'):
                metrics['hand_contrast_' + key] = self.telemetry['hand_contrast_' + key].clone()
            metrics['contrast_heading_active'] = self.contrast.get('heading_weight', self.contrast['forward_weight']) > 0
            metrics['contrast_hand_active'] = self.telemetry['hand_contrast_hand_active'] > 0
            # Retain the internal denominator key; it now counts every valid zone.
            metrics['contrast_zone_seen'], metrics['contrast_zone_required'] = self._zone_progress_counts()
        if getattr(self,'balance_settings',None) is not None:
            flat=self.bank.flat_balance[self.scene_ids];t=self.telemetry
            valid=flat&t['balance_walking']&t['balance_compliant']&~terminated
            self.balance_streak=torch.where(valid,self.balance_streak+1,0)
            metrics.update({'balance/flat':flat,'balance/walking':t['balance_walking'],
                'balance/compliant':t['balance_compliant'],'balance/speed':t['balance_speed'],
                'balance/nominal':((d.qpos[:,22:]-self.nominal[15:]).square().mean(-1).sqrt()<.15)&~t['balance_compliant'],
                'balance/foot_balance':components['foot_balance']*self.dt,
                'balance/bonus':components['flat_balance_posture_bonus']*self.dt,
                'balance/streak':self.balance_streak.clone(),
                'balance/heights':torch.cat((d.qpos[:,2:3],i['positions'][:,5:7,2]),-1)})
        for k,v in self.episode.items():metrics['episode/'+k]=v.clone()
        i['last_joint_vel']=d.qvel[:,6:].clone()
        i['previous_previous_upper']=i['previous_upper'].clone();i['previous_upper']=targets[:,12:].clone()
        if style_now is not None:
            # Transition (s_t, s_t+1) is captured BEFORE autoreset, so terminal steps are real motion.
            from .style_prior import gate_weights
            hs=getattr(self,'_heading_sdf',None)
            if hs is None:hs=torch.full((self.num_envs,4),2.,device=self.device)
            from .style_prior import context_probe_points
            cmd=i['command'][:,1:3];norm=torch.linalg.vector_norm(cmd,dim=-1,keepdim=True)
            yaw=torch.atan2(i['navi'][:,1,0],i['navi'][:,0,0]);facing=torch.stack((yaw.cos(),yaw.sin()),-1)
            direction=torch.where(norm>1e-3,cmd/norm.clamp_min(1e-6),facing)   # standing: probe where it faces
            probes=context_probe_points(i['positions'][:,1,:2],direction)
            psdf=self.bank.sample('sdf',probes,self.scene_ids).reshape(self.num_envs,-1)
            weights,gap=gate_weights(hs,psdf[:,:2].amin(-1),psdf[:,2:].amin(-1))
            first=style_now
            stride=int(getattr(self.style_prior,'stride',1))
            if stride>1:
                # Pairs (s_t+1-k, s_t+1): a 0.1 s stride judges motion, not 50 Hz vibration.
                # Ring of the last k step-start features; worlds younger than k steps get zero weight.
                if getattr(self,'_style_hist',None) is None or self._style_hist.shape[0]!=stride:
                    self._style_hist=torch.zeros((stride,*style_now.shape),device=self.device)
                    self._style_ptr=0;self._style_age=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
                self._style_hist[self._style_ptr]=style_now;self._style_age+=1
                self._style_ptr=(self._style_ptr+1)%stride
                valid=self._style_age>=stride
                first=torch.where(valid[:,None],self._style_hist[self._style_ptr],style_now)
                weights=weights*valid[:,None]
            near=None
            if self.config.get('style_free_near_obstacles'):
                # Hands within 0.2-0.3 m of an obstacle: tucking or raising the arms there is the job, not
                # bad style. The style reward is blended to neutral and these steps are not shown to the
                # discriminator (human walking has the arms down; moving the body away was cheaper).
                clearance=i['sdf'][:,5:7,0].amin(-1)
                blend=((clearance-.20)/.10).clamp(0.,1.);near=1.-blend.square()*(3-2*blend)
            self.style_transition=(first,self._style_features(),weights,near)
            metrics['style/gap_m']=gap
        # Only physically completed worlds reset; cached terminal observations
        # and terminal rewards retain the transition before reset.
        self.reset(torch.nonzero(done,as_tuple=False).flatten())
        return dict(obs=self.obs,reward=reward,terminated=terminated,truncated=truncated,done=done,
                    terminal_obs=terminal_obs,metrics=metrics)

    def sampling_state(self):
        """Shared, physics-independent state transferable between backends.

        The caller must verify the exact scene-bank hash before transferring
        these arrays: scene indices and curriculum levels belong to that bank.
        Logits are returned up to the irrelevant categorical additive constant.
        """
        state={'pf_episode_ema':self.scene_episode_ema,
               'pf_success_ema':self.scene_success_ema,
               'pf_sampling_logits':torch.where(self.probabilities>0,self.probabilities.log(),-torch.inf),
               'pf_navigation_outcome_counts':self.navigation_counts,
               'pf_contrast_outcome_counts':self.contrast_counts}
        if getattr(self.bank,'width_curriculum',None) is not None:
            state.update(pf_width_curriculum_stage=self.curriculum_stage,
                         pf_width_curriculum_completed=self.curriculum_completed,
                         pf_width_curriculum_goals=self.curriculum_goals)
        if self.bank.levels is not None:
            state.update(pf_hand_curriculum_stage=self.curriculum_stage,
                         pf_hand_curriculum_completed=self.curriculum_completed,
                         pf_hand_curriculum_goals=self.curriculum_goals)
        return {key:value.detach().clone() for key,value in state.items()}

    @torch.no_grad()
    def restore_sampling_state(self,state,*,resample=True):
        """Restore one shared copy of the source sampler after bank-hash checks.

        Physics and temporal histories deliberately restart on a backend
        migration. Resampling draws those fresh episodes from the *preserved*
        curriculum and keeps the original startup horizon staggering. Validate
        every shared array before changing any live task state.
        """
        required={'pf_episode_ema','pf_success_ema','pf_sampling_logits'}
        hand={'pf_hand_curriculum_stage','pf_hand_curriculum_completed','pf_hand_curriculum_goals'}
        optional={'pf_navigation_outcome_counts','pf_contrast_outcome_counts'}
        width=getattr(self.bank,'width_curriculum',None)
        width_keys={'pf_width_curriculum_stage','pf_width_curriculum_completed','pf_width_curriculum_goals'}
        if width is not None:required|=width_keys
        if self.bank.levels is not None:required|=hand
        if not required<=set(state) or set(state)-(required|optional):
            raise ValueError('Incomplete or unknown shared sampling state')

        def array(name,shape,*,integer=False,logits=False):
            value=torch.as_tensor(state[name],device=self.device).detach().clone()
            if value.shape!=shape or value.is_complex() or value.dtype==torch.bool:
                raise ValueError(f'Invalid shared sampling shape/type: {name}')
            finite=torch.isfinite(value)
            if logits:
                if torch.isnan(value).any() or torch.isposinf(value).any() or not finite.any():
                    raise ValueError(f'Invalid shared sampling logits: {name}')
            elif not finite.all() or (value<0).any():
                raise ValueError(f'Invalid shared sampling values: {name}')
            if integer:
                if value.is_floating_point() and not torch.equal(value,value.round()):
                    raise ValueError(f'Noninteger shared sampling counts: {name}')
                value=value.long()
                if (value<0).any():raise ValueError(f'Overflowing shared sampling counts: {name}')
            else:
                value=value.float()
                if not logits and not torch.isfinite(value).all():
                    raise ValueError(f'Overflowing shared sampling values: {name}')
            return value

        episodes=array('pf_episode_ema',(self.bank.count,))
        successes=array('pf_success_ema',(self.bank.count,))
        if (successes>episodes+1e-5).any():raise ValueError('Scene successes exceed completed episodes')
        logits=array('pf_sampling_logits',(self.bank.count,),logits=True)
        probabilities=torch.softmax(logits,dim=0)
        if not torch.isfinite(probabilities).all():raise ValueError('Invalid normalized scene distribution')
        stage=self.curriculum_stage.clone()
        completed=self.curriculum_completed.clone();goals=self.curriculum_goals.clone()
        if self.bank.levels is not None:
            stage=array('pf_hand_curriculum_stage',(),integer=True)
            completed=array('pf_hand_curriculum_completed',(3,),integer=True)
            goals=array('pf_hand_curriculum_goals',(3,),integer=True)
            if stage>2 or (goals>completed).any():raise ValueError('Inconsistent hand curriculum state')

        if width is not None:
            count=len(width['rung_widths_m'])
            stage=array('pf_width_curriculum_stage',(),integer=True)
            completed=array('pf_width_curriculum_completed',(count,),integer=True)
            goals=array('pf_width_curriculum_goals',(count,),integer=True)
            if stage>=count or (goals>completed).any():raise ValueError('Inconsistent width curriculum state')

        # Preserve adaptive weights while checking immutable role/family mass
        # and locked-level support. Specialist hand-only banks have raw mass
        # .5; categorical sampling normalizes it, as it did in the JAX task.
        expected=self.bank.probabilities(torch.ones_like(episodes),stage=stage)
        expected=expected/expected.sum()
        if ((expected==0)&(probabilities!=0)).any():raise ValueError('Sampling enables a locked scene')
        if self.bank.roles is not None:groups=getattr(self.bank,'sampling_ids',self.bank.roles)
        elif self.bank.levels is not None:
            groups=self.bank.groups*2+(self.bank.levels>=0).long()
        else:groups=getattr(self.bank,'groups',None)
        if groups is not None:
            for group in groups.unique():
                selected=groups==group
                if not torch.allclose(probabilities[selected].sum(),expected[selected].sum(),atol=2e-6,rtol=2e-5):
                    raise ValueError('Sampling state changes a fixed scene-group mass')

        counts={}
        for name,shape in (('pf_navigation_outcome_counts',(4,2)),('pf_contrast_outcome_counts',(3,2))):
            if name not in state:continue
            counts[name]=array(name,shape,integer=True)
            if (counts[name][:,1]>counts[name][:,0]).any():raise ValueError('Success counts exceed resolved outcomes')
        navigation=counts.get('pf_navigation_outcome_counts',self.navigation_counts)
        if not torch.equal(navigation[3],navigation[:3].sum(0)):
            raise ValueError('Navigation outcome populations do not sum to the total')

        self.scene_episode_ema=episodes;self.scene_success_ema=successes
        self.probabilities=probabilities
        self.curriculum_stage=stage;self.curriculum_completed=completed;self.curriculum_goals=goals
        self.navigation_counts=navigation.clone()
        self.contrast_counts=counts.get('pf_contrast_outcome_counts',self.contrast_counts).clone()
        if resample:
            horizon_offsets=self.info['wrapper_steps'].clone()
            self.reset()
            self.info['wrapper_steps'].copy_(horizon_offsets)

    def state_dict(self):
        """Torch task state; simulation and learner states are saved separately."""
        names=('info','navigation','contrast','episode','telemetry','obs','scene_ids','scene_episode_ema','scene_success_ema',
               'curriculum_stage','curriculum_completed','curriculum_goals','probabilities','navigation_counts','contrast_counts',
               'policy_ids','role_counts','zone_steps','outcome_counted','episode_reward')
        if getattr(self,'balance_settings',None) is not None:names+=('balance_streak',)
        if hasattr(self,'speed_state'):names+=('speed_state',)
        if hasattr(self,'tolerance_state'):names+=('tolerance_state','tolerance_episode_rung','tolerance_zone_steps')
        if self.analytic_objects is not None:names+=('analytic_object_state',)
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
        restored_objects=None
        if self.analytic_objects is not None:
            from .analytic_objects import AnalyticObjects
            if self.reactive_objects is not None:
                from .reactive import StandingObjects
                restored_objects=StandingObjects(bank=self.reactive_objects.path,num_envs=self.num_envs,model=self.model,device=self.device)
                restored_objects.state=restored['analytic_object_state']
            else:restored_objects=AnalyticObjects(**restored['analytic_object_state'])
        # Validate everything before mutating the live task.
        self.generator.set_state(state['rng'].cpu())
        for name,value in restored.items():setattr(self,name,value)
        if restored_objects is not None:
            self.analytic_objects=restored_objects
            if self.reactive_objects is not None:self.reactive_objects=restored_objects
