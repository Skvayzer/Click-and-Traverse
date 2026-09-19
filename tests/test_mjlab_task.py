"""Numerical contracts against the existing JAX task, before GPU integration."""
from copy import deepcopy
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from cat_mjlab import task_math as tm
from cat_mjlab import navigation as nav


def tensor(x):return torch.as_tensor(np.asarray(x).copy())
def close(actual,expected,atol=2e-6):np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),atol=atol,rtol=3e-5)


def test_exact_wholebody_configuration_scales():
    from ml_collections import config_dict
    from cat_ppo.furniture.generalist_config import released_config
    from cat_ppo.envs.g1.env_cat_wholebody import wholebody_config as legacy
    from cat_mjlab.config import wholebody_config
    base=released_config()['env_config']
    for protection,contrast in ((False,False),(True,False),(True,True)):
        expected=legacy(config_dict.ConfigDict(base),stabilization=True,hand_protection=protection,hand_contrast=contrast).to_dict()
        actual=wholebody_config(base,stabilization=True,hand_protection=protection,hand_contrast=contrast)
        for key in expected:
            if key=='wholebody':continue
            assert actual[key]==expected[key],key


def test_motor_targets_preserve_incremental_legs_and_slew_limited_upper():
    rng=np.random.default_rng(42)
    action=rng.uniform(-2,2,(8,29)).astype('f')
    previous=rng.uniform(-1,1,(8,29)).astype('f')
    nominal=rng.uniform(-.4,.4,29).astype('f')
    lower=np.full(29,-1.3,'f');upper=np.full(29,1.2,'f')
    from cat_ppo.envs.g1.env_cat_wholebody import _WholeBodyTask
    from ml_collections import config_dict
    legacy=object.__new__(_WholeBodyTask)
    legacy.compatibility_mode=False;legacy._ctrl_dt=.02;legacy._config=config_dict.ConfigDict(dict(action_scale=.5,upper_action_scale=.8,upper_target_rate=2.,ctrl_dt=.02,sim_dt=.002))
    legacy._default_qpos=jp.asarray(nominal);legacy._soft_lowers=jp.asarray(lower);legacy._soft_uppers=jp.asarray(upper)
    expected=jax.vmap(legacy._motor_targets)(jp.asarray(action),jp.asarray(previous))
    close(tm.motor_targets(*map(tensor,(action,previous,nominal,lower,upper))),expected)


def test_commands_delayed_body_transforms_and_rpy_match_jax():
    from cat_ppo.envs.g1.env_cat import G1CatEnv,delay_body_pos
    import jaxlie
    rng=np.random.default_rng(9)
    q=rng.normal(size=(5,7)).astype('f');q[:,3:]/=np.linalg.norm(q[:,3:],axis=-1,keepdims=True)
    odom=rng.normal(size=(5,7)).astype('f');odom[:,3:]/=np.linalg.norm(odom[:,3:],axis=-1,keepdims=True)
    positions=rng.normal(size=(5,11,3)).astype('f')
    expected=jax.vmap(delay_body_pos)(jp.array(q[:,:3]),jp.array(q[:,3:]),jp.array(odom[:,:3]),jp.array(odom[:,3:]),jp.array(positions))
    close(tm.delay_body_pos(tensor(q),tensor(odom),tensor(positions)),expected)
    gf=rng.normal(size=(5,11,3)).astype('f');bf=rng.normal(size=(5,11,3)).astype('f')
    legacy=object.__new__(G1CatEnv);legacy._stop_cmd=jp.zeros(4)
    expected=jax.vmap(legacy.compute_cmd_from_rtf)(jp.array(gf[:,1]),jp.array(gf[:,[0,3,4,5,6]]),jp.array(bf[:,[0,3,4,5,6]]))
    close(tm.compute_cmd_from_rtf(tensor(gf[:,1]),tensor(gf[:,[0,3,4,5,6]]),tensor(bf[:,[0,3,4,5,6]])),expected)
    matrices=np.asarray(jax.vmap(lambda a:jaxlie.SO3(a).as_matrix())(jp.array(q[:,3:])))
    angles=jax.vmap(lambda a:jp.asarray(jaxlie.SO3.from_matrix(a).as_rpy_radians()))(jp.array(matrices))
    close(tm.matrix_rpy(tensor(matrices)),angles)


def test_ordered_navigation_and_thin_rotated_obstacles_match_jax():
    from cat_ppo.furniture import room_navigation as old
    batch=7
    route=np.array([[0,0],[1,0],[1,1],[2,1]],np.float32)
    obs=np.array([[.6,.6,.07,.2,.8,.6],[1.5,.6,.1,.1,1,0]],np.float32)
    roots=np.array([[0,0],[.95,.0],[1,.6],[1.5,1],[2,1],[.6,.6],[0,1]],np.float32)
    previous=roots-np.array([.03,0],np.float32)
    segments=np.array([0,0,1,2,2,0,0],np.int32)
    expected=jax.vmap(lambda a,b,s:old.route_context(a,b,s,jp.array(False),jp.array(route),jp.int32(4),jp.array(obs),jp.int32(2),radius=.14))(jp.array(roots),jp.array(previous),jp.array(segments))
    actual=nav.route_context(tensor(roots),tensor(previous),tensor(segments),torch.zeros(batch,dtype=torch.bool),
        tensor(np.broadcast_to(route,(batch,*route.shape))),torch.full((batch,),4),
        tensor(np.broadcast_to(obs,(batch,*obs.shape))),torch.full((batch,),2),radius=torch.full((batch,),.14))
    for key,value in expected.items():close(actual[key],value)


def test_contrast_regions_and_phase_match_jax():
    from cat_ppo.furniture.contrastive_rewards import hand_contrast_context,reward_terms,pack_hand_contrast
    metadata=pack_hand_contrast([None])
    metadata['enabled'][:]=True;metadata['zone_valid'][:,0]=True
    metadata['start_m'][:,0]=1;metadata['end_m'][:,0]=3;metadata['fade_m'][:,0]=.2
    metadata['forward_weight'][:,0]=1;metadata['hand_active'][:,0]=True;metadata['region_valid'][:,0]=True
    metadata['hand_regions_min'][:,0,:,:,2]=1.;metadata['hand_regions_max'][:,0]=metadata['hand_regions_min'][:,0]+.15
    progress=np.array([.7,1.,1.1,1.5,2.9,3.1],np.float32);batch=len(progress)
    tangent=np.tile([.8,.6],(batch,1)).astype('f')
    repeated={k:np.repeat(v,batch,axis=0) for k,v in metadata.items()}
    expected=jax.vmap(hand_contrast_context)(jax.tree.map(jp.asarray,repeated),jp.array(progress),jp.array(tangent))
    actual=nav.hand_contrast_context({k:tensor(v) for k,v in repeated.items()},tensor(progress),tensor(tangent))
    for key in actual:close(actual[key],expected[key])
    rng=np.random.default_rng(88);hands=rng.uniform(-.2,1.2,(batch,2,3)).astype('f')
    roots=np.zeros((batch,2),'f');pf=np.tile([1,0,0],(batch,1)).astype('f');tf=np.tile([.8,.6,0],(batch,1)).astype('f')
    a,b=nav.contrast_reward_terms(actual,*map(tensor,(hands,roots,pf,tf)))
    ea,eb=jax.vmap(reward_terms)(expected,*map(jp.asarray,(hands,roots,pf,tf)))
    for key in a:close(a[key],ea[key])
    for key in b:close(b[key],eb[key])


@pytest.mark.parametrize('protection',[False,True])
def test_upper_stability_and_clearance_pressure_match_jax(protection):
    from cat_ppo.furniture.wholebody_stability import upper_stability_terms
    rng=np.random.default_rng(63)
    target=rng.normal(size=(11,17)).astype('f')*.2
    previous=target+rng.normal(size=target.shape).astype('f')*.01
    earlier=previous+rng.normal(size=target.shape).astype('f')*.01
    nominal=np.zeros(17,'f');hand=rng.uniform(-.1,.5,(11,2,1)).astype('f');elbow=rng.uniform(-.1,.5,(11,2)).astype('f')
    active=rng.random((11,2))>.5
    kw=dict(dt=.02,hand_protection=protection)
    expected=jax.vmap(lambda t,p,e,h,a,c:upper_stability_terms(t,p,e,jp.array(nominal),h,a,contrast_arm_active=c,**kw))(
        *map(jp.asarray,(target,previous,earlier,hand,elbow,active)))
    actual=tm.upper_stability_terms(*map(tensor,(target,previous,earlier,nominal,hand,elbow)),contrast_arm_active=tensor(active),**kw)
    for act,exp in zip(actual,expected):
        for key,value in exp.items():close(act[key],value,atol=3e-5)


@pytest.fixture(scope='module')
def reference_task(tmp_path_factory):
    from cat_ppo.envs.g1.env_cat_wholebody import _WholeBodyTask,wholebody_config
    from cat_ppo.furniture.generalist_config import released_config
    from ml_collections import config_dict
    path=tmp_path_factory.mktemp('mjlab-task-reference')
    grid=np.indices((20,21,16),dtype=np.float32)
    fields={'sdf':1+.02*grid[0]+.01*grid[2],
            'gf':np.stack((.7+.01*grid[0],.05*grid[1],.01*grid[2]),-1),
            'bf':np.stack((.1*grid[0],.1*grid[1],.1*grid[2]),-1)}
    for key,value in fields.items():np.save(path/(key+'.npy'),value)
    cfg=config_dict.ConfigDict(released_config()['env_config'])
    cfg.pf_config.path=str(path);cfg.pf_config.origin=[-2.,-2.,-.5];cfg.pf_config.dx=.25
    cfg.noise_config.level=0.
    task=_WholeBodyTask(config=wholebody_config(cfg,stabilization=True,hand_protection=True))
    state=jax.jit(task.reset)(jax.random.PRNGKey(5))
    return task,state


def test_observation_feature_order_matches_existing_checkpoint_contract(reference_task):
    task,state=reference_task;i=state.info;d=state.data
    batch=lambda x:tensor(np.asarray(x)[None])
    positions=jp.concatenate((i['head_pos'][None],i['pelv_pos'][None],i['tors_pos'][None],i['feet_pos'],i['hands_pos'],i['knees_pos'],i['shlds_pos']))
    velocity=jp.zeros_like(positions).at[0].set(i['head_vel']).at[3:5].set(i['feet_vel']).at[5:7].set(i['hands_vel'])
    def fields(suffix):
        return tuple(jp.concatenate([i[group+kind+suffix] for group in ('head','pelv','tors','feet','hands','knees','shlds')]) for kind in ('gf','bf','df'))
    gf,bf,sdf=fields('');gfd,bfd,sdfd=fields('_delay')
    elbows=task._elbow_fields(d.site_xpos[task._elbow_site_ids],i,actor=False)
    elbows_actor=task._elbow_fields(d.site_xpos[task._elbow_site_ids],i,actor=True)
    contact=jp.zeros(2)
    actual=tm.observations(joint_pos=batch(d.qpos[7:]),joint_vel=batch(d.qvel[6:]),nominal=tensor(task._default_qpos),
        gyro=batch(task.get_gyro(d,'pelvis')),gravity=batch(d.site_xmat[task._pelvis_imu_site_id].T@jp.array([0,0,-1.])),
        linear_velocity=batch(task.get_local_linvel(d,'pelvis')),noise={k:torch.zeros((1,n)) for k,n in (('gyro',3),('gravity',3),('joint_pos',29),('joint_vel',29))},
        last_action=batch(i['last_act']),targets=batch(i['motor_targets']),command=batch(i['command']),command_delay=batch(i['command_delay']),
        foot_height=batch(i['foot_height']),phase=batch(i['phase']),navi=batch(i['navi2world_pose'][:3,:3]),
        gf=batch(gf),bf=batch(bf),sdf=batch(sdf),gf_delay=batch(gfd),bf_delay=batch(bfd),sdf_delay=batch(sdfd),
        positions=batch(positions),velocities=batch(velocity),torso_rpy=batch(i['navi_torso_rpy']),gait=batch(i['gait_mask']),contacts=batch(contact),
        kp=batch(i['kp_scale']),kd=batch(i['kd_scale']),rfi=batch(i['rfi_lim_scale']),elbow_true=batch(elbows),elbow_actor=batch(elbows_actor))
    expected=task._get_obs(d,deepcopy(i),contact)
    for key in expected:close(actual[key][0],expected[key])


def test_all_native_reward_terms_match_reference(reference_task):
    from cat_ppo.envs.g1.env_cat import G1CatEnv
    task,state=reference_task;d=state.data;i=deepcopy(state.info)
    # Test nonzero motion and field penalties on one valid compiled robot state.
    i['navi_pelvis_rpy']=jp.array([.1,-.1,0]);i['navi_torso_rpy']=jp.array([.2,-.3,0])
    i['navi_torso_ang_vel']=jp.array([.15,-.2,.3]);i['global_lin_vel']=jp.array([.3,.2,.1])
    i['last_act']=jp.linspace(-.3,.3,29);i['last_last_act']=jp.linspace(.2,-.2,29)
    i['last_joint_vel']=jp.linspace(-.1,.3,29);i['gait_mask']=jp.array([1.,-1.])
    i['head_vel']=jp.array([.2,.3,.1]);i['hands_vel']=jp.ones((2,3))*.2;i['feet_vel']=jp.ones((2,3))*.1
    for group in ('head','pelv','tors','feet','hands','knees','shlds'):i[group+'df']=i[group+'df']*.03
    positions=jp.concatenate((i['head_pos'][None],i['pelv_pos'][None],i['tors_pos'][None],i['feet_pos'],i['hands_pos'],i['knees_pos'],i['shlds_pos']))
    velocities=jp.zeros_like(positions).at[0].set(i['head_vel']).at[3:5].set(i['feet_vel']).at[5:7].set(i['hands_vel'])
    fields=lambda suffix:jp.concatenate([i[group+suffix] for group in ('head','pelv','tors','feet','hands','knees','shlds')])
    action=jp.linspace(-.2,.3,29);contacts=jp.array([True,False]);batch=lambda x:tensor(np.asarray(x)[None])
    expected=G1CatEnv._get_reward(task,d,action,i,jp.array(False),contacts)
    actual=tm.native_rewards(action=batch(action),last_action=batch(i['last_act']),last_last_action=batch(i['last_last_act']),
        joint_pos=batch(d.qpos[7:]),joint_vel=batch(d.qvel[6:]),last_joint_vel=batch(i['last_joint_vel']),lower=tensor(task._soft_lowers),upper=tensor(task._soft_uppers),
        actuator_force=batch(d.actuator_force),command=batch(i['command']),pelvis_rpy=batch(i['navi_pelvis_rpy']),torso_rpy=batch(i['navi_torso_rpy']),
        head_z=batch(i['head_pos'][2]),torso_height=task._config.torso_height[1],global_velocity=batch(i['global_lin_vel']),torso_angvel=batch(i['navi_torso_ang_vel']),
        navi=batch(i['navi2world_pose'][:3,:3]),leg_rotations=batch(jp.concatenate((d.xmat[task.body_ids_left_leg],d.xmat[task.body_ids_right_leg]))),
        feet_pos=batch(i['feet_pos']),feet_sensor_velocity=batch(d.sensordata[task._foot_linvel_sensor_adr]),subtree_com=batch(d.subtree_com[task.body_id_pelvis]),
        feet_contact=batch(contacts),gait=batch(i['gait_mask']),foot_height=batch(i['foot_height']),foot_height_stance=task._config.reward_config.foot_height_stance,
        gf=batch(fields('gf')),positions=batch(positions),velocities=batch(velocities),sdf=batch(fields('df')),crossed=batch(task._crossed_goal(positions)),
        dt=.02,max_yaw=abs(task._config.ang_vel_yaw[1]))
    assert set(actual)==set(expected)
    for key,value in expected.items():close(actual[key][0],value,atol=2e-5)


class _CPUSimulation:
    """Test-only real MuJoCo bridge; production always uses mjlab Simulation."""
    def __init__(self,num_envs):
        import mujoco
        from cat_mjlab.model import assemble_training_xml
        self.model=mujoco.MjModel.from_xml_string(assemble_training_xml())
        self.model.opt.timestep=.002
        self.raw=[mujoco.MjData(self.model) for _ in range(num_envs)]
        self.names=('qpos','qvel','ctrl','site_xpos','site_xmat','xpos','xmat','subtree_com','sensordata','actuator_force','qacc_warmstart','time')
        self.data=SimpleNamespace(**{k:tensor(np.stack([np.asarray(getattr(d,k)) for d in self.raw])).float() for k in self.names})
    def _write(self,index):
        d=self.raw[index]
        for name in ('qpos','qvel','ctrl','qacc_warmstart'):getattr(d,name)[:]=getattr(self.data,name)[index].numpy()
        d.time=float(self.data.time[index])
    def _read(self,index):
        for name in self.names:getattr(self.data,name)[index]=tensor(np.asarray(getattr(self.raw[index],name))).float()
    def step(self):
        import mujoco
        for index,d in enumerate(self.raw):self._write(index);mujoco.mj_step(self.model,d);self._read(index)
    def reset_data(self,ids):
        import mujoco
        for index in ids.tolist():mujoco.mj_resetData(self.model,self.raw[index]);self._read(index)
    def forward(self,ids=None):
        import mujoco
        for index in (range(len(self.raw)) if ids is None else ids.tolist()):
            self._write(index);mujoco.mj_forward(self.model,self.raw[index]);self._read(index)
    def contact_flags(self,pairs):
        flags=torch.zeros((len(self.raw),len(pairs)),dtype=torch.bool)
        for row,d in enumerate(self.raw):
            for column,(first,second) in enumerate(pairs.tolist()):
                flags[row,column]=any(c.dist<0 and set(c.geom)=={first,second} for c in d.contact)
        return flags
    def final_collision_data(self):
        import mujoco
        positions=[];rotations=[]
        for qpos in self.data.qpos:
            d=mujoco.MjData(self.model);d.qpos[:]=qpos.numpy();mujoco.mj_kinematics(self.model,d)
            positions.append(d.xpos.copy());rotations.append(d.xmat.copy())
        return SimpleNamespace(xpos=tensor(np.stack(positions)).float(),xmat=tensor(np.stack(rotations)).float())


def _tiny_bank(num_scenes=2):
    from cat_ppo.furniture.room_navigation import pack_room_scenes
    from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
    from cat_ppo.envs.g1.constants import DEFAULT_QPOS
    bank=SimpleNamespace(count=num_scenes,device=torch.device('cpu'),has_contrast=False,levels=None,roles=None,
        is_cat=torch.ones(num_scenes,dtype=torch.bool),reset_is_cat=torch.ones(num_scenes,dtype=torch.bool),
        crossed_is_plane=torch.ones(num_scenes,dtype=torch.bool),reset_yaws=torch.zeros(num_scenes),
        starts=torch.zeros((num_scenes,3)),reset_xy_scale=torch.ones((num_scenes,2)),goals=torch.tensor([[2.,0.,.7]]).expand(num_scenes,-1),
        origins=torch.tensor([[-5.,-5.,-.5]]).expand(num_scenes,-1),shapes=torch.tensor([[60,60,30]]).expand(num_scenes,-1),
        dxs=torch.full((num_scenes,),.2),episode_lengths=torch.full((num_scenes,),3),navigation_groups=torch.zeros(num_scenes,dtype=torch.long),
        reset_pool=tensor(np.asarray(DEFAULT_QPOS)).float()[None,None].expand(num_scenes,1,-1),
        rooms={k:tensor(v) for k,v in pack_room_scenes([None]*num_scenes).items()},
        contrast={k:tensor(v) for k,v in pack_hand_contrast([None]*num_scenes).items()})
    bank.probabilities=lambda weights=None,stage=None:torch.ones(num_scenes)/num_scenes if weights is None else weights/weights.sum()
    def sample(name,positions,scene_ids):
        if name=='sdf':return torch.ones((*positions.shape[:-1],1))
        result=torch.zeros_like(positions)
        if name=='gf':result[:,:,0]=.7
        return result
    bank.sample=sample
    return bank


def test_task_real_cpu_physics_reset_histories_and_terminal_penalty():
    from cat_mjlab.task import CATTask
    from cat_mjlab.config import wholebody_config
    class Collision:
        active=False
        def __call__(self,scenes,data):
            result=torch.zeros((len(scenes),6),dtype=torch.bool)
            if self.active:result[0,0]=True
            return result
    collision=Collision();sim=_CPUSimulation(2)
    cfg=wholebody_config();cfg['randomize_initial_episode_steps']=False
    task=CATTask(sim,_tiny_bank(),cfg,collision=collision,seed=13)
    assert task.obs['state'].shape==(2,222)
    assert task.obs['privileged_state'].shape==(2,310)
    assert torch.isfinite(task.obs['state']).all()
    result=task.step(torch.zeros((2,29)))
    assert not result['done'].any()
    # Original true projection uses raw GF; delayed actor command uses the
    # normalized held-pose GF. They intentionally differ on a .7-magnitude field.
    close(task.info['command'][:,1],np.full(2,.7*.7*.75))
    close(task.info['command_delay'][:,1],np.full(2,.7/(.7+1e-6)*.7*.75))
    prior_step=task.info['step'].clone();collision.active=True
    result=task.step(torch.zeros((2,29)))
    assert result['terminated'][0] and not result['truncated'][0]
    assert result['reward'][0]<0 and result['metrics']['reward/body_collision_event'][0]==-1.
    assert task.info['step'][0]==0 and task.info['step'][1]==prior_step[1]+1
    assert torch.equal(task.info['velocities'][0],torch.zeros_like(task.info['velocities'][0]))
    assert torch.equal(task.info['previous_upper'][0],task.nominal[12:])
    assert torch.equal(result['metrics']['navigation_counts'][0],torch.tensor([1,0]))
    assert not torch.equal(result['terminal_obs']['state'][0],result['obs']['state'][0])
    collision.active=False
    result=task.step(torch.zeros((2,29)))
    assert result['truncated'][1] and not result['terminated'][1]


@pytest.mark.parametrize('compile_task',[False,True])
def test_task_state_resume_preserves_random_stream_and_temporal_history(compile_task):
    from cat_mjlab.task import CATTask
    from cat_mjlab.config import wholebody_config
    cfg=wholebody_config();cfg['randomize_initial_episode_steps']=False
    first=CATTask(_CPUSimulation(2),_tiny_bank(),cfg,seed=23)
    first.step(torch.full((2,29),.04))
    saved=deepcopy(first.state_dict())
    physics={k:v.clone() for k,v in vars(first.data).items()}
    expected=first.step(torch.full((2,29),-.02))
    second=CATTask(_CPUSimulation(2),_tiny_bank(),cfg,seed=5)
    for key,value in physics.items():getattr(second.data,key).copy_(value)
    second.load_state_dict(saved)
    if compile_task:second.enable_compilation(backend='eager')
    actual=second.step(torch.full((2,29),-.02))
    close(actual['reward'],expected['reward'])
    for key in actual['obs']:close(actual['obs'][key],expected['obs'][key])


def test_field_sampler_compiles_as_one_graph_for_dynamic_reset_batches():
    from cat_mjlab.fields import sample_ragged_field
    from cat_mjlab.compilation import compile_batched_kernel
    compiled=compile_batched_kernel(sample_ragged_field,batch_arg='pos',backend='eager')
    field=torch.arange(4*5*6*3,dtype=torch.float32).reshape(-1,3)
    for batch in (2,1,3):
        positions=torch.full((batch,13,3),1.35)
        arguments=dict(origin=torch.zeros((batch,3)),dx=torch.ones(batch),
            shape=torch.tensor([[4,5,6]]).expand(batch,-1),offset=torch.zeros(batch,dtype=torch.long))
        close(compiled(field,positions,**arguments),sample_ragged_field(field,positions,**arguments))


def test_single_world_dispatch_keeps_eager_quaternions_and_correct_field_batch(monkeypatch):
    from cat_mjlab.compilation import compile_batched_kernel
    from cat_mjlab.fields import sample_ragged_field
    compiled_calls=[]
    def compile_spy(function,**options):
        def compiled(*args,**kwargs):
            compiled_calls.append(function.__name__)
            return function(*args,**kwargs)
        return compiled
    monkeypatch.setattr(torch,'compile',compile_spy)
    delayed=compile_batched_kernel(tm.delay_body_pos)
    sample=compile_batched_kernel(sample_ragged_field,batch_arg='pos')
    # Non-tensor metadata before the first tensor must not select the batch.
    context=compile_batched_kernel(nav.hand_contrast_context)
    from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
    for size in (3,1,2):
        before=len(compiled_calls)
        qpos=torch.zeros((size,36));qpos[:,3]=1.;poses=torch.rand((size,11,3))
        close(delayed(qpos=qpos,odom=qpos[:,:7],positions=poses),poses)
        fields=torch.ones((4*5*6,1));pos=torch.full((size,11,3),1.25)
        actual=sample(fields,pos,origin=torch.zeros((size,3)),dx=torch.ones(size),
                      shape=torch.tensor([[4,5,6]]).expand(size,-1),offset=torch.zeros(size,dtype=torch.long))
        close(actual,torch.ones((size,11,1)))
        metadata={key:tensor(value) for key,value in pack_hand_contrast([None]*size).items()}
        context(metadata,torch.zeros(size),torch.ones((size,2)))
        assert len(compiled_calls)-before==(0 if size==1 else 3)


def test_initial_horizon_offsets_do_not_advance_native_perception_or_grace_age():
    from cat_mjlab.task import CATTask
    from cat_mjlab.config import wholebody_config
    task=CATTask(_CPUSimulation(5),_tiny_bank(),wholebody_config(),seed=43)
    assert (task.info['step']==0).all()
    assert (task.info['wrapper_steps']>=0).all() and (task.info['wrapper_steps']<1000).all()
    assert task.info['wrapper_steps'].unique().numel()>1
    task.reset(torch.tensor([0,1]))
    assert (task.info['wrapper_steps'][:2]==0).all()


def test_backend_migration_preserves_sampler_curriculum_and_initial_horizon_offsets():
    from cat_mjlab.task import CATTask
    from cat_mjlab.config import wholebody_config
    from cat_mjlab.scene_bank import SceneBank
    bank=_tiny_bank()
    bank.levels=torch.tensor([0,2]);bank.groups=torch.tensor([2,2])
    bank.group_masses=torch.tensor([0.,0.,1.,0.]);bank.weights=torch.ones(2)
    bank.probabilities=lambda weights=None,stage=None:SceneBank.probabilities(bank,weights,stage)
    task=CATTask(_CPUSimulation(3),bank,wholebody_config(),seed=12)
    assert (task.scene_ids==0).all()
    offsets=task.info['wrapper_steps'].clone()
    transferred={
        'pf_episode_ema':np.array([12.,30.],np.float32),
        'pf_success_ema':np.array([5.,19.],np.float32),
        'pf_sampling_logits':np.array([-np.inf,0.],np.float32),
        'pf_hand_curriculum_stage':np.array(2,np.int32),
        'pf_hand_curriculum_completed':np.array([92,110,64],np.int32),
        'pf_hand_curriculum_goals':np.array([80,90,32],np.int32),
        'pf_navigation_outcome_counts':np.array([[10,4],[2,1],[8,7],[20,12]],np.int32),
        'pf_contrast_outcome_counts':np.array([[2,1],[7,4],[3,0]],np.int32)}
    task.restore_sampling_state(transferred)
    assert (task.scene_ids==1).all()
    assert torch.equal(task.info['wrapper_steps'],offsets)
    assert (task.info['step']==0).all()
    for name,value in task.sampling_state().items():close(value,transferred[name])
    # Returned shared state must be independent of live mutable counters.
    exported=task.sampling_state();exported['pf_episode_ema'].zero_()
    close(task.scene_episode_ema,transferred['pf_episode_ema'])
    before=deepcopy(task.state_dict())
    invalid=deepcopy(transferred);invalid['pf_hand_curriculum_stage']=np.array(0)
    with pytest.raises(ValueError,match='locked scene'):task.restore_sampling_state(invalid)
    for name in ('probabilities','scene_episode_ema','curriculum_stage','navigation_counts','rng'):
        assert torch.equal(task.state_dict()[name],before[name])
    invalid=deepcopy(transferred);invalid['pf_hand_curriculum_goals'][0]=1000
    with pytest.raises(ValueError,match='curriculum'):task.restore_sampling_state(invalid)


def test_sampler_transfer_rejects_changed_fixed_group_mass_and_partial_state():
    from cat_mjlab.task import CATTask
    from cat_mjlab.config import wholebody_config
    from cat_mjlab.scene_bank import SceneBank
    bank=_tiny_bank(4);bank.roles=torch.arange(4);bank.weights=torch.ones(4)
    bank.probabilities=lambda weights=None,stage=None:SceneBank.probabilities(bank,weights,stage)
    task=CATTask(_CPUSimulation(1),bank,wholebody_config(),seed=7)
    state=task.sampling_state()
    state['pf_sampling_logits']=torch.tensor([0.,0.,0.,1.])
    with pytest.raises(ValueError,match='group mass'):task.restore_sampling_state(state,resample=False)
    state=task.sampling_state();state.pop('pf_success_ema')
    with pytest.raises(ValueError,match='Incomplete'):task.restore_sampling_state(state,resample=False)


def _contrast_outcome_task(roles,policy_ids):
    """Exercise real outcome accounting without physics or scene generation."""
    from cat_mjlab.task import CATTask
    from cat_mjlab.scene_bank import SceneBank
    task=object.__new__(CATTask)
    task.device=torch.device('cpu');task.num_envs=len(roles)
    task.config={'pf_config':{'sampling_ema_decay':.95,'sampling_alpha':1.}}
    task.bank=SimpleNamespace(count=4,roles=torch.arange(4),levels=None,
        groups=None,device=task.device,weights=torch.ones(4),navigation_groups=torch.full((4,),2))
    task.bank.probabilities=lambda weights=None,stage=None:SceneBank.probabilities(task.bank,weights,stage)
    task.scene_ids=torch.tensor(roles,dtype=torch.long)
    task.navigation_counts=torch.zeros((4,2),dtype=torch.long)
    task.contrast_counts=torch.zeros((3,2),dtype=torch.long)
    task.role_counts=torch.zeros((4,2),dtype=torch.long)
    task.policy_ids=torch.zeros(task.num_envs,dtype=torch.long)
    task.set_policy_ids(torch.tensor(policy_ids))
    task.outcome_counted=torch.zeros(task.num_envs,dtype=torch.bool)
    task.hand_contrast=True
    required=torch.zeros((task.num_envs,6),dtype=torch.bool)
    required[:,0]=(task.scene_ids==1)|(task.scene_ids==3)
    task.contrast=dict(role=task.scene_ids,zone_index=torch.zeros(task.num_envs,dtype=torch.long),
        core_active=torch.ones(task.num_envs,dtype=torch.bool),
        required_forward_zones=required,required_hand_zones=required.clone())
    task.zone_steps=torch.zeros((task.num_envs,6,3),dtype=torch.long)
    task.telemetry={name:torch.ones(task.num_envs) for name in
        ('hand_contrast_heading_good','hand_contrast_hand_good')}
    task.episode={name:torch.zeros(task.num_envs,dtype=torch.bool) for name in
        ('fall','obstacle','self_contact','numerical','body_collision','hand_violation','elbow_violation','outside_bounds','goal_reached')}
    task.scene_episode_ema=torch.zeros(4);task.scene_success_ema=torch.zeros(4)
    task.curriculum_stage=torch.zeros((),dtype=torch.long)
    return task


def test_success_counts_describe_leader_but_sampler_uses_every_policy():
    task=_contrast_outcome_task([0,1,2,3]*2,[0]*4+[1]*4)
    task.episode['goal_reached'][:]=True
    task.episode['fall'][1]=True
    task.telemetry['hand_contrast_heading_good'][3]=0.
    done=torch.zeros(8,dtype=torch.bool);done[1]=True
    resolved,clean=task._outcomes(done,torch.zeros_like(done))
    assert resolved.all() and clean.sum()==7
    assert torch.equal(task.navigation_counts[3],torch.tensor([4,3]))
    assert torch.equal(task.contrast_counts,torch.tensor([[1,0],[1,1],[1,0]]))
    assert torch.equal(task.role_counts,torch.tensor([[1,1],[1,0],[1,1],[1,0]]))
    # Physical follower outcomes still inform task sampling. A leader's clean
    # sideways transition fails the posture objective even though it navigated.
    close(task.scene_episode_ema,[2,2,2,2])
    close(task.scene_success_ema,[2,1,2,1])
    before=task.role_counts.clone()
    again,_=task._outcomes(done,torch.zeros_like(done))
    assert not again.any() and torch.equal(task.role_counts,before)
    with pytest.raises(ValueError,match='after recording outcomes'):
        task.set_policy_ids(torch.zeros(8,dtype=torch.long))


def test_protected_sampling_requires_each_zone_and_ninety_percent_posture():
    # Two examples of each role, all reaching clean goals. Even open/narrow
    # bad posture remains successful: only protected/transition sampling changes.
    task=_contrast_outcome_task([0,0,1,1,2,2,3,3],[0]*8)
    task.episode['goal_reached'][:]=True
    task.zone_steps[:,0]=torch.tensor([9,8,8])
    task.telemetry['hand_contrast_heading_good'][:]=0.
    task.telemetry['hand_contrast_hand_good'][:]=0.
    # One protected world achieves exactly 90%, its matched one only 80%.
    task.telemetry['hand_contrast_heading_good'][2]=1.
    task.telemetry['hand_contrast_hand_good'][2]=1.
    # First transition has perfect posture in visited zone zero but never
    # visits required zone one. Second qualifies in both zones.
    for key in ('required_forward_zones','required_hand_zones'):
        task.contrast[key][6:,1]=True
    task.zone_steps[6:,0]=torch.tensor([9,9,9])
    task.telemetry['hand_contrast_heading_good'][6:]=1.
    task.telemetry['hand_contrast_hand_good'][6:]=1.
    task.zone_steps[7,1]=torch.tensor([10,9,9])
    task._outcomes(torch.zeros(8,dtype=torch.bool),torch.zeros(8,dtype=torch.bool))
    close(task.scene_success_ema,[2,1,2,1])
    assert torch.equal(task.role_counts,torch.tensor([[2,2],[2,1],[2,2],[2,1]]))
    # The fixed behavior-role masses survive adaptation.
    close(task.probabilities,torch.full((4,),.25))


def test_leader_population_assignment_and_role_counts_survive_native_resume():
    from cat_mjlab.task import CATTask
    from cat_mjlab.config import wholebody_config
    first=CATTask(_CPUSimulation(2),_tiny_bank(),wholebody_config(),seed=3)
    first.set_policy_ids(torch.tensor([0,5]))
    first.role_counts.copy_(torch.tensor([[3,1],[4,2],[5,3],[6,4]]))
    second=CATTask(_CPUSimulation(2),_tiny_bank(),wholebody_config(),seed=7)
    second.load_state_dict(deepcopy(first.state_dict()))
    assert torch.equal(second.policy_ids,torch.tensor([0,5]))
    assert torch.equal(second.role_counts,first.role_counts)
    for invalid in (torch.tensor([0.,1.]),torch.tensor([-1,0]),torch.tensor([True,False]),torch.tensor([0])):
        with pytest.raises(ValueError,match='integer policy ID'):second.set_policy_ids(invalid)


def test_legacy_sampler_still_uses_completed_episode_survival():
    task=_contrast_outcome_task([0,1],[0,1])
    task.hand_contrast=False;task.bank.roles=None
    task.episode['goal_reached'][0]=True
    # Leader reaches a goal but continues its episode; follower survives to the
    # horizon. Native legacy sampling counts only that completed episode.
    done=torch.tensor([False,True]);task._outcomes(done,done.clone())
    close(task.scene_episode_ema,[0,1,0,0])
    close(task.scene_success_ema,[0,1,0,0])
    assert torch.equal(task.navigation_counts[3],torch.tensor([1,1]))
    assert not task.role_counts.any()
