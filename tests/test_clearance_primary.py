"""CPU objective and assigned-trial regression contracts; never train."""
from copy import deepcopy
import pytest
import torch
from cat_mjlab.config import wholebody_config
from cat_mjlab.task_math import hand_clearance_pressure
from cat_mjlab.acceptance import ClearancePassageGate, TrainingAcceptance
from test_mjlab_acceptance import geometry, step, status, fake_task, transition


def test_cli_defaults_overrides_and_validation():
    from train_cat_mjlab import parser
    common=['run','--from-scratch','--bank-manifest','bank','--body-collision-bank','body',
            '--body-collision-resets','reset','--run-dir','run']
    args=parser().parse_args(common)
    assert args.disable_hand_contrast is True and args.hand_clearance_weight is None
    args=parser().parse_args(common+['--disable-hand-contrast','--hand-clearance-weight','-20',
        '--hand-clearance-target','.05','--hand-clearance-anticipation','.25','--hand-clearance-near-weight','.5'])
    keys=['disable_hand_contrast','hand_clearance_weight','hand_clearance_target','hand_clearance_anticipation','hand_clearance_near_weight']
    c=wholebody_config(hand_contrast=True,**{k:getattr(args,k) for k in keys})
    assert c['reward_config']['scales']['wholebody_hand_contrast_heading']==0
    assert c['reward_config']['scales']['wholebody_hand_contrast_region']==0
    assert c['reward_config']['scales']['wholebody_hand_clearance']==-20
    assert c['hand_protection_target_clearance']==.05
    inherited=wholebody_config(c,hand_contrast=True)
    assert inherited['clearance_objective_parameters']==c['clearance_objective_parameters']
    for kw in [dict(hand_clearance_weight=1),dict(hand_clearance_weight=float('nan')),
               dict(hand_clearance_target=0),dict(hand_clearance_target=.3),
               dict(hand_clearance_anticipation=.02),dict(hand_clearance_near_weight=1.1),dict(hand_clearance_target=True)]:
        with pytest.raises(ValueError):wholebody_config(**kw)


def test_known_pressure_and_scale_magnitudes():
    distances=torch.tensor([[.2,1.],[.1,1.],[.04,1.],[0.,1.]])
    p,_=hand_clearance_pressure(distances,near_weight=.5)
    torch.testing.assert_close(-20*p*.02,torch.tensor([0.,-.025,-.064,-.2]))


@pytest.mark.parametrize('cohort',['flat','clutter','cat','narrow'])
def test_default_flags_preserve_real_cpu_transition(cohort):
    from test_mjlab_task import _CPUSimulation,_tiny_bank
    from cat_mjlab.task import CATTask
    old=wholebody_config();old['randomize_initial_episode_steps']=False
    new=wholebody_config(disable_hand_contrast=False,hand_clearance_weight=-.5,
        hand_clearance_target=.04,hand_clearance_anticipation=.20,hand_clearance_near_weight=.8)
    new['randomize_initial_episode_steps']=False
    def make(cfg):
        b=_tiny_bank(1)
        # Exercise legacy CAT reset and the three room reset/group paths.
        b.reset_is_cat[:]=cohort=='cat';b.is_cat[:]=cohort=='cat'
        b.navigation_groups[:]=dict(cat=0,flat=1,clutter=1,narrow=2)[cohort]
        return CATTask(_CPUSimulation(1),b,cfg,seed=17)
    a,b=make(old),make(new)
    for _ in range(3):
        x,y=a.step(torch.zeros(1,29)),b.step(torch.zeros(1,29))
        for key in x['obs']:torch.testing.assert_close(x['obs'][key],y['obs'][key],rtol=0,atol=0)
        torch.testing.assert_close(x['reward'],y['reward'],rtol=0,atol=0)
        torch.testing.assert_close(a.data.qpos,b.data.qpos,rtol=0,atol=0)


def test_disabled_boxes_keep_metrics_and_clearance_survives_floor():
    from test_mjlab_task import _CPUSimulation,_tiny_bank
    from test_mjlab_hand_approach import context
    from cat_mjlab.task import CATTask
    from cat_mjlab.lateral_corridor import ArmGeometry
    cfg=wholebody_config(hand_contrast=False,disable_hand_contrast=True,hand_clearance_weight=-20)
    task=CATTask(_CPUSimulation(1),_tiny_bank(1),cfg,seed=17)
    task.hand_contrast=True;task.contrast=context(1.)
    task.bank.acceptance=geometry();task.corridor_arms=ArmGeometry(task.model,task.device)
    # Force positive diagnostic box cost and a pre-floor negative ledger.
    task.info['sdf'][:,5:7]=0.;task.info['pelvis_rpy'][:]=2.
    # Directly overriding scales must not re-enable disabled box rewards.
    cfg['reward_config']['scales'].update(wholebody_hand_contrast_region=-1000.,wholebody_hand_contrast_heading=-1000.)
    r,p=task._rewards(torch.zeros(1,29),torch.zeros(1,2,dtype=torch.bool))
    assert task.telemetry['hand_contrast_region_cost'].item()>0
    assert p['wholebody_hand_contrast_region'].item()==0
    assert p['wholebody_hand_contrast_heading'].item()==0
    assert p['wholebody_hand_clearance'].item()==-20
    # Independently recover exact floor-plus-clearance result.
    base=sum(v for k,v in p.items() if k!='wholebody_hand_clearance')*.02
    torch.testing.assert_close(r,base.clamp(0,10000)-.4)
    assert task.obs['state'].shape==(1,222)
    assert task.obs['privileged_state'].shape==(1,310)


def clear_gate():
    return ClearancePassageGate(geometry(),torch.zeros(1,2),20.,initial_clearance=torch.tensor([[.3,.3]]))


def advance(g,x,t,d=(.3,.3),**flags):
    names=('clean_goal','done','hand_collision','body_collision','fall')
    g.advance(torch.tensor([[x,flags.pop('y',0.)]]),t,
        hand_clearance=None if d is None else torch.tensor([d]),
        **{k:torch.tensor([flags.get(k,False)]) for k in names})


@pytest.mark.parametrize('strategy',['refuse_entry','crawl','bypass','fall_first','contact','missing','too_close'])
def test_clearance_cannot_game_s_gate(strategy):
    g=clear_gate()
    if strategy=='refuse_entry':advance(g,0,20,done=True)
    elif strategy=='crawl':
        advance(g,1,1);advance(g,1.5,7);advance(g,3,9,clean_goal=True)
    elif strategy=='bypass':advance(g,3,3,y=2.,clean_goal=True)
    elif strategy=='fall_first':advance(g,.1,1,fall=True,done=True)
    elif strategy=='contact':advance(g,3,3,hand_collision=True,clean_goal=True)
    elif strategy=='missing':advance(g,3,3,d=None,clean_goal=True)
    else:advance(g,3,3,d=(.03,.3),clean_goal=True)
    assert status(g)=='FAIL'
    assert g.counts()['assigned_count'].item()==1
    assert not g.counts()['pass_count'].item()


def test_clearance_duration_minimum_pending_and_freeze():
    g=clear_gate();advance(g,.5,1,d=(.1,.3));advance(g,3,3,d=(.04,.3),clean_goal=True)
    assert status(g)=='PASS'
    c=g.counts();assert c['clearance_observed_s']==3
    assert c['clearance_inside_20cm_s']==3
    assert c['clearance_below_4cm_s']==0
    assert float(g.clearance_min[0])==pytest.approx(.04)
    advance(g,4,4,d=(-.1,.3),hand_collision=True)
    assert status(g)=='PASS' and g.counts()['hand_collision_count']==0
    assert g.counts()['clearance_observed_s']==3
    pending=clear_gate();advance(pending,0,1)
    assert status(pending)=='PENDING' and pending.counts()['assigned_count']==1


def test_clearance_observer_terminal_capture_and_resume():
    task=fake_task();task.config={'disable_hand_contrast':True};task.info['sdf']=torch.ones(2,11,1)
    obs=TrainingAcceptance();obs.before_step(task,torch.zeros(2,dtype=torch.long))
    event=transition(task,3,goal=True);event['metrics']['acceptance/hand_clearance']=torch.tensor([[.1,.3],[.02,.3]])
    obs.after_step(task,event);m=obs.metrics();p='training/passage_acceptance_leader_'
    assert m[p+'success_rate']==.5
    assert m[p+'clearance_inside_20cm_fraction']==1
    assert m[p+'clearance_below_4cm_fraction']==.5
    restored=TrainingAcceptance();restored.load_state_dict(deepcopy(obs.state_dict()))
    assert isinstance(restored.gate,ClearancePassageGate)
    assert restored.metrics()==m


def test_real_bank_flat_clutter_cat_narrow_defaults_and_flat_demotion():
    """Five immutable real fields, actual collision detector and certified resets."""
    import json
    from pathlib import Path
    import numpy as np
    from test_mjlab_task import _CPUSimulation,_tiny_bank
    from cat_mjlab.task import CATTask
    from cat_mjlab.fields import sample_ragged_field
    from cat_mjlab.collision import CollisionChecker
    from cat_mjlab.acceptance import pack_geometry
    from cat_mjlab.passage_rewards import passage_parameters
    from cat_ppo.furniture.generalist_fields import scene_directory
    from cat_ppo.furniture.room_navigation import pack_room_scenes
    from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
    root=Path('data/furniture');path=root/'cat_flat_hand_balance_v2_20260921/manifest.json'
    if not path.exists():pytest.skip('Immutable v2 bank not present')
    m=json.loads(path.read_text());records=m['scenes']
    predicates=[lambda r:r['source'].get('flat_balance'),
        lambda r:r['family']=='generic_clutter' and not r['source'].get('flat_balance') and not r['source'].get('hand_contrast'),
        lambda r:r['family']=='original_cat',
        lambda r:r['source'].get('hand_contrast',{}).get('role')=='narrow',
        lambda r:r['source'].get('hand_contrast',{}).get('role')=='forward_protected']
    ids=[next(i for i,r in enumerate(records) if pred(r)) for pred in predicates]
    records=[records[i] for i in ids];n=len(ids);b=_tiny_bank(n)
    dirs=[scene_directory(m,path,r) for r in records]
    scenes=[json.loads((d/'scene.json').read_text()) if r['task_kind']=='room' else None for r,d in zip(records,dirs)]
    t=lambda a,dtype=None:torch.as_tensor(np.asarray(a).copy(),dtype=dtype)
    for attr,key,dtype in [('starts','start',torch.float32),('goals','goal',torch.float32),('origins','origin',torch.float32),
        ('shapes','shape',torch.long),('dxs','dx',torch.float32),('reset_xy_scale','reset_xy_scale',torch.float32),
        ('reset_yaws','reset_yaw',torch.float32),('episode_lengths','episode_length',torch.long)]:
        setattr(b,attr,t([r[key] for r in records],dtype))
    b.is_cat=t([r['task_kind']=='cat' for r in records]);b.reset_is_cat=t([r['reset_mode']=='cat' for r in records]);b.crossed_is_plane=t([r['crossed_mode']=='x_plane' for r in records])
    b.rooms={k:t(v) for k,v in pack_room_scenes(scenes).items()}
    b.contrast={k:t(v) for k,v in pack_hand_contrast(scenes).items()}
    b.contrast.update({k:t(v) for k,v in passage_parameters(scenes,bank_path=path).items()})
    b.acceptance={k:t(v) for k,v in pack_geometry(scenes).items()}
    b.has_contrast=True;b.has_sdf_reward_overrides=False;b.balance_settings=m['flat_balance']['settings']
    b.flat_balance=t([bool(r['source'].get('flat_balance')) for r in records]);b.navigation_groups=t([1,1,0,2,2],torch.long)
    sizes=[np.prod(r['shape']) for r in records];offsets=t(np.cumsum([0]+sizes[:-1]),torch.long)
    fields={name:t(np.concatenate([np.load(d/(name+'.npy'),mmap_mode='r').reshape(-1,ch) for d in dirs])) for name,ch in [('sdf',1),('bf',3),('gf',3)]}
    b.sample=lambda name,pos,ids:sample_ragged_field(fields[name],pos,origin=b.origins[ids],dx=b.dxs[ids],shape=b.shapes[ids],offset=offsets[ids])
    reset_path=root/'cat_flat_hand_balance_v2_20260921_resets/manifest.json'
    reset=json.loads(reset_path.read_text());b.reset_pool=t(np.load(reset_path.parent/reset['file'],mmap_mode='r')[ids])
    sim=_CPUSimulation(n);checker=CollisionChecker(sim.model,root/'cat_flat_hand_balance_v2_20260921_collision/manifest.json',field_manifest=path,device='cpu')
    collision=lambda scenes,data:checker(t(ids,torch.long)[scenes],data)
    source=json.loads(Path('outputs/cat_recover_23_30720_20260922/run.json').read_text())['contract']['environment_config']
    old=wholebody_config(source,hand_contrast=True);old['randomize_initial_episode_steps']=False
    new=wholebody_config(source,hand_contrast=True,disable_hand_contrast=False,hand_clearance_weight=-.5,
        hand_clearance_target=.04,hand_clearance_anticipation=.20,hand_clearance_near_weight=.8)
    new['randomize_initial_episode_steps']=False
    a=CATTask(sim,b,old,collision=collision,seed=19)
    z=CATTask(_CPUSimulation(n),b,new,collision=collision,seed=19)
    a.reset(scene_ids=torch.arange(n));z.reset(scene_ids=torch.arange(n))
    for _ in range(5):
        x,y=a.step(torch.zeros(n,29)),z.step(torch.zeros(n,29))
        torch.testing.assert_close(x['reward'],y['reward'],atol=0,rtol=0)
        for key in x['obs']:torch.testing.assert_close(x['obs'][key],y['obs'][key],atol=0,rtol=0)
        torch.testing.assert_close(a.data.qpos,z.data.qpos,atol=0,rtol=0)
    # Legacy flag changes cannot revive any box reward: flat bonus stays zero,
    # diagnostic posture cost and collision detector remain available.
    a.config['disable_hand_contrast']=True
    _,parts=a._rewards(torch.zeros(n,29),torch.zeros(n,2,dtype=torch.bool))
    for name in ('flat_balance_posture_bonus','wholebody_hand_contrast_region','wholebody_hand_contrast_heading'):
        assert torch.count_nonzero(parts[name])==0
    assert 'balance_cost' in a.telemetry and 'hand_contrast_region_cost' in a.telemetry


def test_selection_does_not_use_box_compliance():
    from cat_mjlab.clearance_objective import clearance_selection
    p='training/passage_acceptance_leader_'
    m={p+'assigned_count':10,p+'clearance_observed_s':100,p+'pass_count':5,p+'success_rate':.5}
    h={'success/cat_goal_success_rate':.8,'selection/flat_walking':.9}
    assert clearance_selection(m,h)==pytest.approx(.64)
    h['selection/flat_walking_compliant']=0;h['selection/protected_core_compliant']=0
    assert clearance_selection(m,h)==pytest.approx(.64)
    assert clearance_selection({},h) is None


def test_prepared_clearance_launch_parses_without_launching():
    import shlex
    from pathlib import Path
    from train_cat_mjlab import parser
    text=Path('configs/pilots/clearance_primary_50.sh').read_text().replace('\\\n',' ')
    line=next(line for line in text.splitlines() if 'train_cat_mjlab.py run' in line)
    args=parser().parse_args(shlex.split(line)[2:])
    assert args.disable_hand_contrast and args.hand_clearance_weight==-20
    assert args.from_scratch and args.checkpoint_native is None and args.checkpoint_npz is None
    assert args.wandb_mode=='online' and args.wandb_entity=='skvayzer' and args.wandb_project=='CAT-wholebody'
    for path in (args.bank_manifest,args.body_collision_bank,args.body_collision_resets):assert path.is_file()


def test_clearance_reference_evaluator_includes_reset_and_fails_without_evidence():
    from cat_mjlab.acceptance import evaluate_assigned_trials
    room=dict(route=[[0.,0.],[10.,0.]],hand_contrast=dict(zones=[dict(start_m=1.,end_m=2.)],modules=[dict(width_m=1.)]))
    row=dict(trial_id='one',room=room,start_xy=[0.,0.],deadline_s=20.,cohort='upstream',
        reset_body_collision=False,reset_hand_collision=False,initial_hand_clearance=[.03,.3])
    event=dict(xy=[3.,0.],time_s=3.,hand_clearance=[.3,.3],clean_goal=True,done=False,
        hand_collision=False,body_collision=False,fall=False,other=False)
    result=evaluate_assigned_trials([row],lambda _: [event],clearance_primary=True)
    assert result['trials']['one']['status']=='FAIL'
    assert result['cohorts']['upstream']['minimum_hand_clearance_m']==pytest.approx(.03)
    assert result['cohorts']['upstream']['hand_collision_incidence']==0
    with pytest.raises(ValueError,match='reset'):
        ClearancePassageGate(geometry(),torch.zeros(1,2),20.)
