"""CPU-only speed, gait, trial anti-cheating and hysteresis regression checks."""
import copy
from types import SimpleNamespace
import pytest
import torch
from cat_mjlab import speed_curriculum as sc
from cat_mjlab.config import wholebody_config
from cat_mjlab.task import CATTask
from cat_mjlab import task_math
from test_mjlab_hand_approach import context

CFG=dict(window=4,threshold=.75,demote_threshold=.4)


def test_default_noop_and_mutual_exclusion():
    base=wholebody_config(hand_contrast=True)
    assert sc.configure(copy.deepcopy(base))==base
    enabled=sc.configure(copy.deepcopy(base),enabled=True)
    assert sc.configure(enabled)==base
    from cat_mjlab.tolerance_curriculum import configure
    with pytest.raises(ValueError,match='separately'):
        sc.configure(configure(copy.deepcopy(base),enabled=True),enabled=True)
    for kwargs in ({'window':0},{'threshold':float('nan')},{'threshold':.2,'demote_threshold':.4}):
        with pytest.raises(ValueError):sc.configure(base,**kwargs)
    with pytest.raises(ValueError):sc.configure({},enabled=True)
    with pytest.raises(ValueError):sc.configure(dict(base,hand_contrast_metric_tolerance=.2),enabled=True)


def feed(state, values, role=1, seeded=False, generation=None):
    n=len(values)
    sc.record(state,CFG,roles=torch.full((n,),role),rungs=state['stage'].expand(n),
        generations=torch.full((n,),int(state['generation']) if generation is None else generation),
        eligible=torch.ones(n,dtype=torch.bool),success=torch.tensor(values,dtype=torch.bool),
        compliance=torch.tensor(values,dtype=torch.float),seeded=torch.full((n,),seeded))


def test_progression_demotion_rolling_and_generations():
    s=sc.initialize(CFG,1,'cpu')
    feed(s,[1]*8,seeded=True)
    assert s['stage']==0 and s['count'].sum()==0
    feed(s,[0]*4);feed(s,[1]*3)
    feed(s,[1]*4,role=3)
    assert s['stage']==1  # rolling 3/4, lifetime 3/7
    feed(s,[1]*4,generation=0)
    assert s['count'].sum()==0
    feed(s,[0]*4)
    assert s['stage']==1
    feed(s,[0]*4)
    assert s['stage']==0 and s['demotions']==1
    for rung in range(5):
        feed(s,[1]*4);feed(s,[1]*4,role=3)
        assert s['stage']==min(rung+1,4)
    assert sc.metrics(s)['hand_speed/command_m_s']==.6
    # A large simultaneous batch cannot hide two failed windows behind a good tail.
    feed(s,[0]*8+[1]*4)
    assert s['stage']==3


def test_scope_caps_and_actual_task_fields():
    n=7;c={k:v.repeat((n,)+(1,)*(v.ndim-1)) for k,v in context(1.).items()}
    c['role']=torch.tensor([-1,0,2,1,3,3,1])
    c['hand_active'][5]=False  # transition's narrow module
    c['enabled'][6]=False  # protected scene outside its zones
    s=sc.initialize(CFG,n,'cpu');s['episode_rung'].fill_(2)
    ids=torch.arange(n)
    active,cap,hold=sc.speed_limit(s,c,ids,.02)
    assert active.tolist()==[False,False,False,True,True,False,False]
    t=object.__new__(CATTask);t.scene_ids=ids;t.contrast=c;t.dt=.02
    t.hand_radii=torch.zeros(2)
    fields={'gf':torch.ones(n,11,3)*.2,'bf':torch.zeros(n,11,3),'sdf':torch.ones(n,11,1)}
    t.bank=SimpleNamespace(sample=lambda key,positions,scenes:fields[key].clone())
    t.navigation=dict(target=torch.tensor([[10.,0.]]).repeat(n,1),guidance=torch.tensor([[.6,0.,0.]]).repeat(n,1),
        blocked=torch.zeros(n,dtype=torch.bool),violation=torch.zeros(n,dtype=torch.bool),enabled=torch.ones(n,dtype=torch.bool))
    t._room_meta=lambda ids:dict(obstacles=None,obstacle_count=None,navigation_radius=None)
    t.swept_root_clearance=lambda *args,**kwargs:torch.ones(n)
    t.math=task_math
    args=(torch.zeros(n,11,3),torch.zeros(n,2),ids)
    baseline=t._fields(*args)
    t.speed_state=s
    revised=t._fields(*args)
    torch.testing.assert_close(revised[3][:,1],torch.tensor([.6,.6,.6,.3,.3,.6,.6]))
    for before,after in zip(baseline,revised):
        torch.testing.assert_close(before[~active],after[~active],rtol=0,atol=0)
    s['episode_rung'].zero_()
    stopped=t._fields(*args)
    assert (stopped[3][active]==0).all() and (stopped[0][active]==0).all()
    s['ticks'].fill_(225)
    released=t._fields(*args)
    torch.testing.assert_close(released[3][active,1],torch.full((2,),.2))


def test_real_stance_and_release_gait():
    cmd=torch.tensor([[.75,.0,.0,.0],[.75,.6,0,0]])
    old=torch.tensor([[1.,.6,0,0],[1.,.6,0,0]])
    phase=torch.tensor([[0.,torch.pi],[0.,torch.pi]])
    a,stop,p,g=task_math.update_phase(cmd,old,torch.full((2,),100),phase,torch.full((2,),.2),.6,torch.ones(2,dtype=torch.bool))
    values=sc.standing_phase(a,cmd,stop,p,g,torch.tensor([True,False]))
    assert (values[0][0]==0).all() and (values[1][0]==0).all()
    assert values[2][0]==0 and (values[3][0]==0).all() and (values[4][0]==1).all()
    for before,after in zip((a,cmd,stop,p,g),values):torch.testing.assert_close(before[1],after[1])
    released=sc.restart_phase(values[3],values[0],torch.tensor([[.75,.2,0,0],[1.,.6,0,0]]),torch.tensor([True,False]))
    assert released[0,1]-released[0,0]==pytest.approx(torch.pi)


def trial(rung, *, velocity=None, good=True, seeded=False, resolved_at=None, fail_at=None):
    s=sc.initialize(CFG,1,'cpu');s['stage'].fill_(rung)
    sc.reset(s,torch.tensor([0]),torch.zeros(1))
    c=context(1.);p=torch.zeros(1)
    for tick in range(340):
        v=(0 if tick<225 else .2) if velocity is None else velocity
        before=p.clone();p=p+v*.02
        sc.observe(s,CFG,context=c,previous_progress=before,progress=p,
            strict_good=torch.tensor([good]),heading_good=torch.tensor([True]),
            failure=torch.tensor([tick==fail_at]),resolved=torch.tensor([tick==resolved_at]),
            leader=torch.tensor([True]),seeded=torch.tensor([seeded]),dt=.02)
        if s['finished'].item():break
    return s


def test_zero_hold_bounded_and_requires_release_progress():
    s=trial(0)
    assert s['successes'][0,0]==1 and 225<s['ticks']<=325
    frozen=trial(0,velocity=0)
    assert frozen['finished'].item() and frozen['ticks']==325
    assert frozen['successes'].sum()==0
    moving=trial(0,velocity=.2)
    assert moving['successes'].sum()==0  # Cannot skip the actual standing trial.
    seeded=trial(0,seeded=True)
    assert seeded['seeded_successes'][0]==1 and seeded['count'].sum()==0


@pytest.mark.parametrize('rung,speed',[(1,.2),(2,.3),(3,.45),(4,.6)])
def test_positive_rungs_require_compliant_motion(rung,speed):
    assert trial(rung,velocity=speed)['successes'][rung,0]==1
    assert trial(rung,velocity=0)['successes'].sum()==0
    assert trial(rung,velocity=2*speed)['successes'].sum()==0
    assert trial(rung,velocity=speed,good=False)['successes'].sum()==0
    assert trial(rung,velocity=speed,fail_at=99)['successes'].sum()==0


def test_early_exit_no_entry_and_checkpoint_roundtrip():
    s=sc.initialize(CFG,1,'cpu');c=context(1.)
    kwargs=dict(settings=CFG,previous_progress=torch.zeros(1),progress=torch.zeros(1),
        strict_good=torch.ones(1,dtype=torch.bool),heading_good=torch.ones(1,dtype=torch.bool),
        failure=torch.zeros(1,dtype=torch.bool),resolved=torch.zeros(1,dtype=torch.bool),
        leader=torch.ones(1,dtype=torch.bool),seeded=torch.zeros(1,dtype=torch.bool),dt=.02)
    sc.observe(s,context=c,**kwargs)
    c['core_active'].zero_()
    sc.observe(s,context=c,**kwargs)
    assert s['completed'][0,0]==1 and s['successes'].sum()==0
    sc.reset(s,torch.tensor([0]),torch.zeros(1))
    kwargs['resolved'].fill_(True)
    sc.observe(s,context=c,**kwargs)
    assert s['completed'][0,0]==2
    # Real CATTask serializer retains every curriculum tensor.
    t=object.__new__(CATTask);t.device='cpu';t.generator=torch.Generator();t.speed_state=s
    for name in ('info','navigation','contrast','episode','telemetry','obs'):setattr(t,name,{})
    for name in ('scene_ids','scene_episode_ema','scene_success_ema','curriculum_stage','curriculum_completed',
                 'curriculum_goals','probabilities','navigation_counts','contrast_counts','policy_ids',
                 'role_counts','zone_steps','outcome_counted','episode_reward'):setattr(t,name,torch.zeros(1))
    saved=copy.deepcopy(t.state_dict());s['completed'].zero_();t.load_state_dict(saved)
    assert t.speed_state['completed'][0,0]==2


def test_actual_outcomes_wire_speed_and_count_no_entry_once():
    from test_mjlab_hand_curriculum_gate import hand_task
    t=hand_task(config={'hand_speed_curriculum':CFG},completed=0,goals=0)
    t.hand_contrast=True;t.num_envs=1;t.dt=.02
    t.bank.levels=None;t.bank.roles=torch.tensor([1])
    t.info={};t.navigation={'progress_m':torch.zeros(1)}
    t.contrast=context(.1)
    t.telemetry=dict(hand_contrast_heading_good=torch.ones(1),hand_contrast_hand_good=torch.zeros(1))
    t.zone_steps=torch.zeros((1,6,3),dtype=torch.long)
    t.contrast_counts=torch.zeros((3,2),dtype=torch.long);t.role_counts=torch.zeros((4,2),dtype=torch.long)
    t.speed_state=sc.initialize(CFG,1,'cpu')
    done=torch.zeros(1,dtype=torch.bool)
    t._outcomes(done,done)
    assert t.speed_state['completed'][0,0]==1 and t.speed_state['successes'].sum()==0
    t._outcomes(done,done)
    assert t.speed_state['completed'][0,0]==1


def test_cli_ladders_mutually_exclusive_before_runtime_import():
    from train_cat_mjlab import parser
    base=['run','--checkpoint-native','best.pt','--fresh-optimizer','--bank-manifest','bank.json',
          '--body-collision-bank','collision.json','--body-collision-resets','resets.json','--run-dir','unused']
    assert not parser().parse_args(base).hand_speed_curriculum
    assert parser().parse_args(base+['--hand-speed-curriculum']).hand_speed_curriculum
    with pytest.raises(SystemExit):parser().parse_args(base+['--hand-speed-curriculum','--hand-tolerance-curriculum'])


def test_progress_during_noncompliant_burst_cannot_pass():
    s=sc.initialize(CFG,1,'cpu');s['episode_rung'].fill_(1);s['stage'].fill_(1)
    c=context(1.);progress=torch.zeros(1)
    for tick in range(100):
        before=progress.clone()
        # Stand compliantly for 90%; rush forward with dropped hands for 10%.
        good=tick<90
        progress+=0. if good else .04
        sc.observe(s,CFG,context=c,previous_progress=before,progress=progress,
            strict_good=torch.tensor([good]),heading_good=torch.tensor([True]),
            failure=torch.tensor([False]),resolved=torch.tensor([False]),
            leader=torch.tensor([True]),seeded=torch.tensor([False]),dt=.02)
    assert s['good']==90 and s['completed'][1,0]==1
    assert s['successes'].sum()==0 and s['compliant_progress']==0
