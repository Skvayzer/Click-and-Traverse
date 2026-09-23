"""CPU-only progression, gate isolation, reward invariance and persistence checks."""
import copy
import pytest
import torch
from cat_mjlab.tolerance_curriculum import RUNGS, configure, initialize, advance, metrics
from cat_mjlab.config import wholebody_config
from cat_mjlab.navigation import contrast_reward_terms
from test_mjlab_hand_approach import context


def feed(s, cfg, values, *, role=1, rung=None, eligible=True):
    n=len(values)
    advance(s,cfg,roles=torch.full((n,),role),
        episode_rungs=torch.full((n,),int(s['stage']) if rung is None else rung),
        eligible=torch.full((n,),eligible),success=torch.tensor(values,dtype=torch.bool))


def test_window_both_roles_stale_episodes_and_final_exact():
    cfg=dict(window=4,threshold=.75);s=initialize(cfg,'cpu')
    assert RUNGS==(.20,.15,.10,.07,.05)
    feed(s,cfg,[1]*4,eligible=False)
    assert not s['completed'].any()
    feed(s,cfg,[1]*4)
    assert int(s['stage'])==0  # Transition evidence is independently required.
    feed(s,cfg,[0]*4,role=3)
    feed(s,cfg,[1]*2,role=3)
    assert int(s['stage'])==0
    feed(s,cfg,[1],role=3)
    assert int(s['stage'])==1  # Rolling 3/4, despite lifetime 3/7.
    feed(s,cfg,[1]*10,rung=0)
    assert not s['count'].any()
    for rung in range(1,5):
        feed(s,cfg,[1]*4)
        feed(s,cfg,[1]*4,role=3)
        assert int(s['stage'])==min(rung+1,4)
    assert metrics(s)['hand_tolerance/tolerance_m']==.05
    restored=copy.deepcopy(s)
    feed(restored,cfg,[0]*4)
    assert int(restored['stage'])==4


def test_default_noop_and_fresh_optin():
    baseline=wholebody_config(hand_contrast=True)
    assert configure(copy.deepcopy(baseline))==baseline
    inherited=configure(copy.deepcopy(baseline),enabled=True)
    assert configure(inherited)==baseline
    with pytest.raises(ValueError):configure(baseline,enabled=True,window=0)
    with pytest.raises(ValueError):configure({},enabled=True)


def test_loose_metric_does_not_relax_reward_or_strict_score_and_compiles():
    c=context(1.)
    hands=torch.full((1,2,3),.5);hands[:,:,0]=.39 # 11 cm box error
    args=(c,hands,torch.zeros(1,2),torch.tensor([[1.,0.,0.]]),torch.tensor([[1.,0.,0.]]))
    strict,sm=contrast_reward_terms(*args)
    loose,lm=contrast_reward_terms(*args,hand_good_distance=torch.tensor([[[.2]]]))
    assert sm['hand_contrast_hand_good'].item()==0
    assert lm['hand_contrast_hand_good'].item()==1
    for key in strict:torch.testing.assert_close(strict[key],loose[key],rtol=0,atol=0)
    compiled=torch.compile(contrast_reward_terms,backend='eager',fullgraph=True)
    _,cm=compiled(*args,hand_good_distance=torch.tensor([[[.2]]]))
    torch.testing.assert_close(cm['hand_contrast_hand_good'],lm['hand_contrast_hand_good'])


def test_v6_only_mixture_and_sampling():
    from pathlib import Path
    from cat_ppo.furniture.generalist_fields import load_generalist_manifest
    from cat_ppo.furniture.balance_bank import sampling_plan
    from cat_ppo.furniture.protected_heavy_bank import validate
    p=Path('data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json')
    m=load_generalist_manifest(p,verify_files=False)
    ids,masses=sampling_plan(m)
    assert len(ids)==2375
    assert masses.sum()==pytest.approx(1)
    assert masses[6:].sum()==pytest.approx(.45)
    assert (ids==6).sum()==12 and (ids==7).sum()==12
    bad=copy.deepcopy(m);bad['scenes'][0]['sampling_weight']=99
    with pytest.raises(ValueError,match='Only reset mixture'):validate(bad,path=p)


def test_task_outcomes_strict_reporting_loose_sampling_and_seed_gate():
    from test_mjlab_hand_curriculum_gate import hand_task
    from cat_mjlab.task import CATTask
    cfg=dict(window=1,threshold=.6)
    t=hand_task(config={'hand_tolerance_curriculum':cfg},completed=0,goals=0)
    t.hand_contrast=True;t.num_envs=1
    t.bank.levels=None;t.bank.roles=torch.tensor([1])
    t.info={'hand_raised_seeded':torch.tensor([True])}
    t.contrast=dict(zone_index=torch.tensor([0]),core_active=torch.tensor([True]),
        required_forward_zones=torch.tensor([[True,False,False,False,False,False]]),
        required_hand_zones=torch.tensor([[True,False,False,False,False,False]]),role=torch.tensor([1]))
    t.telemetry=dict(hand_contrast_heading_good=torch.ones(1),hand_contrast_hand_good=torch.zeros(1),hand_tolerance_good=torch.ones(1))
    t.zone_steps=torch.zeros((1,6,3),dtype=torch.long)
    t.contrast_counts=torch.zeros((3,2),dtype=torch.long);t.role_counts=torch.zeros((4,2),dtype=torch.long)
    t.tolerance_state=initialize(cfg,'cpu');t.tolerance_episode_rung=torch.zeros(1,dtype=torch.long)
    t.tolerance_zone_steps=torch.zeros((1,6),dtype=torch.long)
    done=torch.zeros(1,dtype=torch.bool)
    t._outcomes(done,done)
    assert t.scene_success_ema.item()==1  # Loose success drives within-role sampler.
    assert t.contrast_counts[0].tolist()==[1,0]  # Published 5 cm success stays zero.
    assert not t.tolerance_state['completed'].any()  # Seeded outcomes cannot advance.
    t.outcome_counted.zero_();t.info['hand_raised_seeded'].zero_()
    t._outcomes(done,done)
    assert t.tolerance_state['completed'][0,0]==1
    before=t.tolerance_state['completed'].clone()
    t._outcomes(done,done)
    torch.testing.assert_close(t.tolerance_state['completed'],before)
    # Real task checkpoint methods retain rolling evidence and frozen episode rung.
    for name in ('navigation','obs'):setattr(t,name,{})
    t.episode_reward=torch.zeros(1);t.probabilities=torch.ones(1)
    t.generator=torch.Generator()
    snapshot=copy.deepcopy(t.state_dict())
    t.tolerance_state['completed'].zero_()
    t.load_state_dict(snapshot)
    torch.testing.assert_close(t.tolerance_state['completed'],before)
