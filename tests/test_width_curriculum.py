"""CPU-only width progression, retention isolation and sampler contracts."""
from types import SimpleNamespace
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from cat_mjlab.scene_bank import SceneBank
from cat_mjlab.task import CATTask
from cat_mjlab.runner import SuccessWindow
from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
from cat_ppo.furniture.contrastive_bank import role_balanced_logits


def bank_fixture():
    b=object.__new__(SceneBank);b.device=torch.device('cpu')
    b.roles=torch.tensor([-1,-1,0,1,2,2,3,3]);b.count=8
    b.weights=torch.ones(8);b.sampling_ids=torch.tensor([0,1,4,5,6,6,7,7])
    b.sampling_masses=torch.tensor([.1,.4,0.,0.,.025,.1,.3,.075])
    b.width_levels=torch.tensor([-1,-1,-1,-1,0,1,0,1])
    b.width_curriculum=dict(rung_widths_m=[.7,.4],min_completed=64,success_threshold=.6)
    b.navigation_groups=torch.tensor([0,0,2,2,2,2,2,2]);b.levels=None
    return b


def test_fixed_masses_and_locked_widths_under_extreme_adaptation():
    b=bank_fixture()
    for stage in [0,1]:
        for weights in [torch.ones(8),torch.tensor([.001,10,100,.001,.001,100,1,100.])]:
            p=b.probabilities(weights,stage=torch.tensor(stage))
            torch.testing.assert_close(p.sum(),torch.tensor(1.))
            sums=torch.zeros(8).scatter_add_(0,b.sampling_ids,p)
            torch.testing.assert_close(sums,b.sampling_masses)
            if stage==0:assert p[5]==p[7]==0
            else:assert p[5]>0 and p[7]>0 and p[4]>0 and p[6]>0


def task_fixture(follower=False):
    t=object.__new__(CATTask);t.bank=bank_fixture();t.config={};t.device=torch.device("cpu")
    t.scene_ids=torch.tensor([0,2,4]);t.policy_ids=torch.tensor([0,0,1 if follower else 0]);n=3
    t.hand_contrast=True;t.outcome_counted=torch.zeros(n,dtype=torch.bool)
    t.episode={key:torch.zeros(n,dtype=torch.bool) for key in ('fall','obstacle','self_contact','numerical','body_collision','hand_violation','elbow_violation','outside_bounds')}
    t.episode['goal_reached']=torch.tensor([False,True,True])
    t.navigation_counts=torch.zeros(4,2,dtype=torch.long);t.contrast_counts=torch.zeros(3,2,dtype=torch.long);t.role_counts=torch.zeros(4,2,dtype=torch.long)
    t.zone_steps=torch.zeros(n,6,3,dtype=torch.long)
    t.contrast=dict(enabled=torch.tensor([False,True,True]),role=torch.tensor([-1,0,2]),
        zone_index=torch.zeros(n,dtype=torch.long),core_active=torch.tensor([False,True,True]),
        required_forward_zones=torch.zeros(n,6,dtype=torch.bool),required_hand_zones=torch.zeros(n,6,dtype=torch.bool))
    t.telemetry=dict(hand_contrast_heading_good=torch.ones(n),hand_contrast_hand_good=torch.ones(n))
    t.scene_episode_ema=torch.zeros(8);t.scene_success_ema=torch.zeros(8)
    t.curriculum_stage=torch.tensor(0);t.curriculum_completed=torch.tensor([63,0]);t.curriculum_goals=torch.tensor([38,0])
    return t


def test_only_leader_clean_narrow_outcomes_unlock_and_retention_is_not_open():
    for follower in [False,True]:
        t=task_fixture(follower)
        done=torch.tensor([True,False,False]);trunc=torch.tensor([True,False,False])
        t._outcomes(done,trunc)
        assert t.role_counts[0].tolist()==[1,1]  # Only true contrastive open, not retention.
        assert t.navigation_counts[0].tolist()==[1,0]
        assert t.scene_success_ema[0]==1  # Retention keeps native survival adaptation.
        assert int(t.curriculum_stage)==(0 if follower else 1)
        assert t.curriculum_completed[0]==(63 if follower else 64)
        assert (t.probabilities[5]>0).item()==(not follower)
        if not follower:
            before=t.curriculum_completed.clone();t._outcomes(done,trunc)
            assert torch.equal(before,t.curriculum_completed)  # First-outcome latch.


def test_retention_sentinel_and_metrics_are_not_suppressed():
    p=pack_hand_contrast([None,{}]);assert p['role'].tolist()==[-1,-1]
    window=SuccessWindow();window.append(torch.tensor([[10,4],[20,8],[5,1],[35,13]]),torch.zeros(3,2,dtype=torch.long))
    m=window.metrics(contrastive=True)
    assert m['success/cat_goal_success_rate']==.4
    assert m['success/ordinary_clutter_goal_success_rate']==.4
    assert m['success/hand_protection_goal_success_rate']==.2


def test_configured_jax_role_masses():
    import jax.numpy as jp
    import jax
    roles=jp.array([0,1,1,2,3]);masses=jp.array([.05,.15,.65,.15])
    p=jax.nn.softmax(role_balanced_logits(jp.array([10.,.001,1.,.2,5.]),roles,masses))
    np.testing.assert_allclose(np.bincount(roles,weights=p),masses,atol=1e-7)


def test_generator_width_override_is_explicit_scaffolding_and_default_stays_strict():
    from cat_ppo.furniture.contrastive_passages import generate_contrastive_group
    easy=generate_contrastive_group(8101,certify=False,narrow_width_range=(.7,.7),curriculum_rung=0)
    old=generate_contrastive_group(8101,certify=False)
    assert 'certificate_semantics' not in old[2]['hand_contrast']
    assert .4<=old[2]['hand_contrast']['modules'][0]['width_m']<=.41
    for scene in easy:
        c=scene['hand_contrast']
        assert c['certificate_semantics']=='width-curriculum-scaffold-v1'
        assert c['curriculum_rung']==0
        assert scene['difficulty']=='width_curriculum'
        for module in c['modules']:
            if module['role']=='narrow':assert module['width_m']==.7
    assert easy[0]['scene_id']!=old[0]['scene_id']
    with pytest.raises(ValueError,match='width range'):
        generate_contrastive_group(8101,certify=False,narrow_width_range=(.3,.7),curriculum_rung=0)


def test_width_sampler_state_round_trip_and_locked_rung_rejection():
    t=task_fixture();t._outcomes(torch.tensor([True,False,False]),torch.tensor([True,False,False]))
    state=t.sampling_state()
    target=task_fixture();target.restore_sampling_state(state,resample=False)
    assert target.curriculum_stage==1
    torch.testing.assert_close(target.curriculum_completed,t.curriculum_completed)
    torch.testing.assert_close(target.probabilities,t.probabilities)
    state['pf_width_curriculum_stage']=torch.tensor(0)
    with pytest.raises(ValueError,match='locked'):
        target.restore_sampling_state(state,resample=False)


def test_contrastive_goal_counts_outside_passage_zone():
    t=task_fixture()
    t.contrast['enabled'].zero_();t.contrast['core_active'].zero_()
    t._outcomes(torch.tensor([True,False,False]),torch.tensor([True,False,False]))
    assert t.role_counts[0].tolist()==[1,1]
    assert t.role_counts[2].tolist()==[1,1]
    assert t.navigation_counts[0].tolist()==[1,0]
