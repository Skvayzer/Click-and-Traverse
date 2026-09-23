"""CPU regression tests for earlier hand shaping with unchanged scoring."""
import numpy as np
import pytest
import torch
from cat_mjlab.navigation import hand_contrast_context, contrast_reward_terms
from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
from cat_mjlab.config import wholebody_config


def context(progress, role=1, distance=0.):
    m={k:torch.from_numpy(v) for k,v in pack_hand_contrast([None]).items()}
    m['enabled'][:]=True;m['role'][:]=role;m['zone_valid'][:,0]=True
    m['start_m'][:,0]=.35;m['end_m'][:,0]=2.;m['fade_m'][:,0]=.15
    m['hand_active'][:,0]=True;m['region_valid'][:,0,0]=True;m['forward_weight'][:,0]=1.
    m['hand_regions_min'][:,0,0]=.5;m['hand_regions_max'][:,0,0]=.6
    return hand_contrast_context(m,torch.tensor([progress]),torch.tensor([[1.,0.]]),approach_distance=distance)


def reward(c):
    return contrast_reward_terms(c,torch.zeros(1,2,3),torch.zeros(1,2),torch.tensor([[1.,0.,0.]]),torch.tensor([[1.,0.,0.]]))


def test_early_reward_preserves_heading_and_all_compliance_masks():
    for p in [0.,.1,.34,.35,.4,.5,1.,1.9,2.,2.1]:
        old=context(p);new=context(p,distance=.4)
        for k in old:torch.testing.assert_close(new[k],old[k],rtol=0,atol=0)
        a,ma=reward(old);b,mb=reward(new)
        torch.testing.assert_close(a['wholebody_hand_contrast_heading'],b['wholebody_hand_contrast_heading'])
        for key in ['hand_contrast_hand_good','hand_contrast_heading_good','hand_contrast_hand_active']:
            torch.testing.assert_close(ma[key],mb[key])
        if .0<p<.5:assert b['wholebody_hand_contrast_region']>a['wholebody_hand_contrast_region']
        if p>=.5:torch.testing.assert_close(a['wholebody_hand_contrast_region'],b['wholebody_hand_contrast_region'])
    assert context(.1,distance=.4)['reward_phase_weight'].item()==pytest.approx(.2)


@pytest.mark.parametrize('role',[-1,0,2,3])
def test_other_roles_unchanged(role):
    for p in [.1,.35,.4,.5,1.9,2.1]:
        a,ma=reward(context(p,role));b,mb=reward(context(p,role,.4))
        for key in a:torch.testing.assert_close(a[key],b[key])
        for key in ma:torch.testing.assert_close(ma[key],mb[key])


def test_override_validation_and_inheritance():
    c=wholebody_config(hand_contrast=True,hand_contrast_approach_distance=.4)
    assert wholebody_config(c,hand_contrast=True)['hand_contrast_approach_distance']==.4
    assert wholebody_config(c,hand_contrast=True,hand_contrast_approach_distance=0)['hand_contrast_approach_distance']==0
    for bad in [-1,float('nan'),float('inf'),True]:
        with pytest.raises(ValueError,match='approach'):wholebody_config(hand_contrast=True,hand_contrast_approach_distance=bad)


def test_compiled_reward_graph_cpu():
    c=context(.1,distance=.4)
    args=(c,torch.zeros(1,2,3),torch.zeros(1,2),torch.tensor([[1.,0.,0.]]),torch.tensor([[1.,0.,0.]]))
    compiled=torch.compile(contrast_reward_terms,backend='eager',fullgraph=True)
    a,ma=contrast_reward_terms(*args);b,mb=compiled(*args)
    for key in a:torch.testing.assert_close(a[key],b[key])
    for key in ma:torch.testing.assert_close(ma[key],mb[key])


def test_approach_does_not_replace_previous_zone_target():
    m={k:torch.from_numpy(v) for k,v in pack_hand_contrast([None]).items()}
    m['enabled'][:]=True;m['role'][:]=1;m['zone_valid'][:,:2]=True
    m['start_m'][0,:2]=torch.tensor([.35,2.2]);m['end_m'][0,:2]=torch.tensor([2.,3.])
    m['fade_m'][0,:2]=.15;m['hand_active'][:,:2]=True;m['region_valid'][:,:2,0]=True
    m['hand_regions_min'][:,1]=4.
    for p,expected in [(1.9,0.),(2.1,4.)]:
        c=hand_contrast_context(m,torch.tensor([p]),torch.tensor([[1.,0.]]),approach_distance=1.)
        assert c['reward_hand_regions_min'][0,0,0,0].item()==expected
