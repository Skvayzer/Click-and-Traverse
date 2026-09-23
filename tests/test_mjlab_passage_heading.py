"""CPU-only regression checks for optional narrow-module heading shaping."""
import json
import math

import numpy as np
import pytest
import torch

from cat_mjlab.navigation import hand_contrast_context, contrast_reward_terms
from cat_mjlab.passage_rewards import passage_parameters
from test_mjlab_pilot import scene_fixture


def metadata():
    return dict(enabled=torch.tensor([True]),role=torch.tensor([2]),
        zone_valid=torch.tensor([[True]]),start_m=torch.tensor([[1.]]),end_m=torch.tensor([[3.]]),
        fade_m=torch.tensor([[.5]]),forward_weight=torch.tensor([[0.]]),
        hand_active=torch.zeros(1,1,2,dtype=torch.bool),region_valid=torch.zeros(1,1,2,dtype=torch.bool),
        hand_regions_min=torch.zeros(1,1,2,2,3),hand_regions_max=torch.zeros(1,1,2,2,3))


def context(progress=2., route=0., override=True):
    m=metadata()
    if override:
        m.update(heading_override=torch.tensor([[True]]),heading_target_rad=torch.tensor([[math.pi/2]]),
                 heading_weight=torch.tensor([[1.]]),heading_axis=torch.tensor([[True]]))
    return hand_contrast_context(m,torch.tensor([progress]),torch.tensor([[math.cos(route),math.sin(route)]]))


def evaluate(c, angle, torso=None):
    def forward(a): return torch.tensor([[math.cos(a),math.sin(a),0.]])
    return contrast_reward_terms(c,torch.zeros(1,2,3),torch.zeros(1,2),forward(angle),forward(angle if torso is None else torso))


def test_axis_symmetry_route_covariance_and_both_body_headings():
    for route in [0., .7, -2.]:
        c=context(route=route)
        for offset in [math.pi/2,-math.pi/2]:
            rewards,telemetry=evaluate(c,route+offset)
            assert rewards['wholebody_hand_contrast_heading'].item()==pytest.approx(0.,abs=2e-7)
            assert telemetry['hand_contrast_heading_good'].item()==1
        for offset in [0.,math.pi]:
            rewards,telemetry=evaluate(c,route+offset)
            assert rewards['wholebody_hand_contrast_heading'].item()==pytest.approx(1.,abs=2e-7)
            assert telemetry['hand_contrast_heading_good'].item()==0
        rewards,telemetry=evaluate(c,route+math.pi/2,route)
        assert rewards['wholebody_hand_contrast_heading'].item()==pytest.approx(.5,abs=2e-7)
        assert telemetry['hand_contrast_heading_good'].item()==0
    c=context();c['heading_target_rad'].zero_();c['heading_axis'].fill_(False)
    assert evaluate(c,math.pi)[0]['wholebody_hand_contrast_heading'].item()==2


def test_smooth_boundary_and_exact_legacy_behavior():
    for progress,expected in [(0.,0.),(1.,0.),(1.125,.15625),(1.25,.5),(1.5,1.),(2.,1.),(2.75,.5),(3.,0.),(4.,0.)]:
        c=context(progress)
        assert evaluate(c,0.)[0]['wholebody_hand_contrast_heading'].item()==pytest.approx(expected,abs=1e-7)
        assert not c['required_forward_zones'].any()  # Reward override does not redefine success.
    for progress in [0.,1.,3.,4.]:
        old=evaluate(context(progress,override=False),.4)
        new=evaluate(context(progress),.4)
        for before,after in zip(old,new):
            for key in before: assert torch.equal(before[key],after[key])
    m=metadata();m['forward_weight'].fill_(1.)
    old=hand_contrast_context(m,torch.tensor([1.25]),torch.tensor([[1.,0.]]))
    m.update(heading_override=torch.tensor([[False]]),heading_target_rad=torch.tensor([[math.pi/2]]),
             heading_weight=torch.tensor([[1.]]),heading_axis=torch.tensor([[True]]))
    new=hand_contrast_context(m,torch.tensor([1.25]),torch.tensor([[1.,0.]]))
    for before,after in zip(evaluate(old,.4),evaluate(new,.4)):
        for key in before: assert torch.equal(before[key],after[key])


def test_heading_metadata_scope_and_validation(tmp_path):
    bank=tmp_path/'bank.json';bank.write_text('{}')
    scene=scene_fixture();z=scene['hand_contrast']['zones'][1]
    z.update(heading_target_rad=math.pi/2,heading_weight=1.,heading_axis=True)
    result=passage_parameters([scene,None],bank_path=bank)
    assert result['heading_override'].sum()==1
    assert result['heading_axis'][0,1]
    assert np.all(result['sdf_reward_knee']==np.float32(.05))
    for key,bad in [('heading_weight',-1),('heading_target_rad',float('nan')),('heading_axis',1)]:
        saved=z[key];z[key]=bad
        with pytest.raises(ValueError):passage_parameters([scene],bank_path=bank)
        z[key]=saved
    scene['hand_contrast']['modules'][1]['role']='open'
    with pytest.raises(ValueError,match='narrow'):passage_parameters([scene],bank_path=bank)


def test_combined_config_preserves_knee_baseline_and_targets_64_zones():
    from pathlib import Path
    directory=Path(__file__).resolve().parents[1]/'configs/pilots'
    baseline=json.loads((directory/'contrastive_table_v4_narrow_knee0.json').read_text())
    combined=json.loads((directory/'contrastive_table_v4_narrow_knee0_sideways.json').read_text())
    assert combined['schema']==baseline['schema']=='cat-passage-rewards-v1'
    assert combined['bank_sha256']==baseline['bank_sha256']
    assert combined['scenes'].keys()==baseline['scenes'].keys()
    assert len(combined['scenes'])==32
    assert sum(len(zones) for zones in combined['scenes'].values())==64
    for scene,zones in combined['scenes'].items():
        assert zones.keys()==baseline['scenes'][scene].keys()
        for index,entry in zones.items():
            assert entry['heading_axis'] is True
            assert entry['heading_weight']==1.
            assert entry['heading_target_rad']==math.pi/2
            assert {k:v for k,v in entry.items() if not k.startswith('heading_')}==baseline['scenes'][scene][index]
