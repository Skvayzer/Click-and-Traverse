"""Geometry and production-perception gates; never starts a learner."""
from copy import deepcopy

import pytest
import torch

from cat_mjlab.analytic_objects import AnalyticObjects, primitive_distance_normal, merge_fields


def spec(kind=0):
    return AnalyticObjects(start=torch.tensor([[[.7,0.,1.]]]),
        end=torch.tensor([[[.2,0.,1.]]]),rotations=torch.eye(3)[None,None],
        sizes=torch.tensor([[[.06,.1,.08]]]),kinds=torch.tensor([[kind]]),
        valid=torch.tensor([[True]]),speed=torch.tensor([[.1]]),hold=torch.tensor([[1.]]))


@pytest.mark.parametrize('kind',[0,1,2])
def test_exact_distance_and_normal_against_finite_difference(kind):
    obj=spec(kind);s=obj.state
    points=torch.tensor([[[.95,.07,1.12],[.74,.02,1.],[.8,.15,1.3]]],dtype=torch.float64)
    s={k:v.double() if v.is_floating_point() else v for k,v in s.items()}
    args=(s['start'],s['rotations'],s['sizes'],s['kinds'])
    d,n=primitive_distance_normal(points,*args)
    numerical=[]
    for axis in range(3):
        delta=torch.zeros_like(points);delta[...,axis]=1e-6
        numerical.append((primitive_distance_normal(points+delta,*args)[0]-
                          primitive_distance_normal(points-delta,*args)[0])/2e-6)
    torch.testing.assert_close(n,torch.stack(numerical,-1),atol=1e-7,rtol=1e-6)
    assert torch.isfinite(d).all()


def test_rotated_box_and_subvoxel_rod():
    obj=spec(1)
    obj.state['start'].zero_();obj.state['end'].zero_()
    obj.state['rotations'][0,0]=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]])
    d,n=obj.query(torch.tensor([[[0.,.36,0.]]]),torch.tensor([0]),torch.tensor([0.]))
    torch.testing.assert_close(d,torch.tensor([[[.30]]]))
    torch.testing.assert_close(n,torch.tensor([[[0.,1.,0.]]]))
    rod=spec(2);rod.state['sizes'][0,0]=torch.tensor([.008,.22,.008])
    p=torch.tensor([[[.7,.1,1.]]])
    d,_=rod.query(p,torch.tensor([0]),torch.tensor([0.]))
    torch.testing.assert_close(d,torch.tensor([[[.092]]]))


def test_guidance_projection_static_tie_and_inactive_parity():
    gf=torch.tensor([[[1.,2.,0.]]]);bf=torch.tensor([[[0.,1.,0.]]]);sdf=torch.tensor([[[.4]]])
    normal=torch.tensor([[[-1.,0.,0.]]])
    g,b,d=merge_fields(gf,bf,sdf,torch.tensor([[[.2]]]),normal,.1)
    torch.testing.assert_close(g,torch.tensor([[[0.,2.,0.]]]))
    torch.testing.assert_close(d,torch.tensor([[[.1]]]))
    for distance,radius in ((sdf,0.),(torch.full_like(sdf,torch.inf),.1)):
        actual=merge_fields(gf,bf,sdf,distance,normal,radius)
        for value,expected in zip(actual,(gf,bf,sdf)):
            assert torch.equal(value,expected)


def test_approach_hold_retreat_never_overshoots():
    obj=spec();ids=torch.tensor([0])
    for t,x in ((0,.7),(2,.5),(5,.2),(6,.2),(8,.4),(11,.7),(20,.7)):
        torch.testing.assert_close(obj.centers(ids,torch.tensor([float(t)]))[0,0,0],torch.tensor(x),atol=1e-6,rtol=1e-5)


def test_production_stationary_hand_perception():
    from scripts.verify_analytic_objects import demonstrate
    assert demonstrate()['passed']


def test_absent_and_masked_objects_preserve_existing_task_steps():
    from test_mjlab_task import _CPUSimulation, _tiny_bank
    from cat_mjlab.task import CATTask
    from cat_mjlab.config import wholebody_config
    cfg=wholebody_config();cfg['randomize_initial_episode_steps']=False
    first=CATTask(_CPUSimulation(1),_tiny_bank(1),deepcopy(cfg),seed=12)
    engine=spec();engine.state['valid'].zero_()
    second=CATTask(_CPUSimulation(1),_tiny_bank(1),deepcopy(cfg),seed=12,analytic_objects=engine)
    for _ in range(4):
        a=first.step(torch.zeros((1,29)));b=second.step(torch.zeros((1,29)))
        for key in a['obs']:
            assert torch.equal(a['obs'][key],b['obs'][key])
        assert torch.equal(a['reward'],b['reward'])
        assert torch.equal(a['done'],b['done'])
