"""CPU protection objective and adversarial S-gate checks."""
import math
import torch
import pytest
from cat_mjlab.lateral_corridor import corridor_cost, lateral, combined_pose, arm_clearance, MARGIN
from test_mjlab_acceptance import gate, step, status


def test_sign_gradient_and_root_drift():
    y=torch.tensor([[.2726,-.2726]],dtype=torch.float64,requires_grad=True)
    cost=corridor_cost(y,torch.tensor([.1865]))
    cost.backward()
    assert cost.item()==pytest.approx(.741321,abs=1e-6)
    assert y.grad.tolist()[0]==pytest.approx([8.61,-8.61],abs=1e-5)
    assert (-3*.02*y.grad).abs()[0,0].item()==pytest.approx(.5166,abs=1e-6)
    inside=torch.tensor([[.1,-.1]],requires_grad=True)
    corridor_cost(inside,torch.tensor([.18])).sum().backward()
    assert not inside.grad.any()
    world=torch.tensor([[[0.,.2,0.],[0.,-.1,0.]]])
    yy=lateral(world,torch.zeros(1,2),torch.tensor([[1.,0.]]))
    assert corridor_cost(yy,torch.tensor([.18])).item()>0


def test_whole_arm_radius_and_margin_enforced():
    ep=torch.tensor([[[[0.,.2,1.],[.1,.25,1.]]]])
    c=arm_clearance(ep,torch.tensor([.05]),torch.zeros(1,2),torch.tensor([[1.,0.]]),torch.tensor([-.37]),torch.tensor([.37]))
    assert c.item()==pytest.approx(.07)
    assert not combined_pose(torch.zeros(1,2),torch.tensor([.18]),torch.tensor([True]),c).item()
    assert combined_pose(torch.zeros(1,2),torch.tensor([.18]),torch.tensor([True]),torch.tensor([MARGIN])).item()


def protected_gate():
    g=gate();g.geometry['protection'][:]=True
    return g


def observe(g,x,t,good=True,**flags):
    g.advance(torch.tensor([[x,flags.pop('y',0.)]]),t,protection_good=torch.tensor([good]),
        **{k:torch.tensor([flags.get(k,False)]) for k in ('done','clean_goal','fall','hand_collision','body_collision')})


@pytest.mark.parametrize('strategy',['refuse','crawl','bypass','fall','sideways','elbow','missing','safe'])
def test_combined_gate(strategy):
    g=protected_gate()
    if strategy=='refuse': observe(g,0.,20.,done=True)
    elif strategy=='crawl':
        observe(g,1.,1.);observe(g,1.5,7.);observe(g,3.,9.,clean_goal=True)
    elif strategy=='bypass': observe(g,3.,3.,y=2.,clean_goal=True)
    elif strategy=='fall': observe(g,.1,1.,fall=True,done=True)
    elif strategy=='missing': step(g,3.,3.,clean_goal=True)
    else:
        heading=strategy!='sideways'
        clearance=.06641 if strategy=='elbow' else .284228
        good=combined_pose(torch.tensor([[-.005374,-.005374]]),torch.tensor([.184793]),torch.tensor([heading]),torch.tensor([clearance])).item()
        observe(g,3.,3.,good,clean_goal=True)
    assert status(g)==('PASS' if strategy=='safe' else 'FAIL')


def test_cli_default_off_and_reward_configuration():
    from cat_mjlab.upper_control import configure
    from cat_mjlab.config import wholebody_config
    from types import SimpleNamespace
    c=wholebody_config(hand_contrast=True);before=c['reward_config']['scales'].copy()
    configure(c,SimpleNamespace())
    assert not c['lateral_corridor'] and c['reward_config']['scales']==before
    configure(c,SimpleNamespace(lateral_corridor=True))
    assert c['reward_config']['scales']['wholebody_lateral_corridor']==-3


def test_production_reward_integration_and_other_role_isolation():
    import sys, json
    import numpy as np
    from pathlib import Path
    from types import SimpleNamespace
    from cat_mjlab.config import wholebody_config
    from cat_mjlab.upper_control import configure
    from cat_mjlab.acceptance import pack_geometry
    from cat_mjlab.lateral_corridor import ArmGeometry
    from cat_mjlab.collision import PROPOSAL
    from cat_ppo.furniture.room_navigation import pack_room_scenes
    from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
    sys.path.insert(0,str(Path('scripts').resolve()))
    from audit_hand_posture_rewards import Probe
    audit=json.loads(Path('outputs/lateral_corridor_cpu/audit.json').read_text())
    scene=json.loads(Path(audit['geometry'][0]['scene_file']).read_text())
    bank=SimpleNamespace(scene=scene,
        rooms={k:torch.as_tensor(v) for k,v in pack_room_scenes([scene]).items()},
        contrast={k:torch.as_tensor(v) for k,v in pack_hand_contrast([scene]).items()},
        acceptance={k:torch.as_tensor(v) for k,v in pack_geometry([scene]).items()},
        crossed_is_plane=torch.tensor([False]),goals=torch.tensor([[*scene['route'][-1],.7]]))
    def sample(name,pos,ids):
        if name=='sdf': return torch.ones((*pos.shape[:-1],1))
        value=torch.zeros_like(pos)
        if name=='gf':value[...,0]=.6
        return value
    bank.sample=sample
    config=wholebody_config(hand_contrast=True)
    probe=Probe(config,json.loads(PROPOSAL.read_text()))
    q=probe.qpose();t=np.diff(scene['route'],axis=0)[0];t/=np.linalg.norm(t)
    xy=np.asarray(scene['route'][0])+t*.7
    task=probe.task(bank,xy[None],np.array([np.arctan2(t[1],t[0])]),q[None])
    task.corridor_arms=ArmGeometry(task.model,'cpu')
    # The same production reward function used in live tasks, no dynamics/policy.
    configure(config,SimpleNamespace())
    before=probe.rewards(task)['components']
    configure(config,SimpleNamespace(lateral_corridor=True))
    after=probe.rewards(task)['components']
    assert after['wholebody_hand_contrast_region'].item()==0
    assert after['wholebody_lateral_corridor'].item()<0
    assert after['wholebody_hand_contrast_heading'].item()==before['wholebody_hand_contrast_heading'].item()
    assert task.telemetry['hand_contrast_region_cost'].item()>0 # box diagnostic survives
    # Masked-out flat/clutter/CAT/narrow rows preserve every existing component.
    bank.acceptance['protection'][:]=False
    for role in (-1,0,2,3):
        task.contrast['role'][:]=role
        configure(config,SimpleNamespace());a=probe.rewards(task)['components']
        configure(config,SimpleNamespace(lateral_corridor=True));b=probe.rewards(task)['components']
        for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        assert b['wholebody_lateral_corridor'].item()==0
        assert b['wholebody_corridor_arm'].item()==0
    assert not torch.cuda.is_initialized()


def test_pilot_parses_without_launch():
    import shlex
    from pathlib import Path
    from train_cat_mjlab import parser
    text=Path('configs/pilots/lateral_corridor_50.sh').read_text()
    args=parser().parse_args(shlex.split(text.split('train_cat_mjlab.py ',1)[1].replace('\\\n',' ')))
    assert args.lateral_corridor and args.upper_gravity_compensation and args.protected_hand_sdf_margin
    assert args.max_updates==50 and args.wandb_mode=='online'
    assert args.wandb_project=='CAT-wholebody' and args.wandb_entity=='skvayzer'
    assert str(args.checkpoint_native)=='outputs/cat_recover_23_30720_20260922/best.pt'


def test_actual_scene_bounds_and_fk_sideways_witnesses():
    import json
    from pathlib import Path
    from cat_mjlab.acceptance import pack_geometry
    report=json.loads(Path('outputs/lateral_corridor_combined_cpu/audit.json').read_text())
    for row in report['geometry']:
        room=json.loads(Path(row['scene_file']).read_text())
        actual=pack_geometry([room])
        assert actual['bound'][0]==pytest.approx(row['symmetric_bound_m'],abs=1e-10)
    sideways=[r for r in report['static_nominal_heading_witnesses'] if r['heading_deg']==90]
    assert len(sideways)==12
    assert all(r['hand_compliant'] and not r['combined_pose_pass'] for r in sideways)
    assert min(r['min_arm_lateral_clearance_m'] for r in sideways)>.284


def test_geometry_rotates_and_forward_tracking_is_required():
    # A translated/rotated route gives the same signed lateral coordinate.
    world=torch.tensor([[[4.8,8.,1.],[5.1,8.,1.]]])
    yy=lateral(world,torch.tensor([[5.,7.]]),torch.tensor([[0.,1.]]))
    torch.testing.assert_close(yy,torch.tensor([[.2,-.1]]))
    from cat_mjlab.config import wholebody_config
    from cat_mjlab.upper_control import configure
    from types import SimpleNamespace
    c=wholebody_config(hand_contrast=True,hand_contrast_heading_weight=0)
    configure(c,SimpleNamespace(lateral_corridor=True))
    assert c['reward_config']['scales']['wholebody_hand_contrast_heading']==0
    c['reward_config']['scales']['tracking_root_field']=0
    with pytest.raises(ValueError,match='forward tracking'):configure(c,SimpleNamespace(lateral_corridor=True))
