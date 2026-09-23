"""CPU checks for zone isolation, monotonicity, collision signs and launch wiring."""
import json
from pathlib import Path
import shlex
import pytest
import torch
from cat_mjlab.reward_floor import hand_reward_floor


def context(n, role=1, phase=1.):
    return dict(role=torch.full((n,),role), phase_weight=torch.full((n,),phase),
        enabled=torch.ones(n,dtype=torch.bool),hand_active=torch.ones(n,2,dtype=torch.bool),
        region_valid=torch.ones(n,2,dtype=torch.bool))


@pytest.mark.parametrize('role', [-1,0,1,2,3])
def test_signs_and_ordering_including_collision(role):
    x=torch.linspace(-3,3,12001,requires_grad=True)
    old=x.clamp(0,10000)
    new,clipped,_,slope=hand_reward_floor(x,context(len(x),role),.2)
    assert (new>=0).all()
    assert not ((new-1)*(old-1)<0).any()  # Production collision event is -1.
    assert (torch.diff(new)>=0).all()
    if role in (1,3):
        assert (torch.diff(new)>0).all()
        assert not clipped.any()
        new.sum().backward()
        torch.testing.assert_close(x.grad,slope)
        assert (slope>0).all()
    else:
        assert torch.equal(new,old)


def test_default_disabled_and_phase_boundaries():
    x=torch.tensor([-2.,-.1,0.,.1,.2,1.,10001.])
    assert torch.equal(hand_reward_floor(x,{},0.)[0],x.clamp(0,10000))
    c=context(len(x));full=hand_reward_floor(x,c,.2)[0]
    c['phase_weight'].zero_();assert torch.equal(hand_reward_floor(x,c,.2)[0],x.clamp(0,10000))
    c['phase_weight'].fill_(.5)
    torch.testing.assert_close(hand_reward_floor(x,c,.2)[0],(full+x.clamp(0,10000))/2)
    c['reward_enabled']=torch.zeros(len(x),dtype=torch.bool)
    assert torch.equal(hand_reward_floor(x,c,.2)[0],x.clamp(0,10000))


def test_better_hand_pose_is_rewarded_even_when_other_costs_dominate():
    from cat_mjlab.navigation import contrast_reward_terms
    c=context(2);c.update(route_tangent=torch.tensor([[1.,0.]]).repeat(2,1),
        forward_weight=torch.ones(2), hand_regions_min=torch.zeros(2,2,2,3),
        hand_regions_max=torch.full((2,2,2,3),.1))
    hands=torch.stack((torch.full((2,3),.6),torch.full((2,3),.2))).requires_grad_()
    forward=torch.tensor([[1.,0.,0.]]).repeat(2,1)
    terms,_=contrast_reward_terms(c,hands,torch.zeros(2,2),forward,forward)
    pre=-1.5-.06*terms['wholebody_hand_contrast_region']
    assert torch.equal(pre.clamp_min(0),torch.zeros(2))
    reward,_,_,_=hand_reward_floor(pre,c,.2)
    assert reward[1]>reward[0]>0
    reward.sum().backward()
    assert (hands.grad<0).all()


def test_v6_retention_regions_inactive():
    """Changing region -20 to -3 cannot affect any retention row's reward."""
    from cat_ppo.furniture.generalist_fields import scene_directory
    path=Path('data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json').resolve()
    manifest=json.loads(path.read_text()); counts={}
    for record in manifest['scenes']:
        directory=scene_directory(manifest,path,record)
        scene_path=directory/'scene.json'
        scene=json.loads(scene_path.read_text()) if scene_path.exists() else {}
        meta=scene.get('hand_contrast',{})
        role=meta.get('role','retention')
        if role in ('forward_protected','transition'):continue
        counts[role]=counts.get(role,0)+1
        assert not any(any(z['hand_active']) for z in meta.get('zones',[]))
    assert counts.get('narrow')==12
    assert sum(counts.values())==2351


def test_config_and_launch():
    from cat_mjlab.config import wholebody_config
    from train_cat_mjlab import parser
    cfg=wholebody_config(hand_contrast=True,hand_reward_soft_floor=.2,
        hand_contrast_region_weight=-3,hand_contrast_heading_weight=-5)
    assert cfg['hand_reward_soft_floor']==.2
    assert (cfg['num_obs'],cfg['num_pri'])==(222,310)
    for bad in (-1,1,float('nan'),float('inf'),True):
        with pytest.raises(ValueError):wholebody_config(hand_contrast=True,hand_reward_soft_floor=bad)
    script=Path('configs/pilots/hand_reward_floor_v6_50.sh').read_text()
    args=parser().parse_args(shlex.split(script.split('train_cat_mjlab.py ',1)[1].replace('\\\n','')))
    assert args.from_scratch and args.max_updates==50 and args.max_action_std==0
    assert args.hand_reward_soft_floor==.2 and args.hand_contrast_region_weight==-3
    assert args.hand_contrast_heading_weight==-5 and args.hand_raised_reset_fraction==.5
    assert args.checkpoint_selection_weights==[.6,.2,.1,.1]
    assert args.wandb_mode=='online' and args.wandb_project=='CAT-wholebody' and args.wandb_entity=='skvayzer'
    assert 'v6' in str(args.bank_manifest)


def test_compiled_floor_and_measured_production_rewards():
    compiled=torch.compile(hand_reward_floor,backend='eager',fullgraph=True)
    x=torch.tensor([-2.,-.1,0.,.1,.2,1.])
    for actual,expected in zip(compiled(x,context(len(x)),.2),hand_reward_floor(x,context(len(x)),.2)):
        torch.testing.assert_close(actual,expected)
    report=json.loads(Path('outputs/protected_reward_floor_cpu/audit.json').read_text())
    fixed=[r for r in report['summary'] if r['width']==.2]
    assert [r['n'] for r in fixed]==[252,480]
    assert all(r['clipped']==0 and r['slope_min']>0 for r in fixed)
    assert report['measured_final_reward_sign_inversions']==0
    assert report['static_delta_at_region_minus3']['0.2']['post']>0
    baseline=next(r for r in report['summary'] if r['weight']==-20 and r['through_first_termination'])
    assert baseline['clipped']==165
