"""CPU regression checks for commandability, objective isolation and selection."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from cat_mjlab.runner import SuccessWindow, selection_weights
from cat_mjlab.balance import posture_terms
from cat_mjlab.raised_reset import commandable_limits, raised_pose
from cat_mjlab import constants, task_math


def test_commandable_bounds_and_controller_hold():
    import mujoco
    from cat_mjlab.model import assemble_training_xml
    from cat_ppo.furniture.generalist_fields import scene_directory
    model=mujoco.MjModel.from_xml_string(assemble_training_xml())
    path=Path('data/furniture/cat_flat_hand_balance_v5_20260921/manifest.json').resolve()
    if not path.exists():pytest.skip('v5 bank unavailable')
    manifest=json.loads(path.read_text())
    record=next(r for r in manifest['scenes'] if r.get('source',{}).get('hand_contrast',{}).get('role')=='forward_protected')
    scene=json.loads((scene_directory(manifest,path,record)/'scene.json').read_text())
    q=raised_pose(model,scene['hand_contrast']['zones'][0]);lo,hi=commandable_limits(model)
    assert np.all(q[7:]>=lo-1e-6) and np.all(q[7:]<=hi+1e-6)
    action=torch.zeros(1,29);nominal=torch.tensor(constants.DEFAULT_QPOS[7:]);target=torch.tensor(q[7:])[None]
    action[:,12:]=(target[:,12:]-nominal[12:])/.8
    assert action.abs().max()<=1
    for _ in range(20):
        updated=task_math.motor_targets(action,target,nominal,torch.tensor(lo,dtype=torch.float32),torch.tensor(hi,dtype=torch.float32))
        torch.testing.assert_close(updated,target)
    small_lo,small_hi=commandable_limits(model,.3)
    assert np.max(small_hi[12:]-nominal[12:].numpy())<=.300001
    assert np.max(nominal[12:].numpy()-small_lo[12:])<=.300001


def test_flat_bonus_zero_for_protected_even_at_flat_target():
    lower=torch.tensor([[.4,.1,.85],[.4,-.1,.85]])
    upper=lower+.04;hands=((lower+upper)/2)[None].repeat(2,1,1)
    bonus,_=posture_terms(torch.tensor([True,False]),hands,torch.zeros(2,2),
        torch.tensor([[1.,0.]]).repeat(2,1),torch.tensor([[.6,0.,0.]]).repeat(2,1),lower,upper,bonus_scale=10.)
    torch.testing.assert_close(bonus,torch.tensor([10.,0.]))


def window(protected=None):
    w=SuccessWindow();dense={
        'selection/flat_walking_compliant':torch.tensor([4.,10.]),
        'selection/flat_soft_progress':torch.tensor([8.,10.])}
    if protected is not None:dense['selection/protected_core_compliant']=torch.tensor([protected*10,10.])
    w.append(torch.tensor([[10,6],[0,0],[0,0],[0,0]]),torch.zeros(3,2),dense=dense)
    return w


def test_priority_missing_evidence_legacy_and_configuration():
    assert window().flat_balance_score() is None
    assert window().legacy_flat_balance_score()==pytest.approx(.55)
    assert window(.5).flat_balance_score()==pytest.approx(.52)
    assert window(1).flat_balance_score()-window(0).flat_balance_score()==pytest.approx(.6)
    assert window(.5).flat_balance_score((1,0,0,0))==pytest.approx(.5)
    for weights in [(1,1,1,1),(-.1,.5,.5,.1),(float('nan'),0,0,1),(1,0)]:
        with pytest.raises(ValueError):selection_weights(weights)


def test_audit_ledger_and_conditional_denominator():
    path=Path('outputs/protected_objective_audit_cpu/audit.json')
    if not path.exists():pytest.skip('Run CPU audit first')
    report=json.loads(path.read_text())
    assert len(report['scenes'])==12
    assert report['protected_core_steps']==480
    assert report['net_pre_delta']==pytest.approx(sum(v for k,v in report['mean_raised_minus_nominal_ledger'].items() if k!='body_collision_event'))
    for scene in report['scenes']:
        assert scene['commandable_action_max']<=1
        assert scene['rows']['commandable']['ledger']['flat_balance_posture_bonus']==0
    assert report['config']['reward_config']['scales']['wholebody_hand_contrast_region']==-20
    assert report['config']['reward_config']['scales']['wholebody_hand_contrast_heading']==-5


def test_prepared_launch_parses_without_running():
    import shlex
    from train_cat_mjlab import parser
    script=Path('configs/pilots/hand_priority_fixed_50.sh').read_text()
    argv=shlex.split(script.split('train_cat_mjlab.py ',1)[1].replace('\\\n',''))
    args=parser().parse_args(argv)
    assert args.checkpoint_selection_weights==[.6,.2,.1,.1]
    assert args.hand_contrast_region_weight==-20
    assert args.hand_contrast_heading_weight==-5
    assert args.flat_bonus_scale==1
    assert args.max_updates==50 and args.hand_raised_reset_fraction==.5
