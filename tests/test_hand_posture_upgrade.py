"""CPU metadata, sampling, interpolation and selection contracts for option C."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from cat_ppo.furniture.generalist_fields import load_generalist_manifest,sampling_groups
from cat_ppo.furniture.hand_posture_upgrade import validate_hand_posture_upgrade
from cat_ppo.furniture.hand_curriculum import curriculum_levels
from cat_ppo.furniture.contrastive_bank import contrastive_roles
from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
from cat_mjlab.scene_bank import SceneBank
from cat_mjlab.runner import SuccessWindow
from cat_mjlab.fields import sample_ragged_field
from scripts.build_hand_posture_bank import exact_field_box_bound

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def bank():
    path=ROOT/'data/furniture/cat_hand_posture_v1_20260919/manifest.json'
    if not path.exists():pytest.skip('Built option C artifact not installed')
    m=load_generalist_manifest(path,verify_files=False)
    parent=load_generalist_manifest(m['external_retention_bank']['manifest'],verify_files=False)
    return path,m,parent


def test_bank_composition_retention_and_region_choices(bank):
    path,m,parent=bank
    assert m['scenes'][:2338]==parent['scenes'][:2338]
    assert len(m['scenes'])==2362
    assert contrastive_roles(m) is None  # Keep the legacy family/hand sampler.
    scenes=[json.loads((path.parent/r['path']/'scene.json').read_text()) for r in m['scenes'][2338:]]
    packed=pack_hand_contrast([None]+scenes)
    assert packed['role'].tolist()==[-1]+[1]*24
    assert sum(s['hand_protection']['kind']=='hand_table_aisle' for s in scenes)==12
    assert all(s['hand_contrast']['zones'][0]['region_valid']==[True,False] for s in scenes[:12])
    assert [s['hand_contrast']['zones'][0]['region_valid'][1] for s in scenes[12:]]==[True]*4+[False]*8
    validate_hand_posture_upgrade(m,parent,path=path,verify_files=True)


def test_fixed_family_and_hand_mass_at_every_stage_under_adaptation(bank):
    _,m,_=bank;b=object.__new__(SceneBank);b.device=torch.device('cpu');b.roles=None
    b.levels=torch.tensor(curriculum_levels(m,enabled=True));groups,masses=sampling_groups(m)
    b.groups=torch.tensor(groups);b.group_masses=torch.tensor(masses);b.weights=torch.ones(len(groups))
    for stage in range(3):
        for w in [b.weights,torch.linspace(.001,10,len(groups))]:
            p=b.probabilities(w,stage=torch.tensor(stage))
            torch.testing.assert_close(p.sum(),torch.tensor(1.,dtype=p.dtype))
            for group,mass in enumerate([.2,.4,.25,.15]):
                assert float(p[b.groups==group].sum())==pytest.approx(mass)
            assert float(p[(b.groups==2)&(b.levels>=0)].sum())==pytest.approx(.125)
            assert float(p[(b.groups==3)&(b.levels>=0)].sum())==pytest.approx(.075)
            assert not p[b.levels>stage].any()


@pytest.mark.parametrize('mutation', ['drop','retention','mass','fields','table_tuck','dual'])
def test_upgrade_rejects_changed_contracts(bank,mutation):
    _,m,parent=bank;m=deepcopy(m)
    if mutation=='drop':m['scenes'].pop()
    elif mutation=='retention':m['scenes'][0]['sampling_weight']=7
    elif mutation=='mass':m['sampling_group_masses']['original_cat']=.1
    elif mutation=='fields':m['scenes'][-1]['fields']['sdf']['sha256']='0'*64
    elif mutation=='table_tuck':m['scenes'][2338]['source']['hand_contrast']['zones'][0]['region_valid']=[True,True]
    else:m['width_curriculum']={}
    with pytest.raises(ValueError):validate_hand_posture_upgrade(m,parent)


def test_checkpoint_selection_never_requires_absent_roles():
    window=SuccessWindow();window.append(torch.zeros(4,2,dtype=torch.long),torch.tensor([[10,3],[0,0],[0,0]]),torch.tensor([[0,0],[10,3],[0,0],[0,0]]))
    assert window.role_balanced_score() is None
    assert window.role_balanced_score((1,))==pytest.approx(.3)


def test_clipped_cell_bound_preserves_actual_cat_corner_order():
    rng=np.random.default_rng(8);sdf=rng.normal(size=(8,8,8)).astype('f');origin=np.zeros(3)
    centers=np.array([[[.117,.131,.128]]]);halves=np.array([[[.012,.018,.007]]]);radii=np.array([[.103]])
    lower=exact_field_box_bound(sdf,origin,centers,halves,radii)
    points=centers.reshape(3)+rng.uniform(-1,1,(10000,3))*halves.reshape(3)
    actual=sample_ragged_field(torch.tensor(sdf.reshape(-1,1)),torch.tensor(points,dtype=torch.float32),origin=torch.zeros(3),dx=torch.tensor(.04),shape=torch.tensor([8,8,8]),offset=torch.tensor(0))[:,0]-.103
    assert lower<=float(actual.min())+2e-6
    assert float(actual.min())-lower<.2


def test_shelf_probe_tracks_reached_route_segment(bank):
    from scripts.audit_hand_posture_rewards import Probe
    from cat_mjlab.config import wholebody_config
    path,m,_=bank;rec=m['scenes'][2350];directory=path.parent/rec['path'];scene=json.loads((directory/'scene.json').read_text())
    probe=Probe(wholebody_config(hand_contrast=True),json.loads((ROOT/'docs/assets/collision-proxy-proposal-20260916/proposal.json').read_text()))
    b=probe.bank(rec,directory,scene);s,xy,yaw=probe.route(scene,spacing=.1);o=probe.task(b,xy,yaw,np.repeat(probe.qpose('raised')[None],len(xy),axis=0))
    active=(s>.9)&(s<2.6)
    assert (o.navigation['segment'][active]>0).all()
    assert (o.info['command'][active,1:3]*o.navigation['tangent'][active]).sum(-1).min()>.59
