import copy,json
from pathlib import Path
import pytest
import torch
from cat_mjlab.balance import posture_terms,BalanceMetrics
from cat_ppo.furniture.balance_bank import sampling_plan,validate_balance_manifest
ROOT=Path(__file__).resolve().parents[1]
BANK=ROOT/'data/furniture/cat_flat_balance_v1_20260920/manifest.json'

def inputs():
    center=torch.tensor([[.429,.125,.870],[.429,-.125,.870]])
    half=torch.tensor([.018,.010,.018])
    return (torch.tensor([True,False,True,True]),center[None].repeat(4,1,1),torch.zeros(4,2),
            torch.tensor([[1.,0.]]).repeat(4,1),torch.tensor([[.6,0,0],[.6,0,0],[0.,0,0],[-.6,0,0]]),center-half,center+half)

def test_flat_reward_isolation_speed_and_absolute_height():
    args=inputs();reward,report=posture_terms(*args)
    torch.testing.assert_close(reward,torch.tensor([3.,0.,0.,0.]))
    lowerhands=args[1].clone();lowerhands[0,:,2]-=.2
    smaller,report=posture_terms(args[0],lowerhands,*args[2:])
    assert smaller[0]<reward[0] and not report['compliant'][0]
    assert torch.equal(smaller[1:],reward[1:])

def test_compiled_cpu_objective_matches():
    f=torch.compile(posture_terms,backend='inductor',fullgraph=True,dynamic=True)
    a,b=posture_terms(*inputs());c,d=f(*inputs())
    torch.testing.assert_close(a,c)
    for key in b:torch.testing.assert_close(b[key],d[key])

def test_composition_hashes_and_fixed_masses():
    from cat_ppo.furniture.generalist_fields import load_generalist_manifest
    m=load_generalist_manifest(BANK,verify_files=False)
    ids,masses=sampling_plan(m)
    weights=torch.linspace(.001,1,len(ids));ids=torch.tensor(ids);mass=torch.tensor(masses)
    totals=torch.zeros_like(mass).scatter_add_(0,ids,weights)
    probs=weights/totals[ids]*mass[ids]
    torch.testing.assert_close(torch.zeros_like(mass).scatter_add_(0,ids,probs),mass)
    assert list(torch.bincount(ids))==[64,2226,24,24,12,1]
    bad=copy.deepcopy(m);bad['scenes'][-1]['source']['flat_balance']='fake'
    with pytest.raises(ValueError):validate_balance_manifest(bad,path=BANK)
    bad=copy.deepcopy(m);bad['flat_balance']['masses'][0]=.1
    with pytest.raises(ValueError):validate_balance_manifest(bad,path=BANK)

def test_leader_denominators_and_fall_exposure():
    z=torch.zeros(4);b=torch.tensor([True,True,False,True])
    m={'balance/flat':b,'balance/walking':torch.tensor([True,False,True,True]),
       'balance/compliant':torch.tensor([True,True,True,False]),'balance/nominal':torch.tensor([False,False,False,True]),
       'balance/speed':torch.tensor([.6,0.,.6,.4]),'episode/fall':torch.tensor([False,True,False,True]),
       'balance/foot_balance':torch.tensor([-.02,-.03,-.04,-.01]),'balance/bonus':z,
       'balance/streak':torch.tensor([5,0,10,0]),'balance/heights':torch.ones(4,3)}
    a=BalanceMetrics();a.append(m,torch.tensor([False,True,True,True]),torch.tensor([True,True,True,False]),.02)
    r=a.result();assert r['balance/leader_step_count']==2
    assert r['balance/leader_walking_step_count']==1
    assert r['balance/leader_walking_steps_compliant_fraction']==1
    assert r['balance/leader_all_steps_walking_compliant_fraction']==.5
    assert r['balance/leader_fall_rate_per_episode']==1
    assert r['balance/leader_fall_rate_per_simulated_minute']==pytest.approx(1500)
    assert r['balance/leader_longest_raised_walking_seconds']==.1
    assert r['balance/leader_raised_step_count']==2
    assert r['balance/leader_nominal_step_count']==0

def test_composed_reset_pools_preserve_source_poses():
    import numpy as np
    from cat_ppo.furniture.generalist_fields import sha256
    from cat_mjlab.constants import DEFAULT_QPOS
    root=BANK.parent.with_name(BANK.parent.name+'_resets')
    meta=json.loads((root/'manifest.json').read_text())
    assert meta['field_manifest_sha256']==sha256(BANK)
    assert meta['sha256']==sha256(root/meta['file'])
    pool=np.load(root/meta['file']);offset=0
    for pin in meta['sources']:
        parent=Path(pin['manifest']);assert sha256(parent)==pin['sha256']
        source=json.loads(parent.read_text());assert sha256(parent.parent/source['file'])==source['sha256']
        expected=np.load(parent.parent/source['file'])[pin['indices']]
        np.testing.assert_array_equal(pool[offset:offset+len(expected)],expected);offset+=len(expected)
    assert offset==2350 and pool.shape==(2351,32,36)
    np.testing.assert_array_equal(pool[-1],np.repeat(np.asarray(DEFAULT_QPOS,np.float32)[None],32,axis=0))
