"""Safety and inactive-path regression tests for opt-in standing scenes."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from cat_mjlab.reactive import StandingObjects,capsule_query,validate_bank
from cat_mjlab.model import assemble_training_xml
ROOT=Path(__file__).resolve().parents[1]
BANK=ROOT/'data/furniture/reactive_standing_approval_20260922/manifest.json'

def spec():
    torch.set_num_threads(2)
    m=mujoco.MjModel.from_xml_string(assemble_training_xml());d=mujoco.MjData(m)
    e=StandingObjects(bank=BANK,num_envs=1,model=m,device='cpu');e.force_rows=torch.tensor([0])
    _,_,q=e.choose(torch.tensor([0]),torch.zeros(1,2),torch.tensor([0]));d.qpos[:]=q[0].numpy();mujoco.mj_forward(m,d)
    data=SimpleNamespace(xpos=torch.tensor(d.xpos[None],dtype=torch.float32),xmat=torch.tensor(d.xmat[None],dtype=torch.float32))
    return m,e,data

def test_manifest_composition_and_all_full_paths():
    meta=validate_bank(json.loads(BANK.read_text()));assert len(meta['scenes'])==30
    for name in ('danger','anticipation','negative'):
        assert sum(r['bucket']==name for r in meta['scenes'])==10
    assert meta['retained_mass']==.75 and meta['retained_scene_count']==2375
    assert meta['above_share_of_reactive']<.02
    cert=json.loads((ROOT/'docs/assets/reactive-approval-20260922/certification.json').read_text())
    assert cert['all_passed'] and len(cert['scenes'])==30
    for c in cert['scenes']:
        assert c['min_floor_clearance_m']>=.002
        assert max(abs(x-c['target_surface_m']) for x in c['achieved_surface_m'])<1e-5
        assert min(x['continuous_body_lower_bound_m'] for x in c['object_paths'])>0
    for r in meta['scenes']:
        if r['direction'] in ('right','front_right','back_right'):assert r['target']=='right'

def test_object_motion_certified_and_never_changes_robot():
    _,e,data=spec();before={k:v.clone() for k,v in vars(data).items()}
    # Approach, stop and return at .5 m/s, including clearance guard at the endpoint.
    e.state['speed'][:]=.5
    for _ in range(180):
        p=e.state['position'].clone();ids,spheres,gap=e.advance(data,.01)
        displacement=torch.linalg.vector_norm(e.state['position']-p,dim=-1)
        assert (gap-displacement[:,None]).min()>0
    for k,v in before.items():assert torch.equal(getattr(data,k),v)
    assert e.state['retreating'][0,0]

def test_robot_lunge_is_logged_without_rejecting_motion():
    _,e,data=spec();ids,before,gap=e.advance(data,.002)
    after=SimpleNamespace(xpos=data.xpos.clone(),xmat=data.xmat.clone())
    # Inject a robot movement into the fixed object as a collision detector unit test.
    hand=e.geometry.spheres(data,ids)[:,e.geometry.hand_mask][:,0]
    displacement=e.state['position'][:,0]-hand
    after.xpos+=displacement[:,None]
    saved=after.xpos.clone();regions,robot=e.contacts(after,ids,before,gap)
    assert regions.any() and robot.item()
    assert torch.equal(after.xpos,saved)

def test_forearm_interior_detects_thin_rod_elbow_point_misses():
    ep=torch.tensor([[[[0.,0.,0.],[.3,0.,0.]]]])
    d,n=capsule_query(ep,torch.tensor([.03]),torch.tensor([[[.15,.05,0.]]]),torch.eye(3)[None,None],torch.tensor([[[.008,.1,.008]]]),torch.tensor([[2]]),torch.tensor([[True]]))
    assert abs(float(d)-.012)<.0001
    assert float(d)<np.hypot(.15,.05)-.008-.03

def test_inactive_standing_engine_preserves_task_bitwise():
    from test_mjlab_task import _CPUSimulation,_tiny_bank
    from cat_mjlab.task import CATTask
    from cat_mjlab.config import wholebody_config
    cfg=wholebody_config();cfg['randomize_initial_episode_steps']=False
    a=CATTask(_CPUSimulation(1),_tiny_bank(1),deepcopy(cfg),seed=12)
    sim=_CPUSimulation(1);engine=StandingObjects(bank=BANK,num_envs=1,model=sim.model,device='cpu')
    # A disabled composite must consume no additional RNG and change no results.
    def disabled(ids,random,scenes):
        engine.state['active'][ids]=False;engine.state['valid'][ids]=False
        return scenes,engine.state['active'][ids],engine.table['qpos'][:len(ids)]
    engine.choose=disabled
    b=CATTask(sim,_tiny_bank(1),deepcopy(cfg),seed=12,analytic_objects=engine)
    # Additional reset selection uses an independent object sampler, not task RNG.
    for _ in range(4):
        x=a.step(torch.zeros(1,29));y=b.step(torch.zeros(1,29))
        for key in x['obs']:assert torch.equal(x['obs'][key],y['obs'][key])
        assert torch.equal(x['reward'],y['reward'])
        assert torch.equal(x['done'],y['done'])
