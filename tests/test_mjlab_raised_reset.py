"""CPU tests for certified seeding, controller memory, and default behavior."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cat_mjlab.config import wholebody_config
from cat_mjlab.raised_reset import certify_scene, build_raised_reset_pool
from cat_mjlab import task_math as tm


def test_fraction_validation_and_inheritance():
    assert wholebody_config()['hand_raised_reset_fraction'] == 0
    base = wholebody_config(hand_contrast=True, hand_raised_reset_fraction=.5)
    assert wholebody_config(base, hand_contrast=True)['hand_raised_reset_fraction'] == .5
    assert wholebody_config(base, hand_contrast=True, hand_raised_reset_fraction=0)['hand_raised_reset_fraction'] == 0
    for value in [-.1, 1.1, float('nan'), float('inf'), True]:
        with pytest.raises(ValueError, match='hand_raised_reset_fraction'):
            wholebody_config(hand_contrast=True, hand_raised_reset_fraction=value)
    with pytest.raises(ValueError, match='contrastive bank'):
        wholebody_config(hand_raised_reset_fraction=.5)


@pytest.fixture(params=[3, 5])
def scene_record(request):
    from cat_ppo.furniture.generalist_fields import scene_directory
    path = Path(f'data/furniture/cat_flat_hand_balance_v{request.param}_20260921/manifest.json').resolve()
    if not path.exists(): pytest.skip('Local integration bank unavailable')
    manifest = json.loads(path.read_text())
    record = next(r for r in manifest['scenes'] if r.get('source', {}).get('hand_contrast', {}).get('role') == 'forward_protected')
    directory = scene_directory(manifest, path, record)
    return json.loads((directory/'scene.json').read_text()), record, directory


def test_certificate_rejects_unreachable_region(scene_record):
    from test_mjlab_task import _CPUSimulation
    scene, record, directory = scene_record
    model = _CPUSimulation(1).model
    qpos, _, clearance = certify_scene(model, scene, record, directory)
    assert qpos.dtype == np.float32 and qpos.shape == (4, 36) and clearance > 0
    for zone in scene['hand_contrast']['zones']:
        zone['hand_regions_min'] = (np.array(zone['hand_regions_min'])+10).tolist()
        zone['hand_regions_max'] = (np.array(zone['hand_regions_max'])+10).tolist()
    with pytest.raises(ValueError, match='compliance certificate'):
        certify_scene(model, scene, record, directory)


def make_task(monkeypatch, scene_record, fraction, seed=23):
    from test_mjlab_task import _CPUSimulation
    from cat_mjlab.task import CATTask
    from cat_ppo.furniture.room_navigation import pack_room_scenes
    from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
    scene, record, directory = scene_record
    sim = _CPUSimulation(8)
    from cat_mjlab.constants import DEFAULT_QPOS
    bank=SimpleNamespace(count=2,device=torch.device('cpu'),levels=None,roles=None,
        is_cat=torch.ones(2,dtype=torch.bool),reset_is_cat=torch.ones(2,dtype=torch.bool),
        crossed_is_plane=torch.ones(2,dtype=torch.bool),reset_yaws=torch.zeros(2),
        starts=torch.zeros((2,3)),reset_xy_scale=torch.ones((2,2)),goals=torch.tensor([[8.,0.,.7]]).expand(2,-1),
        origins=torch.tensor([[-5.,-5.,-.5]]).expand(2,-1),shapes=torch.tensor([[80,80,30]]).expand(2,-1),
        dxs=torch.full((2,),.2),episode_lengths=torch.full((2,),1000),navigation_groups=torch.zeros(2,dtype=torch.long),
        reset_pool=torch.tensor(DEFAULT_QPOS)[None,None].expand(2,1,-1))
    bank.probabilities=lambda weights=None,stage=None:torch.full((2,),.5)
    def sample(name,positions,scene_ids):
        if name=='sdf': return torch.ones((*positions.shape[:-1],1))
        result=torch.zeros_like(positions)
        if name=='gf': result[:,:,0]=.7
        return result
    bank.sample=sample
    poses, _, _ = certify_scene(sim.model, scene, record, directory)
    pool = np.stack([poses, poses]); eligible = np.array([True, False])
    def build(*args, **kwargs):
        assert fraction > 0, 'Disabled seeding must never construct a pool'
        return pool, eligible, {'test': 'controller wiring; scene separately certified'}
    monkeypatch.setattr('cat_mjlab.raised_reset.build_raised_reset_pool', build)
    bank.has_contrast=True; bank.manifest={}; bank.path=directory
    bank.rooms={k: torch.from_numpy(v) for k,v in pack_room_scenes([scene,None]).items()}
    bank.contrast={k: torch.from_numpy(v) for k,v in pack_hand_contrast([scene,None]).items()}
    class Collision:
        proposal_path=None
        active=False
        def __call__(self, scenes, data):
            return torch.full((len(scenes), 6), self.active, dtype=torch.bool)
    collision=Collision()
    cfg=wholebody_config(hand_contrast=True, hand_raised_reset_fraction=fraction)
    cfg['randomize_initial_episode_steps']=False
    task=CATTask(sim,bank,cfg,collision=collision,seed=seed)
    return task, collision


def test_reset_only_eligible_and_matches_controller_memory(monkeypatch, scene_record):
    task, collision=make_task(monkeypatch,scene_record,1.)
    scenes=torch.tensor([0,1]*4)
    task.reset(scene_ids=scenes)
    selected=scenes==0
    assert torch.equal(task.info['hand_raised_seeded'],selected)
    assert torch.count_nonzero(task.data.qvel[selected])==0
    q=task.data.qpos[selected,7:]
    torch.testing.assert_close(task.info['motor_targets'][selected],q)
    for key in ('previous_upper','previous_previous_upper'):
        torch.testing.assert_close(task.info[key][selected],q[:,12:])
    action=task.info['last_act'][selected]
    targets=tm.motor_targets(action,q,task.nominal,task.lower,task.upper)
    torch.testing.assert_close(targets,q)
    costs,_=tm.upper_stability_terms(q[:,12:],task.info['previous_upper'][selected],
        task.info['previous_previous_upper'][selected],task.nominal[12:],torch.ones(4,2),torch.ones(4,2),
        hand_protection=True,contrast_arm_active=torch.ones(4,2,dtype=torch.bool))
    assert all(torch.count_nonzero(value)==0 for value in costs.values())
    torch.testing.assert_close(task.info['motor_targets'][~selected],task.nominal.expand(4,-1))
    assert task.contrast['core_active'][selected].all()
    # Fail closed rather than silently substituting an old nominal pool pose.
    collision.active=True
    with pytest.raises(RuntimeError,match='runtime collision check'):
        task.reset(scene_ids=scenes)


def test_zero_disabled_and_fraction_reproducible(monkeypatch,scene_record):
    first,_=make_task(monkeypatch,scene_record,0.)
    assert 'hand_raised_seeded' not in first.info
    assert 'hand_raised_reset_certificate' not in first.config
    first,_=make_task(monkeypatch,scene_record,.5)
    second,_=make_task(monkeypatch,scene_record,.5)
    scenes=torch.zeros(8,dtype=torch.long)
    hits=0
    for _ in range(12):
        first.reset(scene_ids=scenes);second.reset(scene_ids=scenes)
        assert torch.equal(first.info['hand_raised_seeded'],second.info['hand_raised_seeded'])
        torch.testing.assert_close(first.data.qpos,second.data.qpos,rtol=0,atol=0)
        hits+=int(first.info['hand_raised_seeded'].sum())
    assert 24 < hits < 72


def test_gate_releases_only_posture():
    nominal=torch.zeros(17);current=torch.full((1,17),.01);current[:,:3]=0
    args=(current,torch.zeros_like(current),torch.zeros_like(current),nominal,torch.ones(1,2),torch.ones(1,2))
    ungated,_=tm.upper_stability_terms(*args,hand_protection=True)
    gated,_=tm.upper_stability_terms(*args,hand_protection=True,contrast_arm_active=torch.ones(1,2,dtype=torch.bool))
    assert gated['wholebody_upper_clear_posture'].item()==0
    assert ungated['wholebody_upper_clear_posture'].item()>0
    for key in ('wholebody_upper_target_velocity','wholebody_upper_target_acceleration'):
        assert gated[key].item()>0
        torch.testing.assert_close(gated[key],ungated[key])


def test_v5_pool_certifies():
    from test_mjlab_task import _CPUSimulation
    path = Path('data/furniture/cat_flat_hand_balance_v5_20260921/manifest.json').resolve()
    if not path.exists():
        pytest.skip('Local v5 bank unavailable')
    pool, eligible, certificate = build_raised_reset_pool(
        _CPUSimulation(1).model, json.loads(path.read_text()), path,
        path.parent.with_name(path.parent.name + '_collision')/'manifest.json')
    assert eligible.sum() == 12 and certificate['poses'] == 48
    assert np.isfinite(pool).all()
    for row in certificate['per_scene']:
        assert row['minimum_box_interior_margin_m'] > 0
        assert row['minimum_native_field_clearance_m'] > 0
        assert row['hand_region_cost_max'] == 0
        assert row['compliance_fraction'] == 1
        assert row['root_route_valid'] and row['body_collision_count'] == 0
        print(row['scene_id'], row['minimum_box_interior_margin_m'], row['minimum_native_field_clearance_m'])


@pytest.mark.parametrize('fraction', [None, 0., .5])
def test_launch_fraction_overrides_checkpoint(monkeypatch, fraction):
    from cat_mjlab import runner
    from cat_mjlab import sim, scene_bank, collision, task
    from cat_ppo.furniture import contrast_preflight
    monkeypatch.setattr(contrast_preflight, 'contrast_preflight', lambda *a, **k: None)
    monkeypatch.setattr(sim, 'CATSimulation', lambda *a, **k: SimpleNamespace(model=None))
    monkeypatch.setattr(scene_bank, 'SceneBank', lambda *a, **k: SimpleNamespace(has_contrast=True, balance_settings=None))
    monkeypatch.setattr(collision, 'CollisionChecker', lambda *a, **k: None)
    class Captured(Exception): pass
    def capture(sim, bank, config, **kwargs):
        assert config['hand_raised_reset_fraction'] == (fraction or 0.)
        raise Captured
    monkeypatch.setattr(task, 'CATTask', capture)
    args = SimpleNamespace(bank_manifest='unused', num_envs=1, device='cpu', nconmax=64,
        njmax=256, body_collision_resets='unused', body_collision_bank='unused', seed=0,
        hand_raised_reset_fraction=fraction)
    with pytest.raises(Captured):
        runner.create_task(args, environment_config=wholebody_config(hand_contrast=True, hand_raised_reset_fraction=.5))


def test_reset_rejects_uncommandable_pool_history(monkeypatch,scene_record):
    task,_=make_task(monkeypatch,scene_record,1.)
    task.raised_reset_pool[0,:,22]=task.nominal[15]-1.0
    with pytest.raises(ValueError,match='commandable action set'):
        task.reset(scene_ids=torch.zeros(8,dtype=torch.long))
