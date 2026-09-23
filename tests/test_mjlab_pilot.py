"""Small CPU checks for scoped rewards and fresh native pilot initialization."""
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import json
import math

import numpy as np
import pytest
import torch

from cat_mjlab import runner, task_math as tm
from cat_mjlab.learning import Learner, gaussian_parameters, log_probability
from cat_mjlab.passage_rewards import passage_knees, blended_sdf_knee
from test_mjlab_runner import FakeTask, FakeSimulation, tiny_config
from train_cat_mjlab import parser


def scene_fixture():
    from cat_ppo.furniture.scenes import _digest
    boxes, dimensions, route = [], [9., 3.6, 1.8], [[1., 1.8], [8., 1.8]]
    certificate = dict(schema='hand-contrast-certificate-v1', route_transition_validated=True,
        navigation_radius_m=.15, primitive_count=35, geometry_hash=_digest(dict(boxes=boxes, room_dimensions=dimensions)),
        route_hash=_digest(route), body_proxy_sha256='a'*64, route_sample_count=1,
        transition_sample_count=1, body_min_separation_m=.03, hand_field_min_clearance_m=.03)
    return dict(scene_id='transition', boxes=boxes, room_dimensions=dimensions, route=route,
        hand_contrast=dict(schema='hand-contrast-v1', navigation_radius_m=.15, certificate=certificate,
            modules=[dict(role='forward_protected'), dict(role='narrow')],
            zones=[dict(fade_m=.15), dict(fade_m=.15)]))


def test_metadata_requires_correct_bank_certified_narrow_zone_and_fade(tmp_path):
    bank = tmp_path/'bank.json'; bank.write_text('{}')
    scene = scene_fixture()
    overlay = tmp_path/'rewards.json'
    doc = dict(schema='cat-passage-rewards-v1', bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
               scenes={'transition': {'1': {'sdf_reward_knee': 0.}}})
    overlay.write_text(json.dumps(doc))
    result = passage_knees([None, scene], bank_path=bank, override_path=overlay)
    assert result[1, 1] == 0
    assert np.all(result[0] == np.float32(.05))
    assert result[1, 0] == np.float32(.05)
    doc['scenes']['transition'] = {'0': {'sdf_reward_knee': 0.}}
    overlay.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='narrow'):
        passage_knees([scene], bank_path=bank, override_path=overlay)
    doc['bank_sha256'] = 'wrong'; overlay.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='hash'):
        passage_knees([scene], bank_path=bank, override_path=overlay)
    scene['hand_contrast']['zones'][1]['sdf_reward_knee'] = .01
    assert passage_knees([scene], bank_path=bank)[0, 1] == np.float32(.01)
    scene['hand_contrast']['zones'][1]['fade_m'] = 0
    with pytest.raises(ValueError, match='fade'):
        passage_knees([scene], bank_path=bank)


def test_zone_ramp_and_legacy_rows_are_exact():
    metadata = {'sdf_reward_knee': torch.tensor([[0., .05]])}
    context = dict(zone_index=torch.tensor([0, 0, 0, 0, 0, 1]),
                   phase_weight=torch.tensor([0., .25, .5, .75, 1., 1.]))
    knees = blended_sdf_knee(metadata, torch.zeros(6, dtype=torch.long), context)
    torch.testing.assert_close(knees, torch.tensor([.05, .0421875, .025, .0078125, 0., .05]))
    sdf = torch.full((6, 2, 1), .05)
    legacy = tm.sdf_reward(sdf)
    modified = tm.sdf_reward(sdf, knees)
    assert torch.equal(legacy[[0, 5]], modified[[0, 5]])
    assert modified[4].item() == pytest.approx(-1.5777947)
    assert legacy[4].item() == pytest.approx(-13.8629436)
    # No optional argument leaves the historical arithmetic bit-for-bit intact.
    expected = (-20 * torch.nn.functional.softplus((.05-sdf)/.02)).flatten(1).mean(-1)
    assert torch.equal(legacy, expected)


def test_native_reward_changes_only_df_inside_selected_rows():
    n = 3
    z = lambda *shape: torch.zeros(n, *shape)
    eye = torch.eye(3).repeat(n, 1, 1)
    kwargs = dict(action=z(29),last_action=z(29),last_last_action=z(29),joint_pos=z(29),joint_vel=z(29),
        last_joint_vel=z(29),lower=torch.full((29,),-2.),upper=torch.full((29,),2.),actuator_force=z(29),
        command=torch.tensor([[1.,.6,0.,0.]]).repeat(n,1),pelvis_rpy=z(3),torso_rpy=z(3),head_z=torch.ones(n),
        torso_height=1.,global_velocity=z(3),torso_angvel=z(3),navi=eye,leg_rotations=eye[:,None].repeat(1,4,1,1),
        feet_pos=z(2,3),feet_sensor_velocity=z(2,3),subtree_com=z(3),feet_contact=torch.ones(n,2,dtype=torch.bool),
        gait=torch.tensor([[1.,-1.]]).repeat(n,1),foot_height=torch.full((n,),.07),foot_height_stance=0.,
        gf=torch.ones(n,11,3),positions=z(11,3),velocities=z(11,3),sdf=torch.full((n,11,1),.05),
        crossed=torch.zeros(n,11,dtype=torch.bool))
    before = tm.native_rewards(**kwargs)
    after = tm.native_rewards(**kwargs, sdf_knee=torch.tensor([.05,0.,.05]))
    for key in before:
        assert torch.equal(before[key][[0,2]],after[key][[0,2]])
        if key.endswith('df'):
            assert after[key][1] > before[key][1]
        else:
            assert torch.equal(before[key],after[key])


def test_sigma_cap_and_reset_preserve_mean_and_all_policy_likelihoods():
    config = tiny_config()
    plain = Learner(config, device='cpu')
    capped = Learner(replace(config, max_action_std=.15), device='cpu')
    capped.model.load_state_dict(plain.model.state_dict())
    obs = dict(state=torch.randn(6,3), privileged_state=torch.randn(6,4))
    ids = torch.tensor([0,0,1,1,2,2])
    before = capped.model.logits(obs['state'],ids)[...,:2].clone()
    capped.reset_action_std(.05)
    logits = capped.model.logits(obs['state'],ids)
    assert torch.equal(before,logits[...,:2])
    torch.testing.assert_close(gaussian_parameters(logits)[1],torch.full((6,2),.05))
    with torch.no_grad():
        capped.model.actor.layers[-1].bias[2:] = 20
    acted = capped.act(obs,policy_ids=ids)
    assert bool((acted['action_std'] <= .15000001).all())
    torch.testing.assert_close(acted['log_prob'], log_probability(capped.model.logits(obs['state'],ids),acted['raw_action']))
    assert torch.equal(plain.model.logits(obs['state'],ids),plain.model.actor(plain.model.condition(obs['state'],ids)))
    with pytest.raises(ValueError):
        capped.reset_action_std(.2)


@pytest.mark.parametrize('reset_sigma', [True, False])
def test_native_warm_start_is_fresh_and_strict_resume_rejects_source_changes(tmp_path,monkeypatch,reset_sigma):
    source = Learner(tiny_config(),device='cpu')
    source.updates=334;source.env_steps=525336576
    source_state=source.state_dict()
    source_state['config'].pop('max_action_std')  # old native source
    native = tmp_path/'source.pt'
    torch.save(dict(schema='cat-mjlab-runtime-v1',learner=source_state,
                    contract=dict(environment_config={'preserved': True})),native)
    # Unit-test task only; no MuJoCo/Warp or actual training job is constructed.
    monkeypatch.setattr(runner,'create_task',lambda args,environment_config: (FakeTask(6),FakeSimulation(),environment_config))
    asset=tmp_path/'bank.json';asset.write_text('{}')
    directory=tmp_path/'pilot'
    argv=['verify','--checkpoint-native',str(native),'--fresh-optimizer','--init-action-std','.05',
        '--max-action-std','.15','--discounting','.995','--bank-manifest',str(asset),
        '--body-collision-bank',str(asset),'--body-collision-resets',str(asset),'--run-dir',str(directory),
        '--num-envs','6','--batch-size','3','--unroll-length','2','--device','cpu']
    if not reset_sigma:
        index = argv.index('--init-action-std')
        del argv[index:index+2]
        argv += ['--num-minibatches', '4']
    # Stop immediately after initialization so we can inspect the saved pilot.
    from contextlib import contextmanager
    @contextmanager
    def stopped(_):
        yield lambda: True
    monkeypatch.setattr(runner,'stop_requests',stopped)
    for extra, message in [
        (['--num-minibatches', '0'], 'num_minibatches must be a positive integer'),
        (['--num-minibatches', '3'], 'must be divisible by num_envs'),
        (['--num-minibatches', '3', '--batch-size', '2'], 'must be divisible by num_policies'),
    ]:
        with pytest.raises(ValueError, match=message):
            runner.run(parser().parse_args(argv + extra))
    runner.run(parser().parse_args(argv))
    saved=torch.load(directory/'resume.pt',weights_only=True)
    assert saved['learner']['env_steps']==saved['learner']['updates']==0
    assert not saved['learner']['optimizer']['state']
    assert saved['learner']['config']['discounting']==.995
    assert saved['learner']['config']['max_action_std']==.15
    expected_minibatches = 2 if reset_sigma else 4
    assert saved['learner']['config']['num_minibatches'] == expected_minibatches
    restored=Learner(replace(tiny_config(),max_action_std=.15,discounting=.995,
                            num_minibatches=expected_minibatches),device='cpu')
    restored.load_state_dict(saved['learner'])
    observations, ids = torch.randn(6,3), torch.tensor([0,0,1,1,2,2])
    expected_std = (torch.full((6,2),.05) if reset_sigma else
                    gaussian_parameters(source.model.logits(observations,ids))[1].clamp_max(.15))
    torch.testing.assert_close(gaussian_parameters(restored.model.logits(observations,ids))[1],expected_std)
    for key,value in source.model.state_dict().items():
        if reset_sigma and key.startswith('actor.layers.1.'):
            assert torch.equal(value[:2],saved['learner']['model'][key][:2])
        else:
            assert torch.equal(value,saved['learner']['model'][key])
    resume=[arg for arg in argv if arg!='--fresh-optimizer']+['--resume']
    runner.run(parser().parse_args(resume))
    monkeypatch.setattr(runner,'_source_identity',lambda:'changed-code')
    with pytest.raises(ValueError,match='contract differs'):
        runner.run(parser().parse_args(resume))
    assert native.stat().st_size > 0


def test_old_learner_without_ceiling_loads_and_new_ceiling_is_checked():
    old=Learner(tiny_config(),device='cpu').state_dict()
    old['config'].pop('max_action_std')
    Learner(tiny_config(),device='cpu').load_state_dict(old)
    with pytest.raises(ValueError,match='configuration'):
        Learner(replace(tiny_config(),max_action_std=.15),device='cpu').load_state_dict(old)
