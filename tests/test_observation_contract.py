"""Native contract, no authored box features and no checkpoint surgery."""
from copy import deepcopy
import pytest
import torch
from cat_mjlab.observation_contract import ACTOR_SIZE, CRITIC_SIZE
from cat_mjlab.navigation import hand_contrast_context
from cat_ppo.furniture.control import mjlab_observation_contract, wholebody_observation_contract

def scene(progress=2., approach=0.):
    from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
    m = {k: torch.as_tensor(v.copy()) for k, v in pack_hand_contrast([None]).items()}
    m['enabled'][:] = True
    m['role'][:] = 1
    m['zone_valid'][:, 0] = True
    m['start_m'][:, 0], m['end_m'][:, 0], m['fade_m'][:, 0] = 1., 3., .2
    m['hand_active'][:, 0] = True
    m['region_valid'][:, 0] = True
    # Both alternatives specify a PAIRED left/right target, in route axes.
    centers = torch.tensor([[[.4, .3, 1.2], [.4, -.3, 1.2]],
                            [[-.2, .3, .9], [-.2, -.3, .9]]])
    half = torch.tensor([.1, .05, .2])
    m['hand_regions_min'][:, 0] = centers-half
    m['hand_regions_max'][:, 0] = centers+half
    p = torch.tensor([progress])
    c = hand_contrast_context(m, p, torch.tensor([[0., 1.]]), approach_distance=approach)
    # root=(10,20), route forward=world +Y, left=world -X.
    # Hand route coordinates are (.2,.1,1.0), (.1,-.2,.8).
    hands = torch.tensor([[[9.9, 20.2, 1.], [10.2, 20.1, .8]]])
    args = (c, m, p, hands, torch.tensor([[10., 20.]]), torch.tensor([[1., 0., 0.]]))
    return args


def test_native_sizes_names_and_geometry_features():
    from cat_mjlab.config import wholebody_config
    from cat_mjlab.learning import LearnerConfig
    from cat_mjlab.task import CATTask
    contract=mjlab_observation_contract()
    assert contract==wholebody_observation_contract()
    assert (ACTOR_SIZE,CRITIC_SIZE)==(222,310)
    assert (CATTask.observation_size,CATTask.privileged_observation_size)==(222,310)
    assert (LearnerConfig().actor_obs,LearnerConfig().critic_obs)==(222,310)
    cfg=wholebody_config();assert (cfg['num_obs'],cfg['num_pri'])==(222,310)
    for key in ('actor_features','critic_features'):
        assert len(contract[key])==len(set(contract[key]))
        assert not any(name.startswith('hand_objective.') for name in contract[key])
        hands=[name for name in contract[key] if name.startswith('pf.hands.')]
        assert len(hands)==14
        assert 'pf.hands.df.0.distance' in hands and 'pf.hands.df.1.distance' in hands


def test_authored_boxes_flags_and_phase_do_not_change_any_observation():
    from test_mjlab_task import _CPUSimulation,_tiny_bank
    from cat_mjlab.task import CATTask
    from cat_mjlab.config import wholebody_config
    cfg=wholebody_config();cfg['noise_config']['level']=0.
    task=CATTask(_CPUSimulation(1),_tiny_bank(1),cfg,seed=7)
    c,m,p,*_=scene();task.bank.contrast=m;task.contrast=c
    contact=torch.zeros(1,2,dtype=torch.bool)
    before={k:v.clone() for k,v in task._observe(task.all_ids,contact).items()}
    task.bank.contrast=deepcopy(m)
    task.bank.contrast['hand_regions_min']+=100;task.bank.contrast['hand_regions_max']+=200
    task.bank.contrast['hand_active'][:]=False;task.bank.contrast['region_valid'][:]=False
    task.contrast=hand_contrast_context(task.bank.contrast,p+.5,torch.tensor([[1.,0.]]))
    after=task._observe(task.all_ids,contact)
    for key in before:torch.testing.assert_close(before[key],after[key],atol=0,rtol=0)


def test_evaluator_requires_native_sizes_without_slicing():
    from dataclasses import replace
    from cat_mjlab.learning import LearnerConfig
    from scripts.evaluate_paired_cat import actor_observation_for_policy
    state=torch.randn(3,222);cfg=LearnerConfig(algorithm='ppo')
    assert actor_observation_for_policy(cfg,state) is state
    with pytest.raises(ValueError,match='Unsupported'):
        actor_observation_for_policy(replace(cfg,actor_obs=222+32,critic_obs=310+32),state)
    with pytest.raises(ValueError,match='native actor'):
        actor_observation_for_policy(cfg,torch.zeros(1,222+32))

def test_scratch_launch_initializes_current_contract_without_checkpoint_or_training(tmp_path, monkeypatch):
    from cat_mjlab import runner
    from train_cat_mjlab import parser
    args = parser().parse_args(['run', '--from-scratch', '--algorithm', 'ppo',
        '--num-envs', '30720', '--batch-size', '768', '--num-minibatches', '40',
        '--max-action-std', '0', '--device', 'cpu', '--bank-manifest', 'unused',
        '--body-collision-bank', 'unused', '--body-collision-resets', 'unused',
        '--run-dir', str(tmp_path/'run')])
    made = []
    original = runner.Learner
    def learner(config, **kwargs):
        result = original(config, **kwargs)
        made.append(result)
        return result
    class StopBeforePhysics(Exception):
        pass
    def stop(*args, **kwargs):
        assert kwargs['environment_config'] is None
        raise StopBeforePhysics
    monkeypatch.setattr(runner, 'Learner', learner)
    monkeypatch.setattr(runner, 'create_task', stop)
    monkeypatch.setattr(runner, 'read_array_archive', lambda *a: pytest.fail('Scratch read a checkpoint'))
    with pytest.raises(StopBeforePhysics):
        runner.run(args)
    assert len(made) == 1
    assert (made[0].config.actor_obs, made[0].config.critic_obs) == (222, 310)
    assert made[0].config.max_action_std is None
    assert not made[0].optimizer.state


def test_obsolete_native_training_rejected_before_learner_or_physics(tmp_path, monkeypatch):
    from dataclasses import asdict, replace
    from cat_mjlab import runner
    from cat_mjlab.learning import LearnerConfig
    from train_cat_mjlab import parser
    args = parser().parse_args(['run', '--checkpoint-native', 'unused', '--fresh-optimizer',
        '--bank-manifest', 'unused', '--body-collision-bank', 'unused',
        '--body-collision-resets', 'unused', '--run-dir', str(tmp_path/'run')])
    old = replace(LearnerConfig(), actor_obs=222+32, critic_obs=310+32)
    monkeypatch.setattr(runner, 'native_initialization', lambda *a: (asdict(old), {}, {}))
    monkeypatch.setattr(runner, 'Learner', lambda *a, **k: pytest.fail('Allocated a learner'))
    monkeypatch.setattr(runner, 'create_task', lambda *a, **k: pytest.fail('Allocated physics'))
    with pytest.raises(ValueError, match='--from-scratch'):
        runner.run(args)


