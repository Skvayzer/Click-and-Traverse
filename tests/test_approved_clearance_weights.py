"""CPU-only checks for the approved warm-start reward overrides; never launch."""
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cat_mjlab.config import wholebody_config
from cat_mjlab.task_math import hand_clearance_pressure


def launch_args():
    from train_cat_mjlab import parser
    script = Path('configs/pilots/clearance_weight_warmstart.sh').read_text().replace('\\\n', ' ')
    line = next(x for x in script.splitlines() if 'train_cat_mjlab.py run' in x)
    return parser().parse_args(shlex.split(line)[2:])


def approved():
    a = launch_args()
    return wholebody_config(**{k: getattr(a, k) for k in (
        'hand_clearance_weight', 'arm_clearance_weight', 'hand_clearance_target',
        'hand_clearance_anticipation', 'hand_clearance_near_weight')})


def test_defaults_and_only_approved_config_deltas():
    old, new = wholebody_config(), approved()
    assert old['reward_config']['scales']['wholebody_hand_clearance'] == -.5
    assert old['reward_config']['scales']['wholebody_arm_clearance'] == -2
    assert old['hand_protection_target_clearance'] == .04
    for key in ('hand_protection_anticipation_distance', 'hand_protection_near_weight'):
        assert old[key] == new[key]
    expected = {'wholebody_hand_clearance': -20., 'wholebody_arm_clearance': -8.}
    assert {k: v for k, v in new['reward_config']['scales'].items()
            if v != old['reward_config']['scales'][k]} == expected
    for key in old:
        if key not in ('reward_config', 'hand_protection_target_clearance'):
            assert old[key] == new[key]
    assert new['hand_protection_target_clearance'] == .09
    restored = wholebody_config(json.loads(json.dumps(new)))
    assert restored == new
    assert wholebody_config(hand_protection=False)['reward_config']['scales']['wholebody_hand_clearance'] == -5


@pytest.mark.parametrize('value', [1., float('nan'), float('inf'), True])
def test_arm_sign_validation(value):
    with pytest.raises(ValueError, match='arm_clearance_weight'):
        wholebody_config(arm_clearance_weight=value)


def test_penalties_and_pose_ordering():
    d = torch.tensor([.1402, .09, .05, 0.], dtype=torch.float64)
    p, _ = hand_clearance_pressure(d[:, None].expand(-1, 2), target=.09, anticipation=.20, near_weight=.8)
    rates = -20*p
    torch.testing.assert_close(rates, torch.tensor([-.357604, -1.21, -5.410493827160494, -20.], dtype=torch.float64))
    assert torch.all(rates[:-1] > rates[1:])
    assert torch.all(rates <= 0)
    print('Both-hand costs: rate / step', list(zip((-rates).tolist(), (-rates*.02).tolist())))
    arm = -8*(.08-torch.tensor([.07899, .06641, .05, 0.], dtype=torch.float64)).clamp_min(0).square()*.02
    assert torch.all(arm[:-1] > arm[1:])
    print('Both-elbow step costs at .07899/.06641/.05/0:', (-arm).tolist())


@pytest.mark.parametrize('cohort', ['flat', 'clutter', 'cat', 'narrow'])
def test_only_clearance_ledger_changes(cohort):
    from test_mjlab_task import _CPUSimulation, _tiny_bank
    from cat_mjlab.task import CATTask
    tasks = []
    for cfg in (wholebody_config(), approved()):
        cfg['randomize_initial_episode_steps'] = False
        bank = _tiny_bank(1)
        bank.reset_is_cat[:] = cohort == 'cat'
        bank.is_cat[:] = cohort == 'cat'
        bank.navigation_groups[:] = dict(cat=0, flat=1, clutter=1, narrow=2)[cohort]
        tasks.append(CATTask(_CPUSimulation(1), bank, cfg, seed=17))
    for distance in (.1402, .09, .05, 0.):
        ledgers = []
        for task in tasks:
            task.info['sdf'][:, 5:7] = distance
            task.info['elbow_clearance'][:] = .05
            reward, parts = task._rewards(torch.zeros(1, 29), torch.zeros(1, 2, dtype=torch.bool))
            hand = parts['wholebody_hand_clearance']*.02
            pre = sum(v for k, v in parts.items() if k != 'wholebody_hand_clearance')*.02
            torch.testing.assert_close(reward, pre.clamp(0, 10000)+hand)
            ledgers.append(parts)
        for key in ledgers[0]:
            if key not in ('wholebody_hand_clearance', 'wholebody_arm_clearance'):
                torch.testing.assert_close(ledgers[0][key], ledgers[1][key], rtol=0, atol=0)
        torch.testing.assert_close(ledgers[1]['wholebody_arm_clearance'], 4*ledgers[0]['wholebody_arm_clearance'])


def test_run_environment_contract_plumbing(monkeypatch):
    from cat_mjlab import runner, sim, scene_bank, collision, task
    from cat_ppo.furniture import contrast_preflight
    monkeypatch.setattr(contrast_preflight, 'contrast_preflight', lambda *a, **k: None)
    monkeypatch.setattr(sim, 'CATSimulation', lambda *a, **k: SimpleNamespace(model=None))
    monkeypatch.setattr(scene_bank, 'SceneBank', lambda *a, **k: SimpleNamespace(has_contrast=True, balance_settings=None))
    monkeypatch.setattr(collision, 'CollisionChecker', lambda *a, **k: None)
    monkeypatch.setattr(task, 'CATTask', lambda s, b, c, **k: SimpleNamespace(config=c))
    args = launch_args()
    args.compile_task = False
    _, _, cfg = runner.create_task(args)
    # run() records this returned config as contract.environment_config.
    contract = json.loads(json.dumps(dict(environment_config=cfg)))
    c = contract['environment_config']
    assert c['reward_config']['scales']['wholebody_hand_clearance'] == -20
    assert c['reward_config']['scales']['wholebody_arm_clearance'] == -8
    assert [c['hand_protection_'+k] for k in ('target_clearance', 'anticipation_distance', 'near_weight')] == [.09, .20, .8]
    assert str(args.checkpoint_native) == 'outputs/cat_flat_balance_ppo_37632_20260920/resume.pt'
    assert args.checkpoint_native.is_file()
    assert args.fresh_optimizer
    assert (args.wandb_mode, args.wandb_project, args.wandb_entity) == ('online', 'CAT-wholebody', 'skvayzer')
