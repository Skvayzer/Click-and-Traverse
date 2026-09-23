"""CPU checks for hand-task gate configuration and first-outcome progression."""
from types import SimpleNamespace

import pytest
import torch

from cat_mjlab.config import hand_curriculum_threshold, wholebody_config
from cat_mjlab.task import CATTask
from train_cat_mjlab import parser


def test_hand_gate_defaults_metadata_and_explicit_run_override():
    assert hand_curriculum_threshold({}) == .6
    assert hand_curriculum_threshold({}, {'hand_protection_curriculum': True}) == .6
    manifest = {'hand_protection_curriculum': {'clean_goal_success_threshold': .4}}
    assert hand_curriculum_threshold({}, manifest) == .4
    config = wholebody_config(hand_curriculum_success_threshold=.35)
    assert config['hand_curriculum_success_threshold'] == .35
    assert hand_curriculum_threshold(config, manifest) == .35
    assert 'hand_curriculum_success_threshold' not in wholebody_config()
    assert wholebody_config(config)['hand_curriculum_success_threshold'] == .35
    args = parser().parse_args(['run', '--checkpoint-npz', 'weights.npz',
        '--bank-manifest', 'bank.json', '--body-collision-bank', 'collision.json',
        '--body-collision-resets', 'resets.json', '--run-dir', 'unused',
        '--hand-curriculum-success-threshold', '.35'])
    assert args.hand_curriculum_success_threshold == .35


@pytest.mark.parametrize('value', [0, -.1, 1.01, float('nan'), float('inf'), True, '0.35'])
def test_hand_gate_rejects_invalid_thresholds(value):
    with pytest.raises(ValueError, match='hand_curriculum_success_threshold'):
        wholebody_config(hand_curriculum_success_threshold=value)
    with pytest.raises(ValueError, match='hand_curriculum_success_threshold'):
        hand_curriculum_threshold({}, {'hand_protection_curriculum': {'clean_goal_success_threshold': value}})


def hand_task(*, config, completed, goals, stage=0):
    t = object.__new__(CATTask)
    t.config = config
    t.device = torch.device('cpu')
    t.hand_contrast = False
    t.bank = SimpleNamespace(roles=None, levels=torch.tensor([stage]),
        navigation_groups=torch.tensor([2]), probabilities=lambda weights, stage: torch.ones(1))
    t.scene_ids = torch.zeros(1, dtype=torch.long)
    t.policy_ids = torch.zeros(1, dtype=torch.long)
    t.outcome_counted = torch.zeros(1, dtype=torch.bool)
    t.episode = {key: torch.zeros(1, dtype=torch.bool) for key in (
        'fall', 'obstacle', 'self_contact', 'numerical', 'body_collision',
        'hand_violation', 'elbow_violation', 'outside_bounds')}
    t.episode['goal_reached'] = torch.ones(1, dtype=torch.bool)
    t.navigation_counts = torch.zeros(4, 2, dtype=torch.long)
    t.curriculum_stage = torch.tensor(stage)
    t.curriculum_completed = torch.zeros(3, dtype=torch.long)
    t.curriculum_goals = torch.zeros(3, dtype=torch.long)
    t.curriculum_completed[stage] = completed
    t.curriculum_goals[stage] = goals
    t.scene_episode_ema = torch.zeros(1)
    t.scene_success_ema = torch.zeros(1)
    return t


@pytest.mark.parametrize('config,completed,goals,stage,expected', [
    ({}, 63, 22, 0, 0),  # 23/64 does not meet the legacy 60% gate.
    ({'hand_curriculum_success_threshold': .35}, 63, 22, 0, 1),
    ({'hand_curriculum_success_threshold': .35}, 62, 62, 0, 0),  # Keep 64 minimum.
    ({'hand_curriculum_success_threshold': .35}, 63, 21, 0, 0),  # 22/64 < .35.
    ({}, 63, 38, 0, 1),
    ({'hand_curriculum_success_threshold': .35}, 63, 22, 1, 2),
    ({'hand_curriculum_success_threshold': .35}, 63, 22, 2, 2),  # Final level capped.
])
def test_actual_outcomes_use_configured_gate_and_count_only_once(config, completed, goals, stage, expected):
    t = hand_task(config=config, completed=completed, goals=goals, stage=stage)
    done = torch.zeros(1, dtype=torch.bool)
    t._outcomes(done, done.clone())  # First clean arrival, no physical termination.
    assert t.curriculum_stage.item() == expected
    before = t.curriculum_completed.clone()
    t._outcomes(done, done.clone())
    torch.testing.assert_close(t.curriculum_completed, before)
