"""Logging-only diagnostics, physical leader filtering and legacy history."""
from dataclasses import replace
import pytest
import torch

from cat_mjlab.learning import Learner, ratio_diagnostics
from cat_mjlab.runner import SuccessWindow, collect_rollout
from test_mjlab_runner import FakeTask, tiny_config


def test_dense_window_weighting_eviction_and_legacy_restore():
    window = SuccessWindow(maxlen=2)
    nav, contrast = torch.zeros(4, 2), torch.zeros(3, 2)
    key = 'training/leader_hand_contrast_heading_compliance_fraction'
    window.append(nav, contrast, dense={key: torch.tensor([1., 2.])})
    legacy = window.state_dict()
    legacy.pop('dense')
    restored = SuccessWindow(maxlen=2)
    restored.load_state_dict(legacy)
    assert key not in restored.metrics()
    restored.append(nav, contrast, dense={key: torch.tensor([3., 4.])})
    assert restored.metrics()[key] == .75
    window.append(nav, contrast, dense={key: torch.tensor([3., 4.])})
    assert window.metrics()[key] == pytest.approx(4 / 6)
    assert window.metrics()['training/checkpoint_selection_window_length'] == 2
    window.append(nav, contrast, dense={key: torch.tensor([0., 2.])})
    assert window.metrics()[key] == .5
    restored.load_state_dict(window.state_dict())
    assert restored.metrics() == window.metrics()


def test_ratio_groups_clip_bounds_and_extreme_importance_ess():
    config = tiny_config()
    data = dict(policy_id=torch.tensor([0, 1, 1, 1]), target_policy_id=torch.tensor([0, 1, 0, 0]),
                log_prob=torch.zeros(4), old_target_log_prob=torch.zeros(4),
                log_importance=torch.tensor([0., 0., 1000., -1000.]))
    current = torch.tensor([0., 1., -1., 0.], requires_grad=True)
    metrics = ratio_diagnostics(current, data, config)
    prefix = 'diagnostics/relabeled_leader/'
    assert metrics[prefix + 'importance_ess'] == 1
    assert metrics[prefix + 'importance_ess_fraction'] == .5
    assert metrics[prefix + 'ppo_ratio_clip_fraction'] == .5
    assert metrics[prefix + 'importance_log_weight_max'] == 1000
    assert metrics['diagnostics/on_policy_leader/ppo_ratio_clip_fraction'] == 0
    assert metrics['diagnostics/followers/ppo_ratio_clip_fraction'] == 1
    assert all(torch.isfinite(v) and not v.requires_grad for v in metrics.values())


def test_collector_leader_role_denominators_and_joint_split():
    class Task(FakeTask):
        def step(self, action):
            result = super().step(action)
            result['done'] = torch.ones(6, dtype=torch.bool)
            result['truncated'] = torch.tensor([False, True, True, True, True, True])
            result['metrics'].update({
                'episode/body_collision': torch.tensor([True, False, False, False, False, False]),
                'contrast_role': torch.tensor([1, 2, 1, 2, 1, 2]),
                'protected_core': torch.tensor([True, False, True, False, True, False]),
                'reward_floor_clipped': torch.tensor([True, False, False, False, False, False]),
                'hand_scene_kind': torch.tensor([1, 2, 1, 2, 1, 2]),
                'contrast_unresolved': torch.ones(6, dtype=torch.bool),
                'contrast_heading_active': torch.ones(6, dtype=torch.bool),
                'contrast_hand_active': torch.ones(6, dtype=torch.bool),
                'hand_contrast_heading_good': torch.tensor([1., 0., 1., 1., 1., 1.]),
                'hand_contrast_hand_good': torch.tensor([0., 1., 1., 1., 1., 1.]),
                'hand_contrast_heading_cost': torch.tensor([.2, .4, 1., 1., 1., 1.]),
                'hand_contrast_region_cost': torch.tensor([.4, .6, 1., 1., 1., 1.]),
                'contrast_zone_seen': torch.tensor([1, 0, 1, 1, 1, 1]),
                'contrast_zone_required': torch.ones(6, dtype=torch.long),
            })
            return result
    learner = Learner(replace(tiny_config(), action_size=29), device='cpu')
    original_act = learner.act
    def act(*args, **kwargs):
        result = original_act(*args, **kwargs)
        result['action_std'][:, :12] = 2
        result['action_std'][:, 12:] = 4
        return result
    learner.act = act
    _, collected = collect_rollout(Task(), learner, unroll_length=1, trajectories=6,
                                    policy_ids=torch.tensor([0, 0, 1, 1, 2, 2]))
    metrics = collected['metrics']
    assert metrics['training/body_collision_rate'] == pytest.approx(1/6)
    assert metrics['training/leader_body_collision_rate'] == .5
    assert metrics['training/leader_forward_protected_body_collision_rate'] == 1
    assert metrics['training/leader_narrow_passage_timeout_rate'] == 1
    assert metrics['training/action_std_legs'] == 2
    assert metrics['training/action_std_upper_body'] == 4
    window = SuccessWindow()
    window.append(collected['navigation_counts'], collected['contrast_counts'], collected['role_counts'], collected['dense'])
    metrics = window.metrics()
    assert metrics['training/protected_core_reward_floor_clipping_fraction'] == 1.
    assert metrics['training/protected_core_reward_floor_clipping_fraction_sample_count'] == 1
    assert metrics['selection/protected_core_compliant'] == 0.
    assert metrics['training/leader_hand_contrast_heading_compliance_fraction'] == .5
    assert metrics['training/leader_hand_contrast_zone_reached_progress_ratio'] == .5
    assert metrics['training/leader_hand_contrast_mean_heading_cost'] == pytest.approx(.3)
    assert metrics['training/leader_table_hand_contrast_hand_region_compliance_fraction'] == 0
    assert metrics['training/leader_shelf_hand_contrast_hand_region_compliance_fraction'] == 1


def test_diagnostics_do_not_change_loss_gradients_or_rng(monkeypatch):
    import cat_mjlab.learning as learning
    learner = Learner(tiny_config(), device='cpu')
    rollout, _ = collect_rollout(FakeTask(), learner, unroll_length=2, trajectories=6,
                                policy_ids=torch.tensor([0, 0, 1, 1, 2, 2]))
    data = learning.prepare_sapg_rollout(learner.model, rollout, learner.config, follower_id=1)
    rng = torch.get_rng_state()
    loss, metrics = learning.compute_loss(learner.model, data, learner.config)
    loss.backward()
    gradients = [p.grad.clone() for p in learner.model.parameters()]
    after_rng = torch.get_rng_state()
    learner.model.zero_grad()
    torch.set_rng_state(rng)
    monkeypatch.setattr(learning, 'ratio_diagnostics', lambda *args: {})
    plain_loss, _ = learning.compute_loss(learner.model, data, learner.config)
    plain_loss.backward()
    torch.testing.assert_close(loss, plain_loss, rtol=0, atol=0)
    assert torch.equal(after_rng, torch.get_rng_state())
    for expected, parameter in zip(gradients, learner.model.parameters()):
        torch.testing.assert_close(expected, parameter.grad, rtol=0, atol=0)


def test_narrow_progress_counts_valid_zones_without_posture_requirements():
    from types import SimpleNamespace
    from cat_mjlab.task import CATTask
    task = object.__new__(CATTask)
    task.scene_ids = torch.tensor([0, 0, 1])
    task.bank = SimpleNamespace(contrast={
        'zone_valid': torch.tensor([[True, True, True, False, False, False],
                                    [True, True, True, False, False, False]]),
        'enabled': torch.tensor([True, False]),
    })
    task.zone_steps = torch.zeros(3, 6, 3, dtype=torch.long)
    task.zone_steps[0, [0, 4], 0] = 1  # A padded slot must not count.
    task.zone_steps[1, :3, 0] = 1
    task.zone_steps[2, :3, 0] = 1  # Disabled scenes must not count.
    seen, valid = task._zone_progress_counts()
    assert seen.tolist() == [1, 3, 0]
    assert valid.tolist() == [3, 3, 0]


def test_per_body_part_logging_preserves_role_seeded_and_policy_masks():
    class Task(FakeTask):
        def step(self, action):
            result = super().step(action)
            result['done'][:] = True
            m = result['metrics']
            m.update(contrast_role=torch.tensor([1, 1, 1, 2, 1, 1]),
                hand_raised_seeded=torch.tensor([True, False, True, False, False, True]),
                contrast_unresolved=torch.ones(6, dtype=torch.bool),
                contrast_heading_active=torch.ones(6, dtype=torch.bool),
                contrast_hand_active=torch.ones(6, dtype=torch.bool),
                contrast_zone_seen=torch.zeros(6), contrast_zone_required=torch.ones(6))
            m['episode_length'] = torch.tensor([101, 99, 200, 200, 200, 200])
            for key in ('heading_good', 'hand_good', 'heading_cost', 'region_cost'):
                m['hand_contrast_'+key] = torch.zeros(6)
            for j, part in enumerate(('feet', 'legs', 'trunk', 'head', 'arms', 'hands')):
                m['episode/body_collision_'+part] = torch.tensor([True, j == 5, True, True, True, True])
            return result
    learner = Learner(tiny_config(), device='cpu')
    _, info = collect_rollout(Task(), learner, unroll_length=1, trajectories=6,
                             policy_ids=torch.tensor([0, 0, 1, 1, 2, 2]))
    m = info['metrics']
    assert m['training/body_collision_hands_count'] == 6
    assert m['training/leader_body_collision_hands_count'] == 2
    assert m['training/leader_body_collision_arms_rate'] == .5
    for prefix in ('leader_forward_protected_seeded_', 'leader_forward_protected_unseeded_',
                   'leader_forward_protected_seeded_after100_'):
        assert m['training/'+prefix+'body_collision_hands_count'] == 1
        assert m['training/'+prefix+'body_collision_hands_rate'] == 1
