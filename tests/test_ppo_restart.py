"""CPU regression coverage for native SAPG extraction and flat selection."""
from dataclasses import asdict, replace
import pytest
import torch
from cat_mjlab.conversion import extract_native_leader
from cat_mjlab.learning import Learner, LearnerConfig
from cat_mjlab.runner import SuccessWindow


def test_leader_extraction_preserves_both_trunks_and_scale():
    torch.manual_seed(17)
    cfg = LearnerConfig(actor_obs=3, critic_obs=4, action_size=2,
                        actor_hidden=(8,), critic_hidden=(9,), embedding_dim=2)
    source = Learner(cfg, device='cpu')
    with torch.no_grad():
        source.model.actor.layers[0].weight[:, -2:].normal_()
        source.model.critic.layers[0].weight[:, -2:].normal_()
        source.model.policy_embeddings.normal_()
    weights = source.model.state_dict()
    before = {k: v.clone() for k, v in weights.items()}
    target = Learner(replace(cfg, algorithm='ppo', num_policies=1), device='cpu')
    target.model.load_state_dict(extract_native_leader(weights, asdict(cfg)), strict=True)
    x, y = torch.randn(21, 3), torch.randn(21, 4)
    torch.testing.assert_close(target.model.logits(x), source.model.logits(x, 0))
    torch.testing.assert_close(target.model.value(y), source.model.value(y, 0))
    for key in weights:
        torch.testing.assert_close(weights[key], before[key])
    assert not target.optimizer.state
    assert target.model.policy_embeddings is None
    with pytest.raises(ValueError, match='embedding shape'):
        extract_native_leader(weights, dict(asdict(cfg), num_policies=5))


def append(window, strict, soft, resolved, success, steps=100):
    navigation = torch.zeros(4, 2, dtype=torch.long)
    navigation[0] = torch.tensor([resolved, success])
    window.append(navigation, torch.zeros(3, 2, dtype=torch.long), dense={
        'selection/flat_walking_compliant': torch.tensor([strict * steps, steps]),
        'selection/flat_soft_progress': torch.tensor([soft * steps, steps])})


def test_score_missing_evidence_weighting_monotonicity_and_restore():
    empty = SuccessWindow()
    assert empty.legacy_flat_balance_score() is None
    append(empty, 0, .4, 0, 0)
    assert empty.legacy_flat_balance_score() is None
    scores = []
    for strict, soft, retention in [(0, .4, 2), (0, .5, 2), (.1, .5, 2), (.1, .5, 3)]:
        window = SuccessWindow()
        append(window, strict, soft, 10, retention)
        scores.append(window.legacy_flat_balance_score())
    assert scores == sorted(set(scores))
    assert scores[0] == pytest.approx(.15)
    append(window, .5, .9, 30, 21, steps=300)
    assert window.legacy_flat_balance_score() == pytest.approx(.5*.4 + .25*.8 + .25*.6)
    restored = SuccessWindow()
    restored.load_state_dict(window.state_dict())
    assert restored.legacy_flat_balance_score() == window.legacy_flat_balance_score()


def test_native_ppo_runner_updates_best_and_names_population(tmp_path, monkeypatch):
    import json
    from cat_mjlab import runner
    from train_cat_mjlab import parser
    from test_mjlab_runner import FakeTask, FakeSimulation, tiny_config
    cfg = tiny_config()
    source = Learner(cfg, device='cpu')
    checkpoint = tmp_path / 'source.pt'
    torch.save(dict(schema='cat-mjlab-best-v1', config=asdict(cfg),
                    model=source.model.state_dict(), contract={'environment_config': {}}), checkpoint)
    asset = tmp_path / 'asset.json'
    asset.write_text('{}')
    def factory(args, **kwargs):
        task = FakeTask(args.num_envs)
        task.balance_settings = {}
        return task, FakeSimulation(), {'checkpoint_selection_weights':(.6,.2,.1,.1)}
    monkeypatch.setattr(runner, 'create_task', factory)
    original_collect = runner.collect_rollout
    count = 0
    def collect(task, *args, **kwargs):
        nonlocal count
        task.balance_settings = None
        rollout, info = original_collect(task, *args, **kwargs)
        task.balance_settings = {}
        count += 1
        info['navigation_counts'][0] = torch.tensor([10, count])
        info['dense']['selection/protected_core_compliant']=torch.tensor([count*10.,100.])
        info['dense'].update({'selection/flat_walking_compliant': torch.tensor([0., 100.]),
                             'selection/flat_soft_progress': torch.tensor([count * 10., 100.])})
        return rollout, info
    monkeypatch.setattr(runner, 'collect_rollout', collect)
    directory = tmp_path / 'run'
    args = parser().parse_args(['verify', '--checkpoint-native', str(checkpoint), '--fresh-optimizer',
        '--algorithm', 'ppo', '--num-envs', '6', '--batch-size', '3', '--num-minibatches', '2',
        '--unroll-length', '2', '--max-updates', '2', '--device', 'cpu',
        '--bank-manifest', str(asset), '--body-collision-bank', str(asset),
        '--body-collision-resets', str(asset), '--run-dir', str(directory)])
    runner.run(args)
    rows = [json.loads(row) for row in (directory / 'metrics.jsonl').read_text().splitlines()]
    assert rows[1]['training/checkpoint_selection_score'] > rows[0]['training/checkpoint_selection_score'] > 0
    best = torch.load(directory / 'best.pt', weights_only=True)
    assert best['step'] == 24
    assert best['config']['num_policies'] == 1
    assert 'policy_embeddings' not in best['model']
    assert rows[-1]['training/metric_population_envs'] == 6
    assert 'success/policy_cat_goal_success_rate' in rows[-1]
    assert not any('leader_' in key or '/followers/' in key for key in rows[-1])
