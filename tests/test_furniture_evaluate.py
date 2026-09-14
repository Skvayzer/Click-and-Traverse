"""Evaluator accounting fixtures; no MuJoCo, inference, or performance episodes."""

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import evaluate_furniture as evaluation


def _scene():
    return {"scene_id": "fixture", "start": [0, 0, 0], "goal": [1, 0],
            "route": [[0, 0], [1, 0]], "time_budget": 1, "bottlenecks": []}


def _fake_jax(monkeypatch):
    fake = SimpleNamespace(jit=lambda fn: fn, device_get=lambda value: value,
                           random=SimpleNamespace(PRNGKey=lambda key: key, split=lambda key: (key + 1, key + 2)))
    monkeypatch.setitem(sys.modules, "jax", fake)
    return fake


def _record(monkeypatch, *, position=(.1, 0), **updates):
    _fake_jax(monkeypatch)
    info = dict(numerical_failure=False, furniture_contact=False, hand_contact=False,
                fall=False, self_collision=False, nonfoot_floor_contact=False,
                goal_reached=False, minimum_clearance=.2, first_contact_part=0, first_contact_time=-1.)
    info.update(updates)
    initial = SimpleNamespace(obs={})
    returned = SimpleNamespace(obs={}, info=info, data=SimpleNamespace(qpos=np.asarray(position)), done=True)
    env = SimpleNamespace(reset=lambda key: initial, step=lambda state, action: returned,
                          dt=.02, _config=SimpleNamespace(goal_radius=.25))
    return evaluation.run_episode(env, lambda obs, key: np.zeros(29), scene=_scene(),
        case_id="case", training_seed=0, episode_seed=0, controller="fixture")


@pytest.mark.parametrize("physical_fall", [False, True])
def test_numerical_failure_preserves_real_fall_flag(monkeypatch, physical_fall):
    record = _record(monkeypatch, numerical_failure=True, fall=physical_fall)
    assert record.numerical_failure and record.fall == physical_fall
    assert not record.strict_success and record.termination_reason == "numerical_failure"


def test_invalid_position_keeps_contact_evidence_and_substep_timestamp(monkeypatch):
    record = _record(monkeypatch, position=(float("nan"), 0), hand_contact=True,
                     furniture_contact=True, first_contact_part=2, first_contact_time=.005)
    assert record.numerical_failure and not record.fall
    assert record.hand_contact and record.furniture_contact
    assert record.first_contact_time == .005 and record.first_contact_part == "hand"
    assert record.path_length == 0


def test_false_goal_flag_cannot_produce_success_and_valid_arrival_can(monkeypatch):
    far = _record(monkeypatch, goal_reached=True)
    assert not far.strict_success and far.termination_reason == "environment_termination"
    arrived = _record(monkeypatch, position=(1, 0), goal_reached=True)
    assert arrived.strict_success and arrived.termination_reason == "goal"


def test_source_provenance_uses_repository_cwd_and_hashes_untracked_sources(tmp_path, monkeypatch):
    repo = tmp_path / "repository"
    package = repo / "cat_ppo"
    package.mkdir(parents=True)
    script = repo / "evaluate_furniture.py"
    script.write_text("# evaluation source\n")
    source = package / "new_untracked_module.py"
    source.write_text("VALUE = 1\n")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    monkeypatch.chdir(foreign)
    monkeypatch.setattr(evaluation, "__file__", str(script))
    calls = []

    def run(command, *, cwd, capture_output, check):
        calls.append((command, Path(cwd)))
        return SimpleNamespace(stdout=b"0123456789\n" if command[1] == "rev-parse" else b"tracked diff")

    monkeypatch.setattr(evaluation.subprocess, "run", run)
    original = evaluation._code_provenance()
    assert all(cwd == repo for _, cwd in calls)
    assert original["git_commit"] == "0123456789"
    assert original["source_files"]["cat_ppo/new_untracked_module.py"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert original["tracked_dirty_diff_sha256"] == hashlib.sha256(b"tracked diff").hexdigest()
    source.write_text("VALUE = 2\n")
    assert evaluation._code_provenance()["source_snapshot_sha256"] != original["source_snapshot_sha256"]


def test_evaluator_sets_and_records_highest_matmul_precision(monkeypatch):
    fake = _fake_jax(monkeypatch)
    config = SimpleNamespace(jax_default_matmul_precision="default")
    config.update = lambda key, value: setattr(config, key, value)
    fake.config = config
    assert evaluation._configure_numerics() == {"jax_default_matmul_precision": "highest"}


def test_eval_overrides_are_recursive_and_reject_unknown_fields():
    config = {"perception_mode": "oracle", "noise": {"level": .1, "other": 2}}
    evaluation._apply_environment_overrides(config, {"noise": {"level": .2}})
    assert config["noise"] == {"level": .2, "other": 2}
    with pytest.raises(ValueError, match="Unknown"):
        evaluation._apply_environment_overrides(config, {"percepton_mode": "corrupted"})
    with pytest.raises(ValueError, match="Unknown"):
        evaluation._apply_environment_overrides(config, {"noise": {"typo": 1}})


def test_run_and_best_link_resolve_same_checkpoint_without_cleanup(tmp_path):
    from cat_ppo.furniture.checkpoint import BestCheckpointStore
    run = tmp_path / "run"
    store = BestCheckpointStore(run)
    def writer(path):
        path.mkdir()
        (path / "ppo_network_config.json").write_text("{}")
        (path / "weights.bin").write_bytes(b"actor and critic fixture")
    store.consider(step=1, metrics={"proxy_score": 1}, source="training_proxy",
                   write_checkpoint=writer, contract={"action_names": ["joint"]})
    stale = store.generations / "candidate-unused"
    stale.mkdir()
    (stale / "owner.json").write_text('{"owner": "' + store.owner + '"}')
    native, selection = evaluation._resolve_checkpoint(run)
    native_from_link, link_selection = evaluation._resolve_checkpoint(run / "checkpoints" / "best")
    assert native == native_from_link and selection == link_selection
    assert stale.exists()
