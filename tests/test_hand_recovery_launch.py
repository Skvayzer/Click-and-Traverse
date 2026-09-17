"""Recovery profile and launcher wiring using real storage and fake PPO updates."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import train_cat_wholebody as launcher
from cat_ppo.furniture.checkpoint import BestCheckpointStore
from cat_ppo.furniture.generalist_config import training_config
from cat_ppo.furniture.generalist_runtime import atomic_save_runtime, load_runtime
from cat_ppo.furniture.retention_validation import MODES, ValidationResult, retention_selection
from test_hand_protection_launch import archive_fixture
from test_recovery_guard import evidence as recovery_evidence, _set_outcomes
from test_stabilized_launch import writer


def test_recovery_profile_limits_noise_and_extends_leg_preservation_without_changing_tasks(tmp_path):
    original = training_config(finetuning="hand_protection", num_envs=16384)
    config = training_config(finetuning="hand_recovery", num_envs=16384)
    assert config["policy_config"]["learning_rate"] == 1e-5
    assert config["policy_config"]["entropy_cost"] == 0.
    fine = config["fine_tuning"]
    assert fine["retention_validation"]["interval_updates"] == 10
    assert fine["reference_kl"]["action_indices"] == list(range(12))
    assert fine["action_distribution"] == original["fine_tuning"]["action_distribution"]
    assert config["env_config"] == original["env_config"]
    assert fine["effective_batch_geometry"] == original["fine_tuning"]["effective_batch_geometry"]
    env = SimpleNamespace(field_bank_manifest={"scenes": [
        dict(family="original_cat", task_kind="cat"),
        dict(family="published_cat", task_kind="cat"),
        dict(family="procedural_cat", task_kind="cat"),
        dict(family="furniture", task_kind="room"),
        dict(family="generic_clutter", task_kind="room"),
    ]})
    assert launcher.reference_kl_config(env, config)["scene_mask"] == [True] * 5
    assert launcher.reference_kl_config(env, original)["scene_mask"] == [True, True, True, False, False]
    args = launcher.parser().parse_args(["plan", "--finetuning", "hand_recovery"])
    with pytest.raises(ValueError, match="verified selected-best"):
        launcher.plan(args)


@pytest.fixture
def recovery_run(tmp_path, monkeypatch, recovery_evidence):
    source_result, source_selection = recovery_evidence
    for summary in source_result["modes"].values():
        summary.update(clutter_hand_violation_rate=.1, clutter_mean_min_hand_clearance_m=.08)
    source_selection["metrics"]["selection"] = retention_selection(source_result["modes"], source_result["modes"])
    archive = tmp_path / "protected"
    archive_fixture(archive)
    selection_path = archive / "checkpoint/selection.json"
    source_selection["files"] = json.loads(selection_path.read_text())["files"]
    selection_path.write_text(json.dumps(source_selection))
    backup_path = archive / "backup.json"
    backup = json.loads(backup_path.read_text())
    backup["source_checkpoint_step"] = source_selection["step"]
    backup_path.write_text(json.dumps(backup))
    bank = tmp_path / "bank.json"
    bank.write_text("{}")
    args = launcher.parser().parse_args(["run", "--bank-manifest", str(bank),
        "--run-dir", str(tmp_path / "run"), "--wandb-mode", "disabled",
        "--finetuning", "hand_recovery", "--warmstart-best", str(archive)])
    spec = launcher.plan(args)
    events = dict(preparations=[], evaluations=[], learner_calls=[], requests=[],
                  intervals_per_call=1, schedule={}, numerical_steps=set(), fail_before_runtime=False)
    factory = object()
    scenes = []
    for kind_index, kind in enumerate(("hand_table_aisle", "hand_shelf_passage")):
        for level in range(3):
            scenes.append(dict(scene_id=f"hand-{kind_index * 3 + level}", family="furniture",
                task_kind="room", source=dict(hand_protection=dict(kind=kind, level=level))))
    scenes.append(dict(scene_id="cat", family="original_cat", task_kind="cat"))
    interval = (spec["config"]["fine_tuning"]["effective_batch_geometry"]["transitions_per_update"]
                * spec["config"]["fine_tuning"]["retention_validation"]["interval_updates"])
    events["interval"] = interval

    def prepare(args, specification, *, restore_model=True):
        events["preparations"].append(restore_model)
        record = deepcopy(specification)
        record.update(bank_sha256="bank", code={"git_commit": "commit", "source_sha256": "source"},
            observation_contract={"actor_features": ["a"], "critic_features": ["b"], "action_names": ["joint"]},
            warmstart={"source": "protected-26m"} if restore_model else None)
        env = SimpleNamespace(field_bank_manifest={"scenes": scenes})
        return env, factory, "initial" if restore_model else None, record

    class Validator:
        def __init__(self, environment, supplied_factory, *, seeds, scene_ids):
            assert supplied_factory is factory and list(seeds) == list(range(16))
            assert list(scene_ids) == source_result["metadata"]["scene_ids"]

        def evaluate(self, params, *, step, baseline=None):
            events["evaluations"].append(dict(params=params, step=step, baseline=deepcopy(baseline)))
            result = deepcopy(source_result)
            result["step"] = step
            outcomes = events["schedule"].get(step // interval, {})
            for mode in MODES:
                _set_outcomes(result, mode, **outcomes)
                result["modes"][mode].update(clutter_hand_violation_rate=.1,
                                            clutter_mean_min_hand_clearance_m=.08)
            if step in events["numerical_steps"]:
                result["modes"]["stochastic"]["numerical_failure_rate"] = 1 / 352
            selection = retention_selection(result["modes"], baseline) if baseline is not None else None
            metrics = {}
            for mode, summary in result["modes"].items():
                prefix = "validation/" if mode == "deterministic" else "validation/stochastic/"
                metrics.update({prefix + key: value for key, value in summary.items() if key != "scenes"})
            return ValidationResult(step, metrics, result["modes"], {}, selection, result["metadata"])

    def learner(**kwargs):
        events["learner_calls"].append(kwargs)
        if events["fail_before_runtime"]:
            events["fail_before_runtime"] = False
            raise RuntimeError("simulated pre-runtime failure")
        restored = kwargs["restore_runtime_state"]
        previous_step = restored["step"] if restored else 0
        def save(step):
            kwargs["runtime_checkpoint_fn"](step, dict(schema="cat-ppo-runtime-v1", step=step,
                contract={"metadata": kwargs["runtime_metadata"]}))
        if restored is None:
            save(0)
        for current in range(previous_step + interval,
                             previous_step + (events["intervals_per_call"] + 1) * interval, interval):
            metrics = {"training/rollout_reward_mean": .3}
            kwargs["progress_fn"](current, metrics)
            kwargs["scored_checkpoint_fn"](current, None, f"learned:{current}", {}, metrics, "training_proxy")
            events["requests"].append(kwargs["recovery_fn"](current))
            save(current)
        return None, None, {"training/completed_steps": current}

    monkeypatch.setattr(launcher, "prepare", prepare)
    monkeypatch.setattr(importlib.import_module("cat_ppo.furniture.retention_validation"), "RetentionValidator", Validator)
    monkeypatch.setattr(importlib.import_module("cat_ppo.learning.policy.ppo.train"), "train", learner)
    monkeypatch.setattr(importlib.import_module("cat_ppo.furniture.checkpoint"), "native_writer",
                        lambda params, network_config, step: writer(f"{params}".encode(), network_config))
    monkeypatch.setattr(importlib.import_module("cat_ppo.furniture.learning"), "load_native",
                        lambda path: ["restored normalizer", "restored actor", "restored critic"])
    return args, spec, events


def test_two_failed_evaluations_survive_resume_and_restore_source_in_same_run(recovery_run):
    args, spec, events = recovery_run
    events["schedule"] = {1: dict(hand=0), 2: dict(hand=0)}
    launcher.run(args, spec)
    first = load_runtime(args.run_dir / "resume.msgpack")
    assert first["recovery_state"]["consecutive_failures"] == 1
    assert events["requests"] == [None]
    identity = json.loads((args.run_dir / "wandb.json").read_text())["id"]
    baseline = (args.run_dir / "validation_baseline.json").read_bytes()
    args.resume = True
    launcher.run(args, spec)
    request = events["requests"][-1]
    assert tuple(request["params"]) == ("restored normalizer", "restored actor", "restored critic")
    assert request["checkpoint"] == str(args.warmstart_best / "checkpoint/native")
    assert request["learning_rate"] == pytest.approx(5e-6)
    runtime = load_runtime(args.run_dir / "resume.msgpack")
    assert runtime["step"] == 2 * events["interval"]
    assert runtime["recovery_state"]["consecutive_failures"] == 0
    assert runtime["recovery_state"]["recovery_count"] == 1
    assert runtime["recovery_state"]["last_report"]["status"] == "recovered"
    assert events["preparations"] == [True, False]
    assert events["learner_calls"][-1]["restore_params"] is None
    assert events["learner_calls"][-1]["restore_runtime_state"]["recovery_state"]["consecutive_failures"] == 1
    assert json.loads((args.run_dir / "wandb.json").read_text())["id"] == identity
    assert (args.run_dir / "validation_baseline.json").read_bytes() == baseline
    assert sum(row["step"] == 0 for row in events["evaluations"]) == 1
    assert BestCheckpointStore.open_existing(args.run_dir).selected() is None


def test_published_passing_best_becomes_recovery_anchor_and_repeated_recoveries_lower_lr(recovery_run):
    args, spec, events = recovery_run
    events["intervals_per_call"] = 5
    events["schedule"] = {1: dict(hand=70), **{index: dict(hand=0) for index in range(2, 6)}}
    source_hash = hashlib.sha256((args.warmstart_best / "checkpoint/selection.json").read_bytes()).hexdigest()
    launcher.run(args, spec)
    store = BestCheckpointStore.open_existing(args.run_dir)
    assert store.selected()["step"] == events["interval"]
    requests = [request for request in events["requests"] if request is not None]
    assert len(requests) == 2
    assert all(request["checkpoint"] == str(store.best / "native") for request in requests)
    assert [request["learning_rate"] for request in requests] == pytest.approx([5e-6, 2.5e-6])
    state = load_runtime(args.run_dir / "resume.msgpack")["recovery_state"]
    assert state["anchor"]["checkpoint"] == str(store.best / "native")
    assert state["baseline"]["checkpoint"] == str(args.warmstart_best / "checkpoint/native")
    assert state["recovery_count"] == 2
    assert len((args.run_dir / "recovery_events.jsonl").read_text().splitlines()) == 2
    assert hashlib.sha256((args.warmstart_best / "checkpoint/selection.json").read_bytes()).hexdigest() == source_hash


@pytest.mark.parametrize("first_hand", [59, 58])
def test_tied_or_weaker_first_passing_candidate_cannot_replace_source_anchor(recovery_run, first_hand):
    args, spec, events = recovery_run
    events["intervals_per_call"] = 3
    events["schedule"] = {1: dict(hand=first_hand), 2: dict(hand=0), 3: dict(hand=0)}
    launcher.run(args, spec)
    assert BestCheckpointStore.open_existing(args.run_dir).selected() is None
    assert events["requests"][0] is None
    assert events["requests"][-1]["checkpoint"] == str(args.warmstart_best / "checkpoint/native")
    state = load_runtime(args.run_dir / "resume.msgpack")["recovery_state"]
    assert state["anchor"] == state["baseline"]


def test_numerical_evaluation_failure_recovers_on_first_failure(recovery_run):
    args, spec, events = recovery_run
    events["numerical_steps"].add(events["interval"])
    launcher.run(args, spec)
    assert len(events["requests"]) == 1 and events["requests"][0] is not None
    state = load_runtime(args.run_dir / "resume.msgpack")["recovery_state"]
    assert state["recovery_count"] == 1 and state["last_report"]["numerical_failure"]
    assert BestCheckpointStore.open_existing(args.run_dir).selected() is None


def test_repeated_recovery_learning_rate_respects_floor(recovery_run):
    args, spec, events = recovery_run
    events["intervals_per_call"] = 10
    events["schedule"] = {index: dict(hand=0) for index in range(1, 11)}
    launcher.run(args, spec)
    rates = [request["learning_rate"] for request in events["requests"] if request is not None]
    assert rates == pytest.approx([5e-6, 2.5e-6, 1.25e-6, 1e-6, 1e-6])


@pytest.mark.parametrize("corruption", ["missing_recovery_state", "changed_baseline", "unreadable_runtime"])
def test_resume_cannot_silently_reset_recovery_history_or_fresh_start(recovery_run, corruption):
    args, spec, events = recovery_run
    launcher.run(args, spec)
    runtime_path = args.run_dir / "resume.msgpack"
    snapshot = load_runtime(runtime_path)
    if corruption == "unreadable_runtime":
        runtime_path.write_bytes(b"invalid msgpack")
    else:
        if corruption == "missing_recovery_state":
            del snapshot["recovery_state"]
        else:
            snapshot["recovery_state"]["baseline"]["modes"]["deterministic"]["cat_goal_success_rate"] = 0.
        atomic_save_runtime(runtime_path, snapshot)
    args.resume = True
    with pytest.raises((ValueError, TypeError)):
        launcher.run(args, spec)
    assert len(events["learner_calls"]) == 1
    assert events["preparations"] == ([True] if corruption == "unreadable_runtime" else [True, False])


def test_pre_runtime_failure_retries_same_source_baseline_and_logging_identity(recovery_run):
    args, spec, events = recovery_run
    events["fail_before_runtime"] = True
    with pytest.raises(RuntimeError, match="pre-runtime"):
        launcher.run(args, spec)
    assert not (args.run_dir / "resume.msgpack").exists()
    identity = json.loads((args.run_dir / "wandb.json").read_text())["id"]
    baseline = (args.run_dir / "validation_baseline.json").read_bytes()
    args.resume = True
    result = launcher.run(args, spec)
    assert result["startup_retry"]
    assert events["preparations"] == [True, True]
    assert json.loads((args.run_dir / "wandb.json").read_text())["id"] == identity
    assert (args.run_dir / "validation_baseline.json").read_bytes() == baseline
    assert sum(row["step"] == 0 for row in events["evaluations"]) == 1


def test_source_comparison_rejects_weaker_startup_before_any_learner_update(recovery_run):
    args, spec, events = recovery_run
    events["schedule"][0] = dict(hand=0)
    with pytest.raises(ValueError, match="failed protected source-best retention"):
        launcher.run(args, spec)
    assert not events["learner_calls"]
    assert not (args.run_dir / "resume.msgpack").exists()
    assert not json.loads((args.run_dir / "source_best_startup_guard.json").read_text())["eligible"]
