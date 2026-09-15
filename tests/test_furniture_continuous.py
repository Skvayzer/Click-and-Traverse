"""Continuous orchestration tests use real manifests and fake training only."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from cat_ppo.furniture import continuous, curriculum
from cat_ppo.furniture.checkpoint import BestCheckpointStore
from cat_ppo.furniture.scenes import generate_scene, load_scene, write_scene_bundle
from cat_ppo.furniture.scenes import _digest


def test_stage_stream_preserves_rehearsal_and_keeps_generating_dense_seeds():
    seeds = set()
    for round_index in range(10):
        stages = [continuous.stage_spec(12 * round_index + offset, 0) for offset in range(12)]
        domains = [stage["scene"]["domain"] for stage in stages]
        assert domains.count("legacy") == 6
        assert domains.count("generic") == domains.count("furniture") == 3
        for stage in stages[1::2]:
            assert stage["scene"]["seed"] not in seeds
            seeds.add(stage["scene"]["seed"])
            assert stage["scene"]["difficulty"] == ("pilot" if round_index == 0 else "dense")
            assert (stage["environment_config"].get("perception_mode") == "corrupted") == (round_index >= 2)
        assert all(stage["scene"]["difficulty"] <= .8 for stage in stages[::2])
    config = continuous.build_config(num_envs=256)
    assert config["global_step_limit"] is config["global_stage_limit"] is config["global_time_limit"] is None
    assert config["wandb_mode"] == "online"


def test_scene_batch_overrides_respect_dense_and_random_capacity():
    config = continuous.build_config(num_envs=256, legacy_num_envs=1024, pilot_num_envs=1024)
    assert continuous.stage_num_envs(config, continuous.stage_spec(0, 0)) == 1024
    assert continuous.stage_num_envs(config, continuous.stage_spec(1, 0)) == 1024
    assert continuous.stage_num_envs(config, continuous.stage_spec(8, 0)) == 512
    assert continuous.stage_num_envs(config, continuous.stage_spec(13, 0)) == 256
    assert continuous.stage_num_envs(config, continuous.stage_spec(25, 0)) == 256
    with pytest.raises(ValueError, match="divisible"):
        continuous.build_config(num_envs=256, legacy_num_envs=1028)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    config = continuous.build_config(steps_per_stage=16, num_envs=4, unroll_length=2,
                                    checkpoint_epochs=2, wandb_mode="disabled")
    bundle = write_scene_bundle(generate_scene(difficulty="open_floor"), tmp_path / "scene", voxel_size=.2)
    scene = load_scene(bundle)
    monkeypatch.setattr(curriculum, "materialize_stage", lambda spec, cache: bundle)
    calls, selected_paths, controls = [], [], {"stop_after": 2}
    run_dir = tmp_path / "run"

    def train(command, **kwargs):
        del kwargs
        args = dict(zip(command[2::2], command[3::2]))
        stage_index = len(calls)
        calls.append(command)
        directory = Path(args["--run-dir"])
        if controls.get("fail_at") == stage_index:
            directory.mkdir(parents=True)
            (directory / "partial.bin").write_bytes(b"preserved")
            raise subprocess.CalledProcessError(1, command)
        store = BestCheckpointStore(directory)

        def writer(path):
            path.mkdir()
            (path / "weights.bin").write_bytes(f"stage {stage_index} actor critic normalizer".encode())
            (path / "ppo_network_config.json").write_text("{}")

        source = {"kind": "pinned_public_native_checkpoint"}
        if "--warmstart" in args:
            native = Path(args["--warmstart"])
            source = dict(kind="explicit_local_native_checkpoint", path=str(native.resolve()),
                files={str(path.relative_to(native)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in native.rglob("*") if path.is_file()})
        run = dict(args={key.removeprefix("--").replace("-", "_"): value for key, value in args.items()},
            training_scene={key: scene[key] for key in ("scene_id", "geometry_hash", "split")},
            validation_scene=None, selection_source="training_proxy", source=source,
            warmstart={"critic_restored": True},
            environment_config=json.loads(Path(args["--env-config-json"]).read_text()))
        for key in ("steps", "seed", "num_envs", "unroll_length", "checkpoint_epochs", "num_evals", "global_step_offset"):
            run["args"][key] = int(run["args"][key])
        run["args"]["action_dofs"] = 29
        (directory / "run.json").write_text(json.dumps(run))
        if stage_index in controls.get("oom_at", ()):
            failure = dict(schema="cat-furniture-training-failure-v1", exception_type="XlaRuntimeError",
                message="RESOURCE_EXHAUSTED: GPU allocator out of memory", classification=controls.get("classification", "gpu_oom"),
                completed_steps=0, completed_update_evidence=False, metrics_present=False, metrics_last_step=None,
                selected_checkpoint=None, checkpoint_inspection_error=None, metrics_inspection_error=None,
                requested_steps=int(args["--steps"]), global_step_offset=int(args["--global-step-offset"]),
                stage=directory.name, run_dir=str(directory))
            if controls.get("oom_evidence") == "reported_steps":
                failure.update(completed_steps=1, completed_update_evidence=True)
            elif controls.get("oom_evidence") == "metrics":
                (directory / "metrics.jsonl").write_text('{"step": 1, "metrics": {}}\n')
            elif controls.get("oom_evidence") == "candidate":
                (directory / ".checkpoint-generations/candidate-partial").mkdir()
            elif controls.get("oom_evidence") == "uncertain":
                failure.update(metrics_inspection_error="cannot inspect metrics", completed_update_evidence=True)
            (directory / "failure.json").write_text(json.dumps(failure))
            raise subprocess.CalledProcessError(1, command)
        store.consider(step=8, metrics={"proxy_score": 1.0}, source="training_proxy",
                       write_checkpoint=writer, contract={"action_names": ["fixture_joint"]},
                       provenance={key: run.get(key) for key in ("warmstart", "source", "code", "training_scene", "validation_scene")})
        selected = store.selected()
        selected_paths.append(selected["path"])
        interrupted = controls.get("partial_stop_at") == stage_index
        budget = int(args["--steps"])
        summary = dict(requested_steps=budget, actual_steps=budget // 2 if interrupted else budget,
            stopped_by_request=interrupted, selected_step=8, selection_source="training_proxy",
            selected_score=selected["score"], actor_max_abs_parameter_update=.1,
            native_checkpoint=selected["path"])
        (directory / "summary.json").write_text(json.dumps(summary))
        if interrupted or len(calls) == controls["stop_after"]:
            continuous.request_stop(Path(args["--stop-file"]).parent)

    monkeypatch.setattr(continuous, "_run_child", train)
    return SimpleNamespace(config=config, run=run_dir, calls=calls, selected=selected_paths, controls=controls)


def test_runs_beyond_finite_curriculum_and_warmstarts_every_stage(fixture):
    fixture.controls["stop_after"] = 40
    state = continuous.execute(fixture.config, fixture.run)
    assert state["status"] == "stopped_by_request"
    assert len(state["completed"]) == len(fixture.calls) == 40
    assert state["completed_transitions"] == 640
    assert "--warmstart" not in fixture.calls[0]
    for index, command in enumerate(fixture.calls[1:], 1):
        assert command[command.index("--warmstart") + 1] == fixture.selected[index - 1]
        assert int(command[command.index("--global-step-offset") + 1]) == index * 16
        assert command[command.index("--num-evals") + 1] == "0"
    assert len(list(fixture.run.rglob("weights.bin"))) == 1
    assert len(list(fixture.run.rglob("curriculum_receipt.json"))) == 40
    assert json.loads((fixture.run / "runner.json").read_text())["active"] is False


def test_boundary_stop_can_continue_without_reinitializing(fixture):
    continuous.execute(fixture.config, fixture.run)
    previous = fixture.selected[-1]
    (fixture.run / "STOP").unlink()
    fixture.controls["stop_after"] = 3
    state = continuous.execute(fixture.config, fixture.run)
    assert len(state["completed"]) == len(fixture.calls) == 3
    command = fixture.calls[-1]
    assert command[command.index("--warmstart") + 1] == previous
    assert len(list(fixture.run.rglob("weights.bin"))) == 1


def test_batch_override_is_used_and_verified_at_each_handoff(fixture):
    fixture.config = continuous.build_config(steps_per_stage=32, num_envs=4, legacy_num_envs=8,
        pilot_num_envs=8, unroll_length=2, checkpoint_epochs=2, wandb_mode="disabled")
    fixture.controls["stop_after"] = 14
    state = continuous.execute(fixture.config, fixture.run)
    assert len(state["completed"]) == 14
    for index, command in enumerate(fixture.calls):
        expected = continuous.stage_num_envs(fixture.config, continuous.stage_spec(index, 0))
        assert int(command[command.index("--num-envs") + 1]) == expected
    assert int(fixture.calls[13][fixture.calls[13].index("--num-envs") + 1]) == 4


def test_failed_stage_halts_and_preserves_predecessor_without_retry(fixture):
    fixture.controls["fail_at"] = 1
    with pytest.raises(subprocess.CalledProcessError):
        continuous.execute(fixture.config, fixture.run)
    assert len(fixture.calls) == 2
    state = json.loads((fixture.run / "continuous_state.json").read_text())
    assert state["status"] == "failed" and len(state["completed"]) == 1
    assert Path(fixture.selected[0], "weights.bin").exists()
    assert len(list(fixture.run.rglob("partial.bin"))) == 1
    with pytest.raises(RuntimeError, match="Incomplete stage preserved"):
        continuous.execute(fixture.config, fixture.run)
    assert len(fixture.calls) == 2


def test_mid_stage_requested_stop_preserves_both_recoverable_selections(fixture):
    fixture.controls["partial_stop_at"] = 1
    state = continuous.execute(fixture.config, fixture.run)
    assert state["status"] == "stopped_with_partial_stage"
    assert len(state["completed"]) == 1 and len(fixture.calls) == 2
    assert state["partial_stage"]["actual_steps"] == 8
    assert len(list(fixture.run.rglob("weights.bin"))) == 2
    assert not (Path(fixture.selected[0]).parents[2] / "model_retired.json").exists()
    (fixture.run / "STOP").unlink()
    with pytest.raises(RuntimeError, match="Stopped partial stage preserved"):
        continuous.execute(fixture.config, fixture.run)
    assert len(fixture.calls) == 2


def test_existing_configuration_cannot_be_changed_during_continuation(fixture):
    continuous.execute(fixture.config, fixture.run)
    changed = continuous.build_config(steps_per_stage=32, num_envs=4, unroll_length=2,
                                      checkpoint_epochs=2, wandb_mode="disabled")
    with pytest.raises(ValueError, match="different configuration"):
        continuous.execute(changed, fixture.run)
    assert len(fixture.calls) == 2


def test_explicit_domain_batch_overrides_support_restart_sizing():
    config = continuous.build_config(steps_per_stage=8388608, num_envs=512,
        legacy_num_envs=8192, pilot_num_envs=8192, pilot_furniture_num_envs=4096,
        random_num_envs=2048, generic_num_envs=1024, furniture_num_envs=512)
    expected = {0: 8192, 1: 8192, 3: 4096, 8: 2048, 13: 1024, 15: 512}
    for index, count in expected.items():
        assert continuous.stage_num_envs(config, continuous.stage_spec(index, 0)) == count


def test_restart_replays_partial_scene_with_selected_weights_and_all_executed_steps(fixture):
    fixture.controls["partial_stop_at"] = 1
    old_state = continuous.execute(fixture.config, fixture.run)
    old_bytes = {str(path.relative_to(fixture.run)): path.read_bytes()
                 for path in fixture.run.rglob("*") if path.is_file()}
    new_run = fixture.run.parent / "larger_run"
    config = continuous.build_config(steps_per_stage=32, num_envs=8, unroll_length=2,
        checkpoint_epochs=2, wandb_mode="disabled", restart_from_run=fixture.run)
    fixture.controls.update(partial_stop_at=None, stop_after=4)
    state = continuous.execute(config, new_run)
    anchor = json.loads((new_run / "restart_anchor.json").read_text())
    assert anchor["source_selection"] == old_state["partial_stage"]["selected"]
    assert anchor["resume_stage_index"] == state["start_stage_index"] == 1
    assert anchor["global_step_offset"] == state["initial_transition_offset"] == 24
    assert state["completed_transitions"] == 88
    first_command = fixture.calls[2]
    assert first_command[first_command.index("--warmstart") + 1] == str(new_run / "restart_source/native")
    assert first_command[first_command.index("--global-step-offset") + 1] == "24"
    assert Path(first_command[first_command.index("--run-dir") + 1]).name == continuous.stage_spec(1, 0)["name"]
    assert not (new_run / "restart_source").exists()
    assert (new_run / "restart_source_retired.json").exists()
    assert len(list(new_run.rglob("weights.bin"))) == 1
    assert old_bytes == {str(path.relative_to(fixture.run)): path.read_bytes()
                         for path in fixture.run.rglob("*") if path.is_file()}


def test_restart_after_complete_stage_advances_and_can_restart_a_restart(fixture):
    continuous.execute(fixture.config, fixture.run)
    second = fixture.run.parent / "second"
    config = continuous.build_config(steps_per_stage=16, num_envs=4, unroll_length=2,
        checkpoint_epochs=2, wandb_mode="disabled", restart_from_run=fixture.run)
    fixture.controls["stop_after"] = 3
    state = continuous.execute(config, second)
    assert state["start_stage_index"] == 2 and state["initial_transition_offset"] == 32
    third = fixture.run.parent / "third"
    config = continuous.build_config(steps_per_stage=16, num_envs=4, unroll_length=2,
        checkpoint_epochs=2, wandb_mode="disabled", restart_from_run=second)
    fixture.controls["stop_after"] = 4
    state = continuous.execute(config, third)
    assert state["start_stage_index"] == 3 and state["initial_transition_offset"] == 48
    assert state["completed_transitions"] == 64


def test_restart_accepts_original_continuous_config_without_new_optional_fields(fixture):
    fixture.controls["partial_stop_at"] = 1
    continuous.execute(fixture.config, fixture.run)
    path = fixture.run / "continuous_config.json"
    old_config = json.loads(path.read_text())
    for key in ("random_num_envs", "generic_num_envs", "furniture_num_envs", "pilot_furniture_num_envs", "restart_from_run",
                "oom_max_retries", "oom_min_num_envs", "estimated_memory_budget_gib"):
        del old_config[key]
    old_config["sha256"] = _digest({key: value for key, value in old_config.items() if key != "sha256"})
    path.write_text(json.dumps(old_config))
    state_path = fixture.run / "continuous_state.json"
    state = json.loads(state_path.read_text())
    state["plan_sha256"] = old_config["sha256"]
    del state["start_stage_index"], state["initial_transition_offset"]
    for item in state["completed"]:
        receipt_path = fixture.run / "stages" / item["name"] / "curriculum_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["plan_sha256"] = old_config["sha256"]
        receipt_path.write_text(json.dumps(receipt))
        item["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(state))
    path = fixture.run / "runner.json"
    runner = json.loads(path.read_text());runner["config_sha256"] = old_config["sha256"]
    path.write_text(json.dumps(runner))
    config = continuous.build_config(steps_per_stage=16, num_envs=4, unroll_length=2,
        checkpoint_epochs=2, wandb_mode="disabled", restart_from_run=fixture.run)
    fixture.controls.update(stop_after=3, partial_stop_at=None)
    resumed = continuous.execute(config, fixture.run.parent / "from_original_v1")
    assert resumed["start_stage_index"] == 1 and resumed["initial_transition_offset"] == 24


@pytest.mark.parametrize("fault", ["scene", "config", "seed", "provenance"])
def test_partial_restart_binds_saved_scene_configuration_and_selected_provenance(fixture, fault):
    fixture.controls["partial_stop_at"] = 1
    state = continuous.execute(fixture.config, fixture.run)
    directory = fixture.run / "stages" / state["partial_stage"]["name"]
    run_path = directory / "run.json"
    record = json.loads(run_path.read_text())
    if fault == "scene":
        record["training_scene"]["geometry_hash"] = "0" * 64
    elif fault == "config":
        record["environment_config"]["action_dofs"] = 12
    elif fault == "seed":
        record["args"]["seed"] = 123
    else:
        record["code"] = {"modified": True}
    run_path.write_text(json.dumps(record))
    config = continuous.build_config(steps_per_stage=16, num_envs=4, unroll_length=2,
        checkpoint_epochs=2, wandb_mode="disabled", restart_from_run=fixture.run)
    with pytest.raises(ValueError, match="Restart partial"):
        continuous.execute(config, fixture.run.parent / "restarted")
    assert len(fixture.calls) == 2


@pytest.mark.parametrize("fault", ["active", "checkpoint", "offset", "seed"])
def test_restart_refuses_unstopped_or_modified_source(fixture, fault):
    continuous.execute(fixture.config, fixture.run)
    if fault == "active":
        path = fixture.run / "runner.json"
        record = json.loads(path.read_text());record["active"] = True
        path.write_text(json.dumps(record))
    elif fault == "checkpoint":
        Path(fixture.selected[-1], "weights.bin").write_bytes(b"modified")
    elif fault == "offset":
        path = fixture.run / "continuous_state.json"
        record = json.loads(path.read_text());record["completed_transitions"] += 1
        path.write_text(json.dumps(record))
    config = continuous.build_config(steps_per_stage=16, num_envs=4, unroll_length=2,
        checkpoint_epochs=2, wandb_mode="disabled", restart_from_run=fixture.run,
        seed=1 if fault == "seed" else 0)
    with pytest.raises(ValueError):
        continuous.execute(config, fixture.run.parent / "restarted")
    assert len(fixture.calls) == 2


def _oom_config(**kwargs):
    return continuous.build_config(steps_per_stage=64, num_envs=16, unroll_length=2,
        checkpoint_epochs=2, wandb_mode="disabled", oom_min_num_envs=4, **kwargs)


def test_gpu_oom_before_updates_halves_batch_preserves_attempt_and_exact_handoff(fixture):
    fixture.config = _oom_config()
    fixture.controls.update(oom_at={1}, stop_after=3)
    state = continuous.execute(fixture.config, fixture.run)
    assert len(state["completed"]) == 2 and len(fixture.calls) == 3
    assert state["completed_transitions"] == 128
    first_failure, retried = fixture.calls[1:]
    assert first_failure[first_failure.index("--num-envs") + 1] == "16"
    assert retried[retried.index("--num-envs") + 1] == "8"
    assert first_failure[first_failure.index("--warmstart") + 1] == retried[retried.index("--warmstart") + 1]
    assert first_failure[first_failure.index("--global-step-offset") + 1] == retried[retried.index("--global-step-offset") + 1] == "64"
    assert state["completed"][-1]["num_envs"] == 8
    attempt = state["oom_attempts"][0]
    record = json.loads((fixture.run / attempt["record"]).read_text())
    assert Path(record["archive"], "failure.json").is_file()
    assert record["num_envs"] == 16 and record["next_num_envs"] == 8
    assert len(list(fixture.run.rglob("weights.bin"))) == 1
    # The preserved archive and effective batch survive supervisor continuation.
    continuous.execute(fixture.config, fixture.run)
    assert len(fixture.calls) == 3


@pytest.mark.parametrize("evidence", ["reported_steps", "metrics", "candidate", "uncertain"])
def test_gpu_oom_never_retries_when_progress_or_inspection_uncertainty_exists(fixture, evidence):
    fixture.config = _oom_config()
    fixture.controls.update(oom_at={1}, oom_evidence=evidence)
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        continuous.execute(fixture.config, fixture.run)
    assert len(fixture.calls) == 2
    assert Path(fixture.selected[0], "weights.bin").exists()
    assert not list(fixture.run.glob("failed_attempts/*/attempt-*.json"))


def test_non_oom_never_retries_and_oom_retry_count_is_bounded(fixture):
    fixture.config = _oom_config()
    fixture.controls.update(oom_at={0}, classification="other")
    with pytest.raises(subprocess.CalledProcessError):
        continuous.execute(fixture.config, fixture.run)
    assert len(fixture.calls) == 1
    fixture.controls.update(oom_at={1, 2, 3, 4}, classification="gpu_oom", stop_after=10)
    second = fixture.run.parent / "bounded"
    with pytest.raises(subprocess.CalledProcessError):
        continuous.execute(fixture.config, second)
    assert len(fixture.calls) == 4  # One non-OOM + initial OOM + two bounded retries.
    batches = [int(command[command.index("--num-envs") + 1]) for command in fixture.calls[1:]]
    assert batches == [16, 8, 4]


def test_explicit_failed_run_restart_uses_preceding_verified_model_without_losing_offset(fixture):
    fixture.config = _oom_config(oom_max_retries=0)
    fixture.controls.update(oom_at={1})
    with pytest.raises(subprocess.CalledProcessError):
        continuous.execute(fixture.config, fixture.run)
    old_state = json.loads((fixture.run / "continuous_state.json").read_text())
    failed = fixture.run / "stages" / old_state["active_stage"]["stage_name"]
    # The real overnight source predates structured failure metadata.
    (failed / "failure.json").unlink()
    old_bytes = {str(path.relative_to(fixture.run)): path.read_bytes()
                 for path in fixture.run.rglob("*") if path.is_file()}
    fixture.controls.update(oom_at=set(), stop_after=3)
    config = continuous.build_config(steps_per_stage=64, num_envs=8, unroll_length=2,
        checkpoint_epochs=2, wandb_mode="disabled", restart_from_run=fixture.run)
    new_run = fixture.run.parent / "recovery"
    state = continuous.execute(config, new_run)
    anchor = json.loads((new_run / "restart_anchor.json").read_text())
    assert state["start_stage_index"] == 1 and state["initial_transition_offset"] == 64
    assert state["completed_transitions"] == 128
    assert anchor["mode"] == "retry-failed-scene-without-observed-progress"
    assert anchor["failed_stage"]["observed_steps"] == 0
    assert anchor["source_stage"] == continuous.stage_spec(0, 0)["name"]
    assert old_bytes == {str(path.relative_to(fixture.run)): path.read_bytes()
                         for path in fixture.run.rglob("*") if path.is_file()}


def test_failed_source_with_any_metrics_cannot_silently_discard_work(fixture):
    fixture.config = _oom_config(oom_max_retries=0)
    fixture.controls.update(oom_at={1}, oom_evidence="metrics")
    with pytest.raises(subprocess.CalledProcessError):
        continuous.execute(fixture.config, fixture.run)
    config = _oom_config(restart_from_run=fixture.run)
    with pytest.raises(ValueError, match="observed training metrics"):
        continuous.execute(config, fixture.run.parent / "rejected")


def test_failed_first_replay_imports_its_owned_anchor_and_preserves_partial_offset(fixture):
    fixture.controls["partial_stop_at"] = 1
    original = continuous.execute(fixture.config, fixture.run)
    failed_run = fixture.run.parent / "failed_first_replay"
    config = _oom_config(restart_from_run=fixture.run, oom_max_retries=0)
    fixture.controls.update(partial_stop_at=None, oom_at={2})
    with pytest.raises(subprocess.CalledProcessError):
        continuous.execute(config, failed_run)
    failed_state = json.loads((failed_run / "continuous_state.json").read_text())
    assert not failed_state["completed"] and failed_state["initial_transition_offset"] == 24
    previous_anchor = json.loads((failed_run / "restart_anchor.json").read_text())
    old_bytes = {str(path.relative_to(failed_run)): path.read_bytes()
                 for path in failed_run.rglob("*") if path.is_file()}
    fixture.controls.update(oom_at=set(), stop_after=4)
    recovered_run = fixture.run.parent / "recovered_first_replay"
    recovered = continuous.execute(_oom_config(restart_from_run=failed_run), recovered_run)
    anchor = json.loads((recovered_run / "restart_anchor.json").read_text())
    assert anchor["source_selection"] == previous_anchor["imported_selection"]
    assert anchor["source_selection"]["files"] == original["partial_stage"]["selected"]["files"]
    assert anchor["source_checkpoint_location"] == "owned_restart_import"
    assert recovered["start_stage_index"] == 1 and recovered["initial_transition_offset"] == 24
    assert recovered["completed_transitions"] == 88
    assert old_bytes == {str(path.relative_to(failed_run)): path.read_bytes()
                         for path in failed_run.rglob("*") if path.is_file()}


def test_modified_oom_archive_is_rejected_on_continuation(fixture):
    fixture.config = _oom_config()
    fixture.controls.update(oom_at={0}, stop_after=2)
    state = continuous.execute(fixture.config, fixture.run)
    record = json.loads((fixture.run / state["oom_attempts"][0]["record"]).read_text())
    Path(record["archive"], "failure.json").write_text("{}")
    with pytest.raises(ValueError, match="OOM attempt"):
        continuous.execute(fixture.config, fixture.run)


def test_reduced_batch_survives_stop_and_restart_before_retry_launch(fixture, monkeypatch):
    fixture.config = _oom_config()
    fixture.controls.update(oom_at={1}, stop_after=3)
    atomic_json = curriculum._atomic_json
    paused = False

    def pause_after_persisting_retry(path, payload):
        nonlocal paused
        atomic_json(path, payload)
        if not paused and payload.get("status") == "retrying_gpu_oom":
            paused = True
            continuous.request_stop(fixture.run)

    monkeypatch.setattr(curriculum, "_atomic_json", pause_after_persisting_retry)
    stopped = continuous.execute(fixture.config, fixture.run)
    assert len(fixture.calls) == 2 and len(stopped["completed"]) == 1
    assert stopped["effective_stage_batches"][continuous.stage_spec(1, 0)["name"]] == 8
    (fixture.run / "STOP").unlink()
    resumed = continuous.execute(fixture.config, fixture.run)
    assert len(fixture.calls) == 3 and len(resumed["completed"]) == 2
    assert fixture.calls[-1][fixture.calls[-1].index("--num-envs") + 1] == "8"
    assert fixture.calls[-1][fixture.calls[-1].index("--global-step-offset") + 1] == "64"


def test_geometric_capacity_cap_precedes_and_records_actual_batch(fixture, monkeypatch):
    fixture.config = _oom_config(estimated_memory_budget_gib=25)
    monkeypatch.setattr(continuous, "_stage_capacity", lambda scene, requested, config:
        {"num_envs": 8, "requested_num_envs": requested, "estimated_peak_bytes": 123, "enabled": True})
    state = continuous.execute(fixture.config, fixture.run)
    assert all(command[command.index("--num-envs") + 1] == "8" for command in fixture.calls)
    assert all(item["num_envs"] == 8 for item in state["completed"])
    assert all(item["requested_num_envs"] == 16 for item in state["capacity_decisions"].values())
