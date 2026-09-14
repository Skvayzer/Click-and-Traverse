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

        store.consider(step=8, metrics={"proxy_score": 1.0}, source="training_proxy",
                       write_checkpoint=writer, contract={"action_names": ["fixture_joint"]})
        selected = store.selected()
        selected_paths.append(selected["path"])
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
        interrupted = controls.get("partial_stop_at") == stage_index
        budget = int(args["--steps"])
        summary = dict(requested_steps=budget, actual_steps=budget // 2 if interrupted else budget,
            stopped_by_request=interrupted, selected_step=8, selection_source="training_proxy",
            selected_score=selected["score"], actor_max_abs_parameter_update=.1,
            native_checkpoint=selected["path"])
        (directory / "summary.json").write_text(json.dumps(summary))
        if interrupted or len(calls) == controls["stop_after"]:
            continuous.request_stop(run_dir)

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
