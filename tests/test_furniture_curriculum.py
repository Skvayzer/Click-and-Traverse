import json
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from cat_ppo.furniture.curriculum import build_plan, load_plan, _retire_owned_stage
from cat_ppo.furniture.scenes import _digest
from cat_ppo.furniture import curriculum
from cat_ppo.furniture.checkpoint import BestCheckpointStore
from cat_ppo.furniture.scenes import generate_scene, load_scene, write_scene_bundle


def test_plan_rehearses_cat_through_dense_and_corrupted_stages(tmp_path):
    plan = build_plan(steps_per_stage=1024, rounds=3)
    assert len(plan["stages"]) == 36
    assert plan["total_steps"] == 36864
    for first, second in zip(plan["stages"][::2], plan["stages"][1::2]):
        assert first["scene"]["domain"] == "legacy"
        assert second["scene"]["domain"] in ("generic", "furniture")
        assert first["environment_config"]["action_dofs"] == second["environment_config"]["action_dofs"] == 29
    assert plan["stages"][-1]["scene"]["difficulty"] == "dense"
    assert plan["stages"][-1]["environment_config"]["map_latency_steps"] == 3
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert load_plan(path) == plan
    plan["stages"][0]["scene"]["split"] = "test"
    plan["sha256"] = _digest({key: value for key, value in plan.items() if key != "sha256"})
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Only training"):
        load_plan(path)


def test_retirement_preserves_run_records_and_refuses_symlink(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "run.json").write_text("{}")
    (stage / "summary.json").write_text("{}")
    receipt = {"stage_name": stage.name,
               "run_sha256": hashlib.sha256((stage / "run.json").read_bytes()).hexdigest(),
               "summary_sha256": hashlib.sha256((stage / "summary.json").read_bytes()).hexdigest()}
    (stage / "curriculum_receipt.json").write_text(json.dumps(receipt))
    receipt_hash = hashlib.sha256((stage / "curriculum_receipt.json").read_bytes()).hexdigest()
    for name in (".checkpoint-generations", "checkpoints", "export"):
        (stage / name).mkdir()
        (stage / name / "weights").write_bytes(b"model")
    _retire_owned_stage(stage, receipt_sha256=receipt_hash)
    assert (stage / "summary.json").exists()
    assert not (stage / ".checkpoint-generations").exists()
    assert (stage / "model_retired.json").exists()
    symlink = tmp_path / "link"
    symlink.symlink_to(stage)
    with pytest.raises(ValueError, match="unrecognized"):
        _retire_owned_stage(symlink, receipt_sha256=receipt_hash)


def test_budget_is_never_rounded_up():
    with pytest.raises(ValueError, match="divisible"):
        build_plan(steps_per_stage=1025)


@pytest.fixture
def two_stage_fixture(tmp_path, monkeypatch):
    """Mock subprocess training; real tiny checkpoint manifests and scene bytes."""
    plan = build_plan(steps_per_stage=16, rounds=1, num_envs=4, unroll_length=2, checkpoint_epochs=2)
    plan["stages"] = plan["stages"][:2]
    plan["total_steps"] = 32
    plan["sha256"] = _digest({k: v for k, v in plan.items() if k != "sha256"})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    bundle = write_scene_bundle(generate_scene(difficulty="open_floor"), tmp_path / "scene", voxel_size=.2)
    scene = load_scene(bundle)
    monkeypatch.setattr(curriculum, "materialize_stage", lambda spec, cache: bundle)
    calls, controls = [], {}

    def train(command, *, cwd, check):
        arguments = dict(zip(command[2::2], command[3::2]))
        stage_dir = Path(arguments["--run-dir"])
        stage_index = 0 if stage_dir.name == plan["stages"][0]["name"] else 1
        calls.append(command)
        if stage_index == 1 and controls.get("subprocess_failure"):
            if controls.get("partial"):
                stage_dir.mkdir(parents=True)
                (stage_dir / "partial.bin").write_bytes(b"preserve interrupted work")
            raise subprocess.CalledProcessError(1, command)
        store = BestCheckpointStore(stage_dir)
        def writer(path):
            path.mkdir()
            (path / "weights.bin").write_bytes(f"stage {stage_index} actor critic normalizer".encode())
            (path / "ppo_network_config.json").write_text("{}")
        store.consider(step=8, metrics={"proxy_score": 1.0}, source="training_proxy",
                       write_checkpoint=writer, contract={"action_names": ["fixture_joint"]})
        selected = store.selected()
        source = dict(kind="pinned_public_native_checkpoint")
        if "--warmstart" in arguments:
            native = Path(arguments["--warmstart"])
            source = dict(kind="explicit_local_native_checkpoint", path=str(native.resolve()),
                files={str(p.relative_to(native)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in native.rglob("*") if p.is_file()})
        if stage_index == 1 and controls.get("wrong_source"):
            source["files"] = {}
        run = dict(args=dict(steps=16, seed=0, num_envs=4, unroll_length=2, checkpoint_epochs=2,
                            num_evals=0, action_dofs=29, run_dir=str(stage_dir), scene_dir=str(bundle)),
                   training_scene={k: scene[k] for k in ("scene_id", "geometry_hash", "split")},
                   validation_scene=None, selection_source="training_proxy", source=source,
                   warmstart={"critic_restored": True},
                   environment_config=plan["stages"][stage_index]["environment_config"])
        (stage_dir / "run.json").write_text(json.dumps(run))
        summary = dict(requested_steps=16, selected_step=8, selection_source="training_proxy",
                       selected_score=selected["score"], actor_max_abs_parameter_update=.1,
                       native_checkpoint=selected["path"])
        if stage_index == 1 and controls.get("nonfinite_update"):
            summary["actor_max_abs_parameter_update"] = float("nan")
        (stage_dir / "summary.json").write_text(json.dumps(summary))
        (stage_dir / "metrics.jsonl").write_text('{"fixture": true}\n')
        (stage_dir / "export").mkdir()
        (stage_dir / "export" / "policy.onnx").write_bytes(b"fixture export")
        if stage_index == 1 and controls.get("corrupt_selected"):
            (Path(selected["path"]) / "weights.bin").write_bytes(b"corrupted after publication")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(curriculum.subprocess, "run", train)
    return SimpleNamespace(plan=plan, path=plan_path, run=tmp_path / "run", calls=calls, controls=controls)


def _stage_paths(fixture):
    return [fixture.run / "stages" / stage["name"] for stage in fixture.plan["stages"]]


def test_two_stage_handoff_resume_keeps_one_model_and_full_provenance(two_stage_fixture):
    fixture = two_stage_fixture
    first, second = _stage_paths(fixture)
    first_state = curriculum.execute(fixture.path, fixture.run, max_stages=1)
    first_selection = BestCheckpointStore.open_existing(first).selected()
    assert first_state["current_stage"] == first.name and len(fixture.calls) == 1
    state = curriculum.execute(fixture.path, fixture.run, max_stages=1)
    assert len(state["completed"]) == 2 and len(fixture.calls) == 2
    command = fixture.calls[1]
    assert command[command.index("--warmstart") + 1] == first_selection["path"]
    assert not (first / ".checkpoint-generations").exists()
    assert not (first / "checkpoints").exists() and not (first / "export").exists()
    assert all((first / name).is_file() for name in ("run.json", "summary.json", "metrics.jsonl", "curriculum_receipt.json"))
    receipt = json.loads((second / "curriculum_receipt.json").read_text())
    assert receipt["input_selection"] == first_selection
    assert "within-stage" in receipt["selection_limit"]
    assert len(list(fixture.run.rglob("weights.bin"))) == 1
    assert BestCheckpointStore.open_existing(second).selected()["step"] == 8
    assert curriculum.execute(fixture.path, fixture.run) == state
    assert len(fixture.calls) == 2


def test_failed_next_process_preserves_previous_best_and_can_retry(two_stage_fixture):
    fixture = two_stage_fixture
    first, second = _stage_paths(fixture)
    fixture.controls["subprocess_failure"] = True
    with pytest.raises(subprocess.CalledProcessError):
        curriculum.execute(fixture.path, fixture.run)
    assert BestCheckpointStore.open_existing(first).selected()["step"] == 8
    assert not second.exists()
    assert len(json.loads((fixture.run / "curriculum_state.json").read_text())["completed"]) == 1
    fixture.controls.clear()
    assert len(curriculum.execute(fixture.path, fixture.run)["completed"]) == 2
    assert len(list(fixture.run.rglob("weights.bin"))) == 1


def test_incomplete_stage_is_preserved_without_silent_retraining(two_stage_fixture):
    fixture = two_stage_fixture
    first, second = _stage_paths(fixture)
    fixture.controls.update(subprocess_failure=True, partial=True)
    with pytest.raises(subprocess.CalledProcessError):
        curriculum.execute(fixture.path, fixture.run)
    fixture.controls.clear()
    with pytest.raises(RuntimeError, match="Incomplete stage preserved"):
        curriculum.execute(fixture.path, fixture.run)
    assert len(fixture.calls) == 2
    assert (second / "partial.bin").read_bytes() == b"preserve interrupted work"
    assert BestCheckpointStore.open_existing(first).selected()["step"] == 8


@pytest.mark.parametrize("fault", ["wrong_source", "nonfinite_update", "corrupt_selected"])
def test_unverified_handoff_never_retires_previous_best(two_stage_fixture, fault):
    fixture = two_stage_fixture
    fixture.controls[fault] = True
    first, _ = _stage_paths(fixture)
    with pytest.raises(ValueError):
        curriculum.execute(fixture.path, fixture.run)
    assert BestCheckpointStore.open_existing(first).selected()["step"] == 8
    assert not (first / "model_retired.json").exists()
    assert len(json.loads((fixture.run / "curriculum_state.json").read_text())["completed"]) == 1


def test_completed_stage_is_adopted_after_state_commit_interruption(two_stage_fixture, monkeypatch):
    fixture = two_stage_fixture
    original = curriculum._atomic_json
    def interrupted(path, value):
        if path.name == "curriculum_state.json" and len(value.get("completed", [])) == 2:
            raise OSError("state commit interrupted")
        original(path, value)
    monkeypatch.setattr(curriculum, "_atomic_json", interrupted)
    with pytest.raises(OSError, match="state commit"):
        curriculum.execute(fixture.path, fixture.run)
    assert len(fixture.calls) == 2 and len(list(fixture.run.rglob("weights.bin"))) == 2
    monkeypatch.setattr(curriculum, "_atomic_json", original)
    state = curriculum.execute(fixture.path, fixture.run)
    assert len(state["completed"]) == 2 and len(fixture.calls) == 2
    assert len(list(fixture.run.rglob("weights.bin"))) == 1


def test_retirement_is_retried_after_committed_handoff_interruption(two_stage_fixture, monkeypatch):
    fixture = two_stage_fixture
    original = curriculum._retire_owned_stage
    def interrupted(*args, **kwargs):
        raise OSError("retirement interrupted")
    monkeypatch.setattr(curriculum, "_retire_owned_stage", interrupted)
    with pytest.raises(OSError, match="retirement interrupted"):
        curriculum.execute(fixture.path, fixture.run)
    assert len(json.loads((fixture.run / "curriculum_state.json").read_text())["completed"]) == 2
    assert len(list(fixture.run.rglob("weights.bin"))) == 2
    monkeypatch.setattr(curriculum, "_retire_owned_stage", original)
    curriculum.execute(fixture.path, fixture.run)
    assert len(fixture.calls) == 2 and len(list(fixture.run.rglob("weights.bin"))) == 1


def test_curriculum_cache_rejects_matching_geometry_with_wrong_resolution(tmp_path, monkeypatch):
    scene = generate_scene(difficulty="open_floor")
    monkeypatch.setattr(curriculum, "generate_scene", lambda **kwargs: scene)
    key = _digest(dict(scene=scene, dx=.10))
    write_scene_bundle(scene, tmp_path / key, voxel_size=.20)
    with pytest.raises(ValueError, match="full requested"):
        curriculum.materialize_stage(dict(domain="furniture", split="train"), tmp_path)
