"""Launcher integration with a fake learner: no simulator or accelerator work."""
import copy
import importlib
import json
from types import SimpleNamespace

import pytest

import train_cat_wholebody as launcher
from cat_ppo.furniture.generalist_logging import GeneralistLogger


@pytest.fixture
def launched(tmp_path, monkeypatch):
    bank = tmp_path / "bank.json"
    bank.write_text("{}\n")
    args = launcher.parser().parse_args(["run", "--run-dir", str(tmp_path / "run"),
        "--bank-manifest", str(bank), "--wandb-mode", "disabled", "--finetuning", "released"])
    specification = launcher.plan(args)
    preparations, calls, loggers, selections = [], [], [], []

    def prepare(args, spec, *, restore_model=True):
        preparations.append(restore_model)
        record = copy.deepcopy(spec)
        record.update(bank_sha256="bank-hash", code={"source_sha256": "code-hash", "git_commit": "commit"},
            observation_contract={"actor_features": ["a"], "critic_features": ["b"], "action_names": ["joint"]},
            warmstart={"source": "pinned-generalist"} if restore_model else None)
        return object(), object(), "mapped-initial-params" if restore_model else None, record

    class SpyLogger(GeneralistLogger):
        def __init__(self, *args, **kwargs):
            self.config = copy.deepcopy(kwargs.get("config"))
            self.exits = []
            super().__init__(*args, **kwargs)
            loggers.append(self)

        def finish(self, exit_code=0):
            self.exits.append(exit_code)
            super().finish(exit_code)

    class Store:
        def __init__(self, *args, **kwargs):
            pass

        def consider(self, **kwargs):
            selections.append(kwargs)
            return True

        def selected(self, **kwargs):
            return None

    fail = {"next_update": False}

    def learner(**kwargs):
        calls.append(kwargs)
        runtime = kwargs["restore_runtime_state"]
        if runtime is None:
            assert kwargs["restore_params"] == "mapped-initial-params"
            step = 0
            kwargs["runtime_checkpoint_fn"](0, {
                "schema": "cat-ppo-runtime-v1", "step": 0,
                "contract": {"metadata": kwargs["runtime_metadata"]}, "adam_count": 0,
            })
        else:
            assert kwargs["restore_params"] is None
            assert runtime["contract"]["metadata"] == kwargs["runtime_metadata"]
            step = runtime["step"]
        if fail["next_update"]:
            fail["next_update"] = False
            raise RuntimeError("simulated first-update OOM")
        step += kwargs["batch_size"] * kwargs["num_minibatches"] * kwargs["unroll_length"]
        metrics = {"training/rollout_reward_mean": .3, "training/total_loss": .1}
        kwargs["progress_fn"](step, metrics)
        kwargs["scored_checkpoint_fn"](step, None, None, {}, metrics, "training_proxy")
        kwargs["runtime_checkpoint_fn"](step, {
            "schema": "cat-ppo-runtime-v1", "step": step,
            "contract": {"metadata": kwargs["runtime_metadata"]}, "adam_count": step,
        })
        return None, None, {"training/completed_steps": step, "training/stopped_by_request": True}

    monkeypatch.setattr(launcher, "prepare", prepare)
    monkeypatch.setattr(launcher, "GeneralistLogger", SpyLogger)
    checkpoint = importlib.import_module("cat_ppo.furniture.checkpoint")
    monkeypatch.setattr(checkpoint, "BestCheckpointStore", Store)
    native = importlib.import_module("cat_ppo.learning.policy.ppo.train")
    monkeypatch.setattr(native, "train", learner)
    return args, specification, preparations, calls, loggers, selections, fail


def test_launcher_uses_one_continuous_learner_and_resumes_its_state_and_wandb_id(launched):
    args, spec, preparations, calls, loggers, selections, _ = launched
    first = launcher.run(args, spec)
    identity = json.loads((args.run_dir / "wandb.json").read_text())["id"]
    args.resume = True
    second = launcher.run(args, spec)
    assert preparations == [True, False]
    assert len(calls) == 2
    assert first["completed_steps"] == 524288
    assert second["completed_steps"] == 1048576
    for call in calls:
        assert call["continuous"] and call["num_timesteps"] == 0
        assert call["num_evals"] == call["num_resets_per_eval"] == 0
        assert call["batch_size"] == 256 and call["num_minibatches"] == 64
        assert call["num_envs"] == 2048 and call["unroll_length"] == 32
        assert call["save_checkpoint_path"] is None
    assert calls[1]["restore_runtime_state"]["step"] == first["completed_steps"]
    assert json.loads((args.run_dir / "wandb.json").read_text())["id"] == identity
    assert all(logger.exits == [0] for logger in loggers)
    assert len(selections) == 2


def test_first_update_failure_preserves_initial_runtime_and_same_experiment(launched):
    args, spec, preparations, calls, loggers, _, fail = launched
    fail["next_update"] = True
    with pytest.raises(RuntimeError, match="first-update OOM"):
        launcher.run(args, spec)
    status = json.loads((args.run_dir / "status.json").read_text())
    assert status["status"] == "failed" and status["completed_steps"] == 0
    assert loggers[0].exits == [1]
    identity = loggers[0].identity["id"]
    args.resume = True
    result = launcher.run(args, spec)
    assert calls[1]["restore_runtime_state"]["step"] == 0
    assert preparations == [True, False]
    assert result["completed_steps"] == 524288
    assert loggers[1].identity["id"] == identity


def test_resume_keeps_original_warmstart_provenance_in_wandb(launched):
    args, spec, _, _, loggers, _, _ = launched
    launcher.run(args, spec)
    args.resume = True
    launcher.run(args, spec)
    assert loggers[1].config["warmstart"] == loggers[0].config["warmstart"]


def test_stop_marker_prevents_new_learner_call(launched):
    args, spec, _, calls, _, _, _ = launched
    args.run_dir.mkdir()
    (args.run_dir / "STOP").touch()
    with pytest.raises(ValueError, match="STOP"):
        launcher.run(args, spec)
    assert calls == []


@pytest.mark.parametrize("failure_point", ["prepare", "store", "logger", "initial_status", "running_status", "launch_record"])
def test_explicit_zero_progress_startup_retry_retains_identity_and_original_record(launched, monkeypatch, failure_point):
    args, spec, preparations, calls, loggers, _, _ = launched
    with monkeypatch.context() as patch:
        if failure_point == "prepare":
            patch.setattr(launcher, "prepare", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("prepare failed")))
        elif failure_point == "store":
            checkpoint = importlib.import_module("cat_ppo.furniture.checkpoint")
            patch.setattr(checkpoint, "BestCheckpointStore", lambda *args: (_ for _ in ()).throw(RuntimeError("store failed")))
        elif failure_point == "logger":
            class BrokenLogger(launcher.GeneralistLogger):
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("logger failed")
            patch.setattr(launcher, "GeneralistLogger", BrokenLogger)
        else:
            original = launcher.atomic_json
            failed = False
            def write(path, value):
                nonlocal failed
                match = ((failure_point == "launch_record" and path.name == "launch.json") or
                         (failure_point == "initial_status" and path.name == "status.json" and value["status"] == "starting") or
                         (failure_point == "running_status" and path.name == "status.json" and value["status"] == "running"))
                if match and not failed:
                    failed = True
                    raise OSError(f"{failure_point} failed")
                return original(path, value)
            patch.setattr(launcher, "atomic_json", write)
        with pytest.raises((RuntimeError, OSError), match="failed"):
            launcher.run(args, spec)
    failed_status = json.loads((args.run_dir / "status.json").read_text())
    assert failed_status["status"] == "failed"
    assert calls == [] and not (args.run_dir / "resume.msgpack").exists()
    identity_path, record_path = args.run_dir / "wandb.json", args.run_dir / "run.json"
    old_identity = json.loads(identity_path.read_text())["id"] if identity_path.exists() else None
    old_record = record_path.read_bytes() if record_path.exists() else None
    if failure_point == "running_status":
        assert loggers[0].exits == [1]
    args.resume = True
    result = launcher.run(args, spec)
    assert result["startup_retry"] and result["completed_steps"] == 524288
    assert result["last_failure"]["error_type"] in ("RuntimeError", "OSError")
    assert "error" not in result
    assert calls[0]["restore_runtime_state"] is None
    assert calls[0]["restore_params"] == "mapped-initial-params"
    if old_identity is not None:
        assert json.loads(identity_path.read_text())["id"] == old_identity
    if old_record is not None:
        assert record_path.read_bytes() == old_record


def test_corrupt_or_missing_runtime_never_turns_into_fresh_initialization(launched):
    args, spec, preparations, calls, _, _, fail = launched
    fail["next_update"] = True
    with pytest.raises(RuntimeError):
        launcher.run(args, spec)
    runtime = args.run_dir / "resume.msgpack"
    runtime.write_bytes(b"not a runtime snapshot")
    args.resume = True
    with pytest.raises(Exception):
        launcher.run(args, spec)
    assert len(calls) == 1 and preparations == [True]
    runtime.unlink()
    with pytest.raises(ValueError, match="missing; cold restart refused"):
        launcher.run(args, spec)
    assert len(calls) == 1 and preparations == [True]


@pytest.mark.parametrize("evidence", ["metrics", "best", "candidate", "observed_steps", "wandb_step"])
def test_missing_snapshot_with_any_progress_evidence_refuses_cold_restart(launched, monkeypatch, evidence):
    args, spec, preparations, calls, _, _, _ = launched
    with monkeypatch.context() as patch:
        patch.setattr(launcher, "prepare", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("prepare failed")))
        with pytest.raises(RuntimeError):
            launcher.run(args, spec)
    if evidence == "metrics":
        (args.run_dir / "metrics.jsonl").write_text('{"global_step":1}\n')
    elif evidence == "best":
        path = args.run_dir / "checkpoints" / "best"
        path.parent.mkdir()
        path.symlink_to("missing-generation")
    elif evidence == "candidate":
        (args.run_dir / ".checkpoint-generations" / "candidate-test").mkdir(parents=True)
    else:
        path = args.run_dir / ("status.json" if evidence == "observed_steps" else "wandb.json")
        record = json.loads(path.read_text())
        record["observed_steps" if evidence == "observed_steps" else "last_global_step"] = 1
        path.write_text(json.dumps(record))
    args.resume = True
    with pytest.raises(ValueError):
        launcher.run(args, spec)
    assert preparations == [] and calls == []


def test_cleanup_failure_does_not_mask_original_training_failure(launched, monkeypatch):
    args, spec, _, _, loggers, _, fail = launched
    fail["next_update"] = True
    original = launcher.atomic_json
    def write(path, value):
        if path.name == "status.json" and value["status"] == "failed":
            raise OSError("status storage unavailable")
        return original(path, value)
    monkeypatch.setattr(launcher, "atomic_json", write)
    with pytest.raises(RuntimeError, match="first-update OOM") as caught:
        launcher.run(args, spec)
    assert any("status storage unavailable" in note for note in caught.value.__notes__)
    assert loggers[0].exits == [1]


@pytest.mark.parametrize("partial_owner", [None, '{"owner":'])
def test_retry_preserves_incomplete_store_owner_metadata(launched, monkeypatch, partial_owner):
    args, spec, _, _, _, _, _ = launched
    checkpoint = importlib.import_module("cat_ppo.furniture.checkpoint")
    def broken_store(directory):
        generations = directory / ".checkpoint-generations"
        generations.mkdir()
        if partial_owner is not None:
            (generations / "owner.json").write_text(partial_owner)
        raise OSError("owner write interrupted")
    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "BestCheckpointStore", broken_store)
        with pytest.raises(OSError, match="owner write interrupted"):
            launcher.run(args, spec)
    args.resume = True
    launcher.run(args, spec)
    archived = args.run_dir / "startup-artifacts" / "checkpoint-owner-before-attempt-0002"
    assert archived.is_dir()
    if partial_owner is not None:
        assert (archived / "owner.json").read_text() == partial_owner


def test_wandb_failed_registration_retries_same_id_then_requires_established_run(tmp_path):
    calls, finishes = [], []
    class OnlineRun:
        url = "https://example.invalid/run"
        def define_metric(self, *args, **kwargs):
            pass
        def finish(self, exit_code):
            finishes.append(exit_code)
    def init(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("registration unavailable")
        return OnlineRun()
    module = SimpleNamespace(init=init)
    with pytest.raises(RuntimeError, match="registration unavailable"):
        GeneralistLogger(tmp_path, wandb_module=module)
    saved = json.loads((tmp_path / "wandb.json").read_text())
    assert saved["initialization_attempted"] and not saved["initialized"]
    second = GeneralistLogger(tmp_path, resume=True, wandb_module=module)
    second.finish()
    third = GeneralistLogger(tmp_path, resume=True, wandb_module=module)
    third.finish()
    assert len({call["id"] for call in calls}) == 1
    assert [call["resume"] for call in calls] == ["never", "allow", "must"]


def test_wandb_failure_after_init_closes_run_as_failed(tmp_path):
    finishes = []
    class OnlineRun:
        url = "https://example.invalid/run"
        def define_metric(self, *args, **kwargs):
            raise RuntimeError("metric setup failed")
        def finish(self, exit_code):
            finishes.append(exit_code)
    with pytest.raises(RuntimeError, match="metric setup failed"):
        GeneralistLogger(tmp_path, wandb_module=SimpleNamespace(init=lambda **kwargs: OnlineRun()))
    assert finishes == [1]
    assert json.loads((tmp_path / "wandb.json").read_text())["initialized"]


def test_field_bank_provenance_reports_reconstructed_exception():
    scenes = [dict(scene_id=str(index), family="original_cat", source={"arrays_unchanged": True}) for index in range(36)]
    scenes.append(dict(scene_id="missing", family="original_cat", source={"kind": "reconstructed-missing-original", "arrays_unchanged": False}))
    scenes.extend(dict(scene_id=family, family=family, source={}) for family in ("furniture", "generic_clutter"))
    result = launcher.field_bank_summary({"scenes": scenes}, "verified-manifest")
    assert result["scene_count"] == 39 and result["byte_verified_original_count"] == 36
    assert result["reconstructed_original_count"] == 1
    assert result["manifest_sha256"] == "verified-manifest"
    assert result["scenes"][36]["source_kind"] == "reconstructed-missing-original"
    assert result["scenes"][36]["arrays_unchanged"] is False
