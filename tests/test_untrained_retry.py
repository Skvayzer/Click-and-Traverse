import fcntl
import hashlib
import json
from pathlib import Path

import pytest

from cat_ppo.furniture import untrained_retry


def _write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def failed_startup(tmp_path, monkeypatch):
    specification = {"config": {"noise_scale": .02}, "warmstart_best": {"archive": "/saved/best"}}
    fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    destination = {"project": "CAT-wholebody", "entity": "skvayzer", "mode": "online"}
    _write(tmp_path / "launch.json", {
        "schema": "cat-generalist-launch-v1", "owner": "test-owner", "specification": specification,
        "specification_sha256": fingerprint, "wandb": destination,
    })
    _write(tmp_path / "status.json", {
        "schema": "cat-generalist-status-v1", "owner": "test-owner", "status": "failed",
        "phase": "learner_initialization", "pid": 7654321, "specification_sha256": fingerprint,
        "initial_runtime_written": False, "runtime_present_on_entry": False, "observed_steps": 0,
    })
    _write(tmp_path / "run.json", dict(specification, code={"git_commit": "test"}))
    _write(tmp_path / "wandb.json", dict(destination, id="46a8a2af", last_global_step=-1,
        initialized=True, initialization_attempted=True, url="https://wandb.ai/skvayzer/CAT-wholebody/runs/46a8a2af"))
    _write(tmp_path / "validation_baseline.json", {"result": "baseline evaluation is not learner progress"})
    (tmp_path / ".learner.lock").touch()
    (tmp_path / "metrics.jsonl").touch()
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/.selection.lock").touch()
    (tmp_path / ".checkpoint-generations").mkdir()
    _write(tmp_path / ".checkpoint-generations/owner.json", {"owner": "independent-checkpoint-owner"})

    def dead(pid, signal):
        assert pid == 7654321 and signal == 0
        raise ProcessLookupError
    monkeypatch.setattr(untrained_retry.os, "kill", dead)
    return tmp_path


def _edit(directory, filename, **changes):
    path = directory / filename
    _write(path, json.loads(path.read_text()) | changes)


def test_baseline_only_failure_reuses_identity_without_mutating_archive(failed_startup):
    before = {str(p.relative_to(failed_startup)): p.read_bytes() for p in failed_startup.rglob("*") if p.is_file()}
    identity, provenance = untrained_retry.verified_untrained_logging_identity(failed_startup)
    assert identity["id"] == "46a8a2af" and identity["last_global_step"] == -1
    assert provenance["training_steps"] == 0 and provenance["runtime_state_reused"] is False
    assert provenance["previous_warmstart_best"] == {"archive": "/saved/best"}
    for name, expected in provenance["metadata_sha256"].items():
        assert hashlib.sha256(before[name]).hexdigest() == expected
    after = {str(p.relative_to(failed_startup)): p.read_bytes() for p in failed_startup.rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize("changes", [
    {"status": "running"}, {"status": "stopped"}, {"phase": "training"}, {"phase": None},
    {"completed_steps": 1}, {"observed_steps": -1}, {"resume_state_steps": 100}, {"num_updates": 1},
    {"completed_steps": False}, {"observed_steps": "0"}, {"initial_runtime_written": True},
    {"runtime_present_on_entry": True}, {"initial_runtime_written": 0},
    {"owner": "different-owner"}, {"specification_sha256": "wrong"}, {"pid": 0}, {"pid": True},
])
def test_rejects_nonzero_or_unproven_startup_state(failed_startup, changes):
    _edit(failed_startup, "status.json", **changes)
    with pytest.raises(ValueError):
        untrained_retry.verified_untrained_logging_identity(failed_startup)


@pytest.mark.parametrize("filename,changes", [
    ("launch.json", {"owner": "other"}),
    ("launch.json", {"specification_sha256": "wrong"}),
    ("run.json", {"config": {"noise_scale": .05}}),
    ("wandb.json", {"last_global_step": 0}),
    ("wandb.json", {"last_global_step": -2}),
    ("wandb.json", {"project": "other"}),
    ("wandb.json", {"id": "not-an-id"}),
    ("wandb.json", {"initialized": "false"}),
])
def test_rejects_inconsistent_metadata_or_logged_identity(failed_startup, filename, changes):
    _edit(failed_startup, filename, **changes)
    with pytest.raises(ValueError):
        untrained_retry.verified_untrained_logging_identity(failed_startup)


@pytest.mark.parametrize("name", [
    "resume.msgpack", ".resume.msgpack.partial", "initial_runtime.msgpack", "validation_latest.json",
    "checkpoints/best", "checkpoints/.best-pending", ".checkpoint-generations/candidate-123",
])
def test_rejects_runtime_and_checkpoint_artifacts_even_empty_or_dangling(failed_startup, name):
    (failed_startup / name).symlink_to("nonexistent")
    with pytest.raises(ValueError):
        untrained_retry.verified_untrained_logging_identity(failed_startup)


def test_rejects_any_logged_event_including_step_zero(failed_startup):
    (failed_startup / "metrics.jsonl").write_text('{"global_step": 0}\n')
    with pytest.raises(ValueError, match="logged metrics"):
        untrained_retry.verified_untrained_logging_identity(failed_startup)


@pytest.mark.parametrize("name", ["run.json", "launch.json", "status.json", "wandb.json"])
def test_requires_complete_regular_metadata(failed_startup, name):
    path = failed_startup / name
    target = failed_startup / (name + ".target")
    path.rename(target)
    path.symlink_to(target.name)
    with pytest.raises(ValueError, match="regular retry metadata"):
        untrained_retry.verified_untrained_logging_identity(failed_startup)


def test_live_pid_and_permission_denied_are_not_evidence_of_exit(failed_startup, monkeypatch):
    monkeypatch.setattr(untrained_retry.os, "kill", lambda *_: None)
    with pytest.raises(ValueError, match="still live"):
        untrained_retry.verified_untrained_logging_identity(failed_startup)
    def inaccessible(*_):
        raise PermissionError
    monkeypatch.setattr(untrained_retry.os, "kill", inaccessible)
    with pytest.raises(ValueError, match="Cannot establish"):
        untrained_retry.verified_untrained_logging_identity(failed_startup)


def test_run_lock_must_be_released_even_when_pid_is_dead(failed_startup):
    with (failed_startup / ".learner.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="run lock"):
            untrained_retry.verified_untrained_logging_identity(failed_startup)
