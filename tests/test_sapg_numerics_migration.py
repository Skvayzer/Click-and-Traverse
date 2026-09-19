"""Offline code migration preserves every learner byte and fails closed."""

import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

from flax import serialization
import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/migrate_sapg_numerics_resume.py"
SPEC = importlib.util.spec_from_file_location("sapg_numerics_migration", PATH)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def commit(root, message):
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", message], check=True, capture_output=True)
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    for key, value in (("user.name", "Test"), ("user.email", "test@example.invalid")):
        subprocess.run(["git", "-C", str(source), "config", key, value], check=True)
    loss = source / migration.ALLOWED_SOURCE_CHANGE
    loss.parent.mkdir(parents=True)
    loss.write_text("loss_version = 1\n")
    (source / "train_cat_wholebody.py").write_text("launcher_version = 1\n")
    trainer = source / "cat_ppo/learning/policy/ppo/train.py"
    trainer.parent.mkdir()
    trainer.write_text("learner_version = 1\n")
    old_commit = commit(source, "old")
    loss.write_text("loss_version = 2\n")
    new_commit = commit(source, "reviewed numerical change")
    monkeypatch.setattr(migration, "SOURCE_COMMIT", old_commit)
    old_code, _ = migration.source_identity(source, old_commit)
    sapg = dict(num_policies=6, embedding_dim=16, prepare_chunk_size=64)
    config = dict(algorithm="sapg", sapg=sapg, fine_tuning=dict(mode="cat_train_only"),
                  policy_config=dict(num_envs=36864, learning_rate=.0003))
    specification = dict(config=config, bank_manifest="/verified/fields/manifest.json")
    spec_sha = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    record = {**specification, "code": old_code, "bank_sha256": "a" * 64,
              "observation_contract": {"actor": 222, "critic": 310, "actions": 29},
              "warmstart": {"original": True, "checkpoint": "released"}}
    destination = dict(project="CAT-wholebody", entity="skvayzer", mode="online")
    launch = dict(schema="cat-generalist-launch-v1", specification=specification,
                  specification_sha256=spec_sha, owner="owner-1", wandb=destination)
    status = dict(status="failed", owner="owner-1", specification_sha256=spec_sha,
                  resume_state_steps=migration.EXPECTED_STEP, error_type="FloatingPointError",
                  error="Non-finite training metrics: training/policy_loss")
    wandb = dict(id=migration.EXPECTED_WANDB_ID, **destination, initialized=True,
                 last_global_step=migration.EXPECTED_STEP + 1179648)
    metadata = dict(config=config, bank_sha256=record["bank_sha256"],
                    source_sha256=old_code["source_sha256"], contract=record["observation_contract"])
    runtime = dict(schema="cat-ppo-runtime-v1", step=migration.EXPECTED_STEP,
        contract=dict(sapg=sapg, metadata=metadata, learning_rate=.0003, num_envs=36864),
        training_state=dict(paths=["actor", "critic", "Adam.first", "Adam.second", "Adam.count"],
            leaves=[np.arange(36, dtype=np.float32).reshape(6, 6), np.linspace(-1, 1, 20, dtype=np.float32),
                    np.full(20, .05, np.float32), np.full(20, .003, np.float32), np.array(70912, np.int32)]),
        env_state=dict(paths=["qpos", "navigation", "scene_ids"],
            leaves=[np.arange(256, dtype=np.float32).reshape(8, 32), np.arange(8, dtype=np.int32), np.arange(8)]),
        local_key=dict(leaves=[np.array([42, 123], np.uint32)]),
        key_envs=dict(leaves=[np.arange(16, dtype=np.uint32).reshape(8, 2)]),
        metrics_logger=dict(episodes=[1., 2., 3.], last_step=migration.EXPECTED_STEP),
        training_walltime=3123.56)
    run = tmp_path / "run"
    run.mkdir()
    for name, value in (("run.json", record), ("launch.json", launch), ("status.json", status), ("wandb.json", wandb)):
        write_json(run / name, value)
    (run / ".learner.lock").touch()
    (run / "resume.msgpack").write_bytes(serialization.msgpack_serialize(runtime))
    (run / "metrics.jsonl").write_text('{"historical":true}\n')
    backup = tmp_path / "backup"
    shutil.copytree(run, backup)
    monkeypatch.setattr(migration, "EXPECTED_RUNTIME_SHA256", migration._sha(backup / "resume.msgpack"))
    monkeypatch.setattr(migration, "EXPECTED_RUN_SHA256", migration._sha(backup / "run.json"))
    return dict(run=run, backup=backup, source=source, new_commit=new_commit, old_commit=old_commit,
                runtime=runtime, record=record)


def perform(fixture, apply=False):
    return migration.migrate(fixture["run"], fixture["backup"], fixture["source"],
                             fixture["new_commit"], apply=apply)


def content_hashes(directory):
    return {str(path.relative_to(directory)): migration._sha(path)
            for path in directory.rglob("*") if path.is_file()}


def test_default_dry_run_proves_identity_without_writing_anything(fixture):
    before = content_hashes(fixture["run"])
    report = perform(fixture)
    assert report["state"] == "verified-dry-run" and report["publication_needed"]
    assert report["saved_step"] == 326762496 and report["wandb_id"] == "f017f302"
    assert content_hashes(fixture["run"]) == before


def test_apply_changes_only_code_contract_and_retains_all_noncontract_bytes(fixture):
    backups_before = content_hashes(fixture["backup"])
    report = perform(fixture, apply=True)
    assert report["state"] == "complete"
    original = (fixture["backup"] / "resume.msgpack").read_bytes()
    migrated = (fixture["run"] / "resume.msgpack").read_bytes()
    old_header, old_spans = migration.runtime_header(fixture["backup"] / "resume.msgpack")
    new_header, new_spans = migration.runtime_header(fixture["run"] / "resume.msgpack")
    old_first, old_last = old_spans["contract"]
    new_first, new_last = new_spans["contract"]
    old_state = original[:old_first] + original[old_last:]
    assert old_state == migrated[:new_first] + migrated[new_last:]
    assert hashlib.sha256(old_state).hexdigest() == report["runtime"]["preserved_noncontract_bytes_sha256"]
    assert migration._sha(fixture["run"] / "resume.msgpack") == report["runtime"]["after_sha256"]
    expected = copy.deepcopy(old_header)
    expected["contract"]["metadata"]["source_sha256"] = report["new_code"]["source_sha256"]
    assert new_header == expected
    restored = serialization.msgpack_restore(migrated)
    for old, new in zip(fixture["runtime"]["training_state"]["leaves"], restored["training_state"]["leaves"]):
        np.testing.assert_array_equal(old, new)
    new_record = json.loads((fixture["run"] / "run.json").read_text())
    assert new_record.pop("code") == report["new_code"]
    history = new_record.pop("code_migrations")
    assert len(history) == 1 and history[0]["runtime"] == report["runtime"]
    old_record = {key: value for key, value in fixture["record"].items() if key != "code"}
    assert new_record == old_record
    for name in ("launch.json", "status.json", "wandb.json", "metrics.jsonl"):
        assert (fixture["run"] / name).read_bytes() == (fixture["backup"] / name).read_bytes()
    assert content_hashes(fixture["backup"]) == backups_before
    hashes = content_hashes(fixture["run"])
    assert perform(fixture, apply=True) == report
    assert content_hashes(fixture["run"]) == hashes


def test_interrupted_two_file_publication_fails_closed_and_can_finish(fixture, monkeypatch):
    original_atomic = migration._atomic_bytes

    def interrupted(path, payload):
        if path == fixture["run"] / "run.json":
            raise OSError("simulated interruption after runtime publication")
        return original_atomic(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(migration, "_atomic_bytes", interrupted)
        with pytest.raises(OSError, match="simulated interruption"):
            perform(fixture, apply=True)
    header, _ = migration.runtime_header(fixture["run"] / "resume.msgpack")
    record = json.loads((fixture["run"] / "run.json").read_text())
    # Ordinary resume checks would refuse this unmatched source pair.
    assert header["contract"]["metadata"]["source_sha256"] != record["code"]["source_sha256"]
    assert perform(fixture)["publication_needed"]
    report = perform(fixture, apply=True)
    assert report["state"] == "complete"
    record = json.loads((fixture["run"] / "run.json").read_text())
    assert header["contract"]["metadata"]["source_sha256"] == record["code"]["source_sha256"]


def test_active_learner_lock_prevents_any_migration(fixture):
    before = content_hashes(fixture["run"])
    with (fixture["run"] / ".learner.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            perform(fixture, apply=True)
    assert content_hashes(fixture["run"]) == before


def test_unreviewed_runtime_source_file_change_is_rejected(fixture):
    path = fixture["source"] / "cat_ppo/learning/policy/ppo/train.py"
    path.write_text("learner_version = 2\n")
    fixture["new_commit"] = commit(fixture["source"], "forbidden trainer change")
    with pytest.raises(ValueError, match="only the reviewed SAPG"):
        perform(fixture, apply=True)
    assert not (fixture["run"] / "code-migrations").exists()


def test_dirty_reviewed_source_is_rejected(fixture):
    (fixture["source"] / migration.ALLOWED_SOURCE_CHANGE).write_text("unreviewed = True\n")
    with pytest.raises(ValueError, match="differs from its reviewed commit"):
        perform(fixture, apply=True)


@pytest.mark.parametrize("target", ["run.json", "resume.msgpack", "wandb.json"])
def test_active_files_must_match_verified_archive(fixture, target):
    path = fixture["run"] / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="differs from"):
        perform(fixture, apply=True)
    assert not (fixture["run"] / "code-migrations").exists()


def test_another_wandb_identity_is_rejected_even_if_current_matches_backup(fixture):
    for root in (fixture["run"], fixture["backup"]):
        path = root / "wandb.json"
        record = json.loads(path.read_text())
        record["id"] = "ffffffff"
        write_json(path, record)
    with pytest.raises(ValueError, match="authorized existing run"):
        perform(fixture, apply=True)


def test_runtime_configuration_change_cannot_be_migrated(fixture):
    value = copy.deepcopy(fixture["runtime"])
    value["contract"]["metadata"]["config"]["policy_config"]["learning_rate"] = .00001
    for root in (fixture["run"], fixture["backup"]):
        (root / "resume.msgpack").write_bytes(serialization.msgpack_serialize(value))
    with pytest.raises(ValueError, match="Runtime config"):
        perform(fixture, apply=True)


def test_symlinked_runtime_is_rejected(fixture):
    path = fixture["run"] / "resume.msgpack"
    path.unlink()
    path.symlink_to(fixture["backup"] / "resume.msgpack")
    with pytest.raises(ValueError, match="non-symlink"):
        perform(fixture, apply=True)
