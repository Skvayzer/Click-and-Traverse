"""Cross-host moves preserve complete learners and only rebase operational paths."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess

from flax import serialization
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/migrate_cat_host.py"
SPEC = importlib.util.spec_from_file_location("cat_host_migration", SCRIPT)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def setup(tmp_path):
    new_root = (tmp_path / "destination").resolve()
    new_root.mkdir()
    source = new_root / "outputs/sources/fixed"
    source.mkdir(parents=True)
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    for key, value in (("user.name", "Test"), ("user.email", "test@example.invalid")):
        subprocess.run(["git", "-C", str(source), "config", key, value], check=True)
    loss = source / "cat_ppo/learning/policy/sapg/losses.py"
    loss.parent.mkdir(parents=True)
    loss.write_text("stable_loss = True\n")
    (source / "train_cat_wholebody.py").write_text("source_preserved = True\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-m", "fixed source"], check=True,
                   capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    code, _ = migration.io.source_identity(source, commit)
    old_root = Path("/old-host/repository")
    run_dir = new_root / "outputs/continuous-run"
    run_dir.mkdir()
    manifest_sha = {}
    operational = {}
    for key, kind in zip(migration.PATHS[:-1], ("fields", "collision", "resets")):
        relative = Path("data/specialist") / kind / "manifest.json"
        manifest = new_root / relative
        manifest.parent.mkdir(parents=True)
        write_json(manifest, dict(kind=kind, immutable=True))
        manifest_sha[key] = migration.io._sha(manifest)
        operational[key] = str(old_root / relative)
    operational["stop"] = str(old_root / "outputs/continuous-run/STOP")
    step = 451805184
    sapg = dict(num_policies=6, embedding_dim=16, prepare_chunk_size=64)
    config = dict(algorithm="sapg", sapg=sapg, policy_config=dict(num_envs=36864, learning_rate=.0003))
    specification = dict(config=config, **operational)
    record = dict(specification, code=code, bank_sha256=manifest_sha["bank_manifest"],
                  observation_contract=dict(actor=222, critic=310),
                  environment_config=dict(original_path=str(old_root / "data/scene")),
                  warmstart=dict(original_cat=True),
                  code_migrations=[dict(audit_path=str(old_root / "outputs/code-audit.json"))])
    spec_hash = migration._spec_sha(specification)
    destination = dict(project="CAT-wholebody", entity="skvayzer", mode="online")
    launch = dict(schema="cat-generalist-launch-v1", owner="owner-123", specification=specification,
                  specification_sha256=spec_hash, wandb=destination)
    status = dict(status="stopped", owner=launch["owner"], resume_state_steps=step,
                  completed_steps=step, observed_steps=step, specification_sha256=spec_hash,
                  pid=999999, last_failure=dict(error="historical numerical failure"))
    wandb = dict(id="f017f302", initialized=True, last_global_step=step, **destination)
    metadata = dict(config=config, bank_sha256=record["bank_sha256"],
                    source_sha256=code["source_sha256"], contract=record["observation_contract"])
    runtime = dict(schema="cat-ppo-runtime-v1", step=step, contract=dict(metadata=metadata, sapg=sapg),
                   training_state=dict(actor=np.arange(20, dtype=np.float32),
                                       adam=np.linspace(0, 1, 40, dtype=np.float32)),
                   env_state=dict(curriculum=np.arange(4), qpos=np.zeros((6, 29))),
                   local_key=np.array([10, 100], dtype=np.uint32),
                   key_envs=np.array([[12, 42]], dtype=np.uint32),
                   metrics_logger=dict(episodes=123), training_walltime=99123.)
    for name, value in zip(migration.FILES, (record, launch, status, wandb)):
        write_json(run_dir / name, value)
    (run_dir / "resume.msgpack").write_bytes(serialization.msgpack_serialize(runtime))
    (run_dir / "STOP").touch()
    (run_dir / ".learner.lock").touch()
    (run_dir / "metrics.jsonl").write_text('{"history":"preserved"}\n')
    kwargs = dict(run_dir=run_dir, source_root=source, source_repo=old_root, destination_repo=new_root,
                  source_host="tl-server-0", destination_host="dep-1", expected_step=step,
                  expected_runtime_sha256=migration.io._sha(run_dir / "resume.msgpack"),
                  expected_commit=commit, expected_wandb_id=wandb["id"], manifest_sha256=manifest_sha)
    return dict(kwargs=kwargs, run=run_dir, record=record, source=source)


def hashes(directory):
    return {str(path.relative_to(directory)): migration.io._sha(path)
            for path in directory.rglob("*") if path.is_file()}


def test_dry_run_never_changes_any_file(setup):
    before = hashes(setup["run"])
    result = migration.migrate(**setup["kwargs"])
    assert result["state"] == "verified-dry-run" and result["publication_needed"]
    assert hashes(setup["run"]) == before


def test_apply_only_rebases_paths_and_preserves_full_runtime_and_history(setup):
    before = {name: (setup["run"] / name).read_bytes() for name in
              ("resume.msgpack", "wandb.json", "metrics.jsonl", "STOP")}
    result = migration.migrate(**setup["kwargs"], apply=True)
    assert result["state"] == "complete" and result["runtime_bytes_unchanged"]
    for name, value in before.items():
        assert (setup["run"] / name).read_bytes() == value
    updated = json.loads((setup["run"] / "run.json").read_text())
    history = updated.pop("host_migrations")
    assert len(history) == 1 and history[0]["runtime_sha256"] == setup["kwargs"]["expected_runtime_sha256"]
    for key in migration.PATHS:
        assert updated.pop(key).startswith(str(setup["kwargs"]["destination_repo"]))
    original = {key: value for key, value in setup["record"].items() if key not in migration.PATHS}
    assert updated == original  # Including config, prior code migrations, original environment paths.
    launch = json.loads((setup["run"] / "launch.json").read_text())
    status = json.loads((setup["run"] / "status.json").read_text())
    assert status["specification_sha256"] == launch["specification_sha256"] == migration._spec_sha(launch["specification"])
    before_retry = hashes(setup["run"])
    assert migration.migrate(**setup["kwargs"], apply=True) == result
    assert hashes(setup["run"]) == before_retry
    assert not migration.migrate(**setup["kwargs"])["publication_needed"]


def test_interrupted_publication_keeps_stop_and_can_complete(setup, monkeypatch):
    atomic = migration.io._atomic_bytes

    def fail_run(path, value):
        if path == setup["run"] / "run.json":
            raise OSError("simulated interrupted publication")
        return atomic(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(migration.io, "_atomic_bytes", fail_run)
        with pytest.raises(OSError, match="simulated"):
            migration.migrate(**setup["kwargs"], apply=True)
    assert (setup["run"] / "STOP").exists()
    assert migration.migrate(**setup["kwargs"])["publication_needed"]
    assert migration.migrate(**setup["kwargs"], apply=True)["state"] == "complete"


def test_active_lock_blocks_migration(setup):
    before = hashes(setup["run"])
    with (setup["run"] / ".learner.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            migration.migrate(**setup["kwargs"], apply=True)
    assert hashes(setup["run"]) == before


@pytest.mark.parametrize("problem", ["runtime", "step", "source", "wandb", "manifest", "running", "no-stop"])
def test_rejects_mismatched_or_active_transfer(setup, problem):
    kwargs = copy.deepcopy(setup["kwargs"])
    if problem == "runtime":
        kwargs["expected_runtime_sha256"] = "a" * 64
    elif problem == "step":
        kwargs["expected_step"] += 1
    elif problem == "source":
        (setup["source"] / "cat_ppo/learning/policy/sapg/losses.py").write_text("unreviewed = True\n")
    elif problem == "wandb":
        kwargs["expected_wandb_id"] = "another-run"
    elif problem == "manifest":
        kwargs["manifest_sha256"]["body_collision_bank"] = "b" * 64
    elif problem == "running":
        path = setup["run"] / "status.json"
        value = json.loads(path.read_text())
        value["status"] = "running"
        write_json(path, value)
    else:
        (setup["run"] / "STOP").unlink()
    before = hashes(setup["run"])
    with pytest.raises(ValueError):
        migration.migrate(**kwargs, apply=True)
    assert hashes(setup["run"]) == before


def test_rejects_changed_metadata_after_partial_publication(setup, monkeypatch):
    atomic = migration.io._atomic_bytes

    def fail_launch(path, value):
        if path == setup["run"] / "launch.json":
            raise OSError("interrupted")
        return atomic(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(migration.io, "_atomic_bytes", fail_launch)
        with pytest.raises(OSError):
            migration.migrate(**setup["kwargs"], apply=True)
    path = setup["run"] / "run.json"
    value = json.loads(path.read_text())
    value["config"]["policy_config"]["learning_rate"] = .1
    write_json(path, value)
    with pytest.raises(ValueError, match="Unexpected metadata modification"):
        migration.migrate(**setup["kwargs"], apply=True)
