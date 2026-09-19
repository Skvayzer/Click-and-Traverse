#!/usr/bin/env python3
"""Rebase a stopped, byte-identical learner onto another host's filesystem.

The full learner file is never rewritten. Only four operational paths and their
launch hash change. Historical environment/code provenance remains untouched.
Run without --apply first. A destination STOP marker protects interrupted
publication; deliberately retire it only after this tool reports complete.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import time


_spec = importlib.util.spec_from_file_location(
    "cat_migration_io", Path(__file__).with_name("migrate_sapg_numerics_resume.py"))
io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(io)

FILES = ("run.json", "launch.json", "status.json", "wandb.json")
PATHS = ("bank_manifest", "body_collision_bank", "body_collision_resets", "stop")
SCHEMA = "cat-host-migration-v1"


def _spec_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _rebase(value, old_root, new_root):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Operational paths must be canonical absolute paths")
    try:
        return str(new_root / path.relative_to(old_root))
    except ValueError as error:
        raise ValueError(f"Operational path is outside source repository: {value}") from error


def _metadata(directory):
    payloads = {name: io._regular(directory / name).read_bytes() for name in FILES}
    return payloads, {name: json.loads(value) for name, value in payloads.items()}


def _validate(original, header, old_root, new_root, run_dir, expected):
    record, launch, status, wandb = (original[name] for name in FILES)
    spec = launch["specification"]
    if launch.get("schema") != "cat-generalist-launch-v1":
        raise ValueError("Unsupported launch schema")
    if status.get("status") != "stopped":
        raise ValueError("Host migration requires a cleanly stopped learner")
    if status.get("owner") != launch.get("owner"):
        raise ValueError("Run ownership differs")
    if status.get("resume_state_steps") != expected["step"] or header["step"] != expected["step"]:
        raise ValueError("Saved learner step differs from expected source checkpoint")
    if any(value != _spec_sha(spec) for value in
           (launch.get("specification_sha256"), status.get("specification_sha256"))):
        raise ValueError("Original launch specification hash differs")
    if any(record.get(key) != value for key, value in spec.items()):
        raise ValueError("Original run record differs from launch specification")
    if record.get("code") != expected["code"]:
        raise ValueError("Target source differs from checkpoint source identity")
    config = record["config"]
    if config.get("algorithm") != "sapg":
        raise ValueError("This migration is for complete SAPG learners")
    metadata = dict(config=config, bank_sha256=record["bank_sha256"],
                    source_sha256=record["code"]["source_sha256"],
                    contract=record["observation_contract"])
    if header["contract"].get("metadata") != metadata or header["contract"].get("sapg") != config["sapg"]:
        raise ValueError("Full learner contract differs from run record")
    if (wandb.get("id") != expected["wandb_id"] or not wandb.get("initialized")
            or any(wandb.get(key) != value for key, value in launch["wandb"].items())):
        raise ValueError("Original W&B identity differs")
    if wandb.get("last_global_step") != expected["step"]:
        raise ValueError("Cleanly stopped W&B step differs from the full learner")
    rebased = {key: _rebase(spec[key], old_root, new_root) for key in PATHS}
    if rebased["stop"] != str(run_dir / "STOP"):
        raise ValueError("Destination run path differs from the original run's relative location")
    if expected["manifest_sha256"]["bank_manifest"] != record["bank_sha256"]:
        raise ValueError("Expected field manifest differs from the recorded bank")
    for key, fingerprint in expected["manifest_sha256"].items():
        path = Path(rebased[key])
        if path.resolve() != path or io._sha(path) != fingerprint:
            raise ValueError(f"Transferred manifest identity differs: {key}")
    return rebased


def migrate(run_dir, source_root, source_repo, destination_repo, *, source_host,
            destination_host, expected_step, expected_runtime_sha256,
            expected_commit, expected_wandb_id, manifest_sha256, apply=False):
    """Verify the transfer, then journal and rebase only operational metadata."""
    for name, value in (("source_host", source_host), ("destination_host", destination_host)):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(f"Invalid {name}")
    if source_host == destination_host:
        raise ValueError("Source and destination host must differ")
    if expected_step <= 0 or not re.fullmatch(r"[a-f0-9]{64}", expected_runtime_sha256):
        raise ValueError("Explicit positive step and checkpoint SHA256 are required")
    if set(manifest_sha256) != set(PATHS[:-1]):
        raise ValueError("Exactly the field, collision and reset manifest hashes are required")
    if any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in manifest_sha256.values()):
        raise ValueError("Invalid manifest SHA256")
    run_dir, source_root, new_root = map(lambda value: Path(value).absolute(),
                                       (run_dir, source_root, destination_repo))
    old_root = Path(source_repo)
    if not old_root.is_absolute() or ".." in old_root.parts or old_root == new_root:
        raise ValueError("Source and destination repositories must be distinct absolute paths")
    for directory in (run_dir, source_root, new_root):
        if not directory.is_dir() or directory.resolve() != directory:
            raise ValueError(f"Expected canonical regular directory: {directory}")
    # STOP remains in place even after success. Normal launch cannot start from
    # a partially published bundle; its owner retires the marker deliberately.
    if io._regular(run_dir / "STOP").read_bytes() != b"":
        raise ValueError("Migration requires the copied empty STOP marker")
    with io._regular(run_dir / ".learner.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, _ = io.source_identity(source_root, expected_commit, verify_checkout=True)
        expected = dict(step=expected_step, runtime_sha256=expected_runtime_sha256,
                        code=code, wandb_id=expected_wandb_id, manifest_sha256=manifest_sha256)
        request = dict(source_host=source_host, destination_host=destination_host,
                       source_repository=str(old_root), destination_repository=str(new_root),
                       source_checkout=str(source_root), destination_run=str(run_dir), **expected)
        audit_parent = run_dir / "host-migrations"
        audit = audit_parent / f"{source_host}-to-{destination_host}-{expected_step}"
        if audit_parent.is_symlink() or audit.is_symlink():
            raise ValueError("Migration audit directory cannot be a symlink")
        journal = None
        if audit.exists():
            journal = json.loads(io._regular(audit / "migration.json").read_text())
            if journal.get("schema") != SCHEMA or journal.get("request") != request:
                raise ValueError("Existing host migration has different expectations")
            original_bytes, original = _metadata(audit / "original")
            if {name: _hash_bytes(value) for name, value in original_bytes.items()} != journal["before_sha256"]:
                raise ValueError("Archived metadata differs from migration journal")
        else:
            original_bytes, original = _metadata(run_dir)
        current_bytes, _ = _metadata(run_dir)
        for name, value in current_bytes.items():
            permitted = {_hash_bytes(original_bytes[name])}
            if journal:
                permitted.add(journal["after_sha256"][name])
            if _hash_bytes(value) not in permitted:
                raise ValueError(f"Unexpected metadata modification: {name}")
        runtime = run_dir / "resume.msgpack"
        if io._sha(runtime) != expected_runtime_sha256:
            raise ValueError("Transferred learner differs from source checkpoint SHA256")
        header, _ = io.runtime_header(runtime)
        rebased = _validate(original, header, old_root, new_root, run_dir, expected)
        if journal is None:
            entry = dict(schema=SCHEMA, created_epoch=time.time(), **request,
                         operational_paths={key: dict(before=original["run.json"][key], after=value)
                                            for key, value in rebased.items()},
                         runtime_bytes_unchanged=True,
                         historical_environment_and_code_provenance_preserved=True,
                         audit_path=str(audit / "migration.json"))
            migrated = copy.deepcopy(original)
            migrated["launch.json"]["specification"].update(rebased)
            spec_hash = _spec_sha(migrated["launch.json"]["specification"])
            migrated["launch.json"]["specification_sha256"] = spec_hash
            migrated["status.json"]["specification_sha256"] = spec_hash
            migrated["run.json"].update(rebased)
            migrated["run.json"]["host_migrations"] = [*original["run.json"].get("host_migrations", []), entry]
            # W&B identity bytes, not just its parsed value, remain identical.
            after_bytes = {name: original_bytes[name] if name == "wandb.json" else io._json_bytes(value)
                           for name, value in migrated.items()}
            journal = dict(schema=SCHEMA, state="pending", request=request, entry=entry,
                           before_sha256={name: _hash_bytes(value) for name, value in original_bytes.items()},
                           after_sha256={name: _hash_bytes(value) for name, value in after_bytes.items()},
                           after_payload={name: value.decode() for name, value in after_bytes.items()})
        else:
            after_bytes = {name: value.encode() for name, value in journal["after_payload"].items()}
            if set(after_bytes) != set(FILES) or any(_hash_bytes(value) != journal["after_sha256"][name]
                                                    for name, value in after_bytes.items()):
                raise ValueError("Migration output differs from journal hashes")
        complete = all(current_bytes[name] == after_bytes[name] for name in FILES)
        if not apply:
            return dict(state="verified-dry-run", publication_needed=not complete, **request,
                        operational_paths=journal["entry"]["operational_paths"])
        if not audit.exists():
            audit_parent.mkdir(exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".host-migration-", dir=audit_parent))
            (temporary / "original").mkdir()
            for name, value in original_bytes.items():
                io._atomic_bytes(temporary / "original" / name, value)
            io._atomic_bytes(temporary / "original" / "STOP", b"")
            io._atomic_bytes(temporary / "migration.json", io._json_bytes(journal))
            os.rename(temporary, audit)
            io._sync_directory(audit_parent)
        for name in ("status.json", "run.json", "launch.json"):
            if current_bytes[name] != after_bytes[name]:
                io._atomic_bytes(run_dir / name, after_bytes[name])
        if io._sha(runtime) != expected_runtime_sha256:
            raise ValueError("Learner checkpoint changed during host migration")
        if journal["state"] != "complete":
            journal["state"] = "complete"
            io._atomic_bytes(audit / "migration.json", io._json_bytes(journal))
        return dict(state="complete", audit_path=str(audit / "migration.json"),
                    runtime_bytes_unchanged=True, stop_marker_preserved=True, **request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "source-root", "source-repo", "destination-repo"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-host", "destination-host", "expected-runtime-sha256",
                 "expected-commit", "expected-wandb-id", "field-sha256",
                 "collision-sha256", "reset-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    arguments = vars(parser.parse_args())
    arguments["manifest_sha256"] = dict(bank_manifest=arguments.pop("field_sha256"),
        body_collision_bank=arguments.pop("collision_sha256"),
        body_collision_resets=arguments.pop("reset_sha256"))
    print(json.dumps(migrate(**arguments), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
