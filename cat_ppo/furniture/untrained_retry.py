"""Reuse a logging identity only after proving a failed launch never trained.

This is deliberately different from learner recovery: the caller archives the
old directory and creates a fresh learner with an explicitly changed setup.
Only its unused W&B identity may carry over. No files are changed here.
"""
from __future__ import annotations

from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re


_PRETRAINING_PHASES = frozenset({
    "creating", "preparing", "checkpoint_store", "logger", "learner_initialization",
})


def _metadata(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular retry metadata: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid retry metadata: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected retry metadata object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _empty_store(directory, allowed):
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError(f"Unexpected checkpoint storage: {directory}")
    if directory.exists():
        for path in directory.iterdir():
            if path.name not in allowed or path.is_symlink() or not path.is_file():
                raise ValueError(f"Checkpoint evidence prevents unused logging reuse: {path}")


def verified_untrained_logging_identity(previous_run):
    """Return ``(identity, provenance)`` for an archived, failed unused run.

    Run on the training host: its recorded PID must no longer exist. A reused
    PID or inaccessible process is rejected conservatively. A present learner
    lock is also checked without creating or modifying it. Baseline evaluation
    may exist, but even a step-zero learner snapshot or logged event makes the
    identity ineligible: those require exact runtime recovery instead.
    """
    directory = Path(previous_run)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Previous run must be an existing regular directory")
    directory = directory.resolve()
    lock_path = directory / ".learner.lock"
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        raise ValueError("Invalid previous learner lock")
    context = lock_path.open("rb") if lock_path.exists() else nullcontext(None)
    with context as lock:
        if lock is not None:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Previous learner still owns its run lock") from error
        records, hashes = {}, {}
        for name in ("status.json", "launch.json", "run.json", "wandb.json"):
            records[name], hashes[name] = _metadata(directory / name)
        status, launch, record, identity = (records[name] for name in
            ("status.json", "launch.json", "run.json", "wandb.json"))
        if (status.get("schema") != "cat-generalist-status-v1"
                or status.get("status") != "failed"
                or status.get("phase") not in _PRETRAINING_PHASES):
            raise ValueError("Logging reuse requires a failed pretraining launch")
        pid = status.get("pid")
        if type(pid) is not int or pid <= 0:
            raise ValueError("Previous launch must identify its process")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        except OSError as error:
            raise ValueError("Cannot establish that the previous learner exited") from error
        else:
            raise ValueError("Previous learner PID is still live or has been reused")
        for name in ("completed_steps", "resume_state_steps", "observed_steps", "global_step",
                     "training_steps", "env_steps", "num_updates", "updates"):
            value = status.get(name, 0)
            if type(value) is not int or value != 0:
                raise ValueError(f"Previous launch has training progress: {name}")
        for name in ("initial_runtime_written", "runtime_present_on_entry"):
            if status.get(name, False) is not False:
                raise ValueError("A learner runtime was previously created; exact recovery is required")
        owner = launch.get("owner")
        specification = launch.get("specification")
        if (launch.get("schema") != "cat-generalist-launch-v1"
                or not isinstance(owner, str) or not owner or status.get("owner") != owner
                or not isinstance(specification, dict) or not specification):
            raise ValueError("Previous launch/status ownership or specification is invalid")
        spec_hash = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
        if (launch.get("specification_sha256") != spec_hash
                or status.get("specification_sha256") != spec_hash
                or any(key not in record or record[key] != value for key, value in specification.items())):
            raise ValueError("Previous launch, status, and prepared run specification differ")
        if not isinstance(record.get("config"), dict) or not record["config"]:
            raise ValueError("Previous prepared run has no training configuration")
        destination = launch.get("wandb")
        if (not isinstance(destination, dict)
                or any(identity.get(key) != destination.get(key) for key in ("project", "entity", "mode"))
                or any(not isinstance(identity.get(key), str) or not identity[key] for key in ("project", "entity"))
                or identity.get("mode") not in ("online", "disabled")
                or not isinstance(identity.get("id"), str)
                or re.fullmatch(r"[0-9a-f]{8}", identity["id"]) is None
                or type(identity.get("last_global_step")) is not int or identity["last_global_step"] != -1):
            raise ValueError("Previous W&B identity has progress or differs from its launch")
        for name in ("initialized", "initialization_attempted"):
            if name in identity and type(identity[name]) is not bool:
                raise ValueError("Malformed W&B initialization evidence")
        for name in ("resume.msgpack", "initial_runtime.msgpack", "validation_latest.json"):
            path = directory / name
            if path.exists() or path.is_symlink():
                raise ValueError(f"Learner runtime or training validation evidence exists: {name}")
        if any(path.name.startswith(".resume.msgpack") for path in directory.iterdir()):
            raise ValueError("An incomplete learner runtime prevents logging reuse")
        _empty_store(directory / "checkpoints", {".selection.lock"})
        _empty_store(directory / ".checkpoint-generations", {"owner.json"})
        metrics = directory / "metrics.jsonl"
        if metrics.is_symlink() or (metrics.exists() and (not metrics.is_file() or metrics.stat().st_size)):
            raise ValueError("Previously logged metrics prevent unused logging reuse")
        provenance = {
            "schema": "cat-untrained-logging-reuse-v1", "previous_run": str(directory),
            "previous_owner": owner, "previous_pid": pid, "previous_status": status,
            "previous_specification_sha256": spec_hash,
            "previous_config_sha256": hashlib.sha256(json.dumps(record["config"], sort_keys=True).encode()).hexdigest(),
            "previous_warmstart_best": record.get("warmstart_best"),
            "metadata_sha256": hashes, "wandb_id": identity["id"],
            "training_steps": 0, "runtime_state_reused": False,
        }
        return identity, provenance
