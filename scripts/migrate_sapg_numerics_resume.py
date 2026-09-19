"""Offline, audited code-identity migration for one failed SAPG run.

Run as a CPU Slurm job after archiving the stopped run. The default is read-only
verification; --apply changes only run.json code provenance and the runtime
contract's source fingerprint. Normal launcher/resume checks remain unchanged.
No array is deserialized, optimizer reset, configuration changed, or W&B call
made. A pending journal supports an interrupted two-file publication.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import msgpack


SOURCE_COMMIT = "2d2f76acc20536e07979497ef589bbe0e192d646"
EXPECTED_STEP = 326762496
EXPECTED_WANDB_ID = "f017f302"
EXPECTED_RUNTIME_SHA256 = "69add45620a7d5dacb748e3b8123c605896bf6f4a8bcf1a6c9405c42299e2276"
EXPECTED_RUN_SHA256 = "7bbe29ab93a2064071fb74f906384223db0b3ac9a2b35b0418105c3a46a37ba9"
ALLOWED_SOURCE_CHANGE = "cat_ppo/learning/policy/sapg/losses.py"
METADATA_FILES = ("run.json", "launch.json", "status.json", "wandb.json")
AUDIT_SCHEMA = "cat-sapg-numerics-code-migration-v1"
CHUNK = 8 * 1024 * 1024


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular, non-symlink file: {path}")
    return path


def _sha(path):
    digest = hashlib.sha256()
    with _regular(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _atomic_bytes(path, payload):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Unsafe metadata destination: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def source_identity(root, commit, *, verify_checkout=False):
    """Match launcher's fingerprint, including its complete Python file set."""
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("Source commits must be exact lowercase 40-character hashes")
    resolved = _git(root, "rev-parse", "--verify", commit + "^{commit}").decode().strip()
    if resolved != commit:
        raise ValueError("Source commit did not resolve exactly")
    names = _git(root, "ls-tree", "-r", "--name-only", commit, "--", "cat_ppo", "train_cat_wholebody.py")
    paths = sorted(name for name in names.decode().splitlines() if name.endswith(".py"))
    if "train_cat_wholebody.py" not in paths or ALLOWED_SOURCE_CHANGE not in paths:
        raise ValueError("Incomplete CAT source checkout")
    hashes = {name: hashlib.sha256(_git(root, "show", f"{commit}:{name}")).hexdigest() for name in paths}
    if verify_checkout:
        if _git(root, "rev-parse", "HEAD").decode().strip() != commit:
            raise ValueError("Source checkout HEAD is not the reviewed target commit")
        actual_paths = sorted([str(path.relative_to(root)) for path in (root / "cat_ppo").rglob("*.py")]
                              + ["train_cat_wholebody.py"])
        if actual_paths != paths or any(_sha(root / name) != fingerprint for name, fingerprint in hashes.items()):
            raise ValueError("Target runtime source differs from its reviewed commit")
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return dict(git_commit=commit, source_sha256=fingerprint), hashes


def runtime_header(path):
    """Locate small metadata while seeking over opaque Flax ndarray payloads."""
    def integer(stream, width):
        data = stream.read(width)
        if len(data) != width:
            raise ValueError("Truncated msgpack integer")
        return int.from_bytes(data, "big")

    def skip(stream, size):
        pending = 1
        while pending:
            tag = integer(stream, 1)
            pending -= 1
            if tag <= 0x7f or tag >= 0xe0 or tag in (0xc0, 0xc2, 0xc3):
                continue
            if 0x80 <= tag <= 0x8f:
                pending += (tag & 0xf) * 2
            elif 0x90 <= tag <= 0x9f:
                pending += tag & 0xf
            elif 0xa0 <= tag <= 0xbf:
                stream.seek(tag & 0x1f, 1)
            elif tag in (0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xd9, 0xda, 0xdb):
                width = {0xc4: 1, 0xc5: 2, 0xc6: 4, 0xc7: 1, 0xc8: 2, 0xc9: 4,
                         0xd9: 1, 0xda: 2, 0xdb: 4}[tag]
                length = integer(stream, width)
                stream.seek(length + (tag in (0xc7, 0xc8, 0xc9)), 1)
            elif tag in (0xdc, 0xdd, 0xde, 0xdf):
                count = integer(stream, 2 if tag in (0xdc, 0xde) else 4)
                pending += count * (2 if tag in (0xde, 0xdf) else 1)
            else:
                widths = {0xca: 4, 0xcb: 8, 0xcc: 1, 0xcd: 2, 0xce: 4, 0xcf: 8,
                          0xd0: 1, 0xd1: 2, 0xd2: 4, 0xd3: 8,
                          0xd4: 2, 0xd5: 3, 0xd6: 5, 0xd7: 9, 0xd8: 17}
                if tag not in widths:
                    raise ValueError("Invalid msgpack token")
                stream.seek(widths[tag], 1)
            if stream.tell() > size:
                raise ValueError("Msgpack extends beyond file")

    def small(stream, size):
        start = stream.tell()
        skip(stream, size)
        end = stream.tell()
        if end - start > 16 * 1024 * 1024:
            raise ValueError("Unexpectedly large runtime metadata")
        stream.seek(start)
        return msgpack.unpackb(stream.read(end - start), raw=False), (start, end)

    with _regular(path).open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        tag = integer(stream, 1)
        if 0x80 <= tag <= 0x8f:
            count = tag & 0xf
        elif tag in (0xde, 0xdf):
            count = integer(stream, 2 if tag == 0xde else 4)
        else:
            raise ValueError("Runtime must be a msgpack map")
        values, spans, keys = {}, {}, set()
        for _ in range(count):
            key, _ = small(stream, size)
            if not isinstance(key, str) or key in keys:
                raise ValueError("Invalid or duplicate runtime key")
            keys.add(key)
            if key in ("schema", "step", "contract"):
                values[key], spans[key] = small(stream, size)
            else:
                skip(stream, size)
        if stream.tell() != size or values.get("schema") != "cat-ppo-runtime-v1":
            raise ValueError("Invalid runtime schema or trailing bytes")
        if not {"schema", "step", "contract", "training_state", "env_state", "local_key", "key_envs"} <= keys:
            raise ValueError("Incomplete exact learner snapshot")
    return values, spans


def rewrite_runtime(source, destination, new_contract):
    """Change exactly one msgpack object; copy all other bytes verbatim."""
    _, spans = runtime_header(source)
    first, last = spans["contract"]
    old_hash, new_hash, opaque_hash = (hashlib.sha256() for _ in range(3))
    with _regular(source).open("rb") as source_stream, destination.open("xb") as output:
        def copy(count):
            while count:
                chunk = source_stream.read(min(CHUNK, count))
                if not chunk:
                    raise ValueError("Runtime changed while copying")
                output.write(chunk)
                old_hash.update(chunk)
                new_hash.update(chunk)
                opaque_hash.update(chunk)
                count -= len(chunk)

        copy(first)
        old_hash.update(source_stream.read(last - first))
        replacement = msgpack.packb(new_contract, use_bin_type=True)
        output.write(replacement)
        new_hash.update(replacement)
        copy(os.fstat(source_stream.fileno()).st_size - last)
        output.flush()
        os.fsync(output.fileno())
    return dict(before_sha256=old_hash.hexdigest(), after_sha256=new_hash.hexdigest(),
                preserved_noncontract_bytes_sha256=opaque_hash.hexdigest())


def _read_bundle(directory):
    payloads = {name: _regular(directory / name).read_bytes() for name in METADATA_FILES}
    return {name: json.loads(value) for name, value in payloads.items()}, payloads


def _validate_original(bundle, header, old_identity):
    record, launch, status, wandb = (bundle[name] for name in METADATA_FILES)
    if record["code"] != old_identity or record.get("code_migrations"):
        raise ValueError("Run does not have the audited original source identity")
    if header["step"] != EXPECTED_STEP or status.get("resume_state_steps") != EXPECTED_STEP:
        raise ValueError("Saved runtime is not the authorized migration step")
    if wandb.get("id") != EXPECTED_WANDB_ID or wandb.get("mode") != "online":
        raise ValueError("W&B identity is not the authorized existing run")
    if (status.get("status") != "failed" or status.get("error_type") != "FloatingPointError"
            or "Non-finite training metrics" not in status.get("error", "")):
        raise ValueError("Missing the recorded numerical failure")
    if status.get("owner") != launch.get("owner"):
        raise ValueError("Run ownership differs")
    specification = launch["specification"]
    spec_hash = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    if (launch.get("schema") != "cat-generalist-launch-v1"
            or launch.get("specification_sha256") != spec_hash
            or status.get("specification_sha256") != spec_hash
            or any(record.get(key) != value for key, value in specification.items())):
        raise ValueError("Original launch specification/configuration is inconsistent")
    if launch.get("wandb") != {key: wandb[key] for key in ("project", "entity", "mode")}:
        raise ValueError("Logging destination differs from the original launch")
    config = record["config"]
    if config.get("algorithm") != "sapg" or config["fine_tuning"].get("mode") != "cat_train_only":
        raise ValueError("Only the audited SAPG training-only run may migrate")
    expected_metadata = dict(config=config, bank_sha256=record["bank_sha256"],
        source_sha256=old_identity["source_sha256"], contract=record["observation_contract"])
    if header["contract"].get("metadata") != expected_metadata:
        raise ValueError("Runtime config/scene/observation/source identity differs from original run")
    if header["contract"].get("sapg") != config.get("sapg"):
        raise ValueError("Runtime SAPG configuration differs")


def migrate(run_dir, backup_dir, source_root, target_commit, *, apply=False):
    """Verify/apply this one migration; safely finish its interrupted publication."""
    for directory in (run_dir, backup_dir, source_root):
        if Path(directory).is_symlink() or not Path(directory).is_dir():
            raise ValueError(f"Expected real directory: {directory}")
    run_dir, backup_dir, source_root = (Path(value).resolve() for value in (run_dir, backup_dir, source_root))
    if run_dir == backup_dir or backup_dir.is_relative_to(run_dir):
        raise ValueError("Original backup must be outside the active run directory")
    with _regular(run_dir / ".learner.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _migrate_locked(run_dir, backup_dir, source_root, target_commit, apply=apply)


def _migrate_locked(run_dir, backup_dir, source_root, target_commit, *, apply):
    old_identity, old_files = source_identity(source_root, SOURCE_COMMIT)
    new_identity, new_files = source_identity(source_root, target_commit, verify_checkout=True)
    subprocess.run(["git", "-C", str(source_root), "merge-base", "--is-ancestor", SOURCE_COMMIT, target_commit], check=True)
    changed = sorted(name for name in old_files.keys() | new_files.keys() if old_files.get(name) != new_files.get(name))
    if changed != [ALLOWED_SOURCE_CHANGE]:
        raise ValueError(f"Migration allows only the reviewed SAPG numerical loss change: {changed}")
    original, original_bytes = _read_bundle(backup_dir)
    header, _ = runtime_header(backup_dir / "resume.msgpack")
    _validate_original(original, header, old_identity)
    before_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in original_bytes.items()}
    before_hashes["resume.msgpack"] = _sha(backup_dir / "resume.msgpack")
    if (before_hashes["resume.msgpack"] != EXPECTED_RUNTIME_SHA256
            or before_hashes["run.json"] != EXPECTED_RUN_SHA256):
        raise ValueError("Backup differs from the hash-verified authorized failure archive")
    identity = f"sapg-numerics-{EXPECTED_STEP}-{target_commit[:12]}"
    audit_dir = run_dir / "code-migrations" / identity
    audit_path = audit_dir / "migration.json"
    if (run_dir / "code-migrations").is_symlink() or audit_dir.is_symlink():
        raise ValueError("Migration audit storage cannot be a symlink")
    prior = json.loads(_regular(audit_path).read_bytes()) if audit_path.exists() else None
    if prior is not None and (prior.get("schema") != AUDIT_SCHEMA or prior.get("id") != identity
            or prior.get("backup_dir") != str(backup_dir) or prior.get("before_hashes") != before_hashes
            or prior.get("old_code") != old_identity or prior.get("new_code") != new_identity):
        raise ValueError("Existing migration journal belongs to different evidence")
    if audit_dir.exists() and prior is None:
        raise ValueError("Unrecognized migration directory without its journal")
    created = prior["created_utc"] if prior else datetime.now(timezone.utc).isoformat()
    audit = dict(schema=AUDIT_SCHEMA, id=identity, created_utc=created, backup_dir=str(backup_dir),
        run_dir=str(run_dir), source_root=str(source_root), old_code=old_identity, new_code=new_identity,
        allowed_changed_source=changed, source_file_hashes={ALLOWED_SOURCE_CHANGE: {
            "before": old_files[ALLOWED_SOURCE_CHANGE], "after": new_files[ALLOWED_SOURCE_CHANGE]}},
        saved_step=EXPECTED_STEP, wandb_id=EXPECTED_WANDB_ID, before_hashes=before_hashes,
        state="verified", failure=original["status.json"], launch_specification_changed=False,
        preserved="all optimizer, actor/critic/embedding, RNG, normalizer, environment, counter and step bytes")
    # launch/status/W&B are never changed, including on interrupted retry.
    for name in ("launch.json", "status.json", "wandb.json"):
        if _sha(run_dir / name) != before_hashes[name]:
            raise ValueError(f"Current {name} differs from the archived stopped run")
    current_resume_hash = _sha(run_dir / "resume.msgpack")
    current_run_hash = _sha(run_dir / "run.json")
    allowed_runtime = {before_hashes["resume.msgpack"]}
    allowed_run = {before_hashes["run.json"]}
    if prior is not None:
        allowed_runtime.add(prior.get("runtime", {}).get("after_sha256"))
        allowed_run.add(prior.get("after_run_sha256"))
    if current_resume_hash not in allowed_runtime or current_run_hash not in allowed_run:
        raise ValueError("Current run/runtime differs from archived evidence and this migration journal")
    if not apply:
        return {**audit, "state": "verified-dry-run", "publication_needed": (
            prior is None or prior.get("state") != "complete")}
    if prior is not None and prior.get("state") == "complete":
        if current_resume_hash != prior["runtime"]["after_sha256"] or current_run_hash != prior["after_run_sha256"]:
            raise ValueError("Completed migration files were subsequently replaced")
        return prior
    audit_dir.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(audit_path, _json_bytes({**audit, "state": "staging", **({
        "runtime": prior["runtime"], "after_run_sha256": prior["after_run_sha256"]}
        if prior and "runtime" in prior else {})}))
    staged_runtime = audit_dir / "resume.msgpack.staged"
    if staged_runtime.exists():
        _regular(staged_runtime).unlink()
    contract = {**header["contract"], "metadata": {**header["contract"]["metadata"],
                                                    "source_sha256": new_identity["source_sha256"]}}
    proof = rewrite_runtime(backup_dir / "resume.msgpack", staged_runtime, contract)
    if proof["before_sha256"] != before_hashes["resume.msgpack"] or _sha(staged_runtime) != proof["after_sha256"]:
        raise ValueError("Runtime publication failed byte-integrity verification")
    changed_header, _ = runtime_header(staged_runtime)
    if changed_header != {**header, "contract": contract}:
        raise ValueError("Runtime migration changed unexpected metadata")
    entry = dict(schema=AUDIT_SCHEMA, id=identity, created_utc=created,
        reason="Reviewed SAPG loss numerical-stability fix; resume the same complete saved learner",
        from_code=old_identity, to_code=new_identity, saved_step=EXPECTED_STEP,
        wandb_id=EXPECTED_WANDB_ID, backup_dir=str(backup_dir), audit_path=str(audit_path),
        runtime=proof, changed_source=changed)
    new_record = {**original["run.json"], "code": new_identity, "code_migrations": [entry]}
    new_run_bytes = _json_bytes(new_record)
    after_run_sha = hashlib.sha256(new_run_bytes).hexdigest()
    if prior and "runtime" in prior and (prior["runtime"] != proof or prior["after_run_sha256"] != after_run_sha):
        raise ValueError("Interrupted migration reconstruction differs from prepared journal")
    audit.update(state="prepared", runtime=proof, after_run_sha256=after_run_sha)
    _atomic_bytes(audit_path, _json_bytes(audit))
    # A crash between replacements fails ordinary strict resume. Re-running this
    # utility accepts only the exact old/new hashes recorded above and finishes.
    os.replace(staged_runtime, run_dir / "resume.msgpack")
    _sync_directory(run_dir)
    _atomic_bytes(run_dir / "run.json", new_run_bytes)
    audit["state"] = "complete"
    _atomic_bytes(audit_path, _json_bytes(audit))
    return audit


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--run-dir", type=Path, required=True)
    arguments.add_argument("--backup-dir", type=Path, required=True)
    arguments.add_argument("--source-root", type=Path, required=True)
    arguments.add_argument("--target-commit", required=True)
    arguments.add_argument("--apply", action="store_true")
    args = arguments.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        arguments.error("Run this substantial CPU checkpoint operation through Slurm")
    print(json.dumps(migrate(**vars(args)), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
