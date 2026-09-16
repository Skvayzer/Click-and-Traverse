"""One selected native checkpoint per run, with an atomic model/metadata pointer."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import warnings


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _scalar(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Checkpoint selection requires finite metrics")
    return number


def selection_score(metrics, source, *, max_fall_rate=0.0, max_contact_rate=0.0):
    """Return a lexicographic score; no final-step or tie-based replacement."""
    if source == "training_proxy":
        return (_scalar(metrics["proxy_score"]),), {"eligible": None, "proxy": "epoch_mean_training_reward"}
    if source == "retention_validation":
        selection = metrics["selection"]
        score = tuple(_scalar(value) for value in selection["score"])
        if len(score) != 4 or type(selection["eligible"]) is not bool:
            raise ValueError("Malformed retention selection")
        return score, {key: selection[key] for key in (
            "eligible", "max_cat_drop", "max_scene_drop", "ordering")}
    if source != "validation":
        raise ValueError("Selection source must be validation or training_proxy")
    fall, contact, success = (_scalar(metrics[key]) for key in ("fall_rate", "contact_rate", "strict_success_rate"))
    elapsed = _scalar(metrics["completion_time"])
    if not all(0 <= value <= 1 for value in (fall, contact, success, max_fall_rate, max_contact_rate)) or elapsed < 0:
        raise ValueError("Invalid rate or completion time for checkpoint selection")
    eligible = fall <= max_fall_rate and contact <= max_contact_rate
    return (float(eligible), success, -elapsed), {
        "eligible": eligible, "max_fall_rate": max_fall_rate,
        "max_contact_rate": max_contact_rate, "ordering": ["both_safety_gates", "strict_success_rate", "negative_completion_time"],
    }


def _manifest(directory):
    records = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed inside a checkpoint: {path}")
        if path.is_file():
            records[str(path.relative_to(directory))] = {
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    if not records:
        raise ValueError("Native checkpoint writer produced no files")
    return records


class BestCheckpointStore:
    """Publish a complete immutable generation through checkpoints/best.

    A temporary second generation exists only while replacing the incumbent.
    Both native parameters and selection.json switch with one symlink rename.
    Only generations carrying this run's owner ID can be garbage-collected.
    """

    def __init__(self, run_dir, *, max_fall_rate=0.0, max_contact_rate=0.0):
        self.read_only = False
        self.run_dir = Path(run_dir).absolute()
        self.checkpoints = self.run_dir / "checkpoints"
        self.generations = self.run_dir / ".checkpoint-generations"
        self.best = self.checkpoints / "best"
        self.max_fall_rate = max_fall_rate
        self.max_contact_rate = max_contact_rate
        for directory in (self.run_dir, self.checkpoints):
            if directory.is_symlink():
                raise ValueError(f"Refusing symlink directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        if self.generations.is_symlink():
            raise ValueError("Refusing symlink generation directory")
        if self.generations.exists():
            owner_file = self.generations / "owner.json"
            if owner_file.is_symlink() or not owner_file.is_file():
                raise ValueError("Existing generation directory has no valid ownership marker")
            self.owner = json.loads(owner_file.read_text())["owner"]
        else:
            self.generations.mkdir()
            self.owner = str(uuid.uuid4())
            _json(self.generations / "owner.json", {"owner": self.owner, "schema_version": 1})
        if self.best.exists() or self.best.is_symlink():
            with self._lock():
                selected = self.selected(verify=True)
                self._cleanup(Path(selected["generation"]))

    @classmethod
    def open_existing(cls, run_dir):
        """Inspect an existing selection without creating or cleaning any files.

        Consumers such as curriculum handoff verification must not invoke the
        writer constructor, whose recovery behavior can remove old generations.
        """
        self = cls.__new__(cls)
        self.read_only = True
        self.run_dir = Path(run_dir).absolute()
        self.checkpoints = self.run_dir / "checkpoints"
        self.generations = self.run_dir / ".checkpoint-generations"
        self.best = self.checkpoints / "best"
        for directory in (self.run_dir, self.checkpoints, self.generations):
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError(f"Expected existing regular checkpoint directory: {directory}")
        owner_file = self.generations / "owner.json"
        if owner_file.is_symlink() or not owner_file.is_file():
            raise ValueError("Existing generation directory has no valid ownership marker")
        self.owner = json.loads(owner_file.read_text())["owner"]
        return self

    @contextmanager
    def _lock(self):
        path = self.checkpoints / ".selection.lock"
        if path.is_symlink():
            raise ValueError("Refusing symlink checkpoint lock")
        with path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def selected(self, *, verify=True):
        if not self.best.exists() and not self.best.is_symlink():
            return None
        if not self.best.is_symlink():
            raise ValueError("Existing best path is not this store's selection link")
        generation = self.best.resolve(strict=True)
        if generation.parent != self.generations.resolve() or not generation.name.startswith("candidate-"):
            raise ValueError("Selection link points outside this run's owned generations")
        owner_file = generation / "owner.json"
        selection_file = generation / "selection.json"
        if owner_file.is_symlink() or selection_file.is_symlink():
            raise ValueError("Symlink selection metadata is not allowed")
        if json.loads(owner_file.read_text())["owner"] != self.owner:
            raise ValueError("Checkpoint owner mismatch")
        record = json.loads(selection_file.read_text())
        if record["owner"] != self.owner or record["native_path"] != "native":
            raise ValueError("Invalid checkpoint selection record")
        native = generation / "native"
        if native.is_symlink() or not native.is_dir():
            raise ValueError("Invalid native checkpoint directory")
        if verify and _manifest(native) != record["files"]:
            raise ValueError("Selected checkpoint hash manifest does not match its payload")
        return {**record, "path": str(native), "generation": str(generation)}

    def _cleanup(self, keep):
        for generation in self.generations.glob("candidate-*"):
            if generation == keep or generation.is_symlink() or not generation.is_dir():
                continue
            owner_file = generation / "owner.json"
            if owner_file.is_file() and not owner_file.is_symlink() and json.loads(owner_file.read_text()).get("owner") == self.owner:
                shutil.rmtree(generation)

    def consider(self, *, step, metrics, source, write_checkpoint, contract, provenance=None):
        if self.read_only:
            raise ValueError("Cannot write through a read-only checkpoint store")
        if not isinstance(step, int) or step <= 0:
            raise ValueError("A learned checkpoint needs a positive integer training step")
        score, rules = selection_score(metrics, source, max_fall_rate=self.max_fall_rate,
                                       max_contact_rate=self.max_contact_rate)
        if source == "retention_validation" and not rules["eligible"]:
            return False
        # Reject malformed metadata before invoking any expensive checkpoint writer.
        json.dumps({"metrics": metrics, "contract": contract, "provenance": provenance}, allow_nan=False)
        with self._lock():
            previous = self.selected(verify=True)
            if previous and previous["selection_source"] != source:
                raise ValueError("Cannot compare validation scores with training proxies in one run")
            if previous and previous["rules"] != rules:
                # Eligibility changes with results; selection limits/order must not.
                old_rules = {k: v for k, v in previous["rules"].items() if k != "eligible"}
                new_rules = {k: v for k, v in rules.items() if k != "eligible"}
                if old_rules != new_rules:
                    raise ValueError("Cannot change checkpoint selection rules within a run")
            if previous and tuple(score) <= tuple(previous["score"]):
                self._cleanup(Path(previous["generation"]))
                return False
            generation = Path(tempfile.mkdtemp(prefix="candidate-", dir=self.generations))
            _json(generation / "owner.json", {"owner": self.owner})
            published = False
            temporary_link = self.checkpoints / f".best-{uuid.uuid4().hex}"
            try:
                native = generation / "native"
                write_checkpoint(native)
                if native.is_symlink() or not native.is_dir():
                    raise ValueError("Writer must create the supplied native checkpoint directory")
                _json(native / "observation_contract.json", contract)
                record = {"schema_version": 1, "owner": self.owner, "native_path": "native",
                    "step": step, "selection_source": source, "score": list(score), "rules": rules,
                    "metrics": metrics, "provenance": provenance or {}, "files": _manifest(native)}
                _json(generation / "selection.json", record)
                _fsync_directory(native);_fsync_directory(generation)
                os.symlink(os.path.relpath(generation, self.checkpoints), temporary_link)
                os.replace(temporary_link, self.best)
                published = True
                _fsync_directory(self.checkpoints)
            finally:
                if temporary_link.is_symlink():
                    temporary_link.unlink()
                # A signal can arrive immediately after rename, before the next
                # Python assignment. Never delete the generation now selected.
                if not published and self.best.is_symlink() and self.best.resolve() == generation:
                    published = True
                if not published:
                    shutil.rmtree(generation)
            try:
                self._cleanup(generation)
            except OSError as exc:
                warnings.warn(f"Best checkpoint published; old generation cleanup will be retried: {exc}")
            return True


def native_writer(params, network_config, step):
    """Adapt Brax's numbered-directory writer to this store's native directory."""
    def write(destination):
        from brax.training.agents.ppo import checkpoint as brax_checkpoint
        destination = Path(destination)
        staging = destination.parent / "orbax-staging"
        brax_checkpoint.save(staging, step, params, network_config)
        os.rename(staging / f"{step:012d}", destination)
        staging.rmdir()
    return write
