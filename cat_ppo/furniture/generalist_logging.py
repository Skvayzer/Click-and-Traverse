"""One experiment identity and explicit, monotonic CAT training telemetry."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid


def atomic_json(path, value):
    path = Path(path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Metadata must be a regular file: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class GeneralistLogger:
    """Scene diagnostics share one run; charts use committed learner transitions.

    W&B event indices are intentionally independent of global_step: episode and
    optimizer reports can occur at the same step. On recovery from an older
    durable snapshot, already-reported steps are suppressed until caught up.
    """

    @staticmethod
    def reserve_identity(run_dir, *, project, entity, mode, resume=False):
        """Allocate a local ID before preparation, without opening an online run."""
        path = Path(run_dir) / "wandb.json"
        if mode not in ("online", "disabled"):
            raise ValueError("Expected online or disabled W&B mode")
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("W&B identity must be a regular file")
        if path.exists():
            if not resume:
                raise ValueError("Experiment identity already exists; use --resume")
            identity = json.loads(path.read_text())
            for key, value in (("project", project), ("entity", entity), ("mode", mode)):
                if identity[key] != value:
                    raise ValueError(f"Cannot change W&B {key} when resuming")
            if (not re.fullmatch(r"[0-9a-f]{8}", identity["id"]) or
                    type(identity["last_global_step"]) is not int or identity["last_global_step"] < -1):
                raise ValueError("Malformed W&B identity")
        else:
            identity = dict(id=uuid.uuid4().hex[:8], project=project, entity=entity,
                            mode=mode, last_global_step=-1, initialized=False,
                            initialization_attempted=False)
            atomic_json(path, identity)
        return identity

    def __init__(self, run_dir, *, project="CAT-wholebody", entity="skvayzer",
                 mode="online", resume=False, config=None, wandb_module=None,
                 replace_untrained_config=False):
        self.directory = Path(run_dir)
        self._lock = threading.RLock()
        self._run = None
        self._finished = False
        self._path = self.directory / "wandb.json"
        self.identity = self.reserve_identity(self.directory, project=project, entity=entity,
                                              mode=mode, resume=resume)
        self._last_step = int(self.identity["last_global_step"])
        if replace_untrained_config and (not resume or self._last_step != -1):
            raise ValueError("W&B config replacement requires a resumed identity with zero logged training steps")
        if mode != "disabled":
            if wandb_module is None:
                import wandb as wandb_module
            established = self.identity.get("initialized", bool(self.identity.get("url")))
            resume_mode = "must" if established else "allow" if resume else "never"
            self.identity["initialization_attempted"] = True
            atomic_json(self._path, self.identity)
            try:
                self._run = wandb_module.init(
                    id=self.identity["id"], project=project, entity=entity, mode=mode,
                    resume=resume_mode, name=self.directory.name,
                    dir=str(self.directory), config=config,
                    **({"allow_val_change": True} if replace_untrained_config else {}),
                )
                if replace_untrained_config:
                    self._run.config.update(config, allow_val_change=True)
                self.identity.update(initialized=True, url=self._run.url)
                atomic_json(self._path, self.identity)
                self._run.define_metric("global_step")
                self._run.define_metric("*", step_metric="global_step")
            except BaseException as error:
                # Some SDK failures happen after registering the remote run.
                candidate = getattr(wandb_module, "run", None)
                if self._run is None and getattr(candidate, "id", None) == self.identity["id"]:
                    self._run = candidate
                try:
                    self.finish(exit_code=1)
                except BaseException as cleanup_error:
                    error.add_note(f"W&B failure cleanup also failed: {cleanup_error}")
                raise

    def log(self, step, metrics):
        with self._lock:
            step = int(step)
            scalars, nonfinite = {}, []
            for name, value in metrics.items():
                if getattr(value, "ndim", 0) != 0:
                    continue
                number = float(value)
                if math.isfinite(number):
                    scalars[name] = number
                else:
                    nonfinite.append(name)
            if nonfinite:
                # Persist the actual problem, never silently remove bad losses.
                event = dict(global_step=max(step, self._last_step), source_step=step,
                             nonfinite_keys=nonfinite, **{"health/nonfinite": 1})
                self._write(event)
                raise FloatingPointError("Non-finite training metrics: " + ", ".join(nonfinite))
            if step < self._last_step:
                return
            self._write(dict(scalars, global_step=step, **{"health/nonfinite": 0}))

    def _write(self, event):
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        if self._run is not None:
            self._run.log(event)
        self._last_step = int(event["global_step"])
        self.identity["last_global_step"] = self._last_step
        atomic_json(self._path, self.identity)

    def finish(self, exit_code=0):
        if self._run is not None and not self._finished:
            self._run.finish(exit_code=exit_code)
            self._finished = True
