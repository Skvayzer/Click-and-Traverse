#!/usr/bin/env python3
"""Resume one complete SAPG learner using its original launch choices.

This entry point works inside a scheduler allocation or a durable workstation
session. It neither selects GPUs nor changes training/resource settings.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


def resume_arguments(launch, directory):
    """Recover every exposed training choice from the saved launch."""
    specification = launch["specification"]
    config = specification["config"]
    policy, fine, sapg = config["policy_config"], config["fine_tuning"], config["sapg"]
    if config.get("algorithm") != "sapg" or fine["mode"] != "cat_train_only":
        raise ValueError("This entry point resumes SAPG training-only learners")
    arguments = ["run", "--resume", "--run-dir", str(directory),
                 "--algorithm", config["algorithm"],
                 "--finetuning", fine["mode"], "--profile", fine["profile"],
                 "--num-envs", str(policy["num_envs"]),
                 "--batch-size", str(policy["batch_size"]), "--seed", str(policy["seed"]),
                 "--sapg-num-policies", str(sapg["num_policies"]),
                 "--sapg-embedding-dim", str(sapg["embedding_dim"]),
                 "--bank-manifest", specification["bank_manifest"]]
    for key, flag in (("body_collision_bank", "--body-collision-bank"),
                      ("body_collision_resets", "--body-collision-resets")):
        if specification.get(key):
            arguments += [flag, specification[key]]
    for key in ("mode", "project", "entity"):
        arguments += ["--wandb-" + key, launch["wandb"][key]]
    return arguments


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular, non-symlink resume file: {path}")
    return path


def prepare_command(source_root, run_dir, expected_commit, *, executable=None, launcher=None):
    """Validate provenance and exact reconstructed plan before any learner runs."""
    if not re.fullmatch(r"[a-f0-9]{40}", expected_commit):
        raise ValueError("Expected commit must be an exact lowercase 40-character hash")
    source, directory = Path(source_root).absolute(), Path(run_dir).absolute()
    for path in (source, directory):
        if not path.is_dir() or path.resolve() != path:
            raise ValueError(f"Expected canonical regular directory: {path}")
    stop = directory / "STOP"
    if stop.exists() or stop.is_symlink():
        raise ValueError("STOP exists; retire the stop request deliberately before resuming")
    for name in ("run.json", "launch.json", "status.json", "wandb.json", "resume.msgpack"):
        _regular(directory / name)
    if (directory / "resume.msgpack").stat().st_size == 0:
        raise ValueError("Full learner checkpoint is empty")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != expected_commit:
        raise ValueError("Pinned source HEAD differs from expected commit")
    branch = subprocess.run(["git", "-C", str(source), "symbolic-ref", "-q", "HEAD"],
                            capture_output=True, text=True)
    if branch.returncode != 1:
        raise ValueError("Pinned source must be a detached checkout")
    read = lambda name: json.loads((directory / name).read_text())
    launch, record, status, identity = (read(name) for name in
                                      ("launch.json", "run.json", "status.json", "wandb.json"))
    if launch.get("schema") != "cat-generalist-launch-v1":
        raise ValueError("Unsupported launch schema")
    specification = launch["specification"]
    spec_sha = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    if launch.get("specification_sha256") != spec_sha or status.get("specification_sha256") != spec_sha:
        raise ValueError("Launch/status specification hash differs")
    if status.get("owner") != launch.get("owner"):
        raise ValueError("Run status ownership differs")
    if any(record.get(key) != value for key, value in specification.items()):
        raise ValueError("Run record differs from original launch specification")
    if not identity.get("initialized") or identity.get("mode") != "online" or not identity.get("id"):
        raise ValueError("An established online W&B identity is required")
    for key in ("mode", "project", "entity"):
        if identity[key] != launch["wandb"][key]:
            raise ValueError("Stored W&B identity differs from original logging destination")
    if launcher is None:
        sys.path.insert(0, str(source))
        launcher = importlib.import_module("train_cat_wholebody")
    if Path(launcher.__file__).resolve() != source / "train_cat_wholebody.py":
        raise ValueError("Imported launcher is outside the pinned source checkout")
    code = launcher.code_identity()
    if code["git_commit"] != expected_commit or code != record["code"]:
        raise ValueError("Source commit/content differs from the recorded run identity")
    arguments = resume_arguments(launch, directory)
    if launcher.plan(launcher.parser().parse_args(arguments)) != specification:
        raise ValueError("Reconstructed resume arguments change the original launch specification")
    return dict(command=[executable or sys.executable, str(source / "train_cat_wholebody.py"), *arguments],
                source_root=str(source), source=code, wandb_id=identity["id"],
                saved_step=status.get("resume_state_steps"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--print-command", action="store_true",
                        help="Validate and print the exact command without starting training")
    args = parser.parse_args(argv)
    prepared = prepare_command(args.source_root, args.run_dir, args.expected_commit)
    print(json.dumps({key: value for key, value in prepared.items() if key != "command"}, sort_keys=True), flush=True)
    print(shlex.join(prepared["command"]), flush=True)
    if args.print_command:
        return
    # Relative released asset paths resolve within the same pinned source tree;
    # its data directory is prepared by the deployment, not modified here.
    os.chdir(prepared["source_root"])
    os.execv(prepared["command"][0], prepared["command"])


if __name__ == "__main__":
    main()
