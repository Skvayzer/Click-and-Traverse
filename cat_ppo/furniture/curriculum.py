"""Prepare or explicitly execute CAT + generic clutter + furniture fine-tuning.

Each stage is a separately compiled physical scene and PPO run. The next stage
restores the previous selected actor, critic and normalizer, with a fresh
optimizer. This is sequential rehearsal, not a simultaneous mixed-scene batch.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from cat_ppo.furniture.scenes import _digest, generate_scene, load_scene, write_scene_bundle


def build_plan(*, steps_per_stage, rounds=3, seed=0, num_envs=32,
               unroll_length=16, checkpoint_epochs=2, wandb_mode="disabled"):
    quantum = num_envs * unroll_length * checkpoint_epochs
    if min(steps_per_stage, rounds, num_envs, unroll_length, checkpoint_epochs) < 1 or steps_per_stage % quantum:
        raise ValueError(f"Positive budgets required; steps_per_stage must be divisible by {quantum}")
    if num_envs % 4:
        raise ValueError("num_envs must be divisible by four minibatches")
    if wandb_mode not in ("disabled", "offline", "online"):
        raise ValueError("Invalid wandb mode")
    stages = []
    legacy = ["forward", "hurdle1", "side0", "crouch0", "random", "side-hurdle-crouch0"]
    for round_index in range(rounds):
        for pair, scene_type in enumerate(legacy):
            stage_seed = seed * 100000 + round_index * 100 + pair
            original = dict(domain="legacy", kind="random" if scene_type == "random" else "typical",
                            scene_type=scene_type, seed=stage_seed, split="train",
                            difficulty=min(.35 + .15 * round_index, .8))
            stages.append(dict(name=f"r{round_index:02d}-p{pair}-cat-{scene_type}", scene=original,
                               environment_config={"action_dofs": 29}))
            domain = "generic" if pair % 2 == 0 else "furniture"
            clutter = dict(domain=domain, seed=stage_seed, split="train", family="mixed",
                           difficulty="pilot" if round_index == 0 else "dense")
            overrides = {"action_dofs": 29}
            if round_index >= 2:
                overrides.update(perception_mode="corrupted", map_latency_steps=3,
                    map_update_interval_steps=2, map_position_noise_std=.01, unknown_probability=.02)
            stages.append(dict(name=f"r{round_index:02d}-p{pair}-{domain}", scene=clutter,
                               environment_config=overrides))
    plan = dict(schema="cat-whole-body-curriculum-v1", seed=seed, steps_per_stage=steps_per_stage,
                total_steps=steps_per_stage * len(stages), rounds=rounds, num_envs=num_envs,
                unroll_length=unroll_length, checkpoint_epochs=checkpoint_epochs, wandb_mode=wandb_mode,
                stages=stages, native_initialization="pinned-public-CAT-generalist",
                sampling="sequential-rehearsal: 50% original CAT, 25% generic clutter, 25% furniture",
                optimizer="fresh per stage; actor, critic and normalizer restored",
                checkpoint_selection="within-stage training proxy; no validation or test evaluation is launched",
                final_selection="best of the final stage, not best across a retained-skill validation mixture",
                retention="previous stage model pruned only after successful verified handoff",
                execution_status="plan-only; no training started")
    plan["sha256"] = _digest(plan)
    return plan


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != "cat-whole-body-curriculum-v1" or plan.get("sha256") != _digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("Curriculum schema or hash mismatch")
    names = [stage["name"] for stage in plan["stages"]]
    if len(set(names)) != len(names) or any(Path(name).name != name or name in ("", ".", "..") for name in names):
        raise ValueError("Stage names must be unique directory basenames")
    if any(stage["scene"].get("split") != "train" for stage in plan["stages"]):
        raise ValueError("Only training scenes belong in the curriculum")
    return plan


def materialize_stage(spec, cache):
    from cat_ppo.furniture.manifest import _case_specification
    spec = dict(spec)
    domain = spec.pop("domain")
    if spec.get("split") != "train":
        raise ValueError("Curriculum accepts training scenes only")
    if domain == "legacy":
        from cat_ppo.furniture.legacy_scenes import generate_legacy_scene
        scene = generate_legacy_scene(**spec)
        dx = .04
    elif domain == "generic":
        from cat_ppo.furniture.clutter import generate_clutter_scene
        scene = generate_clutter_scene(**spec)
        dx = .10
    elif domain == "furniture":
        scene = generate_scene(**spec)
        dx = .10
    else:
        raise ValueError(f"Unknown training domain: {domain}")
    key = _digest(dict(scene=scene, dx=dx))
    directory = Path(cache) / key
    if not directory.exists():
        write_scene_bundle(scene, directory, voxel_size=dx)
    loaded = load_scene(directory)
    if (_case_specification(loaded) != _case_specification(scene, goal_index=0)
            or loaded["grid"]["voxel_size"] != dx):
        raise ValueError("Cached curriculum scene differs from its full requested specification/resolution")
    return directory.resolve()


def _atomic_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing symlink metadata: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", prefix=f".{path.name}.", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular metadata file: {path}")
    return json.loads(path.read_text())


def _retire_owned_stage(stage_dir, *, receipt_sha256):
    """Prune only models bound to a verified runner receipt; preserve all logs."""
    stage_dir = Path(stage_dir)
    if stage_dir.is_symlink() or not (stage_dir / "run.json").is_file():
        raise ValueError("Refusing to retire an unrecognized stage directory")
    receipt_path = stage_dir / "curriculum_receipt.json"
    receipt = _read_json(receipt_path)
    if (_file_hash(receipt_path) != receipt_sha256 or receipt["stage_name"] != stage_dir.name
            or _file_hash(stage_dir / "run.json") != receipt["run_sha256"]
            or _file_hash(stage_dir / "summary.json") != receipt["summary_sha256"]):
        raise ValueError("Retirement receipt does not match the recorded stage")
    paths = [stage_dir / name for name in (".checkpoint-generations", "checkpoints", "export")]
    # Check every top-level path before deleting any of them.
    if any(path.is_symlink() or (path.exists() and not path.is_dir()) for path in paths):
        raise ValueError("Refusing to prune unexpected model artifact type/symlink")
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
    _atomic_json(stage_dir / "model_retired.json", {
        "reason": "next stage completed and selected checkpoint verified",
        "receipt_sha256": receipt_sha256,
        "selection_limit": "handoff selection uses within-stage training proxy, not a global retained-skill score"})


def _selected(stage_dir):
    from cat_ppo.furniture.checkpoint import BestCheckpointStore
    selection = BestCheckpointStore.open_existing(stage_dir).selected(verify=True)
    if selection is None:
        raise ValueError(f"No verified selected checkpoint in {stage_dir}")
    return selection


def _validate_state(state, plan):
    if (state.get("schema") != "cat-curriculum-state-v1" or not state.get("owner")
            or state.get("plan_sha256") != plan["sha256"]):
        raise ValueError("Existing run uses a different or unrecognized curriculum state")
    completed = state.get("completed", [])
    if len(completed) > len(plan["stages"]) or any(
            item["name"] != stage["name"] for item, stage in zip(completed, plan["stages"])):
        raise ValueError("Completed stages must be an ordered prefix of this plan")
    expected_current = completed[-1]["name"] if completed else None
    if state.get("current_stage") != expected_current:
        raise ValueError("Current stage differs from the last completed stage")
    active = state.get("active_stage")
    if active is not None and (len(completed) == len(plan["stages"])
            or active["stage_name"] != plan["stages"][len(completed)]["name"]):
        raise ValueError("Active stage differs from the next planned stage")


def _reconcile_retirement(state, run_dir):
    """Retry interrupted pruning only while the successor selection still verifies."""
    completed = state["completed"]
    if not completed:
        return
    current = run_dir / "stages" / completed[-1]["name"]
    selection = _selected(current)
    for item in completed:
        stage_dir = run_dir / "stages" / item["name"]
        receipt = _read_json(stage_dir / "curriculum_receipt.json")
        if (_file_hash(stage_dir / "curriculum_receipt.json") != item["receipt_sha256"]
                or receipt["owner"] != state["owner"] or receipt["plan_sha256"] != state["plan_sha256"]
                or _file_hash(stage_dir / "summary.json") != receipt["summary_sha256"]):
            raise ValueError("Completed stage receipt or summary was modified")
        if item is completed[-1] and receipt["selected"] != selection:
            raise ValueError("Current checkpoint differs from its committed handoff receipt")
    for item in completed[:-1]:
        stage_dir = run_dir / "stages" / item["name"]
        artifacts_remain = any((stage_dir / name).exists() or (stage_dir / name).is_symlink()
                              for name in (".checkpoint-generations", "checkpoints", "export"))
        if artifacts_remain or not (stage_dir / "model_retired.json").is_file():
            _retire_owned_stage(stage_dir, receipt_sha256=item["receipt_sha256"])


def _verify_stage(stage_dir, *, plan, stage, scene, scene_dir, previous_selection, active):
    """Verify completed training and exact warm-start lineage before old weights go."""
    if stage_dir.is_symlink() or not stage_dir.is_dir():
        raise ValueError("Expected a regular completed stage directory")
    if not (stage_dir / "summary.json").is_file():
        raise RuntimeError(f"Incomplete stage preserved at {stage_dir}; no optimizer-resume or overwrite is attempted")
    run = _read_json(stage_dir / "run.json")
    summary = _read_json(stage_dir / "summary.json")
    selected = _selected(stage_dir)
    expected_args = dict(steps=plan["steps_per_stage"], seed=plan["seed"], num_envs=plan["num_envs"],
                         unroll_length=plan["unroll_length"], checkpoint_epochs=plan["checkpoint_epochs"],
                         num_evals=0, action_dofs=29)
    if any(run["args"].get(key) != value for key, value in expected_args.items()):
        raise ValueError("Completed stage training arguments differ from the plan")
    if (Path(run["args"]["run_dir"]).absolute() != stage_dir.absolute()
            or Path(run["args"]["scene_dir"]).resolve() != scene_dir.resolve()
            or run["training_scene"]["geometry_hash"] != scene["geometry_hash"]
            or run["training_scene"]["scene_id"] != scene["scene_id"]
            or run["training_scene"]["split"] != "train" or run.get("validation_scene") is not None
            or run.get("selection_source") != "training_proxy"):
        raise ValueError("Completed stage scene or selection provenance differs from the plan")
    for key, value in stage["environment_config"].items():
        if run["environment_config"].get(key) != value:
            raise ValueError(f"Completed stage environment override differs at {key}")
    if previous_selection is not None:
        expected_files = {name: record["sha256"] for name, record in previous_selection["files"].items()}
        source = run.get("source", {})
        if (source.get("kind") != "explicit_local_native_checkpoint"
                or Path(source.get("path", "")).resolve() != Path(previous_selection["path"]).resolve()
                or source.get("files") != expected_files
                or not run.get("warmstart", {}).get("critic_restored")):
            raise ValueError("Stage did not restore the preceding verified actor/critic/normalizer payload")
    update = float(summary.get("actor_max_abs_parameter_update", 0))
    if (not math.isfinite(update) or update <= 0
            or summary.get("requested_steps") != plan["steps_per_stage"]
            or summary.get("selected_step") != selected["step"]
            or not 0 < selected["step"] <= plan["steps_per_stage"]
            or summary.get("selection_source") != "training_proxy"
            or selected.get("selection_source") != "training_proxy"
            or summary.get("selected_score") != selected["score"]
            or Path(summary.get("native_checkpoint", "")).resolve() != Path(selected["path"]).resolve()):
        raise ValueError("Stage summary does not prove a finite learned, selected checkpoint")
    receipt = dict(schema="cat-curriculum-stage-receipt-v1", owner=active["owner"],
        plan_sha256=plan["sha256"], stage_name=stage["name"], stage_sha256=_digest(stage),
        input_selection=previous_selection, selected=selected,
        run_sha256=_file_hash(stage_dir / "run.json"), summary_sha256=_file_hash(stage_dir / "summary.json"),
        selection_limit="within-stage training proxy; no global retained-skill validation score")
    _atomic_json(stage_dir / "curriculum_receipt.json", receipt)
    return selected, receipt


def _execute_locked(plan_path, run_dir, *, max_stages=None):
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path)
    root = Path(__file__).resolve().parents[2]
    run_dir = Path(run_dir).absolute()
    if run_dir.is_symlink():
        raise ValueError("Curriculum run directory cannot be a symlink")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("stages", "scenes"):
        if (run_dir / name).is_symlink():
            raise ValueError(f"Managed curriculum {name} directory cannot be a symlink")
    state_path = run_dir / "curriculum_state.json"
    if state_path.exists():
        state = _read_json(state_path)
    else:
        if any(path.name != ".curriculum.lock" for path in run_dir.iterdir()):
            raise ValueError("New curriculum directory must be empty")
        state = dict(schema="cat-curriculum-state-v1", owner=str(uuid.uuid4()),
                     plan_sha256=plan["sha256"], completed=[], current_stage=None, active_stage=None)
        (run_dir / "plan.json").write_text(plan_path.read_text())
        _atomic_json(state_path, state)
    _validate_state(state, plan)
    _reconcile_retirement(state, run_dir)
    # Incomplete stage directories are preserved. A completed stage whose state
    # commit was interrupted can be verified and adopted without retraining.
    remaining = plan["stages"][len(state["completed"]):]
    if max_stages is not None:
        if max_stages < 1:
            raise ValueError("max-stages must be positive")
        remaining = remaining[:max_stages]
    for stage in remaining:
        scene_dir = materialize_stage(stage["scene"], run_dir / "scenes")
        scene = load_scene(scene_dir)
        stage_dir = run_dir / "stages" / stage["name"]
        config_path = run_dir / (stage["name"] + ".json")
        _atomic_json(config_path, stage["environment_config"])
        command = [sys.executable, str(root / "train_furniture.py"),
            "--scene-dir", str(scene_dir), "--run-dir", str(stage_dir),
            "--steps", str(plan["steps_per_stage"]), "--seed", str(plan["seed"]),
            "--num-envs", str(plan["num_envs"]), "--unroll-length", str(plan["unroll_length"]),
            "--checkpoint-epochs", str(plan["checkpoint_epochs"]),
            "--env-config-json", str(config_path), "--wandb-mode", plan["wandb_mode"], "--num-evals", "0"]
        previous = run_dir / "stages" / state["current_stage"] if state["current_stage"] else None
        previous_selection = _selected(previous) if previous is not None else None
        if previous_selection is not None:
            command += ["--warmstart", str(previous_selection["path"])]
        active = dict(owner=state["owner"], stage_name=stage["name"], stage_sha256=_digest(stage),
                      input_selection=previous_selection, command=command)
        if state.get("active_stage") is not None and state["active_stage"] != active:
            raise ValueError("Interrupted stage intent differs from the requested handoff")
        if stage_dir.exists() and state.get("active_stage") is None:
            raise ValueError("Existing stage directory has no recorded execution intent")
        state["active_stage"] = active
        _atomic_json(state_path, state)
        if not stage_dir.exists():
            subprocess.run(command, cwd=root, check=True)
        selected, receipt = _verify_stage(stage_dir, plan=plan, stage=stage, scene=scene,
            scene_dir=scene_dir, previous_selection=previous_selection, active=active)
        state["completed"].append(dict(name=stage["name"], scene_id=scene["scene_id"],
            selected_step=selected["step"], summary_sha256=receipt["summary_sha256"],
            receipt_sha256=_file_hash(stage_dir / "curriculum_receipt.json")))
        state["current_stage"] = stage["name"]
        state["current_best"] = str(stage_dir / "checkpoints" / "best")
        state["active_stage"] = None
        _atomic_json(state_path, state)
        _reconcile_retirement(state, run_dir)
    return state


def execute(plan_path, run_dir, *, max_stages=None):
    run_dir = Path(run_dir).absolute()
    if run_dir.is_symlink():
        raise ValueError("Curriculum run directory cannot be a symlink")
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / ".curriculum.lock").is_symlink():
        raise ValueError("Curriculum lock cannot be a symlink")
    with (run_dir / ".curriculum.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute_locked(plan_path, run_dir, max_stages=max_stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Write a reviewable plan; starts no training")
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--steps-per-stage", required=True, type=int)
    prepare.add_argument("--rounds", default=3, type=int)
    prepare.add_argument("--seed", default=0, type=int)
    prepare.add_argument("--num-envs", default=32, type=int)
    prepare.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    run = sub.add_parser("run", help="Explicitly execute the budgeted plan")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument("--max-stages", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        plan = build_plan(steps_per_stage=args.steps_per_stage, rounds=args.rounds, seed=args.seed,
                          num_envs=args.num_envs, wandb_mode=args.wandb_mode)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(plan, indent=2) + "\n")
        print(f"Prepared {len(plan['stages'])} stages; {plan['total_steps']:,} total transitions. No training started.")
    else:
        print(json.dumps(execute(args.plan, args.run_dir, max_stages=args.max_stages), indent=2))


if __name__ == "__main__":
    main()
