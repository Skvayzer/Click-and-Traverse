"""Run CAT whole-body rehearsal continuously until an explicit stop request.

Per-scene budgets are switching/checkpoint cadences, never a global stopping
budget. The public CAT checkpoint initializes only the first stage. Every later
stage restores the preceding verified selection; failures halt without retries.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import uuid

from cat_ppo.furniture import curriculum
from cat_ppo.furniture.scenes import _digest, load_scene


def stage_spec(index, seed):
    if type(index) is not int or index < 0 or type(seed) is not int or seed < 0:
        raise ValueError("Stage index and seed must be nonnegative integers")
    round_index, position = divmod(index, 12)
    pair, clutter_stage = divmod(position, 2)
    stage_seed = seed * 100000 + round_index * 100 + pair
    legacy = ("forward", "hurdle1", "side0", "crouch0", "random", "side-hurdle-crouch0")
    overrides = {"action_dofs": 29}
    if not clutter_stage:
        kind = legacy[pair]
        scene = dict(domain="legacy", kind="random" if kind == "random" else "typical",
                     scene_type=kind, seed=stage_seed, split="train",
                     difficulty=min(.35 + .15 * round_index, .8))
        suffix = "cat-" + kind
    else:
        domain = "generic" if pair % 2 == 0 else "furniture"
        scene = dict(domain=domain, seed=stage_seed, split="train", family="mixed",
                     difficulty="pilot" if round_index == 0 else "dense")
        suffix = domain
        if round_index >= 2:
            overrides.update(perception_mode="corrupted", map_latency_steps=3,
                             map_update_interval_steps=2, map_position_noise_std=.01,
                             unknown_probability=.02)
    return dict(name=f"r{round_index:06d}-p{pair}-{suffix}", scene=scene,
                environment_config=overrides)


def build_config(*, steps_per_stage=1048576, num_envs=32, seed=0,
                 unroll_length=16, checkpoint_epochs=16, wandb_mode="online",
                 wandb_group=None, wandb_entity=None, legacy_num_envs=None,
                 pilot_num_envs=None):
    # Reuse the bounded curriculum's exact arithmetic validation; its finite
    # stage list is deliberately not used by the continuous runner.
    curriculum.build_plan(steps_per_stage=steps_per_stage, rounds=1, seed=seed,
        num_envs=num_envs, unroll_length=unroll_length,
        checkpoint_epochs=checkpoint_epochs, wandb_mode=wandb_mode)
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    random_num_envs = min(legacy_num_envs, 2 * num_envs) if legacy_num_envs is not None else None
    for count in (legacy_num_envs, pilot_num_envs, random_num_envs):
        if count is not None:
            curriculum.build_plan(steps_per_stage=steps_per_stage, rounds=1, seed=seed,
                num_envs=count, unroll_length=unroll_length,
                checkpoint_epochs=checkpoint_epochs, wandb_mode=wandb_mode)
    config = dict(schema="cat-continuous-curriculum-v1", steps_per_stage=steps_per_stage,
        num_envs=num_envs, seed=seed, unroll_length=unroll_length,
        checkpoint_epochs=checkpoint_epochs, wandb_mode=wandb_mode,
        wandb_group=wandb_group, wandb_entity=wandb_entity,
        legacy_num_envs=legacy_num_envs, pilot_num_envs=pilot_num_envs,
        initialization="pinned-public-CAT-once; subsequent stages restore previous selection",
        sampling="sequential rehearsal: 50% original CAT, 25% generic, 25% furniture",
        stop_condition="explicit STOP file or SIGINT/SIGTERM; errors halt without retries",
        global_step_limit=None, global_stage_limit=None, global_time_limit=None,
        checkpoint_selection="within-stage training reward proxy; no automatic evaluation")
    config["sha256"] = _digest(config)
    return config


def _validate_config(config):
    expected = build_config(**{key: config[key] for key in (
        "steps_per_stage", "num_envs", "seed", "unroll_length", "checkpoint_epochs",
        "wandb_mode", "wandb_group", "wandb_entity", "legacy_num_envs", "pilot_num_envs")})
    if config != expected:
        raise ValueError("Continuous configuration hash or schema mismatch")


def stage_num_envs(config, stage):
    scene = stage["scene"]
    if scene["domain"] == "legacy" and config["legacy_num_envs"] is not None:
        count = config["legacy_num_envs"]
        # Original random occupancy is substantially larger than the typical
        # templates; cap its optional light-scene batch at twice the base.
        return min(count, 2 * config["num_envs"]) if scene["kind"] == "random" else count
    if scene["difficulty"] == "pilot" and config["pilot_num_envs"] is not None:
        return config["pilot_num_envs"]
    return config["num_envs"]


def _validate_state(state, config):
    if (state.get("schema") != "cat-continuous-state-v1" or not state.get("owner")
            or state.get("plan_sha256") != config["sha256"]):
        raise ValueError("Existing run belongs to a different continuous configuration")
    completed = state["completed"]
    for index, item in enumerate(completed):
        if item["name"] != stage_spec(index, config["seed"])["name"]:
            raise ValueError("Continuous history is not the expected ordered prefix")
    if state["current_stage"] != (completed[-1]["name"] if completed else None):
        raise ValueError("Continuous current stage disagrees with completed history")
    if state["completed_transitions"] != sum(item["actual_steps"] for item in completed):
        raise ValueError("Continuous transition offset disagrees with completed history")
    if state.get("active_stage") and state["active_stage"]["stage_name"] != stage_spec(len(completed), config["seed"])["name"]:
        raise ValueError("Active stage is not the next continuous stage")


def request_stop(run_dir):
    run_dir = Path(run_dir).absolute()
    if run_dir.is_symlink() or not (run_dir / "continuous_state.json").is_file():
        raise ValueError("Stop requests require an existing continuous run")
    target = run_dir / "STOP"
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("Refusing an unexpected STOP path")
    try:
        with target.open("x") as stream:
            stream.write("Explicit stop requested; finish the active checkpoint epoch and save.\n")
    except FileExistsError:
        pass
    return target


def _run_child(command, *, cwd, runner_record, runner_path):
    # A separate session lets the parent translate Ctrl-C into a cooperative
    # stop file, including while the child is still importing/compiling.
    with subprocess.Popen(command, cwd=cwd, start_new_session=True) as process:
        runner_record["child_pid"] = process.pid
        curriculum._atomic_json(runner_path, runner_record)
        result = process.wait()
    runner_record["child_pid"] = None
    curriculum._atomic_json(runner_path, runner_record)
    if result:
        raise subprocess.CalledProcessError(result, command)


def _execute_locked(config, run_dir):
    root = Path(__file__).resolve().parents[2]
    state_path, stop_file = run_dir / "continuous_state.json", run_dir / "STOP"
    for name in ("stages", "scenes", "STOP", "runner.json", "continuous_config.json"):
        if (run_dir / name).is_symlink():
            raise ValueError(f"Refusing managed symlink: {name}")
    if state_path.exists():
        state = curriculum._read_json(state_path)
        if curriculum._read_json(run_dir / "continuous_config.json") != config:
            raise ValueError("Existing continuous run has a different configuration")
    else:
        if any(path.name != ".continuous.lock" for path in run_dir.iterdir()):
            raise ValueError("New continuous run directory must be empty")
        state = dict(schema="cat-continuous-state-v1", owner=str(uuid.uuid4()),
                     plan_sha256=config["sha256"], completed=[], current_stage=None,
                     active_stage=None, completed_transitions=0, status="starting")
        curriculum._atomic_json(run_dir / "continuous_config.json", config)
        curriculum._atomic_json(state_path, state)
    _validate_state(state, config)
    curriculum._reconcile_retirement(state, run_dir)
    runner_path = run_dir / "runner.json"
    runner = dict(pid=os.getpid(), child_pid=None, hostname=socket.gethostname(),
                  owner=state["owner"], config_sha256=config["sha256"], active=True,
                  started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    curriculum._atomic_json(runner_path, runner)
    previous_handlers = {}

    def stop_handler(signum, frame):
        del signum, frame
        request_stop(run_dir)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, stop_handler)
        while True:
            if stop_file.exists():
                state["status"] = "stopped_by_request"
                curriculum._atomic_json(state_path, state)
                return state
            if state.get("partial_stage"):
                raise RuntimeError("Stopped partial stage preserved; use its selected checkpoint in an explicitly configured new run")
            index = len(state["completed"])
            stage = stage_spec(index, config["seed"])
            num_envs = stage_num_envs(config, stage)
            scene_dir = curriculum.materialize_stage(stage["scene"], run_dir / "scenes")
            scene = load_scene(scene_dir)
            # A stop can arrive during expensive field materialization.
            if stop_file.exists():
                continue
            stage_dir = run_dir / "stages" / stage["name"]
            override_path = run_dir / (stage["name"] + ".json")
            curriculum._atomic_json(override_path, stage["environment_config"])
            command = [sys.executable, str(root / "train_furniture.py"),
                "--scene-dir", str(scene_dir), "--run-dir", str(stage_dir),
                "--steps", str(config["steps_per_stage"]), "--seed", str(config["seed"]),
                "--num-envs", str(num_envs), "--unroll-length", str(config["unroll_length"]),
                "--checkpoint-epochs", str(config["checkpoint_epochs"]),
                "--env-config-json", str(override_path), "--wandb-mode", config["wandb_mode"],
                "--num-evals", "0", "--stop-file", str(stop_file),
                "--wandb-group", config["wandb_group"] or run_dir.name,
                "--global-step-offset", str(state["completed_transitions"])]
            if config["wandb_entity"]:
                command += ["--wandb-entity", config["wandb_entity"]]
            previous = run_dir / "stages" / state["current_stage"] if state["current_stage"] else None
            previous_selection = curriculum._selected(previous) if previous is not None else None
            if previous_selection is not None:
                command += ["--warmstart", previous_selection["path"]]
            active = dict(owner=state["owner"], stage_name=stage["name"], stage_sha256=_digest(stage),
                          input_selection=previous_selection, command=command)
            if state.get("active_stage") is not None and state["active_stage"] != active:
                raise ValueError("Interrupted continuous stage intent differs from saved intent")
            if stage_dir.exists() and state.get("active_stage") is None:
                raise ValueError("Stage directory exists without a recorded continuous intent")
            state.update(active_stage=active, status="training")
            curriculum._atomic_json(state_path, state)
            if not stage_dir.exists():
                _run_child(command, cwd=root, runner_record=runner, runner_path=runner_path)
            if not (stage_dir / "summary.json").is_file():
                raise RuntimeError(f"Incomplete stage preserved at {stage_dir}; no silent retry or overwrite")
            summary = curriculum._read_json(stage_dir / "summary.json")
            if summary.get("stopped_by_request"):
                selected = curriculum._selected(stage_dir)
                state.update(status="stopped_with_partial_stage", partial_stage=dict(
                    name=stage["name"], actual_steps=summary["actual_steps"], selected=selected,
                    summary_sha256=curriculum._file_hash(stage_dir / "summary.json")),
                    current_best=str(stage_dir / "checkpoints" / "best"))
                curriculum._atomic_json(state_path, state)
                # The preceding selection is retained: this was not a completed
                # domain handoff and must never retire the recovery checkpoint.
                return state
            if summary.get("actual_steps") != config["steps_per_stage"]:
                raise ValueError("A completed continuous stage did not execute its exact switching cadence")
            run = curriculum._read_json(stage_dir / "run.json")
            if (run["args"].get("global_step_offset") != state["completed_transitions"]
                    or run["args"].get("stop_file") != str(stop_file)
                    or run["args"].get("wandb_group") != (config["wandb_group"] or run_dir.name)
                    or run["args"].get("wandb_mode") != config["wandb_mode"]
                    or run["args"].get("wandb_entity") != config["wandb_entity"]):
                raise ValueError("Stage tracking/stop metadata differs from the continuous intent")
            verification_plan = {**config, "num_envs": num_envs}
            selected, receipt = curriculum._verify_stage(stage_dir, plan=verification_plan, stage=stage,
                scene=scene, scene_dir=scene_dir, previous_selection=previous_selection, active=active)
            state["completed"].append(dict(name=stage["name"], scene_id=scene["scene_id"],
                actual_steps=summary["actual_steps"], selected_step=selected["step"],
                summary_sha256=receipt["summary_sha256"],
                receipt_sha256=curriculum._file_hash(stage_dir / "curriculum_receipt.json")))
            state.update(current_stage=stage["name"], current_best=str(stage_dir / "checkpoints" / "best"),
                         active_stage=None, completed_transitions=state["completed_transitions"] + summary["actual_steps"],
                         status="switching_scene")
            curriculum._atomic_json(state_path, state)
            curriculum._reconcile_retirement(state, run_dir)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        curriculum._atomic_json(state_path, state)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        runner.update(active=False, child_pid=None,
                      exited_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        curriculum._atomic_json(runner_path, runner)


def execute(config, run_dir):
    _validate_config(config)
    run_dir = Path(run_dir).absolute()
    if run_dir.is_symlink():
        raise ValueError("Continuous run directory cannot be a symlink")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".continuous.lock"
    if lock_path.is_symlink():
        raise ValueError("Continuous lock cannot be a symlink")
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute_locked(config, run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Train continuously; stops only on explicit request or error")
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument("--steps-per-stage", type=int, default=1048576)
    run.add_argument("--num-envs", type=int, default=32)
    run.add_argument("--legacy-num-envs", type=int,
                     help="Typical CAT batch override; random CAT is capped at twice --num-envs")
    run.add_argument("--pilot-num-envs", type=int, help="Pilot clutter batch override")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--checkpoint-epochs", type=int, default=16)
    run.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    run.add_argument("--wandb-group")
    run.add_argument("--wandb-entity")
    stop = sub.add_parser("stop", help="Request a saved stop at the next checkpoint epoch")
    stop.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "stop":
        print(request_stop(args.run_dir))
    else:
        values = vars(args).copy()
        values.pop("command")
        run_dir = values.pop("run_dir")
        config = build_config(**values)
        print(json.dumps(execute(config, run_dir), indent=2))


if __name__ == "__main__":
    main()
