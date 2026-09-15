"""Run CAT whole-body rehearsal continuously until an explicit stop request.

Per-scene budgets are switching/checkpoint cadences, never a global stopping
budget. The public CAT checkpoint initializes only the first stage. Every later
stage restores the preceding verified selection. Confirmed GPU memory failures
before observed training progress permit a bounded number of batch halvings.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import uuid

from cat_ppo.furniture import curriculum
from cat_ppo.furniture.checkpoint import _manifest
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
                 pilot_num_envs=None, random_num_envs=None, generic_num_envs=None,
                 furniture_num_envs=None, pilot_furniture_num_envs=None, restart_from_run=None,
                 oom_max_retries=2, oom_min_num_envs=128, estimated_memory_budget_gib=None):
    # Reuse the bounded curriculum's exact arithmetic validation; its finite
    # stage list is deliberately not used by the continuous runner.
    curriculum.build_plan(steps_per_stage=steps_per_stage, rounds=1, seed=seed,
        num_envs=num_envs, unroll_length=unroll_length,
        checkpoint_epochs=checkpoint_epochs, wandb_mode=wandb_mode)
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if type(oom_max_retries) is not int or not 0 <= oom_max_retries <= 8:
        raise ValueError("oom-max-retries must be an integer from zero to eight")
    if type(oom_min_num_envs) is not int or oom_min_num_envs < 4 or oom_min_num_envs % 4:
        raise ValueError("oom-min-num-envs must be positive and divisible by four")
    if estimated_memory_budget_gib is not None and (not math.isfinite(estimated_memory_budget_gib) or estimated_memory_budget_gib <= 0):
        raise ValueError("estimated-memory-budget-gib must be finite and positive")
    default_random_num_envs = min(legacy_num_envs, 2 * num_envs) if legacy_num_envs is not None else None
    for count in (legacy_num_envs, pilot_num_envs, random_num_envs, default_random_num_envs,
                  generic_num_envs, furniture_num_envs, pilot_furniture_num_envs):
        if count is not None:
            curriculum.build_plan(steps_per_stage=steps_per_stage, rounds=1, seed=seed,
                num_envs=count, unroll_length=unroll_length,
                checkpoint_epochs=checkpoint_epochs, wandb_mode=wandb_mode)
    config = dict(schema="cat-continuous-curriculum-v1", steps_per_stage=steps_per_stage,
        num_envs=num_envs, seed=seed, unroll_length=unroll_length,
        checkpoint_epochs=checkpoint_epochs, wandb_mode=wandb_mode,
        wandb_group=wandb_group, wandb_entity=wandb_entity,
        legacy_num_envs=legacy_num_envs, pilot_num_envs=pilot_num_envs,
        random_num_envs=random_num_envs, generic_num_envs=generic_num_envs,
        furniture_num_envs=furniture_num_envs,
        pilot_furniture_num_envs=pilot_furniture_num_envs,
        restart_from_run=str(Path(restart_from_run).absolute()) if restart_from_run is not None else None,
        oom_max_retries=oom_max_retries, oom_min_num_envs=oom_min_num_envs,
        estimated_memory_budget_gib=estimated_memory_budget_gib,
        initialization="pinned-public-CAT-once; subsequent stages restore previous selection",
        sampling="sequential rehearsal: 50% original CAT, 25% generic, 25% furniture",
        stop_condition="explicit STOP file or SIGINT/SIGTERM; bounded GPU OOM retries only before training progress",
        global_step_limit=None, global_stage_limit=None, global_time_limit=None,
        checkpoint_selection="within-stage training reward proxy; no automatic evaluation")
    config["sha256"] = _digest(config)
    return config


def _validate_config(config):
    expected = build_config(**{key: config[key] for key in (
        "steps_per_stage", "num_envs", "seed", "unroll_length", "checkpoint_epochs",
        "wandb_mode", "wandb_group", "wandb_entity", "legacy_num_envs", "pilot_num_envs",
        "random_num_envs", "generic_num_envs", "furniture_num_envs", "pilot_furniture_num_envs", "restart_from_run",
        "oom_max_retries", "oom_min_num_envs", "estimated_memory_budget_gib")})
    if config != expected:
        raise ValueError("Continuous configuration hash or schema mismatch")


def stage_num_envs(config, stage):
    scene = stage["scene"]
    if scene["domain"] == "legacy" and scene["kind"] == "random" and config.get("random_num_envs") is not None:
        return config["random_num_envs"]
    if scene["domain"] == "legacy" and config["legacy_num_envs"] is not None:
        count = config["legacy_num_envs"]
        # Original random occupancy is substantially larger than the typical
        # templates; cap its optional light-scene batch at twice the base.
        return min(count, 2 * config["num_envs"]) if scene["kind"] == "random" else count
    if scene["domain"] == "furniture" and scene["difficulty"] == "pilot" and config.get("pilot_furniture_num_envs") is not None:
        return config["pilot_furniture_num_envs"]
    if scene["difficulty"] == "pilot" and config["pilot_num_envs"] is not None:
        return config["pilot_num_envs"]
    if scene["domain"] in ("generic", "furniture") and config.get(scene["domain"] + "_num_envs") is not None:
        return config[scene["domain"] + "_num_envs"]
    return config["num_envs"]


def _validate_state(state, config):
    if (state.get("schema") != "cat-continuous-state-v1" or not state.get("owner")
            or state.get("plan_sha256") != config["sha256"]):
        raise ValueError("Existing run belongs to a different continuous configuration")
    completed = state["completed"]
    start = state.get("start_stage_index", 0)
    offset = state.get("initial_transition_offset", 0)
    if type(start) is not int or start < 0 or type(offset) is not int or offset < 0:
        raise ValueError("Invalid restart stage/transition offsets")
    for index, item in enumerate(completed):
        if item["name"] != stage_spec(start + index, config["seed"])["name"]:
            raise ValueError("Continuous history is not the expected ordered prefix")
    if state["current_stage"] != (completed[-1]["name"] if completed else None):
        raise ValueError("Continuous current stage disagrees with completed history")
    if state["completed_transitions"] != offset + sum(item["actual_steps"] for item in completed):
        raise ValueError("Continuous transition offset disagrees with completed history")
    if state.get("active_stage") and state["active_stage"]["stage_name"] != stage_spec(start + len(completed), config["seed"])["name"]:
        raise ValueError("Active stage is not the next continuous stage")
    for name, count in state.get("effective_stage_batches", {}).items():
        attempts = [item for item in state.get("oom_attempts", []) if item["stage_name"] == name]
        if not attempts or attempts[-1]["next_num_envs"] != count:
            raise ValueError("Effective stage batch is not bound to an OOM attempt record")


def _load_restart_anchor(run_dir, state, config):
    if config.get("restart_from_run") is None:
        if state.get("restart_anchor_sha256") or state.get("start_stage_index", 0) or state.get("initial_transition_offset", 0):
            raise ValueError("Fresh public initialization cannot carry restart offsets")
        return None
    anchor = curriculum._read_json(run_dir / "restart_anchor.json")
    if (anchor.get("schema") != "cat-continuous-restart-v1"
            or _digest(anchor) != state.get("restart_anchor_sha256")
            or anchor["owner"] != state["owner"]
            or anchor["source_run"] != config["restart_from_run"]
            or anchor["resume_stage_index"] != state["start_stage_index"]
            or anchor["global_step_offset"] != state["initial_transition_offset"]
            or anchor["imported_selection"]["path"] != str(run_dir / "restart_source" / "native")):
        raise ValueError("Restart anchor differs from saved ownership, source or offsets")
    return anchor


def _verify_source_lineage(run, selected, previous_selection):
    if not run.get("warmstart", {}).get("critic_restored"):
        raise ValueError("Restart source did not restore the CAT critic")
    source = run.get("source", {})
    if previous_selection is None:
        if source.get("kind") != "pinned_public_native_checkpoint":
            raise ValueError("Restart source lineage does not begin at pinned public CAT")
    elif (source.get("kind") != "explicit_local_native_checkpoint"
          or source.get("path") != previous_selection["path"]
          or source.get("files") != {name: item["sha256"] for name, item in previous_selection["files"].items()}):
        raise ValueError("Restart source warm-start lineage does not match its predecessor")
    if selected["selection_source"] != "training_proxy":
        raise ValueError("Continuous restart expects the tracked training-proxy selection")


def _verify_partial_intent(run, selected, active, stage, old_config, source_run, directory, offset):
    command = active.get("command", [])
    if len(command) < 2 or len(command[2:]) % 2:
        raise ValueError("Restart source has a malformed saved training command")
    intent = dict(zip(command[2::2], command[3::2]))
    if len(intent) != len(command[2:]) // 2 or active.get("stage_sha256") != _digest(stage):
        raise ValueError("Restart partial intent does not match the expected curriculum stage")
    expected = dict(steps=old_config["steps_per_stage"], seed=old_config["seed"],
        num_envs=active.get("num_envs", stage_num_envs(old_config, stage)), unroll_length=old_config["unroll_length"],
        checkpoint_epochs=old_config["checkpoint_epochs"], num_evals=0, global_step_offset=offset,
        run_dir=str(directory), stop_file=str(source_run / "STOP"), wandb_mode=old_config["wandb_mode"],
        wandb_group=old_config["wandb_group"] or source_run.name)
    if any(run["args"].get(key) != value or intent.get("--" + key.replace("_", "-")) != str(value)
           for key, value in expected.items()):
        raise ValueError("Restart partial run arguments differ from saved command/configuration")
    if (run["args"].get("action_dofs") != 29 or run.get("validation_scene") is not None
            or run.get("selection_source") != "training_proxy"
            or run["args"].get("wandb_entity") != old_config["wandb_entity"]
            or any(run["environment_config"].get(key) != value for key, value in stage["environment_config"].items())):
        raise ValueError("Restart partial task/action/selection configuration differs from its stage")
    scene_path = Path(intent.get("--scene-dir", ""))
    if run["args"].get("scene_dir") != str(scene_path):
        raise ValueError("Restart partial scene path differs from saved intent")
    scene = load_scene(scene_path)
    if any(run["training_scene"].get(key) != scene[key] for key in ("scene_id", "geometry_hash", "split")) or scene["split"] != "train":
        raise ValueError("Restart partial scene provenance differs from its verified scene bundle")
    if selected is not None:
        for key in ("warmstart", "source", "code", "training_scene", "validation_scene"):
            if selected.get("provenance", {}).get(key) != run.get(key):
                raise ValueError(f"Restart partial selected checkpoint provenance differs at {key}")


def _assert_no_progress(directory):
    """Refuse restart/fallback if any completed rollout/checkpoint was observed."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Expected a regular failed stage directory")
    for name in ("summary.json", "checkpoints/best"):
        path = directory / name
        if path.exists() or path.is_symlink():
            raise ValueError("Failed stage has saved progress; automatic replay is refused")
    metrics = directory / "metrics.jsonl"
    if metrics.is_symlink() or (metrics.exists() and metrics.read_text().strip()):
        raise ValueError("Failed stage has observed training metrics; automatic replay is refused")
    generations = directory / ".checkpoint-generations"
    if generations.is_symlink() or (generations.exists() and any(path.name != "owner.json" for path in generations.iterdir())):
        raise ValueError("Failed stage has checkpoint candidate evidence; automatic replay is refused")
    failure_path = directory / "failure.json"
    if failure_path.exists() or failure_path.is_symlink():
        failure = curriculum._read_json(failure_path)
        if (failure.get("completed_steps", 0) != 0 or failure.get("metrics_last_step", 0) not in (0, None)
                or failure.get("completed_update_evidence", False) or failure.get("metrics_present", False)
                or failure.get("selected_checkpoint") is not None or failure.get("checkpoint_inspection_error")
                or failure.get("metrics_inspection_error")):
            raise ValueError("Failed stage error metadata reports progress or an uncertain checkpoint")


def _artifact_manifest(directory):
    """Hash preserved attempt files; record W&B symlinks without following them."""
    records = {}
    for path in sorted(Path(directory).rglob("*")):
        relative = str(path.relative_to(directory))
        if path.is_symlink():
            records[relative] = {"symlink": os.readlink(path)}
        elif path.is_file():
            records[relative] = {"size": path.stat().st_size, "sha256": curriculum._file_hash(path)}
    return records


def _oom_retry_batch(directory, *, num_envs, attempts, config):
    """Return one validated smaller batch, or None; never classify generic errors."""
    if attempts >= config["oom_max_retries"]:
        return None
    path = directory / "failure.json"
    if not path.is_file() or path.is_symlink():
        return None
    failure = curriculum._read_json(path)
    if (failure.get("schema") != "cat-furniture-training-failure-v1"
            or failure.get("classification") != "gpu_oom"
            or type(failure.get("completed_steps")) is not int or failure["completed_steps"] != 0
            or failure.get("completed_update_evidence") is not False
            or failure.get("metrics_present") is not False
            or failure.get("selected_checkpoint") is not None
            or failure.get("checkpoint_inspection_error") or failure.get("metrics_inspection_error")):
        return None
    _assert_no_progress(directory)
    next_count = num_envs // 2
    if num_envs % 2 or next_count < config["oom_min_num_envs"] or next_count % 4:
        return None
    quantum = next_count * config["unroll_length"] * config["checkpoint_epochs"]
    if config["steps_per_stage"] % quantum:
        return None
    return next_count


def _verify_oom_archives(state, run_dir):
    for item in state.get("oom_attempts", []):
        expected_path = Path("failed_attempts") / item["stage_name"] / f"attempt-{item['attempt']:03d}.json"
        if (Path(item["stage_name"]).name != item["stage_name"]
                or Path(item["record"]) != expected_path):
            raise ValueError("Preserved OOM attempt path was modified")
        record_path = run_dir / expected_path
        record = curriculum._read_json(record_path)
        archive = run_dir / expected_path.with_suffix("")
        if (record.get("owner") != state["owner"]
                or record.get("plan_sha256") != state["plan_sha256"]
                or curriculum._file_hash(record_path) != item["sha256"]
                or record.get("next_num_envs") != item["next_num_envs"]
                or record.get("archive") != str(archive)
                or archive.is_symlink() or not archive.is_dir()
                or record["files"] != _artifact_manifest(archive)):
            raise ValueError("Preserved OOM attempt record or files were modified")


def _archive_oom_attempt(state, config, run_dir, stage_dir, active, num_envs, next_count, returncode):
    index = sum(item["stage_name"] == stage_dir.name for item in state.get("oom_attempts", []))
    parent = run_dir / "failed_attempts"
    for directory in (parent, parent / stage_dir.name):
        if directory.is_symlink():
            raise ValueError("Refusing a symlink OOM archive directory")
        directory.mkdir(exist_ok=True)
    record_path = parent / stage_dir.name / f"attempt-{index:03d}.json"
    archive = record_path.with_suffix("")
    if record_path.exists() or record_path.is_symlink() or archive.exists() or archive.is_symlink():
        raise RuntimeError("Interrupted OOM archive preserved; refusing an automatic overwrite")
    record = dict(schema="cat-continuous-oom-attempt-v1", owner=state["owner"],
        plan_sha256=config["sha256"], stage_name=stage_dir.name, attempt=index,
        original_stage_path=str(stage_dir), archive=str(archive), returncode=returncode,
        num_envs=num_envs, next_num_envs=next_count, active_intent=active,
        files=_artifact_manifest(stage_dir), progress="no observed rollout/update/checkpoint")
    # Publish an immutable intent before moving the failed directory. An
    # interrupted archive is preserved for inspection, never silently replayed.
    with record_path.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    stage_dir.rename(archive)
    item = dict(stage_name=stage_dir.name, attempt=index, next_num_envs=next_count,
                record=str(record_path.relative_to(run_dir)), sha256=curriculum._file_hash(record_path))
    state.setdefault("oom_attempts", []).append(item)
    state.setdefault("effective_stage_batches", {})[stage_dir.name] = next_count
    state.update(active_stage=None, status="retrying_gpu_oom")


def _stage_capacity(scene, requested, config):
    if config["estimated_memory_budget_gib"] is None:
        return {"num_envs": requested, "requested_num_envs": requested, "enabled": False}
    from cat_ppo.furniture.capacity import estimated_memory_cap
    decision = estimated_memory_cap(scene, requested, config["estimated_memory_budget_gib"],
                                    min_envs=config["oom_min_num_envs"])
    count = decision["num_envs"]
    if (type(count) is not int or not 0 < count <= requested or count % 4
            or config["steps_per_stage"] % (count * config["unroll_length"] * config["checkpoint_epochs"])):
        raise ValueError("Estimated memory cap produced a batch incompatible with the exact stage cadence")
    return {**decision, "requested_num_envs": requested, "enabled": True}


def _import_restart(source_run, run_dir, config, owner):
    """Copy one verified native selection while holding the stopped source lock.

    Source metadata and models are read-only. The owned copy decouples the new
    process from future source-run cleanup and is retired after verified handoff.
    """
    source_run = Path(source_run).absolute()
    lock_path = source_run / ".continuous.lock"
    if source_run == run_dir or source_run.is_symlink() or lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("Restart requires a distinct existing regular continuous run")
    with lock_path.open("r") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        old_config = curriculum._read_json(source_run / "continuous_config.json")
        old_state = curriculum._read_json(source_run / "continuous_state.json")
        runner = curriculum._read_json(source_run / "runner.json")
        if (old_config.get("schema") != "cat-continuous-curriculum-v1"
                or old_config.get("sha256") != _digest({key: value for key, value in old_config.items() if key != "sha256"})):
            raise ValueError("Restart source configuration hash is invalid")
        _validate_state(old_state, old_config)
        old_anchor = _load_restart_anchor(source_run, old_state, old_config)
        if (runner.get("active") is not False or runner.get("child_pid") is not None
                or runner.get("owner") != old_state["owner"]
                or runner.get("config_sha256") != old_config["sha256"]
                or old_state.get("status") not in ("stopped_by_request", "stopped_with_partial_stage", "failed")):
            raise ValueError("Restart source must be stopped or failed with no active worker")
        _verify_oom_archives(old_state, source_run)
        if old_config["seed"] != config["seed"]:
            raise ValueError("Restart must preserve the source curriculum seed")
        previous_selection = old_anchor["imported_selection"] if old_anchor else None
        for item in old_state["completed"]:
            directory = source_run / "stages" / item["name"]
            receipt = curriculum._read_json(directory / "curriculum_receipt.json")
            run = curriculum._read_json(directory / "run.json")
            summary = curriculum._read_json(directory / "summary.json")
            if (curriculum._file_hash(directory / "curriculum_receipt.json") != item["receipt_sha256"]
                    or receipt["owner"] != old_state["owner"] or receipt["plan_sha256"] != old_config["sha256"]
                    or curriculum._file_hash(directory / "run.json") != receipt["run_sha256"]
                    or curriculum._file_hash(directory / "summary.json") != receipt["summary_sha256"]
                    or summary["actual_steps"] != item["actual_steps"]
                    or receipt["input_selection"] != previous_selection):
                raise ValueError("Restart source completed-stage history was modified")
            _verify_source_lineage(run, receipt["selected"], previous_selection)
            previous_selection = receipt["selected"]
        next_index = old_state.get("start_stage_index", 0) + len(old_state["completed"])
        offset = old_state["completed_transitions"]
        partial = old_state.get("partial_stage")
        if partial:
            stage = stage_spec(next_index, config["seed"])
            if partial["name"] != stage["name"] or old_state.get("active_stage", {}).get("stage_name") != stage["name"]:
                raise ValueError("Restart partial stage differs from its recorded curriculum position")
            directory = source_run / "stages" / stage["name"]
            selected = curriculum._selected(directory)
            summary = curriculum._read_json(directory / "summary.json")
            run = curriculum._read_json(directory / "run.json")
            if (partial["selected"] != selected
                    or partial["summary_sha256"] != curriculum._file_hash(directory / "summary.json")
                    or summary.get("stopped_by_request") is not True
                    or summary.get("actual_steps") != partial["actual_steps"]
                    or summary.get("selected_step") != selected["step"]
                    or summary.get("selected_score") != selected["score"]
                    or type(partial["actual_steps"]) is not int
                    or not selected["step"] <= partial["actual_steps"] <= old_config["steps_per_stage"]
                    or summary["native_checkpoint"] != selected["path"]
                    or run["args"]["global_step_offset"] != offset
                    or run["args"]["steps"] != old_config["steps_per_stage"]
                    or run["args"]["num_envs"] != old_state["active_stage"].get("num_envs", stage_num_envs(old_config, stage))
                    or old_state["active_stage"]["input_selection"] != previous_selection):
                raise ValueError("Restart partial checkpoint, budget or lineage metadata is inconsistent")
            _verify_source_lineage(run, selected, previous_selection)
            _verify_partial_intent(run, selected, old_state["active_stage"], stage, old_config,
                                   source_run, directory, offset)
            offset += partial["actual_steps"]
            mode = "replay-interrupted-scene-from-selected-parameters"
        elif old_state.get("status") == "failed" and old_state.get("active_stage") is not None:
            if previous_selection is None:
                raise ValueError("Failed restart source has no verified model to import")
            stage = stage_spec(next_index, config["seed"])
            failed_directory = source_run / "stages" / stage["name"]
            active = old_state["active_stage"]
            if (active.get("stage_name") != stage["name"] or active.get("stage_sha256") != _digest(stage)
                    or active.get("input_selection") != previous_selection):
                raise ValueError("Failed restart stage intent differs from its expected scene or predecessor")
            _assert_no_progress(failed_directory)
            failed_run_path = failed_directory / "run.json"
            if failed_run_path.exists() or failed_run_path.is_symlink():
                failed_run = curriculum._read_json(failed_run_path)
                _verify_source_lineage(failed_run, previous_selection, previous_selection)
                _verify_partial_intent(failed_run, None, active, stage, old_config,
                                       source_run, failed_directory, offset)
            directory = source_run / "stages" / old_state["current_stage"] if old_state["completed"] else None
            selected = curriculum._selected(directory) if directory is not None else _initial_selection(old_anchor)
            if selected != previous_selection:
                raise ValueError("Failed restart source predecessor differs from its last verified receipt")
            mode = "retry-failed-scene-without-observed-progress"
        else:
            if old_state.get("active_stage") is not None or not old_state["completed"]:
                raise ValueError("Restart source has no clean completed or stopped-partial selection")
            directory = source_run / "stages" / old_state["current_stage"]
            selected = curriculum._selected(directory)
            if selected != previous_selection:
                raise ValueError("Restart source selection differs from its last handoff receipt")
            mode = "continue-at-next-scene"
        destination = run_dir / "restart_source"
        if destination.exists() or destination.is_symlink():
            raise ValueError("Restart import destination already exists")
        destination.mkdir()
        shutil.copytree(selected["path"], destination / "native")
        if _manifest(destination / "native") != selected["files"]:
            raise ValueError("Copied restart checkpoint failed hash verification")
        imported_selection = {**selected, "path": str(destination / "native"), "generation": str(destination)}
        anchor = dict(schema="cat-continuous-restart-v1", owner=owner, source_run=str(source_run),
            source_owner=old_state["owner"], source_config_sha256=old_config["sha256"],
            source_state_sha256=curriculum._file_hash(source_run / "continuous_state.json"),
            source_runner_sha256=curriculum._file_hash(source_run / "runner.json"),
            source_stage=directory.name if directory is not None else old_anchor["source_stage"],
            source_run_metadata_sha256=curriculum._file_hash(directory / "run.json") if directory is not None else None,
            source_summary_sha256=curriculum._file_hash(directory / "summary.json") if directory is not None else None,
            source_selection=selected, imported_selection=imported_selection,
            resume_stage_index=next_index, global_step_offset=offset, mode=mode,
            optimizer="fresh; actor, critic and normalization parameters retained",
            offset_semantics="all executed source transitions, including work after the selected best checkpoint")
        if directory is None:
            anchor.update(source_checkpoint_location="owned_restart_import",
                source_restart_anchor_sha256=curriculum._file_hash(source_run / "restart_anchor.json"))
        if mode == "retry-failed-scene-without-observed-progress":
            anchor["failed_stage"] = dict(name=stage["name"], active_intent=active,
                files=_artifact_manifest(failed_directory), observed_steps=0,
                reason="explicit restart from last verified model; no failed-stage metrics/checkpoint")
        curriculum._atomic_json(destination / "owner.json", {"owner": owner, "anchor_sha256": _digest(anchor)})
        return anchor


def _initial_selection(anchor):
    if anchor is None:
        return None
    selected = anchor["imported_selection"]
    path = Path(selected["path"])
    if path.is_symlink() or not path.is_dir() or _manifest(path) != selected["files"]:
        raise ValueError("Owned restart checkpoint differs from its immutable import manifest")
    return selected


def _retire_restart_copy(run_dir, state, anchor):
    if not anchor or not state["completed"]:
        return
    # Only the new run's import copy is eligible; source_run is never mutated.
    destination = run_dir / "restart_source"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or curriculum._read_json(destination / "owner.json") != {
                "owner": state["owner"], "anchor_sha256": state["restart_anchor_sha256"]}:
            raise ValueError("Refusing to retire an unowned restart import")
        _initial_selection(anchor)
        shutil.rmtree(destination)
    curriculum._atomic_json(run_dir / "restart_source_retired.json", {
        "anchor_sha256": state["restart_anchor_sha256"], "first_verified_stage": state["completed"][0]["name"],
        "reason": "owned import retired after verified stage handoff; original source run preserved"})


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
    for name in ("stages", "scenes", "STOP", "runner.json", "continuous_config.json", "restart_anchor.json", "restart_source"):
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
                     active_stage=None, completed_transitions=0, status="starting",
                     start_stage_index=0, initial_transition_offset=0)
        if config["restart_from_run"] is not None:
            anchor = _import_restart(config["restart_from_run"], run_dir, config, state["owner"])
            state.update(start_stage_index=anchor["resume_stage_index"],
                         initial_transition_offset=anchor["global_step_offset"],
                         completed_transitions=anchor["global_step_offset"], restart_anchor_sha256=_digest(anchor))
            curriculum._atomic_json(run_dir / "restart_anchor.json", anchor)
        curriculum._atomic_json(run_dir / "continuous_config.json", config)
        curriculum._atomic_json(state_path, state)
    _validate_state(state, config)
    _verify_oom_archives(state, run_dir)
    anchor = _load_restart_anchor(run_dir, state, config)
    curriculum._reconcile_retirement(state, run_dir)
    _retire_restart_copy(run_dir, state, anchor)
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
            index = state.get("start_stage_index", 0) + len(state["completed"])
            stage = stage_spec(index, config["seed"])
            requested_envs = stage_num_envs(config, stage)
            attempt_count = sum(item["stage_name"] == stage["name"] for item in state.get("oom_attempts", []))
            pending_record = run_dir / "failed_attempts" / stage["name"] / f"attempt-{attempt_count:03d}.json"
            if pending_record.exists() or pending_record.is_symlink() or pending_record.with_suffix("").exists():
                raise RuntimeError("Interrupted OOM archive preserved; no automatic replay before inspection")
            scene_dir = curriculum.materialize_stage(stage["scene"], run_dir / "scenes")
            scene = load_scene(scene_dir)
            capacity = _stage_capacity(scene, requested_envs, config)
            capacity.update(scene_id=scene["scene_id"], geometry_hash=scene["geometry_hash"],
                            stage_name=stage["name"], plan_sha256=config["sha256"])
            num_envs = min(capacity["num_envs"], state.get("effective_stage_batches", {}).get(stage["name"], capacity["num_envs"]))
            previous_capacity = state.setdefault("capacity_decisions", {}).get(stage["name"])
            if previous_capacity is not None and previous_capacity != capacity:
                raise ValueError("Estimated capacity decision changed within an existing stage")
            state["capacity_decisions"][stage["name"]] = capacity
            capacity_path = run_dir / (stage["name"] + ".capacity.json")
            if capacity_path.exists() and curriculum._read_json(capacity_path) != capacity:
                raise ValueError("Recorded stage capacity decision was modified")
            curriculum._atomic_json(capacity_path, capacity)
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
            previous_selection = curriculum._selected(previous) if previous is not None else _initial_selection(anchor)
            if previous_selection is not None:
                command += ["--warmstart", previous_selection["path"]]
            active = dict(owner=state["owner"], stage_name=stage["name"], stage_sha256=_digest(stage), num_envs=num_envs,
                          capacity=capacity,
                          input_selection=previous_selection, command=command)
            if state.get("active_stage") is not None and state["active_stage"] != active:
                raise ValueError("Interrupted continuous stage intent differs from saved intent")
            if stage_dir.exists() and state.get("active_stage") is None:
                raise ValueError("Stage directory exists without a recorded continuous intent")
            state.update(active_stage=active, status="training")
            curriculum._atomic_json(state_path, state)
            if not stage_dir.exists():
                try:
                    _run_child(command, cwd=root, runner_record=runner, runner_path=runner_path)
                except subprocess.CalledProcessError as error:
                    if stop_file.exists():
                        raise
                    next_count = _oom_retry_batch(stage_dir, num_envs=num_envs, attempts=attempt_count, config=config)
                    if next_count is None:
                        raise
                    failure = curriculum._read_json(stage_dir / "failure.json")
                    if (failure.get("requested_steps") != config["steps_per_stage"]
                            or failure.get("global_step_offset") != state["completed_transitions"]
                            or failure.get("stage") != stage["name"]
                            or failure.get("run_dir") != str(stage_dir)):
                        raise ValueError("OOM metadata differs from the active stage budget/offset") from error
                    if (stage_dir / "run.json").exists():
                        failed_run = curriculum._read_json(stage_dir / "run.json")
                        _verify_source_lineage(failed_run, {"selection_source": "training_proxy"}, previous_selection)
                        _verify_partial_intent(failed_run, None, active, stage, config,
                                               run_dir, stage_dir, state["completed_transitions"])
                    _archive_oom_attempt(state, config, run_dir, stage_dir, active,
                                         num_envs, next_count, error.returncode)
                    curriculum._atomic_json(state_path, state)
                    print(f"GPU OOM before training progress: preserved attempt {attempt_count}; "
                          f"retrying {stage['name']} with {next_count} environments (was {num_envs}).", flush=True)
                    continue
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
                actual_steps=summary["actual_steps"], selected_step=selected["step"], num_envs=num_envs,
                summary_sha256=receipt["summary_sha256"],
                receipt_sha256=curriculum._file_hash(stage_dir / "curriculum_receipt.json")))
            state.update(current_stage=stage["name"], current_best=str(stage_dir / "checkpoints" / "best"),
                         active_stage=None, completed_transitions=state["completed_transitions"] + summary["actual_steps"],
                         status="switching_scene")
            curriculum._atomic_json(state_path, state)
            curriculum._reconcile_retirement(state, run_dir)
            _retire_restart_copy(run_dir, state, anchor)
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
    run.add_argument("--pilot-furniture-num-envs", type=int, help="Separate furniture pilot batch override")
    run.add_argument("--random-num-envs", type=int, help="Explicit original random CAT batch override")
    run.add_argument("--generic-num-envs", type=int, help="Generic dense clutter batch override")
    run.add_argument("--furniture-num-envs", type=int, help="Furniture dense clutter batch override")
    run.add_argument("--restart-from-run", type=Path,
                     help="Import a stopped/failed run's verified parameters and curriculum/step offset")
    run.add_argument("--oom-max-retries", type=int, default=2,
                     help="Bounded batch halvings after GPU OOM before any observed training progress")
    run.add_argument("--oom-min-num-envs", type=int, default=128, help="Minimum automatic OOM fallback batch")
    run.add_argument("--estimated-memory-budget-gib", type=float,
                     help="Optional geometric memory estimate used to cap scene batches before launch")
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
        parser.error("The sequential scene trainer is retired after the CAT audit. "
                     "Use train_cat_wholebody.py run for one persistent mixed-scene learner. "
                     "Existing historical runs can still be stopped with this command.")


if __name__ == "__main__":
    main()
