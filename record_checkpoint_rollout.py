"""Record one native policy episode on CPU, without training or autoreset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def record(args):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import numpy as np
    from cat_ppo.envs.g1.env_furniture import G1FurnitureEnv, default_config
    from cat_ppo.furniture.benchmark import EpisodeTracker
    from cat_ppo.furniture.checkpoint import _manifest
    from evaluate_furniture import _apply_environment_overrides, _configure_numerics, load_controller

    started = time.monotonic()
    numerics = _configure_numerics()
    checkpoint = args.checkpoint.resolve()
    selection = json.loads(args.selection.read_text())
    if _manifest(checkpoint) != selection["files"]:
        raise ValueError("Imported checkpoint payload differs from its saved selection manifest")
    stage = json.loads(args.stage_metadata.read_text())
    root = Path(__file__).resolve().parent
    source_paths = ["cat_ppo/envs/g1/" + name for name in (
        "env_furniture.py", "env_cat.py", "env_loco.py", "base.py", "constants.py")]
    source_paths += ["cat_ppo/furniture/" + name for name in (
        "control.py", "grippers.py", "perception.py", "scenes.py")]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_paths}
    if any(value != stage["code"]["source_files"].get(name) for name, value in hashes.items()):
        raise ValueError("Recording environment implementation differs from the selected scene's training source")
    config = default_config()
    _apply_environment_overrides(config, stage["environment_config"])
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=False)
    print("Loading exact training scene and native checkpoint on CPU", flush=True)
    env = G1FurnitureEnv(args.scene_dir.resolve(), config=config)
    if env.scene["geometry_hash"] != stage["training_scene"]["geometry_hash"]:
        raise ValueError("Scene geometry differs from training metadata")
    policy = load_controller(checkpoint, env)
    key = jax.random.PRNGKey(args.episode_seed)
    print("Compiling CPU reset", flush=True)
    state = jax.jit(env.reset)(key)
    state.data.qpos.block_until_ready()
    step = jax.jit(env.step)
    tracker = EpisodeTracker(env.scene, case_id="overnight-checkpoint-example",
        training_seed=0, episode_seed=args.episode_seed, controller="deterministic_native_CAT",
        goal_tolerance=float(config.goal_radius))
    poses, velocities, times, actions, telemetry = [], [], [], [], []

    def capture(value):
        qpos, qvel, sim_time = jax.device_get((value.data.qpos, value.data.qvel, value.data.time))
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("Nonfinite simulated pose cannot be rendered as a valid trajectory")
        poses.append(qpos.copy()); velocities.append(qvel.copy()); times.append(float(sim_time))

    capture(state)
    print("Compiling and executing CPU policy/physics step", flush=True)
    reason = "timeout"
    taxonomy = {0: "none", 1: "other_body", 2: "hand", 3: "arm"}
    for index in range(math.ceil(env.scene["time_budget"] / env.dt)):
        key, action_key = jax.random.split(key)
        action = policy(state.obs, action_key)
        if not np.isfinite(np.asarray(action)).all():
            raise ValueError("Nonfinite policy action")
        state = step(state, action)
        info = jax.device_get(state.info)
        capture(state)
        actions.append(np.asarray(action))
        contact_time = float(info.get("first_contact_time", -1.0))
        tracker.update(position=poses[-1][:2], elapsed=(index + 1) * env.dt,
            furniture_contact=bool(info["furniture_contact"]), hand_contact=bool(info["hand_contact"]),
            fall=bool(info["fall"]), self_collision=bool(info["self_collision"]),
            nonfoot_floor_contact=bool(info["nonfoot_floor_contact"]),
            goal_reached=bool(info["goal_reached"]) and not bool(info["numerical_failure"]),
            numerical_failure=bool(info["numerical_failure"]),
            first_contact_time=contact_time if contact_time >= 0 else None,
            minimum_clearance=float(info["minimum_clearance"]),
            contact_part=taxonomy.get(int(info["first_contact_part"]), "unclassified"))
        telemetry.append(dict(step=index + 1, time=(index + 1) * env.dt,
            done=bool(state.done), reward=float(state.reward),
            **{name: bool(info[name]) for name in ("furniture_contact", "hand_contact", "fall",
                "self_collision", "nonfoot_floor_contact", "goal_reached", "numerical_failure")},
            minimum_clearance=float(info["minimum_clearance"])))
        if (index + 1) % 50 == 0 or bool(state.done):
            print(json.dumps(dict(steps=index + 1, simulation_seconds=(index + 1) * env.dt,
                position=poses[-1][:3].tolist(), done=bool(state.done))), flush=True)
        if bool(state.done):
            reason = "numerical_failure" if bool(info["numerical_failure"]) else None
            break
    outcome = tracker.finish(termination_reason=reason)
    if outcome.termination_reason == "incomplete":
        outcome.termination_reason = "environment_termination"
    trajectory = output / "trajectory.npz"
    np.savez_compressed(trajectory, qpos=np.asarray(poses), qvel=np.asarray(velocities),
                       time=np.asarray(times), actions=np.asarray(actions))
    metadata = dict(schema="cat-checkpoint-rollout-v1", checkpoint_label="Overnight CAT checkpoint",
        global_step=args.global_step, checkpoint_stage="r000001-p0-cat-forward",
        checkpoint_native=str(checkpoint), checkpoint_selection=selection,
        checkpoint_hashes_verified=True, checkpoint_selection_step=selection["step"],
        episode_seed=args.episode_seed, dt=env.dt, outcome=outcome.to_dict(),
        scene_id=env.scene["scene_id"], geometry_hash=env.scene["geometry_hash"],
        scene_dir=str(args.scene_dir.resolve()), scene_split="training scene seen overnight",
        environment_config=config.to_dict(), environment_source_hashes=hashes,
        environment_source_matches_training=True, recording_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        device=[str(device) for device in jax.devices()], numerics=numerics,
        inference="native checkpoint deterministic tanh(mean), stored observation normalizer",
        physics="unwrapped G1FurnitureEnv.reset/step, MJX CPU; no autoreset",
        observation_noise="unchanged from scene training config, seeded",
        simulation_steps=len(actions), terminal_pose_retained=True, trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        wall_seconds=time.monotonic() - started)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    (output / "telemetry.json").write_text(json.dumps(telemetry, indent=2, allow_nan=False) + "\n")
    print(json.dumps(outcome.to_dict(), indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--stage-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-seed", type=int, default=0)
    parser.add_argument("--global-step", type=int, default=102301696)
    record(parser.parse_args())
