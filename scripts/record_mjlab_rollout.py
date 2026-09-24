#!/usr/bin/env python3
"""Record one explicitly selected mjlab scene from best.pt or resume.pt.

No training, optimizer updates, W&B or checkpoint writes. The output is consumed
by render_clutter_rollouts.py (rooms) or this script's ``render`` command.
Policy actions are deterministic; the task's seeded noise and randomization
remain enabled exactly as recorded in the training configuration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # Existing metadata validators only.


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="Native mjlab best.pt or resume.pt")
    p.add_argument("--bank-manifest", type=Path, required=True)
    p.add_argument("--allow-source-mismatch", action="store_true",
                   help="Record even if cat_mjlab/*.py changed since the checkpoint (reward-only edits do not "
                        "change actions; reactive.py edits DO change object behaviour -- say so with the video)")
    p.add_argument("--reactive-bank", type=Path,
                   help="Optional cat-reactive-standing-v1 manifest; adds approaching objects")
    p.add_argument("--body-collision-bank", type=Path, required=True)
    p.add_argument("--body-collision-resets", type=Path, required=True)
    p.add_argument("--scene-id", required=True)
    p.add_argument("--frames", type=int, required=True, help="Maximum control transitions, 1..4000")
    p.add_argument("--policy-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions from the policy distribution as in training instead of the mean")
    p.add_argument("--reactive-row", type=int,
                   help="Force this reactive-bank row (an approaching object) instead of the .25 reset coin; "
                        "needs --reactive-bank. The object trajectory is exported for the renderer.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", type=Path, required=True, help="New recording directory")
    return p


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_policy(path, *, device, policy_id):
    import torch
    from cat_mjlab.learning import Learner, LearnerConfig
    # Hash and decode one open inode: training may atomically replace best.pt
    # while a future user requests a recording.
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        stream.seek(0)
        snapshot = torch.load(stream, map_location="cpu", weights_only=True)
    schema = snapshot.get("schema")
    if schema == "cat-mjlab-best-v1":
        config, model, step = snapshot["config"], snapshot["model"], snapshot["step"]
    elif schema == "cat-mjlab-runtime-v1":
        state = snapshot["learner"]
        if state.get("schema") != "cat-mjlab-learner-v1":
            raise ValueError("Unsupported learner checkpoint schema")
        config, model, step = state["config"], state["model"], state["env_steps"]
    else:
        raise ValueError("Expected native mjlab best.pt or resume.pt")
    config = LearnerConfig(**config)
    expected_config = json.loads(json.dumps(asdict(config)))
    if snapshot.get("contract", {}).get("learner_config") != expected_config:
        raise ValueError("Saved learner configuration differs from its training contract")
    policies = config.num_policies if config.algorithm == "sapg" else 1
    if not 0 <= policy_id < policies:
        raise ValueError("Requested policy ID is outside this checkpoint")
    if not all(torch.isfinite(value).all() for value in model.values()):
        raise ValueError("Checkpoint model contains nonfinite values")
    learner = Learner(config, device=device)
    learner.model.load_state_dict(model, strict=True)
    learner.model.eval()
    return learner, dict(contract=snapshot["contract"], step=int(step), schema=schema,
                         observation_contract=snapshot.get("observation_contract"),
                         checkpoint_sha256=digest.hexdigest())


def verify_contract(contract, args):
    from cat_mjlab.runner import _source_identity, _versions
    for key, path in (("bank_sha256", args.bank_manifest), ("collision_sha256", args.body_collision_bank),
                      ("resets_sha256", args.body_collision_resets)):
        if contract.get(key) != sha256(path):
            raise ValueError(f"Recording {key} differs from the checkpoint")
    if contract.get("source_sha256") != _source_identity():
        if not args.allow_source_mismatch:
            raise ValueError("Task/learner source differs from this native checkpoint")
        print("WARNING: cat_mjlab source differs from the checkpoint; recording anyway (--allow-source-mismatch)", file=sys.stderr)
    if contract.get("versions") != _versions():
        raise ValueError("Simulator/learner package versions differ from this native checkpoint")


def record_episode(task, learner, *, policy_id, frames, stochastic=False):
    """Copy each final integrated pose before normal task autoreset can erase it."""
    import numpy as np
    import torch
    if task.num_envs != 1 or not 1 <= frames <= 4000:
        raise ValueError("Recording requires one world and 1..4000 control transitions")
    def pose():
        frame = {key: getattr(task.sim.data, key)[0].detach().cpu().numpy().copy()
                 for key in ("qpos", "qvel", "time")}
        objects = getattr(task, "reactive_objects", None)
        if objects is not None and bool(objects.state["active"][0]):
            # Approaching objects are analytic (no MuJoCo geom), so export them or the video cannot show them.
            s = objects.state
            frame.update({"object_" + key: s[name][0].detach().cpu().numpy().copy() for key, name in
                          (("position", "position"), ("rotation", "rotations"), ("sizes", "sizes"),
                           ("kinds", "kinds"), ("valid", "valid"), ("retreating", "retreating"))})
        return frame
    trace = [pose()]
    original_reset = task.reset
    captured = []
    def capture_reset(*args, **kwargs):
        captured.append(pose())
        return original_reset(*args, **kwargs)
    task.reset = capture_reset
    reason, result = "recording_limit", None
    try:
        for _ in range(frames):
            captured.clear()
            action = learner.act(task.obs, policy_ids=policy_id, deterministic=not stochastic)["action"]
            result = task.step(action)
            if len(captured) != 1:
                raise RuntimeError("Task autoreset boundary changed; refusing an ambiguous trace")
            trace.append(captured[0])
            task.sim.capacity_report()
            if bool(result["metrics"]["resolved"][0]):
                reason = "clean_goal" if bool(result["metrics"]["successful"][0]) else "first_outcome_failure"
                break
            if bool(result["done"][0]):
                reason = "native_horizon" if bool(result["truncated"][0]) else "native_failure"
                break
    finally:
        task.reset = original_reset
    arrays = {key: np.stack([frame[key] for frame in trace]) for key in trace[0]}
    if (not all(np.isfinite(value).all() for value in arrays.values())
            or not np.all(np.diff(arrays["time"]) > 0)):
        raise ValueError("Recording contains nonfinite poses or reset-discontinuous timestamps")
    metrics = result["metrics"]
    success = bool(metrics["resolved"][0] & metrics["successful"][0])
    outcome = {key.removeprefix("episode/"): bool(value[0]) for key, value in metrics.items()
               if key.startswith("episode/") and value.dtype == torch.bool}
    outcome.update(goal_reached=success, reached_goal=success, success=success,
                   timeout=bool(result["truncated"][0]), terminated=bool(result["terminated"][0]),
                   resolved=bool(metrics["resolved"][0]), termination_reason=reason,
                   length=len(trace)-1, reward=float(metrics["episode_return"][0]))
    return arrays, outcome


def write_geometry(directory, bank_manifest, record):
    """Reuse exact room boxes or CAT's existing occupancy mesh convention."""
    import numpy as np
    from cat_mjlab.model import assemble_training_xml
    xml = ET.fromstring(assemble_training_xml())
    world = xml.find("worldbody")
    from cat_ppo.furniture.generalist_fields import load_generalist_manifest, scene_directory
    _mp = Path(bank_manifest).resolve()
    scene_dir = scene_directory(load_generalist_manifest(_mp, verify_files=False), _mp, record)
    is_cat = record.get("task_kind", "cat" if record["family"] == "original_cat" else "room") == "cat"
    if is_cat:
        from scripts.evaluate_clutter_checkpoint import cat_obstacle_mesh
        occupancy = np.load(scene_dir / "obs.npy", allow_pickle=False)
        cat_obstacle_mesh(occupancy, record["dx"]).export(directory / "obs.obj")
        ET.SubElement(xml.find("asset"), "mesh", name="recorded_cat_obstacle", file="obs.obj")
        ET.SubElement(world, "geom", name="recorded_cat_obstacle", type="mesh", mesh="recorded_cat_obstacle",
            pos=" ".join(map(str, record["origin"])), contype="0", conaffinity="0", group="1", rgba=".37 .52 .61 1")
        scene = dict(scene_id=record["scene_id"], family=record["family"], source=record.get("source", {}))
    else:
        scene = json.loads((scene_dir / "scene.json").read_text())
        palette = {"tabletop": ".72 .49 .27 1", "table_leg": ".40 .30 .22 1",
                   "chair_seat": ".23 .47 .53 1", "chair_back": ".20 .42 .48 1",
                   "chair_leg": ".20 .26 .30 1", "wall": ".62 .66 .70 1"}
        for index, part in enumerate(scene["boxes"]):
            yaw = part["yaw"]
            ET.SubElement(world, "geom", name=f"clutter_box_{index}", type="box",
                pos=" ".join(map(str, part["center"])), size=" ".join(map(str, part["half_size"])),
                quat=f"{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}",
                rgba=palette.get(part["category"], ".60 .40 .35 1"), contype="0", conaffinity="0",
                group="4" if part["category"] == "wall" else "0")
    (directory / "model.xml").write_text(ET.tostring(xml, encoding="unicode"))
    (directory / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    return scene, is_cat


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["render"]:
        return render_recording(argv[1:])
    args = parser().parse_args(argv)
    if not 1 <= args.frames <= 4000:
        raise ValueError("--frames must lie in 1..4000")
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Recording destination must be new")
    import numpy as np
    import torch
    from cat_mjlab.runner import create_task
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Native mjlab recording requested CUDA but no CUDA device is available")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    learner, saved = load_policy(args.checkpoint, device=args.device, policy_id=args.policy_id)
    contract = saved["contract"]
    verify_contract(contract, args)
    manifest = json.loads(args.bank_manifest.read_text())
    selected = [i for i, scene in enumerate(manifest["scenes"]) if str(scene["scene_id"]) == args.scene_id]
    if len(selected) != 1:
        raise ValueError("--scene-id must identify exactly one scene in the checkpoint bank")
    from cat_mjlab.observation_contract import ACTOR_SIZE, CRITIC_SIZE, actor_size
    ACTOR_SIZE = actor_size(bool(contract['environment_config'].get('sdf_rate_obs', False)))
    if (learner.config.actor_obs, learner.config.critic_obs, learner.config.action_size) != (ACTOR_SIZE, CRITIC_SIZE, 29):
        raise ValueError("Checkpoint differs from native 222/310 observation contract; obsolete checkpoints are unsupported")
    source_hash = saved["checkpoint_sha256"]
    factory_args = SimpleNamespace(**vars(args), num_envs=1, compile_task=False,
                                  nconmax=contract["nconmax"], njmax=contract["njmax"])
    task, sim, _ = create_task(factory_args, environment_config=contract["environment_config"])
    if saved["schema"] == "cat-mjlab-best-v1" and saved["observation_contract"] != task.contract:
        raise ValueError("Named observation/action features differ from the selected policy")
    if args.reactive_row is not None:
        objects = getattr(task, "reactive_objects", None)
        if objects is None:
            raise ValueError("--reactive-row needs --reactive-bank")
        if not 0 <= args.reactive_row < len(objects.rows):
            raise ValueError(f"--reactive-row must be in [0, {len(objects.rows)})")
        objects.force_rows = torch.tensor([args.reactive_row], device=args.device)
    task.reset(scene_ids=torch.tensor(selected, device=args.device))
    arrays, outcome = record_episode(task, learner, policy_id=args.policy_id, frames=args.frames, stochastic=args.stochastic)
    record = manifest["scenes"][selected[0]]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mjlab-recording-", dir=output.parent) as temporary:
        directory = Path(temporary)
        scene, is_cat = write_geometry(directory, args.bank_manifest, record)
        np.savez_compressed(directory / "trajectory.npz", **arrays)
        metadata = dict(schema="cat-mjlab-recording-v1", backend="mjlab/MuJoCo Warp", label="mjlab-trained policy",
                        reactive_row=args.reactive_row,
            scene_id=record["scene_id"], scene_name=record["scene_id"], scene_seed=scene.get("seed"),
            family=record["family"], seed=args.seed, checkpoint_steps=saved["step"], policy_id=args.policy_id,
            checkpoint_path=str(args.checkpoint.resolve()), checkpoint_sha256=source_hash,
            checkpoint_schema=saved["schema"], bank_sha256=contract["bank_sha256"],
            collision_sha256=contract["collision_sha256"], resets_sha256=contract["resets_sha256"],
            source_sha256=contract["source_sha256"], dt=task.dt, inference="deterministic tanh(mean)",
            compile_task=False,
            outcome=outcome, requested_control_frames=args.frames, recorded_control_frames=len(arrays["time"])-1,
            reset="Explicit scene; fresh full episode horizon; seeded native noise/PD/push randomization retained",
            stopping="First clean goal, first fault/native horizon, or explicit recording frame bound",
            final_frame="Integrated pose copied before normal autoreset", capacity=sim.capacity_report(),
            trajectory_sha256=sha256(directory / "trajectory.npz"),
            geometry="Exact saved scene, added to display XML only; no physics or trajectory changes",
            renderer="scripts/record_mjlab_rollout.py render" if is_cat else "scripts/render_clutter_rollouts.py")
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        os.rename(directory, output)
    print(json.dumps(dict(output_dir=str(output), outcome=outcome, renderer=metadata["renderer"]), indent=2))


def render_recording(argv):
    """Generic pose replay for CAT occupancy or room geometry; no new dynamics."""
    p = argparse.ArgumentParser(description=render_recording.__doc__)
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--fps", type=int, default=25)
    args = p.parse_args(argv)
    if not 12 <= args.fps <= 60 or args.output.suffix.lower() != ".mp4":
        raise ValueError("Render requires 12..60 fps and an .mp4 output")
    output = args.output.absolute()
    partial = output.with_name("." + output.stem + ".partial.mp4")
    if any(path.exists() or path.is_symlink() for path in (output, partial)):
        raise FileExistsError("Render outputs must be new")
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")
    import mujoco
    import numpy as np
    import shutil
    import subprocess
    from scripts.render_clutter_rollouts import resolve_xml_assets
    directory = args.input_dir.resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    if sha256(directory / "trajectory.npz") != metadata["trajectory_sha256"]:
        raise ValueError("Recorded trajectory checksum differs")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        qpos, qvel, times = (archive[key] for key in ("qpos", "qvel", "time"))
    xml, _ = resolve_xml_assets(directory / "model.xml")
    model = mujoco.MjModel.from_xml_string(xml)
    if (qpos.ndim != 2 or qpos.shape[1] != model.nq or qvel.shape != (len(qpos), model.nv)
            or times.shape != (len(qpos),) or len(times) < 2 or not np.all(np.diff(times) > 0)
            or not all(np.isfinite(array).all() for array in (qpos, qvel, times))):
        raise ValueError("Invalid saved trajectory/model dimensions or times")
    width, height = 1280, 720
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    data, camera, option = mujoco.MjData(model), mujoco.MjvCamera(), mujoco.MjvOption()
    camera.distance, camera.azimuth, camera.elevation = 3.8, -45, -25
    option.geomgroup[3:] = 0
    option.sitegroup[:] = 0
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_times = np.arange(0, times[-1]-times[0] + 1e-8, 1/args.fps) + times[0]
    indices = np.searchsorted(times, frame_times, side="right") - 1
    if indices[-1] != len(times)-1:
        indices = np.append(indices, len(times)-1)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(partial)]
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            try:
                with mujoco.Renderer(model, width=width, height=height) as renderer:
                    for index in indices:
                        data.qpos[:], data.qvel[:], data.time = qpos[index], qvel[index], float(times[index])
                        mujoco.mj_forward(model, data)
                        camera.lookat[:] = qpos[index, :3] + [0, 0, -.1]
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        process.stdin.write(renderer.render().tobytes())
                process.stdin.close()
                code = process.wait()
                if code:
                    errors.seek(0)
                    raise RuntimeError("ffmpeg failed: " + errors.read(4096).decode(errors="replace"))
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    print(json.dumps(dict(output=str(output), frames=len(indices), additional_physics_steps=0,
                          additional_policy_inference=False, outcome=metadata["outcome"]), indent=2))


if __name__ == "__main__":
    main()
