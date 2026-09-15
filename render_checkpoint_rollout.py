"""Render a recorded native CAT episode; replay poses without new simulation.

The trajectory comes from record_checkpoint_rollout.py. Camera placement,
colors, route markers and highlighted hand bounds are presentation aids only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--trajectory", type=Path, required=True)
    result.add_argument("--metadata", type=Path, required=True)
    result.add_argument("--scene-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--ffmpeg", type=Path)
    result.add_argument("--fps", type=int, default=25)
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from cat_ppo.envs.g1.env_furniture import assemble_scene_xml
    from cat_ppo.furniture.scenes import load_scene

    if not 12 <= args.fps <= 60:
        raise ValueError("fps must be between 12 and 60")
    started = time.monotonic()
    trajectory = args.trajectory.resolve()
    metadata_path = args.metadata.resolve()
    metadata = json.loads(metadata_path.read_text())
    scene_dir = args.scene_dir.resolve()
    scene = load_scene(scene_dir)
    if metadata.get("geometry_hash") != scene["geometry_hash"]:
        raise ValueError("Trajectory metadata and rendering geometry differ")
    if metadata.get("scene_id") != scene["scene_id"]:
        raise ValueError("Trajectory metadata and rendering scene differ")
    outcome = metadata["outcome"]
    if type(outcome.get("strict_success")) is not bool:
        raise ValueError("Episode metadata must declare strict_success")
    with np.load(trajectory, allow_pickle=False) as stored:
        qpos = np.asarray(stored["qpos"], dtype=float)
        qvel = np.asarray(stored["qvel"], dtype=float)
        timestamps = np.asarray(stored["time"], dtype=float)
        actions = np.asarray(stored["actions"], dtype=float)
    if (qpos.ndim != 2 or len(qpos) < 1 or qvel.ndim != 2
            or len(qvel) != len(qpos) or timestamps.shape != (len(qpos),)
            or actions.shape != (len(qpos) - 1, 29)):
        raise ValueError("Expected qpos/qvel/time frames and one action per transition")
    if not all(np.isfinite(value).all() for value in (qpos, qvel, timestamps)):
        raise ValueError("Cannot render nonfinite physical poses")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("Trajectory timestamps must strictly increase")
    elapsed = timestamps - timestamps[0]
    duration = float(elapsed[-1])
    if not math.isclose(duration, float(outcome["elapsed"]), rel_tol=1e-3, abs_tol=.03):
        raise ValueError("Trajectory duration differs from recorded episode outcome")

    output = args.output_dir.absolute()
    if output.is_symlink():
        raise ValueError("Refusing a symlink output directory")
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name for name in (
        "overnight-cat-checkpoint.mp4", ".overnight-cat-checkpoint.partial.mp4",
        "preview.png", "contact-sheet.png", "render-manifest.json")}
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise ValueError(f"Refusing to overwrite recording artifact: {path}")
    ffmpeg = str(args.ffmpeg.resolve()) if args.ffmpeg else shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as error:
            raise RuntimeError("ffmpeg is required; use --ffmpeg /path/to/ffmpeg") from error

    xml = assemble_scene_xml(scene)
    model = mujoco.MjModel.from_xml_string(xml)
    if qpos.shape[1] != model.nq or qvel.shape[1] != model.nv:
        raise ValueError("Trajectory joint dimensions differ from assembled model")
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    model.vis.quality.shadowsize = 4096
    model.vis.headlight.ambient[:] = [.50] * 3
    model.vis.headlight.diffuse[:] = [.65] * 3
    model.vis.headlight.specular[:] = [.12] * 3
    model.vis.map.znear = .003
    data = mujoco.MjData(model)
    palette = {"tabletop": (.66, .43, .23, 1), "table_leg": (.32, .26, .22, 1),
        "chair_seat": (.06, .39, .43, 1), "chair_back": (.08, .44, .48, 1),
        "chair_leg": (.15, .23, .27, 1), "chair_armrest": (.12, .32, .35, 1),
        "overhead": (.81, .52, .20, 1), "overhead_support": (.47, .35, .24, 1),
        "wall": (.57, .65, .70, .12)}
    for index, box in enumerate(scene["boxes"]):
        model.geom_rgba[model.geom(f"furniture_object_{index}").id] = palette.get(
            box["category"], (.43, .49, .53, 1))
    model.geom_rgba[model.geom("floor").id] = (.91, .93, .94, 1)
    for side in ("left", "right"):
        geom = model.geom(f"furniture_{side}_hand_envelope").id
        model.geom_group[geom] = 4
        model.geom_rgba[geom] = (.98, .59, .12, .65)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.geomgroup[4] = 1
    options.sitegroup[:] = 0
    room_x, room_y = scene["room_dimensions"][:2]
    route = np.asarray(scene["route"], dtype=float)
    heading = math.degrees(float(scene["start"][2]))
    main_camera = mujoco.MjvCamera()
    main_camera.distance = 3.45
    main_camera.azimuth = heading - 35
    main_camera.elevation = -19
    overview_camera = mujoco.MjvCamera()
    overview_camera.lookat[:] = [room_x / 2, room_y / 2, .20]
    overview_camera.distance = max(room_x, room_y) * 1.75
    overview_camera.azimuth = -90
    overview_camera.elevation = -67
    fonts = ["/System/Library/Fonts/Supplemental/Arial.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    font_path = next((path for path in fonts if Path(path).is_file()), None)
    font_cache = {}

    def font(size):
        if size not in font_cache:
            font_cache[size] = (ImageFont.truetype(font_path, size) if font_path
                                else ImageFont.load_default(size=size))
        return font_cache[size]

    def text_fitted(draw, position, text, maximum, size, fill):
        while size > 12 and draw.textlength(text, font=font(size)) > maximum:
            size -= 1
        draw.text(position, text, fill=fill, font=font(size))

    success = outcome["strict_success"]
    reason = str(outcome["termination_reason"]).replace("_", " ")
    result_text = "GOAL REACHED · no forbidden contacts" if success else f"EPISODE ENDED · {reason}"
    result_color = "#187D68" if success else "#AF4934"
    transitions = int(metadata["global_step"])
    checkpoint_text = f"{transitions / 1e6:.1f}M training transitions"
    scene_label = "Tables and chairs" if scene.get("counts", {}).get("tables", 0) else "CAT traversal scene"

    def add_markers(renderer, *, overview=False):
        for first, second in zip(route, route[1:]):
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                np.zeros(3), np.eye(3).reshape(-1), np.asarray([.0, .50, .52, .88]))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .021,
                np.array([*first, .016]), np.array([*second, .016]))
            renderer.scene.ngeom += 1
        markers = [(scene["start"][:2], [.20, .43, .79, 1], .09),
                   (scene["goal"], [.20, .64, .34, 1], .11)]
        if overview:
            markers.append((data.qpos[:2], [.94, .49, .12, 1], .13))
        for point, rgba, radius in markers:
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius] * 3), np.array([*point, .08]),
                np.eye(3).reshape(-1), np.asarray(rgba))
            renderer.scene.ngeom += 1

    pose_cache = {}

    def physical_frame(index, main_renderer, overview_renderer):
        # Stored poses only: mj_forward updates transforms for rendering and
        # never integrates a new physical step or changes the saved trajectory.
        if index in pose_cache:
            return pose_cache[index].copy()
        data.qpos[:] = qpos[index]
        data.qvel[:] = qvel[index]
        data.time = timestamps[index]
        mujoco.mj_forward(model, data)
        main_camera.lookat[:] = [data.qpos[0], data.qpos[1], .73]
        main_renderer.update_scene(data, camera=main_camera, scene_option=options)
        add_markers(main_renderer)
        image = Image.fromarray(main_renderer.render())
        overview_renderer.update_scene(data, camera=overview_camera, scene_option=options)
        add_markers(overview_renderer, overview=True)
        inset = Image.fromarray(overview_renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((897, 350, 1265, 620), radius=12, fill="#FFFFFF")
        image.paste(inset, (903, 383))
        draw.text((915, 360), "ROOM OVERVIEW", fill="#486172", font=font(15))
        # Bounded cache: sufficient for terminal holds and short slow replays.
        if len(pose_cache) < 128:
            pose_cache[index] = image.copy()
        return image

    def composed_frame(index, mode, main_renderer, overview_renderer):
        image = physical_frame(index, main_renderer, overview_renderer).convert("RGBA")
        layer = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(layer)
        draw.rectangle((0, 0, 1280, 101), fill=(250, 252, 253, 246))
        draw.rectangle((0, 635, 1280, 720), fill=(21, 44, 63, 247))
        draw.text((26, 16), "Overnight CAT checkpoint", fill="#16324F", font=font(29))
        draw.text((27, 58), f"{checkpoint_text}  ·  {scene_label}  ·  Unitree G1 / Dex3", fill="#4B6578", font=font(19))
        speed = "0.25× SLOW REPLAY" if mode.startswith("replay") else "1× REAL TIME"
        if mode == "initial":
            speed = "EPISODE START"
        elif mode in ("terminal", "replay_terminal"):
            speed = "TERMINAL FRAME" + (" / REPLAY" if mode == "replay_terminal" else "")
        draw.rounded_rectangle((964, 19, 1253, 57), radius=8, fill=(229, 238, 243, 255))
        text_fitted(draw, (979, 27), speed, 262, 18, "#214B63")
        clock = f"Simulation time  {elapsed[index]:05.2f} / {duration:.2f} s"
        draw.rounded_rectangle((24, 118, 332, 155), radius=8, fill=(255, 255, 255, 228))
        draw.text((36, 128), clock, fill="#2C4C60", font=font(16))
        terminal = index == len(qpos) - 1
        if terminal:
            draw.rounded_rectangle((24, 570, 871, 620), radius=10, fill=(255, 255, 255, 241))
            text_fitted(draw, (39, 584), result_text, 809, 21, result_color)
        elif mode.startswith("replay"):
            draw.rounded_rectangle((24, 574, 492, 615), radius=8, fill=(255, 255, 255, 229))
            draw.text((37, 586), "Same recorded motion · quarter speed", fill="#2C4C60", font=font(18))
        draw.text((27, 650), "Learned policy rollout", fill="white", font=font(20))
        draw.text((27, 682), "Teal: guidance route   ·   Amber: hand collision bounds", fill="#BED3DF", font=font(16))
        actual_status = "GOAL REACHED" if success else "GOAL NOT REACHED" if not outcome.get("reached_goal") else "GOAL REACHED WITH CONTACT"
        draw.text((773, 650), f"Final result: {actual_status}", fill="#DDE9EE", font=font(17))
        path_length = float(outcome.get("path_length", 0))
        progress = 100 * float(outcome.get("route_progress", 0))
        draw.text((773, 681), f"Path {path_length:.2f} m   ·   Route progress {progress:.1f}%", fill="#BED3DF", font=font(16))
        return Image.alpha_composite(image, layer).convert("RGB")

    def sampled_indices(speed):
        playback = np.arange(0, duration + 1e-9, speed / args.fps)
        result = np.searchsorted(elapsed, playback, side="left")
        result = np.clip(result, 0, len(elapsed) - 1)
        previous = np.maximum(result - 1, 0)
        result = np.where(np.abs(elapsed[previous] - playback) < np.abs(elapsed[result] - playback), previous, result)
        indices = [int(index) for index in result]
        if not indices or indices[-1] != len(qpos) - 1:
            indices.append(len(qpos) - 1)
        return indices

    schedule = [(0, "initial")] * args.fps
    schedule += [(index, "normal") for index in sampled_indices(1.0)]
    schedule += [(len(qpos) - 1, "terminal")] * (2 * args.fps)
    slow_replay = duration < 8 and len(qpos) > 1
    if slow_replay:
        schedule += [(0, "replay")] * args.fps
        schedule += [(index, "replay") for index in sampled_indices(.25)]
        schedule += [(len(qpos) - 1, "replay_terminal")] * (2 * args.fps)
    temporary = paths[".overnight-cat-checkpoint.partial.mp4"]
    movie = paths["overnight-cat-checkpoint.mp4"]
    process = subprocess.Popen([ffmpeg, "-n", "-loglevel", "error", "-f", "rawvideo",
        "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", str(args.fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)], stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, height=720, width=1280) as main_renderer, \
                mujoco.Renderer(model, height=230, width=356) as overview_renderer:
            for frame_number, (index, mode) in enumerate(schedule):
                image = composed_frame(index, mode, main_renderer, overview_renderer)
                process.stdin.write(image.tobytes())
                if frame_number % (args.fps * 5) == 0:
                    print(f"Rendered {frame_number}/{len(schedule)} video frames", flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg failed while encoding the checkpoint video")
            temporary.replace(movie)
            preview_index = max(0, min(len(qpos) - 1, round((len(qpos) - 1) * .65)))
            composed_frame(preview_index, "normal", main_renderer, overview_renderer).save(paths["preview.png"])
            sheet = Image.new("RGB", (1280, 768), "#E8EFF3")
            sheet_draw = ImageDraw.Draw(sheet)
            sheet_draw.text((20, 12), "Overnight CAT checkpoint · sampled recorded states", fill="#16324F", font=font(24))
            sheet_draw.text((20, 46), f"{checkpoint_text}  ·  Outcome: {reason}", fill=result_color, font=font(17))
            for position, index in enumerate(np.linspace(0, len(qpos) - 1, 6).round().astype(int)):
                panel = composed_frame(int(index), "normal", main_renderer, overview_renderer)
                # Six legible 416x234 panels in a 3-by-2 layout.
                panel = panel.resize((416, 234), Image.Resampling.LANCZOS)
                x, y = 8 + (position % 3) * 424, 89 + (position // 3) * 280
                sheet.paste(panel, (x, y))
                sheet_draw.text((x + 8, y + 241), f"t = {elapsed[index]:.2f} s", fill="#34566B", font=font(18))
            sheet.save(paths["contact-sheet.png"])
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    outputs = {name: {"path": str(paths[name]), "sha256": _sha256(paths[name])}
               for name in ("overnight-cat-checkpoint.mp4", "preview.png", "contact-sheet.png")}
    manifest = {"schema": "cat-checkpoint-rollout-render-v1", "outputs": outputs,
        "trajectory": str(trajectory), "trajectory_sha256": _sha256(trajectory),
        "metadata": str(metadata_path), "metadata_sha256": _sha256(metadata_path),
        "scene_id": scene["scene_id"], "geometry_hash": scene["geometry_hash"],
        "scene_json_sha256": _sha256(scene_dir / "scene.json"),
        "assembled_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "renderer_sha256": _sha256(Path(__file__)), "mujoco_version": mujoco.__version__,
        "checkpoint_label": metadata.get("checkpoint_label"), "global_step": transitions,
        "checkpoint_stage": metadata.get("checkpoint_stage"), "episode_seed": metadata.get("episode_seed"),
        "episode_outcome": outcome, "recorded_simulation_seconds": duration,
        "recorded_pose_count": len(qpos), "additional_simulation_steps": 0,
        "additional_policy_inference": False, "geometry_or_dynamics_modified": False,
        "pose_replay": "nearest recorded pose; mj_forward only; no integration or interpolation",
        "video": {"codec": "H.264", "width": 1280, "height": 720, "fps": args.fps,
                  "frames": len(schedule), "duration_seconds": len(schedule) / args.fps,
                  "initial_hold_seconds": 1, "terminal_hold_seconds": 2,
                  "quarter_speed_replay": slow_replay},
        "visual_only_changes": ["colors", "wall transparency", "hand bound highlighting",
                                "camera tracking", "room overview", "route/start/goal markers", "text overlays"],
        "elapsed_seconds": time.monotonic() - started}
    paths["render-manifest.json"].write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    render(parser().parse_args())
