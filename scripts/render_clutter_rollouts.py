"""Render saved clutter-policy trajectories, without simulation or inference.

Each input directory contains model.xml, scene.json, trajectory.npz (qpos,
qvel, time), and metadata.json. Furniture stays as its exact training boxes;
collision duplicates, field spheres, and perimeter walls are hidden for view.
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
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLORS = {
    "tabletop": "#c49b68", "table_leg": "#645449", "chair_seat": "#548994",
    "chair_back": "#447985", "chair_leg": "#394d56", "chair_armrest": "#3d6670",
    "wall": "#87929b", "crate": "#c09a71", "low_block": "#b36f5f",
    "partition": "#768cab", "shelf_edge": "#819b7a", "support": "#536650",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--fps", type=int, default=25)
    result.add_argument("--azimuth", type=float, default=None)
    result.add_argument("--distance", type=float, default=3.9)
    result.add_argument("--elevation", type=float, default=-27.)
    result.add_argument("--ffmpeg", type=Path)
    result.add_argument("--overwrite", action="store_true")
    return result


def resolve_xml_assets(model_path):
    """Relocate remote repository assets without modifying the recording."""
    xml = ET.parse(model_path).getroot()
    changes = []
    compiler = xml.find("compiler")
    directories = {name: (compiler.get(name, "") if compiler is not None else "")
                   for name in ("meshdir", "texturedir")}
    for element in xml.iter():
        name = element.get("file")
        if not name:
            continue
        original = Path(name)
        directory = directories["meshdir" if element.tag == "mesh" else "texturedir"]
        possibilities = [original, model_path.parent / original,
                         model_path.parent / directory / original]
        parts = original.parts
        for marker in ("data", "cat_ppo"):
            if marker in parts:
                possibilities.append(ROOT.joinpath(*parts[parts.index(marker):]))
        if "unitree_g1" in parts:
            possibilities.append(ROOT / "data/assets" / Path(*parts[parts.index("unitree_g1"):]))
        selected = next((path.resolve() for path in possibilities if path.is_file()), None)
        if selected is None:
            candidates = list((ROOT / "data/assets/unitree_g1").rglob(original.name))
            if len(candidates) == 1:
                selected = candidates[0].resolve()
        if selected is None:
            raise FileNotFoundError(f"Cannot resolve recorded model asset: {name}")
        element.set("file", str(selected))
        if str(selected) != name:
            changes.append({"recorded": name, "local": str(selected), "sha256": sha256(selected)})
    if compiler is not None:
        for attribute in directories:
            compiler.attrib.pop(attribute, None)
    return ET.tostring(xml, encoding="unicode"), changes


def outcome_label(outcome):
    if outcome.get("goal_reached"):
        return "GOAL REACHED", "Root and both feet reached the goal", True
    causes = []
    for key, label in (("fall", "Fall"), ("hand_violation", "Hand clearance violation"),
                       ("elbow_violation", "Elbow clearance violation"),
                       ("self_contact", "Self contact"), ("numerical", "Numerical failure")):
        if outcome.get(key):
            causes.append(label)
    if outcome.get("obstacle") and not (outcome.get("hand_violation") or outcome.get("elbow_violation")):
        causes.append("Body clearance violation")
    if outcome.get("outside_bounds"):
        causes.append("Left room bounds")
    if outcome.get("timeout"):
        causes.append("Time limit")
    fallback = str(outcome.get("termination_reason", outcome.get("reason", "Goal not reached"))).replace("_", " ")
    return "GOAL NOT REACHED", " + ".join(causes) or fallback, False


def render(args):
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    started = time.monotonic()
    directory = args.input_dir.resolve()
    inputs = {name: directory / name for name in ("model.xml", "scene.json", "trajectory.npz", "metadata.json")}
    scene = json.loads(inputs["scene.json"].read_text())
    metadata = json.loads(inputs["metadata.json"].read_text())
    with np.load(inputs["trajectory.npz"], allow_pickle=False) as data:
        qpos, qvel, timestamps = (np.asarray(data[key], dtype=np.float64) for key in ("qpos", "qvel", "time"))
    if (qpos.ndim != 2 or not len(qpos) or qvel.ndim != 2 or len(qvel) != len(qpos)
            or timestamps.shape != (len(qpos),)):
        raise ValueError("Expected qpos[T,nq], qvel[T,nv], and time[T]")
    if not all(np.isfinite(value).all() for value in (qpos, qvel, timestamps)):
        raise ValueError("Cannot render nonfinite recorded poses")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("Replay requires increasing timestamps without episode resets")
    if not 12 <= args.fps <= 60 or not 1.5 <= args.distance <= 15:
        raise ValueError("Require 12–60fps and camera distance 1.5–15m")
    elapsed = timestamps - timestamps[0]
    duration = float(elapsed[-1])
    output = args.output.absolute()
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must be an .mp4")
    outputs = {"video": output, "poster": output.with_suffix(".poster.png"),
               "contact_sheet": output.with_suffix(".contact-sheet.png"),
               "provenance": output.with_suffix(".render.json"),
               "partial": output.with_name(f".{output.stem}.partial.mp4")}
    for path in outputs.values():
        if path.is_symlink() or (path.exists() and not args.overwrite):
            raise FileExistsError(f"Output exists; use --overwrite if intended: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = str(args.ffmpeg) if args.ffmpeg else shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    xml, asset_changes = resolve_xml_assets(inputs["model.xml"])
    model = mujoco.MjModel.from_xml_string(xml)
    if qpos.shape[1] != model.nq or qvel.shape[1] != model.nv:
        raise ValueError("Recorded qpos/qvel dimensions differ from the model")
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Expected exactly one robot free-root joint")
    root_address = int(model.jnt_qposadr[free[0]])
    roots = qpos[:, root_address:root_address + 3]
    qw, qx, qy, qz = qpos[0, root_address + 3:root_address + 7]
    heading = math.degrees(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    camera = mujoco.MjvCamera()
    camera.azimuth = heading - 55 if args.azimuth is None else args.azimuth
    camera.elevation, camera.distance = args.elevation, args.distance
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    model.vis.global_.fovy = 43
    model.vis.quality.shadowsize = 2048
    model.vis.headlight.ambient[:] = [.52] * 3
    model.vis.headlight.diffuse[:] = [.62] * 3
    model.vis.headlight.specular[:] = [.08] * 3
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor >= 0:
        model.geom_matid[floor] = -1
        model.geom_rgba[floor] = [.91, .93, .94, 1.]
    # Hiding only display groups keeps all furniture dimensions and recorded
    # motion intact. No geometry is moved or resized and no mj_step is called.
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[:3] = 1
    options.sitegroup[:] = 0
    data = mujoco.MjData(model)

    fonts = {}
    font_paths = [Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    font_path = next((path for path in font_paths if path.is_file()), None)
    def font(size):
        if size not in fonts:
            fonts[size] = ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default(size=size)
        return fonts[size]
    def text(draw, xy, value, max_width, size=20, fill="#173448"):
        value = str(value)
        while size > 11 and draw.textlength(value, font=font(size)) > max_width:
            size -= 1
        draw.text(xy, value, font=font(size), fill=fill)

    # A vector overhead map makes the complete room and traveled path visible
    # while the 3-D camera stays close enough to inspect the hands and balance.
    map_size = (322, 310)
    plan = Image.new("RGB", map_size, "#FCFCFA")
    plan_draw = ImageDraw.Draw(plan)
    room_x, room_y = scene["room_dimensions"][:2]
    map_scale = min((map_size[0] - 26) / room_x, (map_size[1] - 24) / room_y)
    xpad, ypad = (map_size[0] - room_x * map_scale) / 2, (map_size[1] - room_y * map_scale) / 2
    def pixel(xy):
        return (xpad + float(xy[0]) * map_scale, map_size[1] - ypad - float(xy[1]) * map_scale)
    for box in sorted(scene["boxes"], key=lambda item: item["center"][2]):
        x, y = box["center"][:2]
        hx, hy = box["half_size"][:2]
        c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
        points = [pixel((x + c * dx * hx - s * dy * hy, y + s * dx * hx + c * dy * hy))
                  for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        plan_draw.polygon(points, fill=COLORS.get(box["category"], "#83939a"), outline="#566875")

    scene_seed = metadata.get("scene_seed", scene.get("seed", "?"))
    family = metadata.get("family", scene.get("family", "clutter"))
    title = "Furniture traversal" if family == "furniture" else "Mixed clutter traversal"
    seed = metadata.get("seed", "?")
    steps = metadata.get("checkpoint_steps")
    checkpoint = f"Checkpoint {int(steps):,} steps" if steps is not None else "Recorded checkpoint"
    outcome = metadata.get("outcome", {})
    status, reason, success = outcome_label(outcome)
    status_color = "#127960" if success else "#B34332"
    counts = scene.get("counts", {})
    objects = (f"{counts.get('tables', '?')} tables  /  {counts.get('chairs', '?')} chairs"
               if family == "furniture" else f"{counts.get('generic_objects', '?')} clutter objects")
    last_index, last_base = None, None

    def frame(index, hold, renderer):
        nonlocal last_index, last_base
        if index == last_index:
            image = last_base.copy()
        else:
            data.qpos[:] = qpos[index]
            data.qvel[:] = qvel[index]
            data.time = timestamps[index]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [roots[index, 0], roots[index, 1], .72]
            renderer.update_scene(data, camera=camera, scene_option=options)
            image = Image.fromarray(renderer.render()).convert("RGB")
            last_index, last_base = index, image.copy()
        overlay = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, 1280, 97), fill=(249, 252, 253, 252))
        text(draw, (23, 12), f"CAT whole-body  |  {title}", 1234, 28)
        text(draw, (24, 53), f"Scene {scene_seed}  ·  rollout seed {seed}  ·  {checkpoint}", 1230, 20, "#476274")
        draw.rounded_rectangle((20, 115, 415, 154), radius=8, fill=(252, 253, 253, 239))
        text(draw, (33, 125), f"{elapsed[index]:05.2f} / {duration:.2f} s    |    {'FINAL POSE' if hold else '1× REAL TIME'}", 370, 18)
        draw.rounded_rectangle((915, 112, 1265, 523), radius=12,
                               fill=(250, 252, 253, 249), outline=(155, 174, 183, 255), width=1)
        text(draw, (931, 124), "ROOM MAP  /  TRAVELED PATH", 318, 16)
        current_plan = plan.copy()
        map_draw = ImageDraw.Draw(current_plan)
        path_indices = list(range(0, index + 1, max(1, index // 800)))
        if not path_indices or path_indices[-1] != index:
            path_indices.append(index)
        traveled = [pixel(roots[position, :2]) for position in path_indices]
        if len(traveled) > 1:
            map_draw.line(traveled, fill="#0C9A75", width=3, joint="curve")
        for label, location, fill in (("A", scene["start"][:2], "#2F70BF"),
                                      ("B", scene["goal"][:2], "#E07832")):
            px, py = pixel(location)
            map_draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=fill, outline="white", width=1)
            map_draw.text((px, py), label, font=font(11), anchor="mm", fill="white")
        qw, qx, qy, qz = qpos[index, root_address + 3:root_address + 7]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        cx, cy = pixel(roots[index, :2])
        triangle = [(cx + math.cos(yaw + angle) * radius, cy - math.sin(yaw + angle) * radius)
                    for angle, radius in ((0., 8.), (2.5, 6.), (-2.5, 6.))]
        map_draw.polygon(triangle, fill="#EBDE56", outline="#183546", width=2)
        overlay.paste(current_plan.convert("RGBA"), (929, 153))
        text(draw, (931, 473), f"{room_x:.1f} × {room_y:.1f} m  ·  {objects}", 318, 15)
        text(draw, (931, 499), "A start    B goal    ▲ robot", 318, 15, "#476274")
        draw.rectangle((0, 626, 1280, 720), fill=(245, 249, 251, 253))
        text(draw, (24, 638), f"Recorded outcome: {status}", 1210, 24, status_color)
        text(draw, (25, 673), reason, 1210, 18, "#354F60")
        draw.rectangle((0, 708, 1280, 720), fill=(199, 213, 220, 255))
        if duration:
            draw.rectangle((0, 708, int(1280 * elapsed[index] / duration), 720), fill=(40, 135, 119, 255))
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    playback_times = np.arange(0., duration + 1e-9, 1 / args.fps)
    indices = np.clip(np.searchsorted(elapsed, playback_times), 0, len(elapsed) - 1)
    previous = np.maximum(indices - 1, 0)
    indices = np.where(np.abs(elapsed[previous] - playback_times) < np.abs(elapsed[indices] - playback_times), previous, indices)
    schedule = [(int(index), False) for index in indices]
    if not schedule or schedule[-1][0] != len(qpos) - 1:
        schedule.append((len(qpos) - 1, False))
    schedule.extend([(len(qpos) - 1, True)] * (2 * args.fps))
    command = [ffmpeg, "-y" if args.overwrite else "-n", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", "1280x720", "-r", str(args.fps), "-i", "-", "-an",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(outputs["partial"])]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, height=720, width=1280) as renderer:
            for number, (index, hold) in enumerate(schedule):
                process.stdin.write(frame(index, hold, renderer).tobytes())
                if number % (args.fps * 5) == 0:
                    print(json.dumps({"video": output.name, "frames": number, "total": len(schedule)}), flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg failed")
            outputs["partial"].replace(output)
            poster_index = int(np.argmin(np.abs(elapsed - duration * .6)))
            frame(poster_index, False, renderer).save(outputs["poster"])
            sheet = Image.new("RGB", (1280, 800), "#EAF0F4")
            sheet_draw = ImageDraw.Draw(sheet)
            text(sheet_draw, (16, 12), f"{title}  |  Scene {scene_seed}  |  seed {seed}  |  {status}", 1250, 24, status_color)
            for position, sample in enumerate(np.linspace(0, duration, 4)):
                index = int(np.argmin(np.abs(elapsed - sample)))
                panel = frame(index, False, renderer)
                panel.thumbnail((624, 351), Image.Resampling.LANCZOS)
                x, y = 8 + position % 2 * 640, 54 + position // 2 * 373
                sheet.paste(panel, (x, y))
                text(sheet_draw, (x + 8, y + 351), f"t = {elapsed[index]:.2f} s", 610, 14)
            sheet.save(outputs["contact_sheet"])
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    provenance = {
        "schema": "cat-clutter-rollout-render-v1", "input_directory": str(directory),
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()},
        "outputs": {name: {"path": str(outputs[name]), "sha256": sha256(outputs[name])}
                    for name in ("video", "poster", "contact_sheet")},
        "metadata": metadata, "mujoco_version": mujoco.__version__, "renderer_sha256": sha256(Path(__file__)),
        "asset_path_relocations": asset_changes, "recorded_pose_count": len(qpos),
        "recorded_duration_seconds": duration, "additional_simulation_steps": 0,
        "additional_policy_inference": False, "pose_sampling": "nearest saved pose; mj_forward only; no interpolation",
        "video": {"width": 1280, "height": 720, "fps": args.fps, "frames": len(schedule),
                  "duration_seconds": len(schedule) / args.fps, "terminal_hold_seconds": 2., "playback_speed": 1.},
        "camera": {"azimuth": float(camera.azimuth), "elevation": float(camera.elevation),
                   "distance": float(camera.distance), "target_height": .72},
        "visual_changes": ["camera", "lighting", "floor color", "hidden geom groups3+ (collision duplicates, spheres, walls)",
                           "vector overhead map and recorded root path", "text overlays"],
        "furniture": "opaque canonical training boxes; original sizes and positions; no substituted meshes",
        "wall_seconds": time.monotonic() - started,
    }
    outputs["provenance"].write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"video": str(output), "seconds": len(schedule) / args.fps,
                      "wall_seconds": provenance["wall_seconds"]}), flush=True)
    return provenance


if __name__ == "__main__":
    render(parser().parse_args())
