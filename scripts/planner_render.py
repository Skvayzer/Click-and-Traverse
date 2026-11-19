#!/usr/bin/env python3
"""Videos and stills of planner plans among the scene's obstacles (kinematic replay, no dynamics).

Reads the plans.npz written by `planner_eval.py --save-plans K` and, per scene type, renders each saved situation as
two panels: left the chosen plan (best of the scene-conditioned samples), right the motion the policy itself executed
in that situation. The pelvis paths of all samples are drawn on the floor, green when the sample's whole body stays
clear of the obstacles (sphere cover, tolerance --tol) and red otherwise; the chosen path is yellow, the goal blue.

    python scripts/planner_render.py --plans EVAL/plans.npz --bank data/furniture/tablegap_v1 --out VIDEOS
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import cat_mjlab  # noqa: E402,F401  (path map)

SOURCES = {0: "u150", 1: "E2", 2: "V7", 3: "T2"}


def world_qpos(state, chunk, pose):
    """World qpos [21, 36]: the start pose and the 20 chunk frames, placed by the chunk frame pose (x, y, yaw)."""
    import torch
    from cat_mjlab.planner.data import quat_from_euler
    x0, y0, yaw0 = (float(v) for v in pose)
    c, s = math.cos(yaw0), math.sin(yaw0)
    st = torch.as_tensor(state, dtype=torch.float64)
    ch = torch.as_tensor(chunk, dtype=torch.float64)
    frames = [torch.cat((torch.zeros(2, dtype=torch.float64), st[58:59], torch.zeros(1, dtype=torch.float64),
                         st[59:61], st[:29]))]                      # start: x, y, z, dyaw, roll, pitch, joints
    frames += list(ch)
    f = torch.stack(frames)                                         # [21, 35] in the chunk frame
    wx = x0 + c * f[:, 0] - s * f[:, 1]
    wy = y0 + s * f[:, 0] + c * f[:, 1]
    quat = quat_from_euler(f[:, 4], f[:, 5], yaw0 + f[:, 3])
    return torch.cat((wx[:, None], wy[:, None], f[:, 2:3], quat, f[:, 6:]), -1).numpy()


def interpolate(q, n_out):
    """Resample [21, 36] keyframes (0.1 s apart) to n_out frames: positions and joints linear, quaternion nlerp."""
    t = np.linspace(0, len(q) - 1, n_out)
    i = np.clip(np.floor(t).astype(int), 0, len(q) - 2)
    a = (t - i)[:, None]
    out = (1 - a) * q[i] + a * q[i + 1]
    qa, qb = q[i, 3:7], q[i + 1, 3:7]
    qb = np.where((qa * qb).sum(-1, keepdims=True) < 0, -qb, qb)
    quat = (1 - a) * qa + a * qb
    out[:, 3:7] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    return out


def path_world(chunk, pose):
    x0, y0, yaw0 = (float(v) for v in pose)
    c, s = math.cos(yaw0), math.sin(yaw0)
    xy = np.concatenate((np.zeros((1, 2)), chunk[:, :2]))
    return np.stack((x0 + c * xy[:, 0] - s * xy[:, 1], y0 + s * xy[:, 0] + c * xy[:, 1]), -1)


def add_path(scene, mujoco, xy, rgba, width, z=0.012):
    for a, b in zip(xy[:-1], xy[1:]):
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(b - a) < 1e-4:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.array([a[0], a[1], z]), np.array([b[0], b[1], z]))
        scene.ngeom += 1


def add_sphere(scene, mujoco, pos, rgba, radius):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(pos, float), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))
    scene.ngeom += 1


def scene_geometry(directory, bank_manifest, rec):
    """model.xml with the scene: room boxes or a CAT occupancy mesh (record_mjlab_rollout.write_geometry); packed banks
    keep no CAT occupancy file, so those scenes get the zero surface of their signed distance field instead."""
    from record_mjlab_rollout import write_geometry
    try:
        write_geometry(directory, bank_manifest, rec)
        return
    except FileNotFoundError:
        pass
    import xml.etree.ElementTree as ET
    import trimesh
    from skimage import measure
    from cat_mjlab.model import assemble_training_xml
    from cat_ppo.furniture.generalist_fields import load_generalist_manifest, scene_directory
    mp = Path(bank_manifest).resolve()
    sdf = np.load(scene_directory(load_generalist_manifest(mp, verify_files=False), mp, rec) / "sdf.npy").astype(np.float32)
    v, f, n, _ = measure.marching_cubes(sdf, level=0.0, spacing=(float(rec["dx"]),) * 3)
    trimesh.Trimesh(vertices=v, faces=f, vertex_normals=n, process=True).export(directory / "obs.obj")
    xml = ET.fromstring(assemble_training_xml())
    ET.SubElement(xml.find("asset"), "mesh", name="recorded_cat_obstacle", file="obs.obj")
    ET.SubElement(xml.find("worldbody"), "geom", name="recorded_cat_obstacle", type="mesh", mesh="recorded_cat_obstacle",
                  pos=" ".join(map(str, rec["origin"])), contype="0", conaffinity="0", group="1", rgba=".37 .52 .61 1")
    (directory / "model.xml").write_text(ET.tostring(xml, encoding="unicode"))


def label(frame, lines, font, xy=(14, 10), fill=(255, 255, 255)):
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    x, y = xy
    for text in lines:
        box = d.textbbox((x, y), text, font=font)
        d.rectangle((box[0] - 6, box[1] - 3, box[2] + 6, box[3] + 3), fill=(0, 0, 0, 160))
        d.text((x, y), text, font=font, fill=fill)
        y = box[3] + 8
    return np.asarray(img)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--bank", default="data/furniture/tablegap_v1")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-type", type=int, default=6)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--slow", type=float, default=2.0, help="Playback slow-down (2 s of plan last slow x 2 s)")
    p.add_argument("--hold", type=float, default=0.6, help="Seconds to hold the first and last frame")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=450)
    p.add_argument("--tol", type=float, default=0.01)
    args = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image, ImageFont
    from render_clutter_rollouts import resolve_xml_assets
    z = np.load(args.plans, allow_pickle=False)
    names = [str(n) for n in z["kind_names"]]
    bank_manifest = Path(args.bank + "_packed/manifest.json")
    manifest = json.loads(bank_manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    n_frames = int(round(2.0 * args.slow * args.fps))
    hold = int(round(args.hold * args.fps))
    index = []
    for kind in np.unique(z["kinds"]):
        rows = np.nonzero(z["kinds"] == kind)[0][:args.per_type]
        name = names[int(kind)]
        video = args.out / f"{name}.mp4"
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{2 * args.width}x{args.height}", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", "30", "-movflags", "+faststart", str(video)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        stills = []
        for r in rows:
            sid = int(z["meta"][r, 0]); rec = manifest["scenes"][sid]
            with tempfile.TemporaryDirectory() as tmp:
                scene_geometry(Path(tmp), bank_manifest, rec)
                xml, _ = resolve_xml_assets(Path(tmp) / "model.xml")
                model = mujoco.MjModel.from_xml_string(xml)
            model.vis.global_.offwidth, model.vis.global_.offheight = 2 * args.width, args.height
            data = mujoco.MjData(model)
            option = mujoco.MjvOption(); option.geomgroup[3:] = 0; option.sitegroup[:] = 0
            camera = mujoco.MjvCamera()
            best = int(z["scene_best"][r])
            free = z["scene_min_clear"][r] >= -args.tol
            planned = world_qpos(z["state"][r], z["scene"][r, best], z["pose"][r])
            executed = world_qpos(z["state"][r], z["executed"][r], z["pose"][r])
            paths = [path_world(z["scene"][r, k], z["pose"][r]) for k in range(z["scene"].shape[1])]
            goal = path_world(z["goal"][r][None, :2], z["pose"][r])[-1]
            yaw0 = float(z["pose"][r][2])
            camera.distance, camera.elevation, camera.azimuth = 3.4, -38, math.degrees(yaw0) + 35
            source = SOURCES.get(int(z["meta"][r, 4]), str(int(z["meta"][r, 4])))
            info = (f"{name}  |  {rec['scene_id'][:44]}  |  data from {source}  |  "
                    f"samples clear {int(free.sum())}/{len(free)}")
            clear_p, clear_e = float(z["scene_min_clear"][r, best]), float(z["executed_min_clear"][r, 0])
            panels = (("plan (best of %d)" % len(free), planned, f"min clearance {100 * clear_p:+.1f} cm", True),
                      (f"executed by {source}", executed, f"min clearance {100 * clear_e:+.1f} cm", False))
            seq = {k: interpolate(q, n_frames) for k, (_, q, _, _) in enumerate(panels)}
            order = [0] * hold + list(range(n_frames)) + [n_frames - 1] * hold
            with mujoco.Renderer(model, width=args.width, height=args.height) as renderer:
                for fi, t in enumerate(order):
                    halves = []
                    for k, (title, _, clear, draw_paths) in enumerate(panels):
                        q = seq[k][t]
                        data.qpos[:] = q; mujoco.mj_forward(model, data)
                        camera.lookat[:] = [q[0], q[1], 0.55]
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        if draw_paths:
                            for kk, xy in enumerate(paths):
                                if kk != best:
                                    add_path(renderer.scene, mujoco, xy, (0.2, 0.75, 0.3, 0.8) if free[kk] else (0.9, 0.25, 0.2, 0.8), 0.008)
                            add_path(renderer.scene, mujoco, paths[best], (1.0, 0.85, 0.1, 1.0), 0.016, z=0.016)
                        else:
                            add_path(renderer.scene, mujoco, path_world(z["executed"][r], z["pose"][r]), (0.1, 0.8, 0.95, 1.0), 0.016, z=0.016)
                        add_sphere(renderer.scene, mujoco, [goal[0], goal[1], 0.03], (0.2, 0.45, 1.0, 1.0), 0.05)
                        img = renderer.render().copy()
                        img = label(img, [title, clear], font)
                        halves.append(img)
                    frame = np.concatenate(halves, 1)
                    frame = label(frame, [info], font, xy=(14, args.height - 34))
                    proc.stdin.write(frame.tobytes())
                    if fi == hold + n_frames // 2:
                        stills.append(frame)
            index.append(dict(kind=name, scene=rec["scene_id"], source=source, best=best, samples_clear=int(free.sum()),
                              samples=int(len(free)), plan_min_clear_m=round(clear_p, 4), executed_min_clear_m=round(clear_e, 4),
                              plan_goal_err_m=round(float(z["scene_goal_err"][r, best]), 3)))
        proc.stdin.close(); proc.wait()
        if stills:
            Image.fromarray(np.concatenate(stills[:4], 0)).save(args.out / f"{name}.jpg", quality=85)
        print("RENDERED", name, len(rows), flush=True)
    (args.out / "index.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
