#!/usr/bin/env python3
"""Videos of the planner-with-a-field-reflex policy from scripts/record_plan_reflex.py recordings.

Per recorded world: the robot as simulated (solid), the active whole-body plan 0.5 s ahead (see-through blue ghost),
the plan's pelvis path for the rest of the chunk (blue line), the plan's palm paths (thin lines), and a red marker on a
hand while it is inside the reflex band (the arm is then not tracked and the potential field moves it). One mp4 per
world (real time at 25 fps), a still at the moment of closest hand approach, and index.json.

    render_plan_reflex.py --rec rec.npz --out VIDEOS [--bank data/furniture/tablegap_v1]
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
from planner_render import add_sphere, label, scene_geometry  # noqa: E402
from planner_video import add_ghost, add_line  # noqa: E402

AHEAD = 0.5          # s, the ghost shows the plan this far ahead of the plan clock


def kind_of(scene_id):
    if scene_id.startswith("sidegap"):
        return "side gap"
    if scene_id.startswith("contrast"):
        return "cabinet passage"
    if scene_id.startswith("random"):
        return "cluttered room"
    if "table" in scene_id:
        return "table pair"
    return "hurdle / beam course"


def plan_pose(plan, t):
    """qpos at plan time t (s) from 21 poses 0.1 s apart: linear in position and joints, normalised quaternion."""
    u = min(max(t / 0.1, 0.), 20.)
    k = min(int(u), 19); a = u - k
    q = (1 - a) * plan[k] + a * plan[k + 1]
    if np.dot(plan[k, 3:7], plan[k + 1, 3:7]) < 0:
        q[3:7] = (1 - a) * plan[k, 3:7] - a * plan[k + 1, 3:7]
    q[3:7] /= max(np.linalg.norm(q[3:7]), 1e-9)
    return q


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rec", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bank", default="data/furniture/tablegap_v1")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--max-seconds", type=float, default=60.)
    args = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image, ImageFont
    from render_clutter_rollouts import resolve_xml_assets
    z = np.load(args.rec, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    bank_manifest = Path(args.bank + "_packed/manifest.json")
    manifest = json.loads(bank_manifest.read_text())
    by_id = {r["scene_id"]: r for r in manifest["scenes"]}
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    dt = float(meta["dt"]); stride = max(1, int(round(1 / (args.fps * dt))))
    index = []
    for w, scene_id in enumerate(meta["scenes"]):
        end = int(meta["end"][w]); outcome = meta["outcome"][w] or dict(result="running at the end", hand_contact=False)
        steps = list(range(0, min(end, int(args.max_seconds / dt)) + 1, stride))
        with tempfile.TemporaryDirectory() as tmp:
            scene_geometry(Path(tmp), bank_manifest, by_id[scene_id])
            xml, _ = resolve_xml_assets(Path(tmp) / "model.xml")
            model = mujoco.MjModel.from_xml_string(xml)
        model.vis.global_.offwidth, model.vis.global_.offheight = args.width, args.height
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == 0 and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE:
                model.geom_rgba[g, 3] = 0.55
        data, ghost = mujoco.MjData(model), mujoco.MjData(model)
        option = mujoco.MjvOption(); option.geomgroup[3:] = 0; option.sitegroup[:] = 0
        perturb, camera = mujoco.MjvPerturb(), mujoco.MjvCamera()
        palms = [model.site("left_palm").id, model.site("right_palm").id]
        q0 = z["qpos"][0, w]
        yaw0 = math.atan2(2 * (q0[3] * q0[6] + q0[4] * q0[5]), 1 - 2 * (q0[5] ** 2 + q0[6] ** 2))
        camera.distance, camera.elevation, camera.azimuth = 3.4, -28, math.degrees(yaw0) + 55
        look = q0[:2].copy()
        name = f"{w:02d}_{kind_of(scene_id).replace(' / ', '_').replace(' ', '_')}"
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", "27", "-movflags", "+faststart", str(args.out / f"{name}.mp4")]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        closest, still = 1e9, None
        reflex_steps = 0
        with mujoco.Renderer(model, width=args.width, height=args.height) as renderer:
            for t in steps:
                q = z["qpos"][t, w]
                data.qpos[:] = q; mujoco.mj_forward(model, data)
                look = 0.9 * look + 0.1 * q[:2]
                camera.lookat[:] = [look[0], look[1], 0.6]
                renderer.update_scene(data, camera=camera, scene_option=option)
                sc = renderer.scene
                active = bool(z["has_plan"][t, w])
                if active:
                    plan, pt = z["plan"][t, w], float(z["plan_t"][t, w])
                    add_ghost(sc, model, ghost, mujoco, option, perturb, plan_pose(plan, pt + AHEAD), (0.30, 0.55, 1.0, 0.28))
                    k0 = min(int(pt / 0.1), 20)
                    path = np.concatenate((plan[k0:, :2], np.full((21 - k0, 1), 0.02)), -1)
                    if len(path) > 1:
                        add_line(sc, mujoco, path, (0.20, 0.45, 1.0, 0.95), 0.012)
                released = z["released"][t, w]
                for h in range(2):
                    if released[h]:
                        add_sphere(sc, mujoco, data.site_xpos[palms[h]], (0.95, 0.15, 0.15, 0.85), 0.045)
                reflex_steps += int(released.any())
                frame = renderer.render().copy()
                clear = float(z["hand_clear"][t, w].min())
                status = ("reflex: " + " + ".join(s for s, r in zip(("left hand", "right hand"), released) if r)
                          if released.any() else ("following the plan" if active else "no plan"))
                frame = label(frame, [f"Planner + field reflex  |  {kind_of(scene_id)}  |  t = {t * dt:5.1f} s  |  told "
                                      f"{meta['speed']:.2f} m/s",
                                      f"{status}  |  closest hand clearance {100 * clear:5.1f} cm"], font)
                frame = label(frame, ["blue ghost: plan 0.5 s ahead  |  blue line: planned pelvis path  |  red: hand inside "
                                      f"the {100 * meta['band_m']:.0f} cm reflex band",
                                      f"episode: {outcome['result']}" + ("  (hand contact)" if outcome.get("hand_contact") else "")],
                              font, xy=(14, args.height - 70))
                proc.stdin.write(frame.tobytes())
                if clear < closest:
                    closest, still = clear, frame
        proc.stdin.close(); proc.wait()
        if still is not None:
            Image.fromarray(still).save(args.out / f"{name}.jpg", quality=88)
        index.append(dict(video=f"{name}.mp4", scene=scene_id, kind=kind_of(scene_id), seconds=round(steps[-1] * dt, 1),
                          reflex_share=round(reflex_steps / max(len(steps), 1), 3), closest_hand_clearance_m=round(closest, 4),
                          **outcome))
        print("VIDEO", name, outcome, flush=True)
    (args.out / "index.json").write_text(json.dumps(dict(meta={k: meta[k] for k in ("checkpoint", "mix", "speed")},
                                                         videos=index), indent=1) + "\n")


if __name__ == "__main__":
    main()
