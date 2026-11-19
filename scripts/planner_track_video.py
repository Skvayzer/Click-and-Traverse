#!/usr/bin/env python3
"""Videos of the full-body tracker executing planner plans (from planner_track_eval.py's npz).

Top: the robot as the tracker moved it in physics (solid) and the plan at the same moment (see-through blue), so the
gap between them is the tracking error; coloured paths are the plan's palms, feet and head, grey paths where the tracker
took them. Bottom: all 29 joint angles, plan solid and tracker dashed. Per scene type, clean executions first and one
that touched or fell. The executed motion is replayed from the 10 Hz frames the evaluation stored.

    python scripts/planner_track_video.py --eval EVAL/track_eval.npz --out VIDEOS
    python scripts/planner_track_video.py --eval EVAL/track_eval_probe.npz --out VIDEOS     # with the inserted column
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
import cat_mjlab  # noqa: E402,F401
from planner_render import SOURCES, add_sphere, interpolate, label, path_world, scene_geometry, world_qpos  # noqa: E402
from planner_video import TRAILS, add_ghost, add_line, cursor, joint_panel, site_paths  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--bank", default="data/furniture/tablegap_v1")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-type", type=int, default=3)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--slow", type=float, default=2.0)
    p.add_argument("--hold", type=float, default=0.8)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--view-height", type=int, default=470)
    p.add_argument("--panel-height", type=int, default=250)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--obstacle-alpha", type=float, default=0.45)
    p.add_argument("--radius", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image, ImageFont
    from render_clutter_rollouts import resolve_xml_assets
    from cat_mjlab.runner import SCENE_BUCKETS
    z = dict(np.load(args.eval, allow_pickle=False))
    n = len(z["plan"])
    tracked, alive, clear = z["tracked"][:n], z["alive"][:n], z["min_clear"][:n]
    column = z.get("column")
    probe = column is not None
    usable = (z["exec_clear"] < -args.tol) if probe else np.ones(n, bool)       # probes: the column blocks the policy
    ok = (clear >= -args.tol) & alive
    bank_manifest = Path(args.bank + "_packed/manifest.json")
    manifest = json.loads(bank_manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    rng = np.random.default_rng(args.seed)
    n_frames = int(round(2.0 * args.slow * args.fps)); hold = int(round(args.hold * args.fps))
    order = [0] * hold + list(range(n_frames)) + [n_frames - 1] * hold
    W, H = args.width, args.view_height + args.panel_height
    index = []
    kinds = z["meta"][:, 1]
    for kind in np.unique(kinds):
        good = rng.permutation(np.nonzero((kinds == kind) & usable & ok)[0])
        bad = rng.permutation(np.nonzero((kinds == kind) & usable & ~ok)[0])
        n_bad = min(len(bad), 1 if len(good) else args.per_type)
        rows = np.concatenate((good[:args.per_type - n_bad], bad[:n_bad])).astype(int)
        if not len(rows):
            continue
        name = SCENE_BUCKETS[int(kind)]; stem = name + ("_track_probe" if probe else "_track")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30",
               "-movflags", "+faststart", str(args.out / f"{stem}.mp4")]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        stills = []
        for r in rows:
            rec = manifest["scenes"][int(z["meta"][r, 0])]
            with tempfile.TemporaryDirectory() as tmp:
                scene_geometry(Path(tmp), bank_manifest, rec)
                xml, _ = resolve_xml_assets(Path(tmp) / "model.xml")
                model = mujoco.MjModel.from_xml_string(xml)
            model.vis.global_.offwidth, model.vis.global_.offheight = W, H
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == 0 and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE:
                    model.geom_rgba[g, 3] = args.obstacle_alpha
            data, ghost_data = mujoco.MjData(model), mujoco.MjData(model)
            option = mujoco.MjvOption(); option.geomgroup[3:] = 0; option.sitegroup[:] = 0
            perturb, camera = mujoco.MjvPerturb(), mujoco.MjvCamera()
            plan_key = world_qpos(z["state"][r], z["plan"][r], z["pose"][r])
            trk_key = world_qpos(z["state"][r], tracked[r], z["pose"][r])
            plan_seq, trk_seq = interpolate(plan_key, n_frames), interpolate(trk_key, n_frames)
            names = [nm for nm, _ in TRAILS]
            plan_trails = site_paths(model, data, mujoco, interpolate(plan_key, 81), names)
            trk_trails = site_paths(model, data, mujoco, interpolate(trk_key, 81), names)
            panel, boxes = joint_panel(plan_key[:, 7:], trk_key[:, 7:], W, args.panel_height,
                                       title="All 29 joint angles (rad): the plan solid, the tracker's execution dashed")
            goal = path_world(z["goal"][r][None, :2], z["pose"][r])[-1]
            mid = plan_key[:, :2].mean(0); yaw0 = float(z["pose"][r][2])
            camera.lookat[:] = [mid[0], mid[1], 0.55]
            camera.distance, camera.elevation, camera.azimuth = 3.0, -36, math.degrees(yaw0) + 40
            col_w = path_world(column[r][None], z["pose"][r])[-1] if probe else None
            source = SOURCES.get(int(z["meta"][r, 4]), str(int(z["meta"][r, 4])))
            result = "clean" if ok[r] else ("fell" if not alive[r] else "touched")
            head = "TRACKER executing the plan in physics (solid) | the plan at the same moment (see-through blue)" + \
                   ("  |  column added" if probe else "")
            sub = "coloured paths: the plan's palms, feet, head  |  grey paths: where the tracker took them"
            info = (f"{name}  |  executed min clearance {100 * clear[r]:+.1f} cm (plan {100 * z['plan_clear'][r]:+.1f} cm)  |  "
                    f"result: {result}  |  data from {source}")
            with mujoco.Renderer(model, width=W, height=args.view_height) as renderer:
                for fi, ti in enumerate(order):
                    data.qpos[:] = trk_seq[ti]; mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    sc = renderer.scene
                    add_ghost(sc, model, ghost_data, mujoco, option, perturb, plan_seq[ti], (0.30, 0.60, 1.0, 0.30))
                    for j, (_, rgba) in enumerate(TRAILS):
                        add_line(sc, mujoco, trk_trails[:, j], (0.40, 0.40, 0.40, 0.95), 0.007)
                        add_line(sc, mujoco, plan_trails[:, j], (*rgba, 1.0), 0.011)
                    add_sphere(sc, mujoco, [goal[0], goal[1], 0.03], (0.2, 0.45, 1.0, 1.0), 0.05)
                    if col_w is not None and sc.ngeom < sc.maxgeom:
                        gc = sc.geoms[sc.ngeom]
                        mujoco.mjv_initGeom(gc, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([args.radius, 0.9, 0]),
                                            np.array([col_w[0], col_w[1], 0.9]), np.eye(3).ravel(), np.array([0.85, 0.15, 0.15, 0.8], np.float32))
                        sc.ngeom += 1
                    view = renderer.render().copy()
                    t_s = 2.0 * ti / (n_frames - 1)
                    err = float(np.linalg.norm(trk_seq[ti, :2] - plan_seq[ti, :2]))
                    view = label(view, [head, sub], font)
                    view = label(view, [f"t = {t_s:.1f} s of 2.0 s   pelvis gap to the plan {100 * err:.1f} cm", info], font,
                                 xy=(14, args.view_height - 62))
                    frame = np.concatenate((view, cursor(panel, boxes, ti / (n_frames - 1))), 0)
                    proc.stdin.write(frame.tobytes())
                    if fi == hold + n_frames // 2:
                        stills.append(frame)
            index.append(dict(kind=name, scene=rec["scene_id"], source=source, result=result,
                              executed_min_clear_m=round(float(clear[r]), 4), plan_min_clear_m=round(float(z["plan_clear"][r]), 4)))
        proc.stdin.close(); proc.wait()
        Image.fromarray(stills[0]).save(args.out / f"{stem}.jpg", quality=88)
        print("VIDEO", stem, len(rows), flush=True)
    (args.out / ("index_track_probe.json" if probe else "index_track.json")).write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
