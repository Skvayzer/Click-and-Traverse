#!/usr/bin/env python3
"""Videos of the planner's 29-DoF plans: the whole 2 s plan in one view, plus every joint angle over time.

Top: the scene with the chosen plan (best of the scene-conditioned samples). Translucent copies of the planned body every
--ghost-every s show the whole plan at once; the solid robot moves through them. Trails show where the plan takes the
palms, feet and head; grey trails are the same points in the policy's own motion from the same start.
Bottom: all 29 joint angles of the plan (solid) and of the policy's own motion (dashed), with a cursor at the shown time.

    python scripts/planner_video.py --plans EVAL/plans.npz --out VIDEOS                       # held-out situations
    python scripts/planner_video.py --plans EVAL/plans.npz --probe PROBE/probe.npz --out VIDEOS # inserted-column probes
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
from planner_render import SOURCES, add_path, add_sphere, interpolate, label, path_world, scene_geometry, world_qpos  # noqa: E402

GROUPS = (("Left leg", 0, ("hip pitch", "hip roll", "hip yaw", "knee", "ankle pitch", "ankle roll")),
          ("Right leg", 6, ("hip pitch", "hip roll", "hip yaw", "knee", "ankle pitch", "ankle roll")),
          ("Waist", 12, ("yaw", "roll", "pitch")),
          ("Left arm", 15, ("shoulder pitch", "shoulder roll", "shoulder yaw", "elbow", "wrist roll", "wrist pitch", "wrist yaw")),
          ("Right arm", 22, ("shoulder pitch", "shoulder roll", "shoulder yaw", "elbow", "wrist roll", "wrist pitch", "wrist yaw")))
TRAILS = (("left_palm", (0.86, 0.22, 0.74)), ("right_palm", (0.55, 0.32, 0.92)), ("left_foot", (0.20, 0.70, 0.25)),
          ("right_foot", (0.05, 0.45, 0.15)), ("head", (0.0, 0.62, 0.62)))
PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b")


def joint_panel(plan_q, exec_q, width, height):
    """RGB image of the 29 joint angles over 2 s, and per subplot the pixel box of its data area (for the cursor)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.linspace(0, 2.0, len(plan_q))
    dpi = 100
    fig, axes = plt.subplots(1, 5, figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.80, bottom=0.30, wspace=0.28)
    for ax, (title, start, names) in zip(axes, GROUPS):
        for j, name in enumerate(names):
            c = PALETTE[j % len(PALETTE)]
            ax.plot(t, plan_q[:, start + j], color=c, lw=1.6, label=name)
            ax.plot(t, exec_q[:, start + j], color=c, lw=0.9, ls=(0, (3, 2)), alpha=0.8)
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_xlim(0, 2.0); ax.tick_params(labelsize=7, length=2, pad=1)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0]); ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=6.3, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False,
                  handlelength=1.2, columnspacing=0.8, labelspacing=0.15)
    axes[0].set_ylabel("rad", fontsize=7, labelpad=1)
    fig.text(0.005, 0.985, "All 29 joint angles (rad) over the 2 s plan: plan solid, the policy's own motion dashed", fontsize=8.5, weight="bold", va="top")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    boxes = []
    for ax in axes:
        (x0, y0), (x1, y1) = ax.transData.transform([(0, ax.get_ylim()[0]), (2.0, ax.get_ylim()[1])])
        boxes.append((x0, img.shape[0] - y1, x1, img.shape[0] - y0))       # left, top, right, bottom in image pixels
    plt.close(fig)
    return img, boxes


def cursor(panel, boxes, frac):
    out = panel.copy()
    for left, top, right, bottom in boxes:
        x = int(round(left + frac * (right - left)))
        out[int(top):int(bottom), max(x - 1, 0):x + 1] = (40, 40, 40)
    return out


def site_paths(model, data, mujoco, q_seq, names):
    ids = [model.site(n).id for n in names]
    out = np.zeros((len(q_seq), len(ids), 3))
    for i, q in enumerate(q_seq):
        data.qpos[:] = q; mujoco.mj_kinematics(model, data)
        out[i] = data.site_xpos[ids]
    return out


def add_ghost(scene, model, ghost_data, mujoco, option, perturb, q, rgba):
    ghost_data.qpos[:] = q; mujoco.mj_kinematics(model, ghost_data)
    before = scene.ngeom
    mujoco.mjv_addGeoms(model, ghost_data, option, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
    for g in range(before, scene.ngeom):
        scene.geoms[g].rgba[:] = rgba
        scene.geoms[g].matid = -1


def add_line(scene, mujoco, pts, rgba, width):
    for a, b in zip(pts[:-1], pts[1:]):
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(b - a) < 1e-4:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scene.geoms[scene.ngeom].matid = -1
        scene.ngeom += 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--probe", type=Path)
    p.add_argument("--probe-key", default="scene_g2")
    p.add_argument("--probe-radius", type=float, default=0.10)
    p.add_argument("--bank", default="data/furniture/tablegap_v1")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-type", type=int, default=3)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--slow", type=float, default=2.0)
    p.add_argument("--hold", type=float, default=0.8)
    p.add_argument("--ghost-times", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--strip-height", type=int, default=200)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--view-height", type=int, default=440)
    p.add_argument("--panel-height", type=int, default=250)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--obstacle-alpha", type=float, default=0.45, help="obstacle opacity, so the planned body shows behind them")
    args = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image, ImageFont
    from render_clutter_rollouts import resolve_xml_assets
    z = dict(np.load(args.plans, allow_pickle=False))
    names = [str(n) for n in z["kind_names"]]
    column = None
    if args.probe:
        pr = dict(np.load(args.probe, allow_pickle=False)); key = args.probe_key
        z = dict(kinds=pr["kinds"], meta=pr["meta"], pose=pr["pose"], state=pr["state"], goal=pr["goal"], executed=pr["executed"],
                 scene=pr[f"{key}_chunks"], scene_best=pr[f"{key}_best"], scene_min_clear=pr[f"{key}_clear"],
                 executed_min_clear=pr["executed_clear"][:, None])
        column, cut = pr["column"], pr["cut"]
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
    n_frames = int(round(2.0 * args.slow * args.fps)); hold = int(round(args.hold * args.fps))
    order = [0] * hold + list(range(n_frames)) + [n_frames - 1] * hold
    W, H = args.width, args.view_height + args.strip_height + args.panel_height
    STRIP_T = (0.0, 0.5, 1.0, 1.5, 2.0)
    GHOST_RGBA = ((0.30, 0.60, 1.0, 0.22), (1.0, 0.55, 0.10, 0.22))
    index = []
    for kind in np.unique(z["kinds"]):
        rows = np.nonzero(z["kinds"] == kind)[0]
        if column is not None:
            rows = rows[cut[rows]]
            ok = z["scene_min_clear"][rows, z["scene_best"][rows]] >= -args.tol
            good, bad = rows[ok], rows[~ok]
            n_bad = min(len(bad), 1 if len(good) else args.per_type)
            rows = np.concatenate((good[:args.per_type - n_bad], bad[:n_bad]))
        else:
            rows = rows[:args.per_type]
        if not len(rows):
            continue
        name = names[int(kind)]; stem = name + ("_probe" if column is not None else "")
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
            for g in range(model.ngeom):           # scene obstacles live on the world body; the floor stays opaque
                if model.geom_bodyid[g] == 0 and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE:
                    model.geom_rgba[g, 3] = args.obstacle_alpha
            data, ghost_data = mujoco.MjData(model), mujoco.MjData(model)
            option = mujoco.MjvOption(); option.geomgroup[3:] = 0; option.sitegroup[:] = 0
            perturb, camera = mujoco.MjvPerturb(), mujoco.MjvCamera()
            best = int(z["scene_best"][r]); free = z["scene_min_clear"][r] >= -args.tol
            plan_key = world_qpos(z["state"][r], z["scene"][r, best], z["pose"][r])          # [21, 36] keyframes
            exec_key = world_qpos(z["state"][r], z["executed"][r], z["pose"][r])
            plan_seq, exec_seq = interpolate(plan_key, n_frames), interpolate(exec_key, n_frames)
            trail_names = [n for n, _ in TRAILS]
            plan_trails = site_paths(model, data, mujoco, interpolate(plan_key, 81), trail_names)
            exec_trails = site_paths(model, data, mujoco, interpolate(exec_key, 81), trail_names)
            ghost_frames = [int(round(t / 0.1)) for t in args.ghost_times]
            panel, boxes = joint_panel(plan_key[:, 7:], exec_key[:, 7:], W, args.panel_height)
            goal = path_world(z["goal"][r][None, :2], z["pose"][r])[-1]
            mid = plan_key[:, :2].mean(0); yaw0 = float(z["pose"][r][2])
            camera.lookat[:] = [mid[0], mid[1], 0.55]
            camera.distance, camera.elevation, camera.azimuth = 3.0, -36, math.degrees(yaw0) + 40
            source = SOURCES.get(int(z["meta"][r, 4]), str(int(z["meta"][r, 4])))
            clear_p, clear_e = float(z["scene_min_clear"][r, best]), float(z["executed_min_clear"][r, 0])
            col_w = path_world(column[r][None], z["pose"][r])[-1] if column is not None else None
            head = ("PLAN with a column added to its map" if column is not None else "PLAN") + \
                   ": pelvis + 29 joints over 2 s, from the planner (replayed, no physics)"
            sub = ("see-through bodies: blue at +1 s, orange at +2 s  |  coloured trails: palms, feet, head  |  "
                   f"grey trails: {source}'s own motion")
            right = (f"plan min clearance {100 * clear_p:+.1f} cm   {source}'s motion {100 * clear_e:+.1f} cm   "
                     f"samples clear {int(free.sum())}/{len(free)}")
            sw = W // len(STRIP_T)
            strip = np.full((args.strip_height, W, 3), 245, np.uint8)
            with mujoco.Renderer(model, width=sw, height=args.strip_height) as small:
                cam_s = mujoco.MjvCamera()
                cam_s.distance, cam_s.elevation, cam_s.azimuth = 2.3, -18, math.degrees(yaw0) + 40
                for k, t_k in enumerate(STRIP_T):
                    q = plan_key[int(round(t_k / 0.1))]
                    data.qpos[:] = q; mujoco.mj_forward(model, data)
                    cam_s.lookat[:] = [q[0], q[1], 0.62]
                    small.update_scene(data, camera=cam_s, scene_option=option)
                    if col_w is not None and small.scene.ngeom < small.scene.maxgeom:
                        g = small.scene.geoms[small.scene.ngeom]
                        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([args.probe_radius, 0.9, 0]),
                                            np.array([col_w[0], col_w[1], 0.9]), np.eye(3).ravel(), np.array([0.85, 0.15, 0.15, 0.8], np.float32))
                        small.scene.ngeom += 1
                    tile = label(small.render().copy(), [f"plan at {t_k:.1f} s"], font, xy=(8, 6))
                    strip[:, k * sw:(k + 1) * sw] = tile
            with mujoco.Renderer(model, width=W, height=args.view_height) as renderer:
                for fi, ti in enumerate(order):
                    data.qpos[:] = plan_seq[ti]; mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    sc = renderer.scene
                    for k, fr in enumerate(ghost_frames):
                        add_ghost(sc, model, ghost_data, mujoco, option, perturb, plan_key[fr], GHOST_RGBA[k % len(GHOST_RGBA)])
                    for j, (_, rgba) in enumerate(TRAILS):
                        add_line(sc, mujoco, exec_trails[:, j], (0.45, 0.45, 0.45, 0.9), 0.006)
                        add_line(sc, mujoco, plan_trails[:, j], (*rgba, 1.0), 0.011)
                    add_sphere(sc, mujoco, [goal[0], goal[1], 0.03], (0.2, 0.45, 1.0, 1.0), 0.05)
                    if col_w is not None and sc.ngeom < sc.maxgeom:
                        g = sc.geoms[sc.ngeom]
                        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([args.probe_radius, 0.9, 0]),
                                            np.array([col_w[0], col_w[1], 0.9]), np.eye(3).ravel(), np.array([0.85, 0.15, 0.15, 0.8], np.float32))
                        sc.ngeom += 1
                    view = renderer.render().copy()
                    t_s = 2.0 * ti / (n_frames - 1)
                    view = label(view, [head, sub], font)
                    view = label(view, [f"t = {t_s:.1f} s of 2.0 s", right], font, xy=(14, args.view_height - 62))
                    st = strip.copy()
                    cur = int(round(t_s / 0.5)); x0 = cur * sw
                    st[:4, x0:x0 + sw] = (230, 180, 20); st[-4:, x0:x0 + sw] = (230, 180, 20)
                    st[:, x0:x0 + 4] = (230, 180, 20); st[:, x0 + sw - 4:x0 + sw] = (230, 180, 20)
                    frame = np.concatenate((view, st, cursor(panel, boxes, ti / (n_frames - 1))), 0)
                    proc.stdin.write(frame.tobytes())
                    if fi == hold + n_frames // 2:
                        stills.append(frame)
            index.append(dict(kind=name, scene=rec["scene_id"], source=source, plan_min_clear_m=round(clear_p, 4),
                              executed_min_clear_m=round(clear_e, 4), samples_clear=int(free.sum())))
        proc.stdin.close(); proc.wait()
        Image.fromarray(stills[0]).save(args.out / f"{stem}.jpg", quality=88)
        print("VIDEO", stem, len(rows), flush=True)
    (args.out / ("index_probe.json" if column is not None else "index.json")).write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
