#!/usr/bin/env python3
"""Side-by-side video of a whole-traversal (full-path) oracle plan followed by the current network, next to the matched
plan-off rollout (same scene, speed and start), from scripts/fullpath_rollout.py recordings (docs/fullpath_plan.md, S2).

Left: the full planned start->goal pelvis path on the floor (blue), posture keyframes of the plan as translucent ghosts every
--ghost-spacing m along the whole route, the reference pose the network is asked to reach 0.3 s ahead (stronger ghost), the
robot as simulated; labels: scene type, speed, time, outcome, the robot's distance to the planned path (pelvis) and the
largest body-point deviation from the nearest plan pose. Right: the same network without a reference (plan-off). Both views
follow their robot; real time at 25 fps; an episode that has ended holds its last frame with the outcome banner.
1280x720 H.264 yuv420p, silent AAC, faststart, plus a poster (JPEG).

    render_fullpath.py --oracle-ep ORACLE.npz --off-ep OFF.npz --out DIR --name NAME --title "Side gap"
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

PANEL_W, H = 640, 720


def yaw(q):
    return math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle-ep", type=Path, required=True)
    p.add_argument("--off-ep", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--max-seconds", type=float, default=75.)
    p.add_argument("--ghost-spacing", type=float, default=0.35)
    p.add_argument("--camera", type=float, nargs=3, default=(3.2, -30., 60.))
    a = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from planner_render import add_path, scene_geometry, label
    from planner_video import add_ghost
    from render_clutter_rollouts import resolve_xml_assets
    from cat_mjlab.exact_geometry import ExactGeometry, interpolate_frames
    from cat_mjlab.fullpath import OracleFeeder
    eo, ef = dict(np.load(a.oracle_ep)), dict(np.load(a.off_ep))
    mo, mf = json.loads(str(eo["meta"])), json.loads(str(ef["meta"]))
    assert mo["scene"] == mf["scene"] and mo["speed"] == mf["speed"] and mo["trial"] == mf["trial"]
    if "oracle_qpos" not in eo:
        raise SystemExit("the oracle episode has no oracle plan (reactive fallback)")
    bank = Path(mo["bank"] + "_packed/manifest.json")
    rec = next(r for r in json.loads(bank.read_text())["scenes"] if r["scene_id"] == mo["scene"])
    with tempfile.TemporaryDirectory() as tmp:
        scene_geometry(Path(tmp), bank, rec)
        xml, _ = resolve_xml_assets(Path(tmp) / "model.xml")
        model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.offwidth, model.vis.global_.offheight = PANEL_W, H
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[g, 3] = 0.6
    data, ghost = mujoco.MjData(model), mujoco.MjData(model)
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0; option.sitegroup[:] = 0
    perturb = mujoco.MjvPerturb()
    plan = eo["oracle_qpos"].astype(np.float64)

    class _T:                      # OracleFeeder only needs the progress table here
        pass
    feeder = OracleFeeder(dict(qpos=plan), _T())
    # ghosts every ghost_spacing m of progress along the whole plan
    us = np.arange(0., feeder.u[-1] + 1e-6, a.ghost_spacing)
    ghosts, _ = feeder.at(us)
    path = np.concatenate((feeder.smooth, np.full((len(feeder.smooth), 1), 0.012)), -1)
    geo = ExactGeometry("cpu")
    dense = plan[::2]
    dense_world, _ = geo.primitives(dense)
    dense_xy = feeder.smooth[::2]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17); big = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = big = ImageFont.load_default()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    dt = float(mo["dt"]); stride = max(1, int(round(1 / (a.fps * dt))))
    Lo, Lf = len(eo["qpos"]) - 2, len(ef["qpos"]) - 2              # last filled state index
    T = min(max(Lo, Lf), int(a.max_seconds / dt))
    steps = list(range(0, T + 1, stride))
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / f"{a.name}.mp4"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * PANEL_W}x{H}",
           "-r", str(a.fps), "-i", "-", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0", "-map", "1:a:0",
           "-t", f"{len(steps) / a.fps:.3f}", "-c:a", "aac", "-b:a", "96k", "-c:v", "libx264", "-threads", "4", "-pix_fmt", "yuv420p",
           "-crf", "21", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    q0 = eo["qpos"][0]; az = math.degrees(yaw(q0)) + a.camera[2]
    cams = [mujoco.MjvCamera(), mujoco.MjvCamera()]
    for c in cams:
        c.distance, c.elevation, c.azimuth = a.camera[0], a.camera[1], az
    looks = [eo["qpos"][0, :2].copy(), ef["qpos"][0, :2].copy()]
    res = lambda m: m["outcome"]["result"] + (" (hand contact)" if m["outcome"].get("hand_contact") else "")
    poster, poster_t = None, (min(Lo, T) // 2)
    errs = []
    with mujoco.Renderer(model, width=PANEL_W, height=H) as renderer:
        for t in steps:
            panels = []
            for side, (e, m, L) in enumerate(((eo, mo, Lo), (ef, mf, Lf))):
                tt = min(t, L)
                q = e["qpos"][tt].astype(np.float64)
                data.qpos[:] = q; mujoco.mj_forward(model, data)
                looks[side] = 0.9 * looks[side] + 0.1 * q[:2] if t else q[:2].copy()
                cams[side].lookat[:] = [looks[side][0], looks[side][1], 0.55]
                renderer.update_scene(data, camera=cams[side], scene_option=option)
                sc = renderer.scene
                lines = []
                if side == 0:
                    add_path(sc, mujoco, feeder.smooth, (0.15, 0.45, 1.0, 0.9), 0.012)
                    for g_ in ghosts:
                        add_ghost(sc, model, ghost, mujoco, option, perturb, g_, (0.45, 0.65, 1.0, 0.10))
                    if "ev_t" in e:
                        k = int(np.searchsorted(e["ev_t"], tt, side="left")) - 1
                        if k >= 0:
                            ref = interpolate_frames(e["ev_chosen"][k], [float(e["plan_t"][tt]) + .3])[0]
                            add_ghost(sc, model, ghost, mujoco, option, perturb, ref, (0.20, 0.50, 1.0, 0.33))
                    j = int(np.linalg.norm(dense_xy - q[:2], axis=-1).argmin())
                    pel_err = float(np.linalg.norm(dense_xy[j] - q[:2]))
                    wa, _ = geo.primitives(q[None])
                    body_err = float(np.linalg.norm((wa["centers"][0] - dense_world["centers"][j]).numpy(), axis=-1).max())
                    errs.append((pel_err, body_err))
                    lines = [f"Full-path oracle plan -> current network  |  {a.title}",
                             f"told {m['speed']:.2f} m/s  |  t = {tt * dt:5.1f} s  |  plan margin {100 * (m['oracle'] or {}).get('min_sep', 0):.1f} cm",
                             f"distance to planned path {100 * pel_err:4.1f} cm  |  largest body deviation {100 * body_err:4.1f} cm"]
                else:
                    lines = [f"Same network, no plan (plan-off)  |  {a.title}", f"told {m['speed']:.2f} m/s  |  t = {tt * dt:5.1f} s"]
                frame = renderer.render().copy()
                frame = label(frame, lines, font)
                ended = t >= L
                banner = [("ENDED: " if ended else "episode: ") + res(m)]
                if side == 0:
                    banner.insert(0, "blue line: planned path  |  faint ghosts: planned postures")
                    banner.insert(1, "bright ghost: target pose 0.3 s ahead")
                frame = label(frame, banner, font, xy=(14, H - 30 - 26 * len(banner)),
                              fill=(255, 220, 120) if ended else (255, 255, 255))
                panels.append(frame)
            frame = np.concatenate(panels, 1)
            frame[:, PANEL_W - 1:PANEL_W + 1] = 255
            proc.stdin.write(frame.tobytes())
            if poster is None and t >= poster_t:
                poster = frame
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg failed")
    Image.fromarray(poster if poster is not None else frame).save(a.out / f"{a.name}.jpg", quality=90)
    e = np.array(errs)
    info = dict(video=out.name, poster=f"{a.name}.jpg", scene=mo["scene"], title=a.title, speed=mo["speed"], trial=mo["trial"],
                oracle_outcome=mo["outcome"], plan_off_outcome=mf["outcome"], oracle_source=(mo["oracle"] or {}).get("source"),
                plan_margin_m=(mo["oracle"] or {}).get("min_sep"), seconds=round(len(steps) / a.fps, 1),
                path_error_p50_m=round(float(np.median(e[:, 0])), 3), body_dev_p50_m=round(float(np.median(e[:, 1])), 3))
    (a.out / f"{a.name}.json").write_text(json.dumps(info, indent=1))
    print("VIDEO", json.dumps(info))


if __name__ == "__main__":
    main()
