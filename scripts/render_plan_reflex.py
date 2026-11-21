#!/usr/bin/env python3
"""Videos of the planner-with-a-field-reflex policy from scripts/record_plan_reflex.py recordings.

Per recorded world: the robot as simulated (solid), the active whole-body plan --ghost-ahead s past the plan clock
(see-through blue ghost; 0.3 s is what the network is asked to reach), the chosen plan's pelvis path for the rest of the
chunk (blue line), the other candidates of the latest replan (grey: clear, amber: turned 90 deg, red: overlapping an
obstacle), and a red marker on a hand while it is inside the reflex band (in training that arm is then not paid for
tracking, and the field-driven terms move it). While the stall fallback has dropped the plan, nothing plan-related is
drawn and the status says so. One mp4 per world (real time at 25 fps), a still at the moment of closest hand
approach, and index.json.

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

AHEAD = 0.3          # s, default: the ghost shows the plan this far ahead of the plan clock


def kind_of(scene_id):
    import re
    width = re.search(r"-w([0-9.]+)", scene_id)
    w = f" {float(width.group(1)):.2f} m" if width else ""
    if scene_id.startswith("sidegap"):
        return "side gap" + w
    if scene_id.startswith("contrast"):
        return "cabinet passage" + w
    if scene_id.startswith("random"):
        return "cluttered room"
    if "table" in scene_id:
        return "table pair"
    course = re.match(r"D\d+G(\d+)L(\d+)O(\d+)S\d+$", scene_id)   # original CAT courses: G low bars, O overhead beams
    if course:
        g, _, o = map(int, course.groups())
        parts = (["overhead beam"] if o else []) + (["low bar"] if g else [])
        return "CAT course, " + (" + ".join(parts) if parts else "blocks")
    return "CAT course"


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
    p.add_argument("--ghost-ahead", type=float, default=AHEAD, help="s past the plan clock that the ghost shows")
    p.add_argument("--stills-only", action="store_true",
                   help="clean PNG stills for figures: no text, no candidate lines, no video; one per --times entry, or "
                        "at the closest hand approach")
    p.add_argument("--worlds", type=int, nargs="+", help="render only these recorded worlds")
    p.add_argument("--times", type=float, nargs="+", help="with --stills-only: episode times (s) to render")
    p.add_argument("--camera", type=float, nargs=3, metavar=("DISTANCE", "ELEVATION", "AZIMUTH_OFFSET"),
                   default=(3.4, -28., 55.), help="camera distance (m), elevation and azimuth offset from the start heading (deg)")
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
    has_ev = "ev_t" in z.files
    fallback = z["fallback"] if "fallback" in z.files else np.zeros_like(z["has_plan"])
    body_clear = z["body_clear"] if "body_clear" in z.files else None
    index = []
    for w, scene_id in enumerate(meta["scenes"]):
        if args.worlds and w not in args.worlds:
            continue
        end = int(meta["end"][w]); outcome = meta["outcome"][w] or dict(result="running at the end", hand_contact=False)
        steps = list(range(0, min(end, int(args.max_seconds / dt)) + 1, stride))
        if args.stills_only:
            last = min(end, int(args.max_seconds / dt))
            steps = ([min(int(round(t / dt)), last) for t in args.times] if args.times
                     else [int(np.argmin(z["hand_clear"][:last + 1, w].min(-1)))])
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
        camera.distance, camera.elevation, camera.azimuth = args.camera[0], args.camera[1], math.degrees(yaw0) + args.camera[2]
        look = q0[:2].copy()
        name = f"{w:02d}_" + "".join(ch if ch.isalnum() else "_" for ch in kind_of(scene_id)).strip("_")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", "27", "-movflags", "+faststart", str(args.out / f"{name}.mp4")]
        proc = None if args.stills_only else subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        closest, still = 1e9, None
        reflex_steps = fallback_steps = 0
        if has_ev:
            mine = np.nonzero(z["ev_w"] == w)[0]
            ev_times = z["ev_t"][mine]
        with mujoco.Renderer(model, width=args.width, height=args.height) as renderer:
            for t in steps:
                q = z["qpos"][t, w]
                data.qpos[:] = q; mujoco.mj_forward(model, data)
                look = q[:2].copy() if args.stills_only else 0.9 * look + 0.1 * q[:2]
                camera.lookat[:] = [look[0], look[1], 0.6]
                renderer.update_scene(data, camera=camera, scene_option=option)
                sc = renderer.scene
                active = bool(z["has_plan"][t, w]); dropped = bool(fallback[t, w])
                cand_txt = ""
                if active:
                    plan, pt = z["plan"][t, w], float(z["plan_t"][t, w])
                    add_ghost(sc, model, ghost, mujoco, option, perturb, plan_pose(plan, pt + args.ghost_ahead), (0.30, 0.55, 1.0, 0.28))
                    k0 = min(int(pt / 0.1), 20)
                    path = np.concatenate((plan[k0:, :2], np.full((21 - k0, 1), 0.02)), -1)
                    if len(path) > 1:
                        add_line(sc, mujoco, path, (0.20, 0.45, 1.0, 0.95), 0.012)
                    if has_ev and len(mine) and not args.stills_only:
                        j = np.searchsorted(ev_times, t, side="left") - 1
                        if j >= 0:
                            e = mine[j]; score = z["ev_score"][e]; best = int(np.argmax(score))
                            clear, off = z["ev_clear"][e], z["ev_offset"][e]
                            for c in range(len(score)):
                                if c == best:
                                    continue
                                col = ((0.85, 0.15, 0.15, 0.9) if clear[c] < 0 else (0.95, 0.62, 0.10, 0.9) if abs(off[c]) > 1e-3
                                       else (0.55, 0.55, 0.55, 0.9))
                                cp = z["ev_cands"][e, c].astype(np.float32)
                                add_line(sc, mujoco, np.concatenate((cp[:, :2], np.full((21, 1), 0.015)), -1), col, 0.006)
                            names = ["%+.0f cm%s%s" % (100 * clear[c], " sideways" if abs(off[c]) > 1e-3 else "",
                                                       " (chosen)" if c == best else "") for c in range(len(score))]
                            cand_txt = "last replan " + f"{(t - int(z['ev_t'][e])) * dt:.1f} s ago: candidate clearances " + ", ".join(names)
                released = z["released"][t, w]
                for h in range(2):
                    if released[h]:
                        add_sphere(sc, mujoco, data.site_xpos[palms[h]], (0.95, 0.15, 0.15, 0.85), 0.045)
                reflex_steps += int(released.any()); fallback_steps += int(dropped)
                frame = renderer.render().copy()
                if args.stills_only:
                    Image.fromarray(frame).save(args.out / f"{name}_t{t * dt:05.1f}.png")
                    print("STILL", name, f"t={t * dt:.1f}s", "hand", float(z["hand_clear"][t, w].min()), flush=True)
                    continue
                clear = float(z["hand_clear"][t, w].min())
                if dropped:
                    status = "FALLBACK: no progress for 2 s, plan dropped - the field alone drives the robot"
                elif released.any():
                    status = ("reflex: " + " + ".join(s for s, r in zip(("left arm", "right arm"), released) if r)
                              + " released from the plan (hand within 15 cm)")
                else:
                    status = "following the plan" if active else "no plan"
                body = "" if body_clear is None else f"  |  closest body point {100 * float(body_clear[t, w]):5.1f} cm"
                lines = [f"Planner + field reflex  |  {kind_of(scene_id)}  |  t = {t * dt:5.1f} s  |  told {meta['speed']:.2f} m/s",
                         f"{status}", f"closest hand {100 * clear:5.1f} cm{body}"]
                if cand_txt:
                    lines.append(cand_txt)
                frame = label(frame, lines, font)
                frame = label(frame, [f"blue ghost: the plan {args.ghost_ahead:.1f} s ahead  |  blue line: chosen plan  |  "
                                      f"red dot: hand within {100 * meta['band_m']:.0f} cm",
                                      "other candidates: grey clear, amber turned 90 deg, red overlapping an obstacle",
                                      f"episode: {outcome['result']}" + ("  (hand contact)" if outcome.get("hand_contact") else "")],
                              font, xy=(14, args.height - 100))
                proc.stdin.write(frame.tobytes())
                if clear < closest:
                    closest, still = clear, frame
        if args.stills_only:
            continue
        proc.stdin.close(); proc.wait()
        if still is not None:
            Image.fromarray(still).save(args.out / f"{name}.jpg", quality=88)
        index.append(dict(video=f"{name}.mp4", scene=scene_id, kind=kind_of(scene_id), seconds=round(steps[-1] * dt, 1),
                          reflex_share=round(reflex_steps / max(len(steps), 1), 3),
                          fallback_share=round(fallback_steps / max(len(steps), 1), 3), closest_hand_clearance_m=round(closest, 4),
                          **outcome))
        print("VIDEO", name, outcome, flush=True)
    (args.out / "index.json").write_text(json.dumps(dict(meta={k: meta.get(k) for k in ("checkpoint", "mix", "speed", "fallback")},
                                                         videos=index), indent=1) + "\n")


if __name__ == "__main__":
    main()
