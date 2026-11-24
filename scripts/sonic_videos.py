#!/usr/bin/env python3
"""Diagnostic videos for the SONIC flat-ground gates (E6b).

record CASE OUT.npz   (CPU) run one 12 s flat-ground episode and store the robot and reference poses per control step.
    sonic030 / sonic015  SONIC tracks the shared planner's online plans at 0.30 / 0.15 m/s (gate b)
    ours030              COMBO_FIX seed 0 (planner-reflex) on the same scene, start, planner and speed
    sample               SONIC on NVIDIA's released sample G1 walk (adapter check), from 9 s into the clip
render IN.npz OUT.mp4 (EGL) 1280x720: robot, translucent blue reference ghost, reference pelvis path (blue) and robot
    pelvis trail (orange), feet close-up inset, text overlay.
sbs A.mp4 B.mp4 OUT.mp4 side-by-side (each scaled to 640x360 and padded to 1280x720).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

SECONDS = 12.
TITLES = {
    "sonic030": "SONIC (NVIDIA GEAR-SONIC, frozen) tracking our planner's whole-body plan, 0.30 m/s",
    "sonic015": "SONIC (NVIDIA GEAR-SONIC, frozen) tracking our planner's whole-body plan, 0.15 m/s",
    "ours030": "Our planner-reflex policy (COMBO_FIX seed 0) with the same planner, 0.30 m/s",
    "sample": "SONIC on NVIDIA's own sample walk in our harness (adapter check)",
}


def record(case, out):
    import torch
    import sonic_gates as g

    torch.set_num_threads(2)
    steps = int(SECONDS / .02)
    robot, ref, chunk, err = [], [], [], []
    if case == "sample":
        from types import SimpleNamespace
        from scipy.spatial.transform import Rotation, Slerp
        d = np.load(g.SAMPLE)
        fps = int(d["fps"]); src = np.arange(len(d["dof"])) / fps
        tq = np.arange(int(src[-1] / .02)) * .02
        dof = np.stack([np.interp(tq, src, d["dof"][:, j]) for j in range(29)], -1)
        pos = np.stack([np.interp(tq, src, d["root_trans_offset"][:, j]) for j in range(3)], -1)
        rot = Slerp(src, Rotation.from_quat(d["root_rot"]))(tq).as_quat()[:, [3, 0, 1, 2]]
        clip = torch.tensor(np.concatenate([pos, rot, dof], -1), dtype=torch.float32)[int(9. / .02):]
        task, scene, _ = g.build(g.V7R, .3, plan=False)
        last = len(clip) - 1
        jv = torch.zeros_like(clip[:, 7:]); jv[1:-1] = (clip[2:, 7:] - clip[:-2, 7:]) / .04
        k = {"k": 0}

        def reference():
            idx = (k["k"] + torch.arange(10) * 5).clamp_max(last)
            return clip[idx][None], jv[idx][None]
        v0 = torch.zeros(1, 35); v0[0, 6:] = (clip[1, 7:] - clip[0, 7:]) / .02
        task.reset(scene_ids=scene, initial_pose=clip[:1].clone(), initial_velocity=v0)
        tracker = g.sonic(task, reference=reference)
        for step in range(steps):
            k["k"] = step
            action, target = tracker()
            task.step(action, position_targets=target)
            q = task.data.qpos[0]
            robot.append(q.numpy().copy()); ref.append(clip[step + 1].numpy().copy())
            chunk.append(clip[min(step + 1, last):min(step + 101, last + 1):5, :2].numpy().copy()[:21])
            err.append(float(torch.linalg.vector_norm(q[:2] - clip[step + 1, :2])))
        chunk = [np.pad(c, ((0, 21 - len(c)), (0, 0)), mode="edge") for c in chunk]
    else:
        from cat_mjlab.sonic_tracker import plan_reference
        speed = .30 if case.endswith("030") else .15
        task, scene, start = g.build(g.COMBO, speed, plan=True)
        task.reset(scene_ids=scene, **start)
        t = task.tracker
        if case.startswith("ours"):
            from record_mjlab_rollout import load_policy
            learner, _ = load_policy(g.COMBO, device="cpu", policy_id=0)
        else:
            tracker = g.sonic(task)
        for step in range(steps):
            if case.startswith("ours"):
                with torch.no_grad():
                    out_ = task.step(learner.act(task.obs, policy_ids=0, deterministic=True)["action"])
            else:
                action, target = tracker()
                out_ = task.step(action, position_targets=target)
            q = task.data.qpos[0]
            pose = plan_reference(task, t.plan_t[:, None])[0][0, 0]
            robot.append(q.numpy().copy()); ref.append(pose.numpy().copy())
            chunk.append(t.plan_xy[0].numpy().copy())
            err.append(float(torch.linalg.vector_norm(q[:2] - pose[:2])) if bool(t.has_plan[0]) else float("nan"))
            if bool(out_["done"][0]):
                print("episode ended at", step + 1, flush=True)
                break
    np.savez_compressed(out, robot=np.array(robot), ref=np.array(ref), chunk=np.array(chunk), err=np.array(err),
                        case=case, title=TITLES[case])
    print("recorded", case, len(robot), "steps; median error", np.nanmedian(err), flush=True)


def render(src, out):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from planner_video import add_ghost, add_line

    d = np.load(src)
    robot, ref, chunk, err = d["robot"], d["ref"], d["chunk"], d["err"]
    title, case = str(d["title"]), str(d["case"])
    model = mujoco.MjModel.from_xml_path(str(ROOT / "data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain.xml"))
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    data, ghost = mujoco.MjData(model), mujoco.MjData(model)
    option, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    feet_cam = mujoco.MjvCamera(); feet_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    main_r = mujoco.Renderer(model, height=720, width=1280, max_geom=4000)
    inset_r = mujoco.Renderer(model, height=270, width=420, max_geom=4000)
    start = robot[0, :2].copy()
    walked = np.r_[0., np.cumsum(np.linalg.norm(np.diff(robot[:, :2], axis=0), axis=1))]
    planned = np.r_[0., np.cumsum(np.linalg.norm(np.diff(ref[:, :2], axis=0), axis=1))]
    reference_name = "NVIDIA sample clip" if case == "sample" else "plan"
    speed = .15 if case.endswith("015") else .30
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", "25", "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    trail_len = 10_000
    for i in range(0, len(robot), 2):                      # 50 Hz control -> 25 fps real time
        data.qpos[:] = robot[i]; mujoco.mj_kinematics(model, data)
        focus = robot[i, :3].copy(); focus[2] = .65
        cam.lookat[:] = focus; cam.distance, cam.azimuth, cam.elevation = 3.4, 115., -14.
        main_r.update_scene(data, cam)
        sc = main_r.scene
        add_ghost(sc, model, ghost, mujoco, option, perturb, ref[i], np.array([.15, .45, 1., .35], np.float32))
        z = .01
        add_line(sc, mujoco, np.c_[chunk[i], np.full(len(chunk[i]), z)], [.1, .35, 1., .9], .012)
        trail = robot[max(0, i - trail_len):i + 1:2, :2]
        if len(trail) > 1:
            add_line(sc, mujoco, np.c_[trail, np.full(len(trail), z + .002)], [1., .5, .05, .9], .010)
        frame = main_r.render().copy()
        # feet close-up: low side view of the robot's feet, real robot only
        feet = (data.xpos[model.body("left_ankle_roll_link").id] + data.xpos[model.body("right_ankle_roll_link").id]) / 2
        feet_cam.lookat[:] = [feet[0], feet[1], .06]
        feet_cam.distance, feet_cam.azimuth, feet_cam.elevation = .9, 90., -3.
        inset_r.update_scene(data, feet_cam)
        inset = inset_r.render()
        img = Image.fromarray(frame); draw = ImageDraw.Draw(img)
        img.paste(Image.fromarray(inset), (1280 - 420 - 12, 720 - 270 - 12))
        draw.rectangle((1280 - 420 - 14, 720 - 270 - 14, 1280 - 10, 720 - 10), outline=(30, 30, 30), width=2)
        lz = [data.xpos[model.body(f"{s}_ankle_roll_link").id][2] for s in ("left", "right")]
        draw.rectangle((1280 - 420 - 14, 720 - 270 - 44, 1280 - 10, 720 - 270 - 14), fill=(0, 0, 0))
        draw.text((1280 - 420 - 6, 720 - 270 - 40), f"feet close-up   ankle z  L {lz[0]*100:4.1f}  R {lz[1]*100:4.1f} cm",
                  font=small, fill=(255, 255, 255))
        lines = [title,
                 f"t = {i * .02:5.2f} s     {reference_name} error = {err[i] * 100:5.1f} cm",
                 (f"distance travelled (net) {np.linalg.norm(robot[i, :2] - start):4.2f} m   vs   clip {np.linalg.norm(ref[i, :2] - ref[0, :2]):4.2f} m"
                  if case == "sample" else
                  f"distance travelled (net) {np.linalg.norm(robot[i, :2] - start):4.2f} m   vs   planned {speed * i * .02:4.2f} m"
                  f" ({speed:.2f} m/s x t)"),
                 f"blue ghost + blue line: {reference_name}      orange line: robot pelvis path"]
        y = 10
        for k, text in enumerate(lines):
            f = font if k < 3 else small
            box = draw.textbbox((14, y), text, font=f)
            draw.rectangle((box[0] - 6, box[1] - 3, box[2] + 6, box[3] + 3), fill=(0, 0, 0))
            draw.text((14, y), text, font=f, fill=(255, 255, 255))
            y = box[3] + 9
        enc.stdin.write(np.asarray(img).tobytes())
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("ffmpeg failed")
    print("rendered", out, flush=True)


def side_by_side(a, b, out):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(a), "-i", str(b), "-filter_complex",
           "[0:v]scale=640:360[l];[1:v]scale=640:360[r];[l][r]hstack=inputs=2,pad=1280:720:0:180:color=black,"
           "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Left\\: SONIC tracking our plan"
           "      Right\\: our planner-reflex policy      (flat ground, 0.30 m/s, same start and planner)':"
           "fontcolor=white:fontsize=20:x=20:y=60",
           "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", str(out)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("record", "render", "sbs"))
    p.add_argument("args", nargs="+")
    a = p.parse_args()
    {"record": lambda x: record(x[0], x[1]), "render": lambda x: render(x[0], x[1]),
     "sbs": lambda x: side_by_side(*x)}[a.mode](a.args)
