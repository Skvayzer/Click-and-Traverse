#!/usr/bin/env python3
"""Obstacle videos: plain SONIC (zero residual), SONIC + residual (whole-body / arms-only) and our planner-reflex policy
on the same scenes, starts and operator speed (one world per scene, GPU MuJoCo-Warp, nominal physics).

record POLICY OUT.npz --scenes ID... [--speed .3]
    POLICY: sonic | combo | residual:<checkpoint>:<whole|arms>
    SONIC worlds follow the field-guidance route (tracker field=1, cat_mjlab/sonic_residual.SonicController);
    the planner-reflex policy (COMBO_FIX seed 0) runs as benchmarked (tracker plan=1, shared frozen planner).
    Each world is recorded from the certified benchmark start until its first episode end (goal, contact, fall) or
    --max-steps; per step: qpos, minimum hand signed distance (m); at the end: outcome and time.
render IN.npz OUTDIR  one 1280x720 mp4 per scene (real time, or 2x when the episode is longer than 15 s)
grid OUTDIR NAME A.mp4 B.mp4 C.mp4 D.mp4  2x2 side-by-side of four clips of one scene
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

BANK = Path(os.environ.get("CAT_BANK", "/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab/data/furniture/tablegap_v1"))
COMBO = Path(os.environ.get("COMBO_CKPT", "/tmp/cat-energy-current-videos/inputs/COMBO_FIX_315808_u150.pt"))
PLANNER = Path(os.environ.get("PLANNER_CKPT", "/tmp/cat-plan-blind/planner.pt"))
DEVICE = os.environ.get("SONIC_VIDEO_DEVICE", "cuda:0")      # cpu: CPU MuJoCo, one world per process
SCENES = {
    "published-hurdle1": ("hurdle", "hurdle"),
    "D8G0L1O1S4": ("overhead beam", "beam"),
    "sidegap2-0088-plain-w0.53": ("narrow side gap 0.53 m", "side-gap"),
    "contrast-train-20260920-r2-w0.580000-forward_protected-96a57ca86054": ("cabinet passage 0.58 m", "cabinet"),
    "table-contrast-train-20260924-forward_protected-5faeda4369c6": ("table gap", "table-gap"),
    "D8G1L1O1S8": ("mixed CAT course (low bar + beam)", "cat-course"),
    "random-generic_clutter-dense-train-20261045-50861c5526b3": ("cluttered room (workshop)", "cluttered-room"),
    "random-furniture-dense-train-004004-b43029cf3054": ("furnished room (living room)", "living-room"),
    "random-furniture-dense-train-004008-73ed3af356e1": ("furnished room (dining room)", "dining-room"),
    "random-furniture-dense-train-20261213-a64e30780040": ("furnished room (meeting room)", "meeting-room"),
}


def build(policy, scene_ids, speed):
    import torch
    from cat_mjlab.runner import create_task
    from eval_per_scene import benchmark_start

    contract = torch.load(COMBO, map_location="cpu", weights_only=True, mmap=True)["contract"]
    env = copy.deepcopy(contract["environment_config"])
    for key in ("side_gap_curriculum", "sgs", "refusal_training", "robustness"):
        env.pop(key, None)
    env["push_config"]["enable"] = False
    env["eval_goal_stop"] = True
    env["tracker"]["mix"] = [["plan" if policy == "combo" else "field", 1.]]
    env["tracker"]["planner"]["path"] = str(PLANNER)
    if policy != "combo":
        env["dm_rand_config"]["enable_pd"] = False
    env["operator_speed"] = dict(min=speed, max=speed, nominal=.525, cutout=.05, near_sdf=.35, near_share=0.,
                                 resample=None, length_scale_max=.525 / .1, anchor_free_below=0., stall_fraction=.2,
                                 rest_start_share=0.)
    if "teleop" in env:
        env["teleop"].update(fraction=0., cat_fraction=0., cat_scenes=False)
    args = SimpleNamespace(bank_manifest=Path(str(BANK) + "_packed/manifest.json"),
                           body_collision_bank=Path(str(BANK) + "_collision/manifest.json"),
                           body_collision_resets=Path(str(BANK) + "_resets/manifest.json"), device=DEVICE, seed=1,
                           num_envs=len(scene_ids), compile_task=False, nconmax=contract["nconmax"],
                           njmax=contract["njmax"], reactive_bank=None, reactive_row=None,
                           goal_seed_scene_ids=set(scene_ids))
    sim_class = None
    if DEVICE == "cpu":
        from cat_mjlab.sim_cpu import CPUSimulation
        sim_class = CPUSimulation
    task, _, _ = create_task(args, environment_config=env, sim_class=sim_class)
    records = task.bank.manifest["scenes"]
    index = {scene["scene_id"]: i for i, scene in enumerate(records)}
    ids = torch.tensor([index[s] for s in scene_ids], device=task.device)
    task.raised_reset_fraction = 0.
    task.bank.episode_lengths[ids] = 100000
    start = benchmark_start(task, ids, torch.zeros(len(ids), dtype=torch.long, device=task.device), 1)
    task.reset(scene_ids=ids, **start)
    return task, [records[int(i)] for i in ids]


def record(policy, out, scene_ids, speed, max_steps):
    import torch
    from record_mjlab_rollout import load_policy

    kind = policy.split(":")[0]
    task, rows = build(kind, scene_ids, speed)
    n = len(scene_ids)
    if kind == "combo":
        learner, _ = load_policy(COMBO, device=DEVICE, policy_id=0)
        act = lambda: (learner.act(task.obs, policy_ids=0, deterministic=True)["action"], None)
        label = "Our planner-reflex policy (COMBO_FIX seed 0)"
    elif kind in ("gait", "latent"):
        from cat_mjlab.sonic_latent import LatentController
        adaptor = load_policy(policy.split(":", 1)[1], device=DEVICE, policy_id=0)[0] if kind == "latent" else None
        act = LatentController(task, adaptor=adaptor)
        label = ("SONIC (frozen) + latent adaptor" if adaptor is not None
                 else "Plain SONIC (frozen) on its own gait, along our route")
    else:
        from cat_mjlab.sonic_residual import SonicController
        residual, mode = None, "whole"
        if kind == "residual":
            _, path, mode = policy.split(":")
            residual = load_policy(path, device=DEVICE, policy_id=0)[0]
        controller = SonicController(task, residual=residual, mode=mode)
        act = controller
        label = ("SONIC (frozen) + residual reflex, " + ("whole body" if mode == "whole" else "arms only")
                 if residual is not None else "Plain SONIC (frozen), driven along our route")
    qpos = np.zeros((max_steps + 1, n, 36), np.float32)
    hand = np.zeros((max_steps + 1, n), np.float32)
    qpos[0] = task.data.qpos.cpu().numpy(); hand[0] = task.info["sdf"][:, 5:7, 0].amin(-1).cpu().numpy()
    length = np.full(n, max_steps, int)
    outcome = ["timeout"] * n
    alive = np.ones(n, bool)
    for t in range(max_steps):
        with torch.no_grad():
            action, targets = act()
            out_ = task.step(action) if targets is None else task.step(action, position_targets=targets)
        m = out_["metrics"]
        done = out_["done"].bool().cpu().numpy()
        zero = torch.zeros_like(out_["done"]).bool()
        goal = m.get("episode/goal_reached", zero).bool().cpu().numpy()
        obstacle = m.get("episode/obstacle", zero).bool().cpu().numpy()
        fall = m.get("episode/fall", zero).bool().cpu().numpy()
        hands = m.get("episode/body_collision_hands", zero).bool().cpu().numpy()
        q = task.data.qpos.cpu().numpy(); h = task.info["sdf"][:, 5:7, 0].amin(-1).cpu().numpy()
        for i in np.flatnonzero(alive):
            if done[i]:
                alive[i] = False; length[i] = t + 1
                outcome[i] = ("clean" if goal[i] and not obstacle[i] else "hand contact" if hands[i] else
                              "contact" if obstacle[i] else "fall" if fall[i] else "refused")
                qpos[t + 1:, i] = qpos[t, i]; hand[t + 1:, i] = hand[t, i]
            else:
                qpos[t + 1, i] = q[i]; hand[t + 1, i] = h[i]
        if not alive.any():
            break
    steps = int(length.max())
    np.savez_compressed(out, qpos=qpos[:steps + 1], hand=hand[:steps + 1], length=length, outcome=np.array(outcome),
                        scenes=np.array(scene_ids), label=label, speed=speed, policy=policy,
                        rows=json.dumps(rows, default=str))
    print(json.dumps(dict(policy=policy, outcomes=dict(zip(scene_ids, outcome)), seconds=(length * .02).tolist())),
          flush=True)


def scene_model(row):
    import mujoco
    from planner_render import scene_geometry
    from render_clutter_rollouts import resolve_xml_assets
    from render_plan_reflex import furnish_room

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        scene_geometry(directory, Path(str(BANK) + "_packed/manifest.json"), row)
        xml, _ = resolve_xml_assets(directory / "model.xml")
        if row.get("task_kind") == "room" and (directory / "scene.json").exists():
            xml = furnish_room(xml, json.loads((directory / "scene.json").read_text()))
        model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    model.vis.quality.shadowsize = 4096
    return model


def render(src, outdir):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    d = np.load(src)
    rows = json.loads(str(d["rows"]))
    label, speed = str(d["label"]), float(d["speed"])
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    tag = {"sonic": "plain-sonic", "combo": "planner-reflex", "gait": "plain-sonic-gait",
           "latent": "sonic-latent"}.get(str(d["policy"]).split(":")[0])
    if tag is None:
        tag = "sonic-residual-" + str(d["policy"]).rsplit(":", 1)[1]
    outputs = []
    for i, scene in enumerate(d["scenes"]):
        scene = str(scene)
        kind, short = SCENES.get(scene, (scene, scene))
        model = scene_model(rows[i])
        data = mujoco.MjData(model)
        steps = int(d["length"][i])
        factor = 1 if steps <= 750 else 2
        frames = list(range(0, steps + 1, 2 * factor)) + [steps] * 25           # hold the end for 1 s
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        q0 = d["qpos"][0, i]
        yaw0 = math.degrees(2 * math.atan2(q0[6], q0[3]))
        path = Path(outdir) / f"{short}__{tag}.mp4"
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720",
                                "-r", "25", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        outcome = str(d["outcome"][i])
        min_hand = np.inf
        with mujoco.Renderer(model, height=720, width=1280) as renderer:
            for f in frames:
                data.qpos[:36] = d["qpos"][f, i]
                mujoco.mj_kinematics(model, data)
                cam.lookat[:] = [data.qpos[0], data.qpos[1], .7]
                cam.distance, cam.azimuth, cam.elevation = 3.8, yaw0 + 40., -24.
                renderer.update_scene(data, cam)
                img = Image.fromarray(renderer.render()); draw = ImageDraw.Draw(img)
                h = float(d["hand"][f, i]); min_hand = min(min_hand, h)
                ended = f >= steps
                status = (f"OUTCOME: {outcome.upper()} at t = {steps * .02:.1f} s" if ended else "running")
                lines = [label, f"Scene: {kind}", f"Speed command {speed:.2f} m/s    t = {f * .02:5.1f} s"
                         + (f"   (playback {factor}x)" if factor > 1 else ""),
                         f"Hand clearance {h * 100:5.1f} cm   (minimum so far {min_hand * 100:5.1f} cm)", status]
                y = 10
                for k, text in enumerate(lines):
                    fnt = font if k in (0, 4) else small
                    box = draw.textbbox((14, y), text, font=fnt)
                    fill = (0, 0, 0)
                    if k == 4 and ended:
                        fill = (20, 110, 40) if outcome == "clean" else (150, 30, 30)
                    draw.rectangle((box[0] - 6, box[1] - 3, box[2] + 6, box[3] + 3), fill=fill)
                    draw.text((14, y), text, font=fnt, fill=(255, 255, 255))
                    y = box[3] + 9
                enc.stdin.write(np.asarray(img).tobytes())
        enc.stdin.close()
        if enc.wait():
            raise RuntimeError("ffmpeg failed")
        outputs.append(str(path))
        print("rendered", path, outcome, flush=True)
    return outputs


def grid(outdir, name, clips):
    inputs = []
    for clip in clips:
        inputs += ["-i", clip]
    filt = ";".join(f"[{k}:v]scale=640:360,tpad=stop_mode=clone:stop_duration=30[v{k}]" for k in range(4))
    filt += ";[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:shortest=0[g]"
    longest = max(float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                                 "csv=p=0", c], text=True)) for c in clips)
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", filt, "-map", "[g]", "-t", f"{longest:.2f}",
                    "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(Path(outdir) / name)], check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("record"); r.add_argument("policy"); r.add_argument("out")
    r.add_argument("--scenes", nargs="+", default=list(SCENES)); r.add_argument("--speed", type=float, default=.3)
    r.add_argument("--max-steps", type=int, default=1500)
    v = sub.add_parser("render"); v.add_argument("src"); v.add_argument("outdir")
    g = sub.add_parser("grid"); g.add_argument("outdir"); g.add_argument("name"); g.add_argument("clips", nargs=4)
    a = p.parse_args()
    if a.mode == "record":
        record(a.policy, a.out, a.scenes, a.speed, a.max_steps)
    elif a.mode == "render":
        render(a.src, a.outdir)
    else:
        grid(a.outdir, a.name, a.clips)
