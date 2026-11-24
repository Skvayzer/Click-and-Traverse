#!/usr/bin/env python3
"""Flat-ground sanity gates for the frozen SONIC tracker before any benchmark (single world, CPU MuJoCo).

replay   record the reactive V7r policy walking on the flat benchmark scene at --speed, then let SONIC track that
         recorded whole-body clip open-loop in time from the identical initial state. Reports the median root-xy error
         and joint error over --seconds.
plan     SONIC tracks the shared frozen planner's online plans on the same flat scene (the BRACE flat check):
         root dx/dy, path length, median distance to the plan's current point, joint error to the plan.
Pass criterion (preregistered by the task): median root error <= 10 cm and no fall.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

BANK = Path("/tmp/cat-overnight/planning_v1")
COMBO = Path("/tmp/cat-energy-current-videos/inputs/COMBO_FIX_315808_u150.pt")
V7R = Path("/tmp/cat-overnight/e4-inputs/V7r_u600.pt")
PLANNER = Path("/tmp/cat-plan-blind/planner.pt")
SONIC = Path("/home/konstantinsmirnov/robotics/sonic/hf_gear_sonic")
SAMPLE = Path("/home/konstantinsmirnov/robotics/sonic/hf_sample/walk_A001.npz")


def build(checkpoint, speed, *, plan):
    from cat_mjlab.runner import create_task
    from cat_mjlab.sim_cpu import CPUSimulation
    from eval_per_scene import benchmark_start

    contract = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["contract"]
    env = copy.deepcopy(contract["environment_config"])
    for key in ("robustness", "sgs", "side_gap_curriculum", "refusal_training"):
        env.pop(key, None)
    env["push_config"]["enable"] = False
    env["dm_rand_config"]["enable_pd"] = False
    if plan:
        env["tracker"]["mix"] = [["plan", 1.]]
        env["tracker"]["planner"]["path"] = str(PLANNER)
    env["goal_seed"]["cache"] = str(BANK) + "_goal"
    env["operator_speed"] = dict(min=speed, max=speed, nominal=.525, cutout=.05, near_sdf=.35, near_share=0.,
                                 resample=None, length_scale_max=5.25, anchor_free_below=0., stall_fraction=.2,
                                 rest_start_share=0.)
    ids = Path(str(BANK) + "_packed/scene_ids.txt").read_text().split()
    args = SimpleNamespace(bank_manifest=Path(str(BANK) + "_packed/manifest.json"),
                           body_collision_bank=Path(str(BANK) + "_collision/manifest.json"),
                           body_collision_resets=Path(str(BANK) + "_resets/manifest.json"), device="cpu", seed=1,
                           num_envs=1, compile_task=False, nconmax=contract["nconmax"], njmax=contract["njmax"],
                           reactive_bank=None, reactive_row=None, goal_seed_scene_ids=set(ids))
    task, _, _ = create_task(args, environment_config=env, sim_class=CPUSimulation)
    records = task.bank.manifest["scenes"]
    flat = next(i for i, scene in enumerate(records) if scene["scene_id"] == "flat-balance-walk-v1")
    task.raised_reset_fraction = 0.
    task.bank.episode_lengths[flat] = 100000
    scene = torch.tensor([flat])
    return task, scene, benchmark_start(task, scene, torch.tensor([0]), 1)


def fell(task):
    return float(task.data.qpos[0, 2]) < .5


OPTIONS = {}


def sonic(task, **kw):
    from cat_mjlab.sonic_tracker import SonicTracker
    folder = SONIC / ("sonic_v1_1" if OPTIONS.get("variant") == "v1_1" else "")
    kw.update(OPTIONS)
    return SonicTracker(task, folder / "model_encoder.onnx", folder / "model_decoder.onnx", provider="cpu", **kw)


def track_clip(clip, q0, v0, steps, noise=True):
    """SONIC follows a whole-body clip [T, 36] (50 Hz qpos) open-loop in time from (q0, v0) on the flat scene."""
    task, scene, _ = build(V7R, .3, plan=False)
    last = len(clip) - 1
    joint_vel = torch.zeros_like(clip[:, 7:])
    joint_vel[1:-1] = (clip[2:, 7:] - clip[:-2, 7:]) / (2 * .02)
    counter = {"k": 0}

    def reference():
        idx = (counter["k"] + torch.arange(10) * 5).clamp_max(last)
        return clip[idx][None], joint_vel[idx][None]

    task.reset(scene_ids=scene, initial_pose=q0, initial_velocity=v0)
    if not noise:
        task.info["rfi"].zero_()
    feet = [task.model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    tracker = sonic(task, reference=reference)
    root_err, joint_err, yaw_err, saturated, foot_z = [], [], [], [], []
    fall_step = None
    for step in range(steps):
        counter["k"] = step
        action, target = tracker()
        out = task.step(action, position_targets=target)
        ref = clip[step + 1]
        q = task.data.qpos[0]
        root_err.append(float(torch.linalg.vector_norm(q[:2] - ref[:2])))
        joint_err.append(float((q[7:] - ref[7:]).abs().mean()))
        yaw_err.append(float(torch.remainder(yaw_of(q[3:7]) - yaw_of(ref[3:7]) + torch.pi, 2 * torch.pi) - torch.pi))
        saturated.append(saturation(task, target))
        foot_z.append(task.data.xpos[0, feet, 2].clone())
        if fell(task) or bool(out["done"][0]):
            fall_step = step + 1
            break
    z = torch.stack(foot_z)
    return dict(seconds=len(root_err) * .02, fell_at_step=fall_step, root_err_median_m=float(np.median(root_err)),
                root_err_final_m=root_err[-1], root_err_max_m=float(np.max(root_err)),
                robot_travel_m=float(torch.linalg.vector_norm(task.data.qpos[0, :2] - q0[0, :2])),
                joint_err_median_rad=float(np.median(joint_err)), joint_err_mean_rad=float(np.mean(joint_err)),
                yaw_err_median_abs_rad=float(np.median(np.abs(yaw_err))),
                foot_lift_max_m=(z.max(0).values - z.min(0).values).tolist(),
                torque_saturation_share=float(np.mean(saturated)), torque_noise=noise,
                batched_graph_max_error=tracker.batched_error)


def sample(a):
    """NVIDIA's released sample G1 walk (sample_data/robot_filtered/210531/walk_forward_amateur_001__A001, 30 fps,
    MuJoCo joint order, xyzw root quaternion), resampled to 50 Hz; starts at --start-s s into the clip."""
    from scipy.spatial.transform import Rotation, Slerp

    d = np.load(SAMPLE)
    fps = int(d["fps"]); src = np.arange(len(d["dof"])) / fps
    tq = np.arange(int(src[-1] / .02)) * .02
    dof = np.stack([np.interp(tq, src, d["dof"][:, j]) for j in range(29)], -1)
    pos = np.stack([np.interp(tq, src, d["root_trans_offset"][:, j]) for j in range(3)], -1)
    rot = Slerp(src, Rotation.from_quat(d["root_rot"]))(tq).as_quat()[:, [3, 0, 1, 2]]
    clip = torch.tensor(np.concatenate([pos, rot, dof], -1), dtype=torch.float32)[int(a.start_s / .02):]
    q0 = clip[:1].clone(); v0 = torch.zeros(1, 35)
    v0[0, 6:] = (clip[1, 7:] - clip[0, 7:]) / .02
    steps = int(round(a.seconds / .02))
    result = dict(gate="NVIDIA sample walk", clip=str(SAMPLE), start_s=a.start_s,
                  reference_path_m=float(torch.linalg.vector_norm(clip[1:steps + 1, :2] - clip[:steps, :2], dim=-1).sum()),
                  **track_clip(clip, q0, v0, steps, noise=not a.no_noise))
    result["pass"] = bool(result["fell_at_step"] is None and result["root_err_median_m"] <= .10)
    return result


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def replay(a):
    from record_mjlab_rollout import load_policy

    steps = int(round(a.seconds / .02))
    lookahead = 50
    cache = Path(f"/tmp/cat-sonic/v7r_clip_{a.speed:.2f}_{steps + lookahead}.pt")
    if cache.exists():
        saved = torch.load(cache)
        clip, q0, v0 = saved["clip"], saved["q0"], saved["v0"]
    else:
        task, scene, start = build(V7R, a.speed, plan=False)
        task.reset(scene_ids=scene, **start)
        learner, _ = load_policy(V7R, device="cpu", policy_id=0)
        q0, v0 = task.data.qpos.clone(), task.data.qvel.clone()
        clip = [task.data.qpos[0].clone()]
        for step in range(steps + lookahead):
            with torch.no_grad():
                out = task.step(learner.act(task.obs, policy_ids=0, deterministic=True)["action"])
            if bool(out["done"][0]) or fell(task):
                raise RuntimeError(f"V7r reference ended at step {step + 1}")
            clip.append(task.data.qpos[0].clone())
        clip = torch.stack(clip)                                   # [T, 36]
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(clip=clip, q0=q0, v0=v0), cache)
        del task, learner
    walked = float(torch.linalg.vector_norm(clip[steps, :2] - clip[0, :2]))

    result = dict(gate="replay V7r flat walk", speed=a.speed, reference_path_m=walked,
                  **track_clip(clip, q0, v0, steps, noise=not a.no_noise))
    result["pass"] = bool(result["fell_at_step"] is None and result["root_err_median_m"] <= .10)
    return result


def saturation(task, target):
    """Share of joints whose PD torque at the control step's start exceeds the active torque limit."""
    d = task.data
    tau = task.kps * (target[0] - d.qpos[0, 7:]) - task.kds * d.qvel[0, 6:]
    return float((tau.abs() > task.torque_limit).float().mean())


def plan(a, ours=False):
    """ours=True: the same check with our planner-reflex policy (COMBO_FIX seed 0, frozen) acting instead of SONIC."""
    steps = int(round(a.seconds / .02))
    task, scene, start = build(COMBO, a.speed, plan=True)
    task.reset(scene_ids=scene, **start)
    if ours:
        from record_mjlab_rollout import load_policy
        from cat_mjlab.sonic_tracker import plan_reference
        learner, _ = load_policy(COMBO, device="cpu", policy_id=0)
        policy_q = {}

        def tracker():
            with torch.no_grad():
                action = learner.act(task.obs, policy_ids=0, deterministic=True)["action"]
            t = task.tracker
            policy_q["ref"] = plan_reference(task, t.plan_t[:, None])[0][:, 0, 7:]
            return action, None
        tracker.ref = policy_q
    else:
        tracker = sonic(task)
    if a.no_noise:
        task.info["rfi"].zero_()
    first = task.data.qpos[0, :2].clone(); previous = first.clone()
    path, errors, joint_err, yaw_err, saturated = 0., [], [], [], []
    fall_step = None
    t = task.tracker
    for step in range(steps):
        action, target = tracker()
        out = task.step(action, position_targets=target)
        q = task.data.qpos[0]
        if fell(task) or bool(out["done"][0]):
            fall_step = step + 1
            break
        if bool(t.has_plan[0]):
            xy, yaw, _, _ = t.plan_at(task.all_ids, t.plan_t)
            errors.append(float(torch.linalg.vector_norm(q[:2] - xy[0])))
            yaw_err.append(float(torch.remainder(yaw_of(q[3:7]) - yaw[0] + torch.pi, 2 * torch.pi) - torch.pi))
            ref_q = tracker.ref["ref"] if ours else tracker.ref_q
            joint_err.append(float((q[7:] - ref_q[0]).abs().mean()))
        if target is not None:
            saturated.append(saturation(task, target))
        path += float(torch.linalg.vector_norm(q[:2] - previous)); previous = q[:2].clone()
    result = dict(gate="planner plans, flat", speed=a.speed, seconds=(fall_step or steps) * .02, fell_at_step=fall_step,
                  root_dx_m=float(previous[0] - first[0]), root_dy_m=float(previous[1] - first[1]), path_m=path,
                  plan_err_median_m=float(np.median(errors)) if errors else None,
                  plan_err_p90_m=float(np.quantile(errors, .9)) if errors else None,
                  joint_err_median_rad=float(np.median(joint_err)) if joint_err else None,
                  yaw_err_median_abs_rad=float(np.median(np.abs(yaw_err))) if yaw_err else None,
                  torque_saturation_share=float(np.mean(saturated)) if saturated else None, torque_noise=not a.no_noise,
                  controller="COMBO_FIX seed 0 (ours)" if ours else "SONIC")
    result["pass"] = bool(fall_step is None and result["plan_err_median_m"] is not None
                          and result["plan_err_median_m"] <= .10)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gate", choices=("replay", "plan", "sample", "plan_ours"))
    p.add_argument("--variant", choices=("default", "v1_1"), default="default")
    p.add_argument("--ankle-gain", type=float, default=1.)
    p.add_argument("--no-noise", action="store_true", help="zero the harness's random torque injection")
    p.add_argument("--start-s", type=float, default=5.)
    p.add_argument("--speed", type=float, default=.3)
    p.add_argument("--seconds", type=float, default=8.)
    p.add_argument("--torque-limits", choices=("harness", "sonic"), default="harness")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    OPTIONS.update(variant=a.variant, ankle_gain=a.ankle_gain, torque_limits=a.torque_limits)
    result = dict(replay=replay, plan=plan, sample=sample, plan_ours=lambda a: plan(a, ours=True))[a.gate](a)
    result.update(variant=a.variant, ankle_gain=a.ankle_gain, torque_limits=a.torque_limits)
    text = json.dumps(result, indent=1)
    print("SONIC_GATE " + json.dumps(result), flush=True)
    if a.out:
        a.out.write_text(text)


if __name__ == "__main__":
    main()
