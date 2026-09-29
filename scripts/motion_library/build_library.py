#!/usr/bin/env python3
"""Merge all tagged clip sets into ONE training-ready motion library for the style priors.

Inputs: data/motion_library/{raw, amass_g1}/manifest.json + clips (G1 29-DoF, canonical format).
Output: data/motion_library/library_v1.npz and library_v1.json

Per frame, resampled to the 50 Hz CONTROL rate of the training task (dt = 0.02 s), so a
transition (t, t+1) in the library has the same time step as a transition of the robot:
  qpos      [N,36]   root pos, quat wxyz, 29 joints        (reference-state initialisation)
  features  [N,F]    discriminator observation (see FEATURES below), float16
  next_ok   [N]      frame t+1 is in the same clip (valid transition)
  clip      [N]      clip index into library_v1.json
  skills    [N,7]    per-frame skill masks (walk, sidle, duck, crawl, step_high, stand, carry)
  group     [N]      discriminator group: 0 locomotion (walk/stand/carry/other), 1 sidle,
                     2 duck + step-over, 3 crawl (simulation only), 4 protect (reserved for own capture)
  context   [N,4]    overhead_clearance_m, lateral_gap_m, floor_obstacle_m, near_hand_object

FEATURES (all in the robot's HEADING frame = yaw removed, so they do not depend on where it faces):
  root height (1) | gravity in pelvis frame (3) | root linear velocity (3) | root angular velocity
  in pelvis frame (3) | joint positions (29) | joint velocities (29) | positions relative to the
  pelvis of head, 2 palms, 2 ankles, 2 elbows (21)  -> F = 89.
The same definition must be computed from simulator state during RL (see feature_names).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from motion_io import load_clip  # noqa: E402
from tag_clips import SKILLS, filtered_masks, kinematics, model, order_permutation  # noqa: E402

CONTROL_DT = 0.02
KEY_SITES = ("head", "left_palm", "right_palm")
KEY_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_elbow_link", "right_elbow_link")
GROUPS = ("locomotion", "sidle", "duck_step", "crawl", "protect")
COMMERCIAL_OK = {"phuma": True, "omniretarget": True}          # AMASS subsets are non-commercial


def feature_names():
    names = ["root_z"] + [f"gravity_{a}" for a in "xyz"] + [f"lin_vel_{a}" for a in "xyz"] + [f"ang_vel_{a}" for a in "xyz"]
    from tag_clips import UNITREE_G1_29
    names += [f"q_{j}" for j in UNITREE_G1_29] + [f"qd_{j}" for j in UNITREE_G1_29]
    names += [f"{k}_{a}" for k in KEY_SITES + KEY_BODIES for a in "xyz"]
    return names


def quat_mul(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def resample(clip, dt=CONTROL_DT):
    """Linear interpolation for positions/joints, normalised lerp (short arc) for the root quaternion."""
    fps = float(clip["fps"]); T = len(clip["dof_pos"])
    t_src = np.arange(T) / fps; t_new = np.arange(0.0, t_src[-1] + 1e-9, dt)
    i = np.clip(np.searchsorted(t_src, t_new, side="right") - 1, 0, T - 2); a = ((t_new - t_src[i]) * fps)[:, None]
    lerp = lambda x: x[i] * (1 - a) + x[i + 1] * a
    q0, q1 = clip["root_quat_wxyz"][i], clip["root_quat_wxyz"][i + 1]
    q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
    q = q0 * (1 - a) + q1 * a; q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return dict(fps=1.0 / dt, root_pos=lerp(clip["root_pos"]), root_quat_wxyz=q, dof_pos=lerp(clip["dof_pos"])), i, a[:, 0]


def compute_features(m, qpos, dt=CONTROL_DT):
    import mujoco
    d = mujoco.MjData(m); T = len(qpos)
    sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in KEY_SITES]
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in KEY_BODIES]
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    keys = np.zeros((T, 7, 3)); Rp = np.zeros((T, 3, 3))
    for t in range(T):
        d.qpos[:] = qpos[t]; mujoco.mj_kinematics(m, d)
        keys[t, :3] = d.site_xpos[sid]; keys[t, 3:] = d.xpos[bid]; Rp[t] = d.xmat[pelvis].reshape(3, 3)
    root = qpos[:, :3]; q = qpos[:, 3:7]
    yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    c, s = np.cos(yaw), np.sin(yaw)
    H = np.zeros((T, 3, 3)); H[:, 0, 0], H[:, 0, 1], H[:, 1, 0], H[:, 1, 1], H[:, 2, 2] = c, s, -s, c, 1   # world -> heading
    grad = lambda x: np.gradient(x, dt, axis=0)
    lin_vel = np.einsum("tij,tj->ti", H, grad(root))
    # angular velocity in the pelvis frame from successive orientations: w = vee(R^T dR/dt)
    dR = grad(Rp.reshape(T, 9)).reshape(T, 3, 3); W = np.einsum("tji,tjk->tik", Rp, dR)
    ang_vel = np.stack([W[:, 2, 1] - W[:, 1, 2], W[:, 0, 2] - W[:, 2, 0], W[:, 1, 0] - W[:, 0, 1]], -1) / 2
    gravity = np.einsum("tji,j->ti", Rp, np.array([0, 0, -1.0]))                                          # R^T g
    rel = np.einsum("tij,tkj->tki", H, keys - root[:, None])
    dof = qpos[:, 7:]
    return np.concatenate([root[:, 2:3], gravity, lin_vel, ang_vel, dof, grad(dof), rel.reshape(T, -1)], 1).astype(np.float32)


# Obstacle groups (sidle, duck/step, crawl) take frames ONLY from clips selected for obstacle
# skills (the AMASS filter), never from the everyday-style sources: there a fighting crouch or a
# kick trips the same detectors (measured: ~1/3 of the first duck/step group was fighting,
# kicks, jumps and box pick-ups).
OBSTACLE_LIBRARIES = {"amass_g1"}
EXCLUDE_FROM_OBSTACLE = __import__("re").compile(r"fight|combo|kick|punch|box(?!_)|martial|dance|jump|karate|boxing", __import__("re").I)


def group_of(masks, obstacle_eligible=True):
    g = np.zeros(len(masks["walk"]), np.int8)
    if not obstacle_eligible:
        return g
    g[masks["sidle"]] = 1
    g[masks["duck"] | masks["step_high"]] = 2
    g[masks["crawl"]] = 3
    return g


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libraries", nargs="+", default=["data/motion_library/raw", "data/motion_library/amass_g1"])
    p.add_argument("--out", default="data/motion_library/library_v1")
    p.add_argument("--max-seconds", type=float, default=30.0, help="split longer clips' tails off (keeps batches varied)")
    args = p.parse_args(argv)
    m, names = model(); perm = order_permutation(names)
    parts = {k: [] for k in ("qpos", "features", "next_ok", "clip", "skills", "group", "context")}
    table = []
    for lib in args.libraries:
        lib = Path(ROOT / lib); manifest = json.loads((lib / "manifest.json").read_text())
        for entry in manifest:
            clip = load_clip(lib / entry["path"])
            n_src = min(len(clip["dof_pos"]), int(args.max_seconds * float(clip["fps"])) + 1)
            clip = {k: (v[:n_src] if isinstance(v, np.ndarray) and v.ndim and len(v) > 2 else v) for k, v in clip.items()}
            if n_src < 3:
                continue
            kin = kinematics(clip, m, perm)
            masks_src = filtered_masks(clip, kin)
            rs, idx, a = resample(clip)
            near = np.where(a < 0.5, idx, idx + 1)                          # nearest source frame for labels
            masks = {k: v[near] for k, v in masks_src.items()}
            qpos = np.concatenate([rs["root_pos"], rs["root_quat_wxyz"], rs["dof_pos"][:, perm]], 1).astype(np.float32)
            T = len(qpos)
            head_like = np.full(T, 2.0); low = masks["duck"] | masks["crawl"]
            head_like[low] = np.minimum(1.9, qpos[low, 2] * 1.55 + 0.1)
            lift = np.abs(kin["foot_z"][:, 0] - kin["foot_z"][:, 1])[near]
            context = np.stack([head_like, np.where(masks["sidle"], 0.45, 2.0), np.where(masks["step_high"], lift, 0.0),
                                np.zeros(T)], 1).astype(np.float16)
            ci = len(table)
            source = str(clip.get("source", entry.get("source", "")))
            table.append(dict(clip_id=entry["clip_id"], library=lib.name, source=source, category=entry.get("category", ""),
                              licence=str(clip.get("licence", "")), commercial_ok=COMMERCIAL_OK.get(source, False),
                              frames=T, seconds=round(T * CONTROL_DT, 2),
                              skill_seconds={k: round(float(v.sum()) * CONTROL_DT, 2) for k, v in masks.items()}))
            parts["qpos"].append(qpos); parts["features"].append(compute_features(m, qpos).astype(np.float16))
            nx = np.ones(T, bool); nx[-1] = False; parts["next_ok"].append(nx)
            parts["clip"].append(np.full(T, ci, np.int32)); parts["skills"].append(np.stack([masks[k] for k in SKILLS], 1))
            eligible = lib.name in OBSTACLE_LIBRARIES and not EXCLUDE_FROM_OBSTACLE.search(entry["clip_id"] + " " + entry.get("source_member", ""))
            table[-1]["obstacle_eligible"] = bool(eligible)
            parts["group"].append(group_of(masks, eligible)); parts["context"].append(context)
            if len(table) % 250 == 0:
                print(f"{len(table)} clips", flush=True)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    out = ROOT / args.out
    np.savez(str(out) + ".npz", **arrays)
    summary = dict(clips=len(table), frames=int(len(arrays["qpos"])), hours=round(len(arrays["qpos"]) * CONTROL_DT / 3600, 2),
                   control_dt=CONTROL_DT, feature_names=feature_names(), skills=list(SKILLS), groups=list(GROUPS),
                   context=["overhead_clearance_m", "lateral_gap_m", "floor_obstacle_m", "near_hand_object"],
                   minutes_per_skill={k: round(float(arrays["skills"][:, i].sum()) * CONTROL_DT / 60, 1) for i, k in enumerate(SKILLS)},
                   minutes_per_group={g: round(float((arrays["group"] == i).sum()) * CONTROL_DT / 60, 1) for i, g in enumerate(GROUPS)},
                   commercial_ok_minutes=round(sum(c["seconds"] for c in table if c["commercial_ok"]) / 60, 1))
    Path(str(out) + ".json").write_text(json.dumps(dict(summary=summary, clips=table), indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "feature_names"}, indent=1))


if __name__ == "__main__":
    main()
