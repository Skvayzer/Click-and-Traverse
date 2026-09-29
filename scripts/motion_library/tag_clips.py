#!/usr/bin/env python3
"""Tag library clips with skills and a context vector, using kinematics of THIS repository's G1 model.

Skills are detected from windows of motion, not from dataset labels (PHUMA has none that match ours):
  walk       upright, forward speed 0.3-1.6 m/s
  sidle      lateral speed > 0.2 m/s and > 1.5x forward speed, sustained >= 1 s
  duck       pelvis height < 0.80 x standing height while moving, sustained >= 0.5 s
  step_high  a foot lifted > 0.22 m above the lower foot while moving
  stand      root speed < 0.05 m/s for >= 1 s, upright
  carry      both hands in front of the pelvis (x > 0.20 m in the pelvis frame) and above the pelvis - 0.15 m
Context vector (the motion's implied situation), used by the conditioned discriminator:
  overhead_clearance_m  (duck: pelvis-derived head height + 0.1; else 2.0)
  lateral_gap_m         (sidle: 0.45; else 2.0)
  floor_obstacle_m      (step_high: max foot lift; else 0.0)
  near_hand_object      (0 for all public clips; set by own capture)
Also verifies that the clip's joint order matches the model (see --check-order).

Usage: python scripts/motion_library/tag_clips.py --library data/motion_library/raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_io import load_clip  # noqa: E402

UNITREE_G1_29 = [  # Unitree standard order used by PHUMA / OmniRetarget / GMR
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]


def model():
    import mujoco
    from cat_mjlab.model import assemble_training_xml
    m = mujoco.MjModel.from_xml_string(assemble_training_xml())
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)
             if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    return m, [n.removesuffix("_joint") for n in names]


def order_permutation(model_names):
    """dataset index for each model joint; asserts the name sets match."""
    assert sorted(model_names) == sorted(UNITREE_G1_29), f"joint name mismatch: {set(model_names) ^ set(UNITREE_G1_29)}"
    return np.array([UNITREE_G1_29.index(n) for n in model_names])


def runs(mask, min_len):
    """Total length of True runs that are at least min_len long."""
    total, cur = 0, 0
    for v in list(mask) + [False]:
        if v:
            cur += 1
        else:
            total += cur if cur >= min_len else 0
            cur = 0
    return total


SKILLS = ("walk", "sidle", "duck", "crawl", "step_high", "stand", "carry")
G1_STAND_PELVIS = 0.78          # nominal G1 pelvis height (m)


def kinematics(clip, m, perm):
    """Per-frame FK quantities from THIS repository's G1 model."""
    import mujoco
    d = mujoco.MjData(m)
    T = len(clip["dof_pos"])
    body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    site = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    feet = [body("left_ankle_roll_link"), body("right_ankle_roll_link")]
    palms = [site("left_palm"), site("right_palm")]
    pelvis, torso = body("pelvis"), body("torso_link")
    head = site("head")
    out = dict(foot_z=np.zeros((T, 2)), hand_local=np.zeros((T, 2, 3)), hand_z=np.zeros((T, 2)), torso_tilt=np.zeros(T),
               head_z=np.zeros(T))
    for t in range(T):
        d.qpos[:3] = clip["root_pos"][t]; d.qpos[3:7] = clip["root_quat_wxyz"][t]; d.qpos[7:36] = clip["dof_pos"][t][perm]
        mujoco.mj_kinematics(m, d)
        out["foot_z"][t] = d.xpos[feet, 2]
        R = d.xmat[pelvis].reshape(3, 3)
        out["hand_local"][t] = (d.site_xpos[palms] - d.xpos[pelvis]) @ R
        out["hand_z"][t] = d.site_xpos[palms, 2]
        out["head_z"][t] = d.site_xpos[head, 2]
        out["torso_tilt"][t] = np.degrees(np.arccos(np.clip(d.xmat[torso].reshape(3, 3)[2, 2], -1, 1)))
    return out


def frame_masks(clip, kin):
    """Per-frame skill masks (before minimum-duration filtering)."""
    fps = float(clip["fps"])
    q = clip["root_quat_wxyz"]
    yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    v = np.gradient(clip["root_pos"][:, :2], axis=0) * fps
    fwd = v[:, 0] * np.cos(yaw) + v[:, 1] * np.sin(yaw); lat = -v[:, 0] * np.sin(yaw) + v[:, 1] * np.cos(yaw)
    speed = np.hypot(fwd, lat); z = clip["root_pos"][:, 2]
    upright = (z > 0.85 * G1_STAND_PELVIS) & (kin["torso_tilt"] < 30)
    moving = speed > 0.15
    hands_down = (kin["hand_z"] < 0.25).any(-1)
    return dict(
        walk=(fwd > 0.3) & (fwd < 1.6) & upright,
        sidle=(np.abs(lat) > 0.2) & (np.abs(lat) > 1.5 * np.abs(fwd)) & (z > 0.75 * G1_STAND_PELVIS),
        # duck = travelling on the feet with the HEAD lowered (by bending and/or crouching), hands off the
        # floor. People duck under beams mostly by pitching the trunk, so pelvis height alone misses it.
        duck=(kin["head_z"] < 1.02) & (z > 0.40) & moving & ~hands_down,
        # crawl = travelling low with a hand on/near the floor or the torso pitched over
        crawl=(z < 0.60) & moving & (hands_down | (kin["torso_tilt"] > 50)),
        # step over = a foot lifted 0.22-0.55 m (a kick goes higher) while the BODY keeps travelling forward
        step_high=(np.abs(kin["foot_z"][:, 0] - kin["foot_z"][:, 1]) > 0.22) & (np.abs(kin["foot_z"][:, 0] - kin["foot_z"][:, 1]) < 0.55)
                  & (fwd > 0.2) & (z > 0.75 * G1_STAND_PELVIS),
        stand=(speed < 0.05) & upright,
        # carry = both hands in front of the body at chest/waist height, upright (not reaching to the floor)
        carry=(kin["hand_local"][:, :, 0] > 0.20).all(-1) & (kin["hand_local"][:, :, 2] > -0.15).all(-1)
              & (kin["hand_z"] > 0.60).all(-1) & upright,
    )


MIN_SECONDS = dict(walk=1.0, sidle=1.0, duck=0.5, crawl=1.0, step_high=0.2, stand=1.0, carry=1.0)


def filtered_masks(clip, kin):
    """Masks with runs shorter than the skill's minimum duration removed."""
    fps = float(clip["fps"]); out = {}
    for k, mk in frame_masks(clip, kin).items():
        keep = np.zeros_like(mk); n = max(1, int(MIN_SECONDS[k] * fps)); t = 0
        while t < len(mk):
            if mk[t]:
                u = t
                while u < len(mk) and mk[u]:
                    u += 1
                if u - t >= n:
                    keep[t:u] = True
                t = u
            else:
                t += 1
        out[k] = keep
    return out


def tag(clip, m, perm):
    kin = kinematics(clip, m, perm)
    masks = filtered_masks(clip, kin)
    fps = float(clip["fps"])
    seconds = {k: round(float(mk.sum()) / fps, 2) for k, mk in masks.items()}
    z = clip["root_pos"][:, 2]; lift = float(np.abs(kin["foot_z"][:, 0] - kin["foot_z"][:, 1]).max())
    context = dict(overhead_clearance_m=round(float(z[masks["duck"] | masks["crawl"]].min() * 1.55 + 0.1), 2)
                   if (masks["duck"] | masks["crawl"]).any() else 2.0,
                   lateral_gap_m=0.45 if masks["sidle"].any() else 2.0,
                   floor_obstacle_m=round(lift, 2) if masks["step_high"].any() else 0.0, near_hand_object=0)
    return seconds, context


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library", default="data/motion_library/raw")
    p.add_argument("--check-order", action="store_true", help="only verify joint order and exit")
    args = p.parse_args(argv)
    m, names = model()
    perm = order_permutation(names)
    print("model joint order matches dataset names; identity" if (perm == np.arange(29)).all() else f"permutation {perm.tolist()}")
    if args.check_order:
        return
    lib = Path(args.library); manifest = json.loads((lib / "manifest.json").read_text())
    for i, entry in enumerate(manifest):
        entry["skill_seconds"], entry["context"] = tag(load_clip(lib / entry["path"]), m, perm)
        entry["skills"] = [k for k, s in entry["skill_seconds"].items() if s > 0]
        if (i + 1) % 100 == 0:
            print(f"tagged {i + 1}/{len(manifest)}", flush=True)
    (lib / "manifest.json").write_text(json.dumps(manifest, indent=1))
    totals = {}
    for e in manifest:
        for k, s in e["skill_seconds"].items():
            totals[k] = totals.get(k, 0.) + s
    print(json.dumps({k: f"{v / 60:.1f} min" for k, v in totals.items()}, indent=1))


if __name__ == "__main__":
    main()
