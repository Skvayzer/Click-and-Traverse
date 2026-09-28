"""Canonical clip format for the motion library.

One clip = one .npz:
  fps               float
  root_pos          [T,3]  world, z up, metres
  root_quat_wxyz    [T,4]
  dof_pos           [T,29] in the G1 joint order of this repository's model (verified by check_joint_order)
  source, source_member, licence   strings
Skill labels and context vectors are added later by tag_clips.py (stored in the library manifest,
not in the clip, so re-tagging never rewrites motion data).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_clip(path: Path, clip: dict, **meta) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fps=np.float32(clip["fps"]), root_pos=clip["root_pos"].astype(np.float32),
                        root_quat_wxyz=clip["root_quat_wxyz"].astype(np.float32),
                        dof_pos=clip["dof_pos"].astype(np.float32), **{k: np.str_(v) for k, v in meta.items()})
    return path


def load_clip(path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


def summarize_clip(clip: dict) -> dict:
    T = len(clip["dof_pos"])
    fps = float(clip["fps"])
    pos = clip["root_pos"]
    return dict(frames=int(T), fps=fps, seconds=round(T / fps, 2),
                path_m=round(float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum()), 2),
                root_z_min=round(float(pos[:, 2].min()), 3), root_z_mean=round(float(pos[:, 2].mean()), 3))
