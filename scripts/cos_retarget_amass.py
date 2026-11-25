#!/usr/bin/env python3
"""Retarget every AMASS SMPL-X sequence of the extracted raw subsets to the Unitree G1 with GMR (docs/cos_obstacles.md).

Same per-clip procedure as the September library (Click-and-Traverse-Mjlab scripts/motion_library/retarget_amass.py,
commit 1cfc154): GMR src_human="smplx", tgt_robot="unitree_g1", 30 fps, lowest body point over the clip on the floor,
xy origin at the first frame. Differences: all *_stageii.npz files (not the September filter), deterministic output
names (<subset>/<relative path with / -> __>.npz), sharding across jobs (--shard k/n), one manifest line per clip.

  PYTHONPATH=$T/pydeps:$T/GMR python scripts/cos_retarget_amass.py --raw EXTRACTED --out OUT --body-models $T/body_models \
      --subsets CMU KIT BMLmovi SFU --workers 12 --shard 0/2
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

LICENCE = "AMASS/{} (non-commercial research licence)"
G1_29 = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
         "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
         "waist_yaw", "waist_roll", "waist_pitch",
         "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
         "left_wrist_pitch", "left_wrist_yaw",
         "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
         "right_wrist_pitch", "right_wrist_yaw"]


def gmr_model():
    import mujoco
    from general_motion_retargeting.params import ROBOT_XML_DICT
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).removesuffix("_joint")
             for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    return m, names


def floor_offset(model, qpos):
    import mujoco
    d = mujoco.MjData(model)
    lowest = np.inf
    for q in qpos[::3]:
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        lowest = min(lowest, float(d.xpos[1:, 2].min()))
    return lowest


def work(job):
    src, subset, rel, out, body_models = job
    target = Path(out) / subset / (rel.replace("/", "__").removesuffix(".npz") + ".npz")
    if target.exists():
        return dict(status="exists", path=str(target), member=rel, subset=subset)
    try:
        from general_motion_retargeting import GeneralMotionRetargeting as GMR
        from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast, load_smplx_file
        data, body_model, smpl_out, height = load_smplx_file(src, body_models)
        frames, fps = get_smplx_data_offline_fast(data, body_model, smpl_out, tgt_fps=30)
        if len(frames) < 30:
            return dict(status="short", member=rel, subset=subset)
        r = GMR(src_human="smplx", tgt_robot="unitree_g1", actual_human_height=height, verbose=False)
        qpos = np.array([r.retarget(fr).copy() for fr in frames])
        model, _ = gmr_model()
        qpos[:, 2] -= floor_offset(model, qpos)
        qpos[:, :2] -= qpos[0, :2]
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.stem + ".tmp.npz")
        np.savez_compressed(tmp, fps=np.float32(fps), root_pos=qpos[:, :3].astype(np.float32),
                            root_quat_wxyz=qpos[:, 3:7].astype(np.float32), dof_pos=qpos[:, 7:36].astype(np.float32),
                            source=f"amass_{subset}", source_member=f"{subset}/{rel}", licence=LICENCE.format(subset),
                            retarget="GMR smplx->unitree_g1 (cos_retarget_amass.py)")
        tmp.rename(target)
        return dict(status="ok", path=str(target), member=rel, subset=subset, frames=len(qpos), fps=float(fps),
                    seconds=round(len(qpos) / float(fps), 2), height=float(height))
    except Exception as e:  # noqa: BLE001 - keep going, report
        return dict(status="error", member=rel, subset=subset, error=f"{type(e).__name__}: {e}"[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder with the extracted subsets <raw>/<subset>/...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--body-models", required=True, help="folder containing smplx/SMPLX_*.npz")
    ap.add_argument("--subsets", nargs="+", default=["SFU", "BMLmovi", "CMU", "KIT"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    _, names = gmr_model()
    assert names == G1_29, f"GMR G1 joint order differs from the library order: {names}"
    k, n = map(int, a.shard.split("/"))
    jobs = []
    for s in a.subsets:
        files = sorted(Path(a.raw, s).rglob("*_stageii.npz"))
        jobs += [(str(f), s, str(f.relative_to(Path(a.raw, s))), a.out, a.body_models) for f in files]
    jobs = jobs[k::n][:a.limit or None]
    print(f"{len(jobs)} sequences in shard {a.shard}, {a.workers} workers", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    log = open(Path(a.out) / f"manifest_shard{k}of{n}.jsonl", "a")
    t, c = time.time(), {}
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=25) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            c[r["status"]] = c.get(r["status"], 0) + 1
            log.write(json.dumps(r) + "\n"); log.flush()
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(jobs)} {c} {time.time() - t:.0f}s", flush=True)
    print("RETARGET DONE", json.dumps(c), f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
