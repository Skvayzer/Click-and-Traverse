#!/usr/bin/env python3
"""Retarget the filtered AMASS clips (SMPL-X, slim 30 fps tars) to the Unitree G1 with GMR.

Runs on CPU in a process pool. GMR and its extra deps live OUTSIDE the training venv
(~/robotics/third_party/{GMR,pydeps}) so the checkpoints' pinned package versions are untouched.

Output: one canonical clip per source clip in data/motion_library/amass_g1/<dataset>/, plus a
manifest.json compatible with tag_clips.py. Per clip:
  * GMR retarget at 30 fps (src_human="smplx", tgt_robot="unitree_g1")
  * height fix: lowest body point over the whole clip placed on the floor (GMR's own convention)
  * xy origin at the first frame
  * joint order checked once against the Unitree 29-DoF order used by the rest of the library

Usage:
  PYTHONPATH=~/robotics/third_party/pydeps:~/robotics/third_party/GMR \\
    .venv-mjlab/bin/python scripts/motion_library/retarget_amass.py --workers 16
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from motion_io import save_clip, summarize_clip  # noqa: E402
from tag_clips import UNITREE_G1_29  # noqa: E402

LICENCE = {"CMU": "AMASS/CMU (non-commercial)", "KIT": "AMASS/KIT (non-commercial)",
           "BMLmovi": "AMASS/BMLmovi (non-commercial)", "SFU": "AMASS/SFU (non-commercial)"}


def gmr_joint_order():
    import mujoco
    from general_motion_retargeting.params import ROBOT_XML_DICT
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).removesuffix("_joint")
             for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    return m, names


def floor_offset(model, qpos: np.ndarray) -> float:
    """Lowest geom-body point over the clip (body origins of the ankle-roll links and everything else)."""
    import mujoco
    d = mujoco.MjData(model)
    lowest = np.inf
    for q in qpos[:: 3]:
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        lowest = min(lowest, float(d.xpos[1:, 2].min()))
    return lowest


def work(job):
    name, raw, dataset, out_dir, body_models = job
    target = Path(out_dir) / dataset / (Path(name).stem.replace("_stageii", "") + "__" + str(abs(hash(name)) % 10**8) + ".npz")
    if target.exists():
        return dict(status="exists", name=name, path=str(target))
    try:
        from general_motion_retargeting import GeneralMotionRetargeting as GMR
        from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            f.write(raw); f.flush()
            data, body_model, out, height = load_smplx_file(f.name, body_models)
            frames, fps = get_smplx_data_offline_fast(data, body_model, out, tgt_fps=30)
        retargeter = GMR(src_human="smplx", tgt_robot="unitree_g1", actual_human_height=height, verbose=False)
        qpos = np.array([retargeter.retarget(fr).copy() for fr in frames])     # [T, 7+29], quat wxyz
        model, _ = gmr_joint_order()
        qpos[:, 2] -= floor_offset(model, qpos)
        qpos[:, :2] -= qpos[0, :2]
        clip = dict(fps=float(fps), root_pos=qpos[:, :3], root_quat_wxyz=qpos[:, 3:7], dof_pos=qpos[:, 7:36])
        save_clip(target, clip, source=f"amass_{dataset}", source_member=name, licence=LICENCE.get(dataset, "AMASS"))
        return dict(status="ok", name=name, path=str(target), **summarize_clip(clip))
    except Exception as e:  # keep going; report at the end
        return dict(status="error", name=name, error=f"{type(e).__name__}: {e}"[:300])


def jobs(amass_dir: Path, out_dir: Path, body_models: str):
    for tar_path in sorted(amass_dir.glob("*_filtered.tar")):
        dataset = tar_path.name.split("_filtered")[0]
        with tarfile.open(tar_path) as tar:
            for m in tar:
                if m.isfile() and m.name.endswith(".npz"):
                    yield (m.name, tar.extractfile(m).read(), dataset, str(out_dir), body_models)


def main(argv=None):
    root = HERE.parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amass", default=str(root / "data/motion_library/amass"))
    p.add_argument("--out", default=str(root / "data/motion_library/amass_g1"))
    p.add_argument("--body-models", default=str(root / "data/body_models"), help="folder containing smplx/SMPLX_*.npz")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke test)")
    args = p.parse_args(argv)

    _, names = gmr_joint_order()
    assert names == UNITREE_G1_29, f"GMR G1 joint order differs from the library order: {names}"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    all_jobs = list(jobs(Path(args.amass), out, args.body_models))
    if args.limit:
        all_jobs = all_jobs[: args.limit]
    print(f"{len(all_jobs)} clips, {args.workers} workers", flush=True)
    results, t0 = [], time.time()
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=20) as pool:
        for i, r in enumerate(pool.imap_unordered(work, all_jobs, chunksize=1)):
            results.append(r)
            if (i + 1) % 25 == 0 or i + 1 == len(all_jobs):
                bad = sum(x["status"] == "error" for x in results)
                print(f"{i + 1}/{len(all_jobs)} done, {bad} errors [{time.time() - t0:.0f}s]", flush=True)
    manifest = [dict(clip_id=Path(r["path"]).stem, path=str(Path(r["path"]).relative_to(out)),
                     source="amass", category=Path(r["path"]).parent.name, source_member=r["name"],
                     **{k: r[k] for k in ("frames", "fps", "seconds", "path_m", "root_z_min", "root_z_mean") if k in r})
                for r in results if r["status"] == "ok"]
    old = out / "manifest.json"
    if old.exists():   # merge with an earlier (partial) run
        known = {e["path"] for e in manifest}
        manifest += [e for e in json.loads(old.read_text()) if e["path"] not in known]
    old.write_text(json.dumps(manifest, indent=1))
    errors = [r for r in results if r["status"] == "error"]
    (out / "errors.json").write_text(json.dumps(errors, indent=1))
    print(json.dumps(dict(ok=sum(r["status"] == "ok" for r in results), exists=sum(r["status"] == "exists" for r in results),
                          errors=len(errors), minutes=round(sum(e.get("seconds", 0) for e in manifest) / 60, 1)), indent=1))
    for e in errors[:5]:
        print("ERROR", e["name"], e["error"])


if __name__ == "__main__":
    main()
