#!/usr/bin/env python3
"""Unified 50 Hz G1 motion library from every source (docs/cos_obstacles.md, data preparation).

  python scripts/cos_library.py --kind gmr    --src $C/amass_gmr            --out $C/library/amass_gmr
  python scripts/cos_library.py --kind ember  --src $C/ember_g1/g1          --out $C/library/ember
  python scripts/cos_library.py --kind phuma  --src $C/phuma_raw/data.zip   --out $C/library/phuma
  python scripts/cos_library.py --kind lafan  --src $C/lafan1_g1/g1         --out $C/library/lafan1

Output: qpos_000.npy ... (float32 [frames, 36]: root xyz, root quaternion wxyz, 29 joints in the G1 MuJoCo order of
the training model), clips.jsonl (clip_id, source, subset, member, licence, shard, start, frames, seconds, path_m,
joint_limit_violation), summary.json. Every clip is resampled to 50 Hz (positions linear, quaternions nlerp), its xy
origin moved to the first frame, and checked: joint order, joint limits (share of frames outside the model's range by
more than 0.05 rad), finite values. PHUMA's 4 s chunks of one sequence are re-joined when they are continuous.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np

G1_29 = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
         "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
         "waist_yaw", "waist_roll", "waist_pitch",
         "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
         "left_wrist_pitch", "left_wrist_yaw",
         "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
         "right_wrist_pitch", "right_wrist_yaw"]
DT = 0.02


def model_limits():
    import mujoco
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cat_mjlab.model import assemble_training_xml
    m = mujoco.MjModel.from_xml_string(assemble_training_xml())
    names = [m.joint(i).name.removesuffix("_joint") for i in range(1, 30)]
    assert names == G1_29, names
    return m.jnt_range[1:30].copy()


def resample(fps, pos, quat, dof):
    T = len(pos)
    src = np.arange(T) / fps
    t = np.arange(0., src[-1] + 1e-9, DT)
    u = np.clip(t * fps, 0, T - 1); i = np.minimum(np.floor(u).astype(int), T - 2); f = (u - i)[:, None]
    lerp = lambda x: (1 - f) * x[i] + f * x[i + 1]
    q = quat.astype(np.float64).copy()
    for k in range(1, T):
        if (q[k] * q[k - 1]).sum() < 0:
            q[k] = -q[k]
    qq = lerp(q); qq /= np.linalg.norm(qq, axis=1, keepdims=True)
    out = np.concatenate([lerp(pos.astype(np.float64)), qq, lerp(dof.astype(np.float64))], 1)
    out[:, :2] -= out[0, :2]
    return out.astype(np.float32)


def xyzw_or_wxyz(q):
    """Upright humans: the scalar part has the largest mean magnitude. NOT used for ember/PHUMA: it fails for wxyz clips
    facing away from +x (|z| > |w|); the readers use each dataset's documented order instead."""
    return q[:, [3, 0, 1, 2]] if np.abs(q[:, 3]).mean() >= np.abs(q[:, 0]).mean() else q


# ---------------------------------------------------------------------------------------------- readers
def read_gmr(src):
    for f in sorted(Path(src).rglob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        yield dict(clip_id=f"gmr__{f.parent.name}__{f.stem}", subset=f.parent.name, member=str(z["source_member"]),
                   licence=str(z["licence"]), source=f"amass_gmr_{f.parent.name}", fps=float(z["fps"]),
                   pos=z["root_pos"], quat=z["root_quat_wxyz"], dof=z["dof_pos"])


def read_ember(src):
    """IsaacLab AMP npz: fps, dof_names, body_names, dof_positions, body_positions, body_rotations (wxyz)."""
    for f in sorted(Path(src).rglob("*.npz")):
        z = np.load(f, allow_pickle=True)
        dn = [str(x).removesuffix("_joint") for x in z["dof_names"]]
        bn = [str(x) for x in z["body_names"]]
        idx = [dn.index(j) for j in G1_29]
        root = bn.index("pelvis") if "pelvis" in bn else 0
        rel = f.relative_to(src)
        subset = rel.parts[0]
        yield dict(clip_id="ember__" + "__".join(rel.with_suffix("").parts), subset=subset, member=str(rel),
                   licence=f"AMASS/{subset} (non-commercial research licence); retargeting fleaven/ember-lab CC-BY-4.0",
                   source=f"amass_ember_{subset}", fps=float(z["fps"]), pos=z["body_positions"][:, root],
                   quat=np.asarray(z["body_rotations"][:, root]), dof=z["dof_positions"][:, idx])   # IsaacLab: wxyz


def read_phuma(src):
    """PHUMA data.zip, G1 folder (data/g1/<category>/<seq>_chunk_<k>.npy); continuous chunks are re-joined."""
    z = zipfile.ZipFile(src)
    names = sorted(n for n in z.namelist() if n.startswith("data/g1/") and n.endswith((".npy", ".npz")))
    groups = {}
    for n in names:
        key = re.sub(r"_chunk_\d+$", "", Path(n).stem)
        groups.setdefault((n.split("/")[2], key), []).append(n)

    def load(n):
        obj = np.load(io.BytesIO(z.read(n)), allow_pickle=True)
        d = obj.item() if obj.dtype == object else dict(obj)
        return (float(np.asarray(d.get("fps", 30.))), np.asarray(d["root_trans"], np.float32),
                np.asarray(d["root_ori"], np.float32)[:, [3, 0, 1, 2]], np.asarray(d["dof_pos"], np.float32))  # xyzw

    for (cat, key), ns in sorted(groups.items()):
        ns.sort(key=lambda n: int(re.search(r"_chunk_(\d+)", n).group(1)) if "_chunk_" in n else 0)
        parts = [load(n) for n in ns]
        runs = [[parts[0]]]
        for p in parts[1:]:
            a, b = runs[-1][-1], p
            cont = (np.abs(a[3][-1] - b[3][0]).mean() < 0.05 and np.linalg.norm(a[1][-1] - b[1][0]) < 0.1)
            if cont:
                runs[-1].append(p)
            else:
                runs.append([p])
        for k, r in enumerate(runs):
            yield dict(clip_id=f"phuma__{cat}__{key}" + (f"__r{k}" if len(runs) > 1 else ""), subset=cat,
                       member=";".join(ns), licence="PHUMA Apache-2.0 (source mocap licences apply)", source="phuma",
                       fps=r[0][0], pos=np.concatenate([x[1] for x in r]), quat=np.concatenate([x[2] for x in r]),
                       dof=np.concatenate([x[3] for x in r]))


def read_lafan(src):
    """Unitree LAFAN1 G1 retargeting CSV: 30 fps, root xyz, root quaternion xyzw, 29 joints (Unitree order)."""
    for f in sorted(Path(src).glob("*.csv")):
        a = np.loadtxt(f, delimiter=",", dtype=np.float64)
        assert a.shape[1] == 36, a.shape
        yield dict(clip_id=f"lafan1__{f.stem}", subset="lafan1", member=f.name,
                   licence="LAFAN1 CC BY-NC-ND 4.0 (Ubisoft La Forge); G1 retargeting by Unitree", source="lafan1",
                   fps=30., pos=a[:, :3], quat=a[:, [6, 3, 4, 5]], dof=a[:, 7:36])


def read_clips(src):
    """Any folder of library clips (fps, root_pos, root_quat_wxyz, dof_pos, source, source_member, licence), e.g. the
    September PHUMA subset (Click-and-Traverse-Mjlab data/motion_library/raw/phuma)."""
    for f in sorted(Path(src).rglob("*.npz")):
        z = np.load(f)
        source = str(z["source"])
        yield dict(clip_id=f"{source}__{f.stem}", subset=source, member=str(z["source_member"]), licence=str(z["licence"]),
                   source=source, fps=float(z["fps"]), pos=z["root_pos"], quat=z["root_quat_wxyz"], dof=z["dof_pos"])


READERS = dict(clips=read_clips, gmr=read_gmr, ember=read_ember, phuma=read_phuma, lafan=read_lafan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=READERS, required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-frames", type=int, default=2_000_000)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lim = model_limits()
    buf, shard, start, total = [], 0, 0, 0
    clips = open(out / "clips.jsonl", "w")
    stats = dict(clips=0, frames=0, rejected=0)

    def flush():
        nonlocal buf, shard, start
        if buf:
            np.save(out / f"qpos_{shard:03d}.npy", np.concatenate(buf))
            shard += 1; start = 0; buf = []

    for k, c in enumerate(READERS[a.kind](a.src)):
        if a.limit and k >= a.limit:
            break
        try:
            q = resample(c["fps"], c["pos"], c["quat"], c["dof"])
        except Exception as e:  # noqa: BLE001
            stats["rejected"] += 1; print("REJECT", c["clip_id"], e); continue
        if len(q) < 50 or not np.isfinite(q).all():
            stats["rejected"] += 1; continue
        viol = float(((q[:, 7:] < lim[:, 0] - 0.05) | (q[:, 7:] > lim[:, 1] + 0.05)).any(1).mean())
        rec = dict(clip_id=c["clip_id"], source=c["source"], subset=c["subset"], member=c["member"], licence=c["licence"],
                   shard=shard, start=start, frames=len(q), seconds=round(len(q) * DT, 2),
                   path_m=round(float(np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1).sum()), 2),
                   joint_limit_violation=round(viol, 4))
        clips.write(json.dumps(rec) + "\n")
        buf.append(q); start += len(q); total += len(q)
        stats["clips"] += 1; stats["frames"] += len(q)
        if start >= a.shard_frames:
            flush()
    flush()
    stats["hours"] = round(stats["frames"] * DT / 3600, 2)
    (out / "summary.json").write_text(json.dumps(dict(kind=a.kind, src=a.src, **stats), indent=1))
    print("LIBRARY DONE", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
