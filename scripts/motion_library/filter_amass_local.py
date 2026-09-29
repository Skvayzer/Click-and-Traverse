#!/usr/bin/env python3
"""Filter AMASS SMPL-X archives ON THE LAPTOP before copying (only numpy needed).

Streams through the downloaded .tar.bz2 archives (or extracted folders) without unpacking them
to disk, keeps only clips that are useful for obstacle skills, and writes ONE small tar plus a
CSV report. A clip is kept if ANY rule fires:

  1. whitelist   CMU subjects 107, 108, 127 (duck / stoop / crawl under), 141, 143 (walk sideways)
  2. keyword     file path contains a behaviour word (obstacle, side, duck, crouch, crawl, ...)
  3. low body    pelvis 0.45-0.65 m for >= 0.5 s WHILE TRAVELLING (ducking, crouched walking);
                 below 0.45 m travelling >= 1 s = crawl. Sitting/lying in place are excluded.
  4. sidestep    upright, lateral speed > 0.25 m/s and > 2x forward speed for >= 1 s

Usage (macOS/Linux, python3 + numpy):
  python3 filter_amass_local.py --inputs ~/Downloads/amass/*.tar.bz2 --out ~/Downloads/amass_filtered.tar
  python3 filter_amass_local.py --inputs ~/Downloads/amass/*.tar.bz2 --dry-run      # report only
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

CMU_SUBJECTS = {"107", "108", "127", "141", "143"}
KEYWORDS = re.compile(r"obstacle|avoid|around|side|sidestep|lateral|duck|stoop|crouch|crawl|kneel|squat|"
                      r"step_?over|stepover|hurdle|jump_?over|climb|bend|dodge|squeez|narrow|tunnel|under",
                      re.IGNORECASE)


def rotmat(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues, batched [T,3] -> [T,3,3]."""
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True).clip(1e-8)
    k = axis_angle / theta
    K = np.zeros(axis_angle.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = np.sin(theta)[..., None], np.cos(theta)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def analyse(data) -> dict | None:
    if "trans" not in data:
        return None
    trans = np.asarray(data["trans"], dtype=np.float64)
    if "root_orient" in data:
        root = np.asarray(data["root_orient"], dtype=np.float64)
    elif "poses" in data:
        root = np.asarray(data["poses"], dtype=np.float64)[:, :3]
    else:
        return None
    fps = float(np.asarray(data.get("mocap_frame_rate", data.get("mocap_framerate", 120.))))
    T = len(trans)
    if T < fps * 0.5:
        return None
    R = rotmat(root)
    fwd = R[:, :, 2][:, :2]; lat = R[:, :, 0][:, :2]           # SMPL body: +z forward, +x left; AMASS world is z-up
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True).clip(1e-6); lat /= np.linalg.norm(lat, axis=1, keepdims=True).clip(1e-6)
    v = np.gradient(trans[:, :2], axis=0) * fps
    vf, vl = (v * fwd).sum(1), (v * lat).sum(1)
    z = trans[:, 2]
    stand = np.percentile(z, 95)
    moving = np.hypot(vf, vl) > 0.25
    s = lambda sec: max(1, int(sec * fps))
    # Low-body rules require TRAVELLING while low: sitting, lying and kneeling in place are not
    # ducking or crawling (first pass kept 281/1,864 BMLmovi clips, mostly sit/lie actions).
    travelling = np.hypot(vf, vl) > 0.2
    return dict(
        seconds=round(T / fps, 2), fps=fps, stand_z=round(float(stand), 3), min_z=round(float(z.min()), 3),
        low_body=longest_run((z < 0.65) & (z > 0.45) & travelling) >= s(0.5),
        crawl=longest_run((z < 0.45) & travelling) >= s(1.0),
        sidestep=longest_run((np.abs(vl) > 0.25) & (np.abs(vl) > 2.0 * np.abs(vf)) & (z > 0.8 * stand)) >= s(1.0),
    )


def slim_clip(raw: bytes, fps_out: float) -> bytes:
    """Keep only what G1 retargeting needs, downsampled: ~20x smaller than the AMASS file.
    (Drops markers, jaw/eye/hand pose, DMPLs; hands are fixed grippers on the G1.)"""
    with np.load(io.BytesIO(raw), allow_pickle=False) as d:
        fps = float(np.asarray(d["mocap_frame_rate"] if "mocap_frame_rate" in d.files else d["mocap_framerate"]))
        step = max(1, int(round(fps / fps_out)))
        root = d["root_orient"] if "root_orient" in d.files else d["poses"][:, :3]
        body = d["pose_body"] if "pose_body" in d.files else d["poses"][:, 3:66]
        out = dict(trans=d["trans"][::step].astype(np.float32), root_orient=root[::step].astype(np.float32),
                   pose_body=body[::step].astype(np.float32), mocap_frame_rate=np.float32(fps / step),
                   betas=d["betas"][:16].astype(np.float32) if "betas" in d.files else np.zeros(16, np.float32),
                   gender=d["gender"] if "gender" in d.files else np.str_("neutral"),
                   surface_model_type=d["surface_model_type"] if "surface_model_type" in d.files else np.str_("smplx"))
    buf = io.BytesIO(); np.savez_compressed(buf, **out); return buf.getvalue()


def iter_members(path: Path):
    """Yield (name, bytes) for every .npz in an archive or folder, streaming."""
    if path.is_dir():
        for p in sorted(path.rglob("*.npz")):
            yield str(p.relative_to(path.parent)), p.read_bytes()
        return
    with tarfile.open(path, mode="r|*") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".npz"):
                yield m.name, tar.extractfile(m).read()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help=".tar.bz2 archives or extracted folders")
    p.add_argument("--out", default="amass_filtered.tar")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fps", type=float, default=30., help="output frame rate (retargeting and the prior use 30 fps)")
    args = p.parse_args()
    out_tar = None if args.dry_run else tarfile.open(args.out, "w")
    report = open(Path(args.out).with_suffix(".csv") if not args.dry_run else "amass_filter_report.csv", "w", newline="")
    writer = csv.writer(report)
    writer.writerow(["member", "kept", "reasons", "seconds", "stand_z", "min_z"])
    kept = total = kept_bytes = 0; kept_sec = 0.; t0 = time.time()
    for archive in args.inputs:
        for name, raw in iter_members(Path(os.path.expanduser(archive))):
            total += 1
            parts = name.replace("\\", "/").split("/")
            try:
                with np.load(io.BytesIO(raw), allow_pickle=False) as d:
                    stats = analyse({k: d[k] for k in d.files if k in ("trans", "root_orient", "poses", "mocap_frame_rate", "mocap_framerate")})
            except Exception as e:                       # shape.npz / corrupt entries
                stats = None
            if stats is None:
                continue
            reasons = []
            if any(x.upper() == "CMU" for x in parts) and any(x.split("_")[0] in CMU_SUBJECTS for x in parts[-2:]):
                reasons.append("cmu_whitelist")
            if KEYWORDS.search(name):
                reasons.append("keyword")
            for key in ("crawl", "low_body", "sidestep"):
                if stats[key]:
                    reasons.append(key)
            keep = bool(reasons)
            writer.writerow([name, int(keep), "+".join(reasons), stats["seconds"], stats["stand_z"], stats["min_z"]])
            if keep:
                slim = slim_clip(raw, args.fps)
                kept += 1; kept_sec += stats["seconds"]; kept_bytes += len(slim)
                if out_tar is not None:
                    info = tarfile.TarInfo(name); info.size = len(slim)
                    out_tar.addfile(info, io.BytesIO(slim))
            if total % 500 == 0:
                print(f"{total} clips scanned, {kept} kept ({kept_sec / 60:.1f} min, {kept_bytes / 1e6:.0f} MB)  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    if out_tar is not None:
        out_tar.close()
    report.close()
    print(f"done: {total} clips scanned, {kept} kept = {kept_sec / 60:.1f} min, {kept_bytes / 1e6:.0f} MB -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
