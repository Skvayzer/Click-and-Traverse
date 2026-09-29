#!/usr/bin/env python3
"""Filter AMASS SMPL-X archives ON THE LAPTOP before copying (only numpy needed).

Streams through the downloaded .tar.bz2 archives (or extracted folders) without unpacking them
to disk, keeps only clips that are useful for obstacle skills, and writes ONE small tar plus a
CSV report. A clip is kept if ANY rule fires:

  1. whitelist   CMU subjects 107, 108, 127 (duck / stoop / crawl under), 141, 143 (walk sideways)
  2. keyword     file path contains a behaviour word (obstacle, side, duck, crouch, crawl, ...)
  3. low body    pelvis below 0.65 m for >= 0.5 s  (ducking, crouching, crawling)
  4. sidestep    lateral speed > 0.2 m/s and > 1.5x forward speed for >= 1 s
  5. high step   pelvis vertical excursion suggests stepping over (> 0.12 m bounce while walking)

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
    return dict(
        seconds=round(T / fps, 2), fps=fps, stand_z=round(float(stand), 3), min_z=round(float(z.min()), 3),
        low_body=longest_run(z < 0.65) >= s(0.5),
        crawl=longest_run(z < 0.45) >= s(0.5),
        sidestep=longest_run((np.abs(vl) > 0.2) & (np.abs(vl) > 1.5 * np.abs(vf))) >= s(1.0),
        high_step=bool(moving.mean() > 0.3 and (np.percentile(z[moving], 98) - np.percentile(z[moving], 2) > 0.12)) if moving.any() else False,
    )


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
            for key in ("crawl", "low_body", "sidestep", "high_step"):
                if stats[key]:
                    reasons.append(key)
            keep = bool(reasons)
            writer.writerow([name, int(keep), "+".join(reasons), stats["seconds"], stats["stand_z"], stats["min_z"]])
            if keep:
                kept += 1; kept_sec += stats["seconds"]; kept_bytes += len(raw)
                if out_tar is not None:
                    info = tarfile.TarInfo(name); info.size = len(raw)
                    out_tar.addfile(info, io.BytesIO(raw))
            if total % 500 == 0:
                print(f"{total} clips scanned, {kept} kept ({kept_sec / 60:.1f} min, {kept_bytes / 1e6:.0f} MB)  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    if out_tar is not None:
        out_tar.close()
    report.close()
    print(f"done: {total} clips scanned, {kept} kept = {kept_sec / 60:.1f} min, {kept_bytes / 1e6:.0f} MB -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
