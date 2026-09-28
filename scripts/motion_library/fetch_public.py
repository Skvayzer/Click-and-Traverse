#!/usr/bin/env python3
"""Stage 1 of the motion-prior pipeline: pull a FILTERED subset of the public G1-native motion
datasets without downloading whole archives (disk on this machine is ~7 GB free).

Zip archives on Hugging Face are read with HTTP range requests (fsspec random access + zipfile),
so only the central directory and the selected members are transferred.

Sources (licences: PHUMA Apache-2.0, OmniRetarget MIT):
  * DAVIAN-Robotics/PHUMA          ~70 h G1 29-DoF locomotion, 30 fps
  * omniretarget/OmniRetarget_Dataset  4 h G1 qpos (box carrying = arms-in-front posture)

Usage:
  python scripts/motion_library/fetch_public.py list                 # show archive layout, no data
  python scripts/motion_library/fetch_public.py fetch --out data/motion_library/raw
Every clip is written as one small .npz in a canonical layout (see motion_io.save_clip).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_io import save_clip, summarize_clip  # noqa: E402

HF = "https://huggingface.co"
# Everyday / scene-interaction motion; excluded: fitness, dance, music, kungfu, aist, perform, haa500.
PHUMA_QUOTA = dict(humanml=1200, idea400=500, EgoBody=400, game_motion=300, custom=200, humman=150, GRAB=100)
SOURCES = {
    "phuma": "DAVIAN-Robotics/PHUMA",
    "omniretarget": "omniretarget/OmniRetarget_Dataset",
}


def list_repo(repo: str) -> list[dict]:
    """Recursive file listing of a HF dataset repo (no huggingface_hub dependency)."""
    out, stack = [], [""]
    while stack:
        path = stack.pop()
        url = f"{HF}/api/datasets/{repo}/tree/main" + (f"/{path}" if path else "")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        for item in r.json():
            if item["type"] == "directory":
                stack.append(item["path"])
            else:
                out.append(dict(path=item["path"], size=item.get("size", 0)))
    return out


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests with a small block cache."""

    def __init__(self, url: str, block: int = 1 << 18):
        self.session = requests.Session()
        self.origin = url
        self._resolve()
        self.pos, self.block, self.cache = 0, block, {}

    def _resolve(self):
        # The CDN hands out signed URLs that expire (~1 h); re-resolve from the stable origin URL.
        head = self.session.head(self.origin, allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.url, self.size = head.url, int(head.headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            start = i * self.block; end = min(start + self.block, self.size) - 1
            for attempt in range(6):
                try:
                    r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                    if r.status_code in (401, 403):      # signed URL expired
                        self._resolve(); continue
                    r.raise_for_status(); break
                except requests.RequestException:
                    if attempt == 5: raise
            self.cache[i] = r.content
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos)); out = bytearray()
        while n > 0:
            i, off = divmod(self.pos, self.block); chunk = self._block(i)[off:off + n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data; return len(data)


def open_remote_zip(repo: str, path: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(f"{HF}/datasets/{repo}/resolve/main/{path}"), buffer_size=1 << 20))


def cmd_list(args):
    for name, repo in SOURCES.items():
        files = list_repo(repo)
        print(f"== {name} ({repo}): {len(files)} files")
        for f in files:
            print(f"   {f['size'] / 1e6:9.1f} MB  {f['path']}")
            if f["path"].endswith(".zip") and args.members:
                z = open_remote_zip(repo, f["path"])
                names = z.namelist()
                exts = {}
                for n in names:
                    exts[Path(n).suffix] = exts.get(Path(n).suffix, 0) + 1
                tops = sorted({n.split("/")[1] if n.count("/") > 1 else n.split("/")[0] for n in names})[:40]
                print(f"        members {len(names)}  by ext {exts}")
                print(f"        second-level dirs/names (first 40): {tops}")
                for n in names[:5]:
                    print(f"        e.g. {n}")


# ---------------------------------------------------------------- parsers (verified on fetch)

def parse_member(source: str, name: str, raw: bytes) -> dict | None:
    """Return canonical arrays or None. Layouts per the dataset cards; asserted, not assumed."""
    if name.endswith(".npz"):
        data = dict(np.load(io.BytesIO(raw), allow_pickle=False))
    elif name.endswith(".npy"):
        obj = np.load(io.BytesIO(raw), allow_pickle=True)
        data = obj.item() if obj.dtype == object else {"array": obj}
    else:
        return None
    if "qpos" in data:                                   # OmniRetarget: [qw qx qy qz x y z] + 29 (+7 object)
        q = np.asarray(data["qpos"], dtype=np.float32)
        assert q.ndim == 2 and q.shape[1] in (36, 43), q.shape
        return dict(fps=float(data["fps"]), root_quat_wxyz=q[:, 0:4], root_pos=q[:, 4:7], dof_pos=q[:, 7:36])
    if {"root_trans", "root_ori", "dof_pos"} <= set(data):  # PHUMA
        ori = np.asarray(data["root_ori"], dtype=np.float32)
        assert ori.shape[1] == 4, f"PHUMA root_ori shape {ori.shape}"
        dof = np.asarray(data["dof_pos"], dtype=np.float32)
        assert dof.shape[1] == 29, dof.shape
        # PHUMA documents IsaacGym-style xyzw. Upright humans have |w| ~ 1 and |x|,|y| small, so the
        # scalar part is the component that stays largest in magnitude; check rather than trust.
        wxyz = ori[:, [3, 0, 1, 2]] if np.abs(ori[:, 3]).mean() >= np.abs(ori[:, 0]).mean() else ori
        return dict(fps=float(np.asarray(data.get("fps", 30.))), root_pos=np.asarray(data["root_trans"], np.float32),
                    root_quat_wxyz=wxyz, dof_pos=dof)
    return None


def dedupe_key(clip: dict) -> str:
    """OmniRetarget augments each source clip by translations/rotations; body motion is identical.
    Key on joint angles only (rounded), which augmentation does not change."""
    return hashlib.sha1(np.round(clip["dof_pos"][:: 5], 3).tobytes()).hexdigest()


def cmd_fetch(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = dict(phuma=args.phuma_max_clips, omniretarget=args.omni_max_clips)
    manifest = []
    for source, repo in SOURCES.items():
        seen, kept = set(), 0
        zips = [f for f in list_repo(repo) if f["path"].endswith(".zip")]
        if source == "omniretarget":           # only the carrying subset: arms held in front
            zips = [f for f in zips if f["path"] == "robot-object.zip"]
        rng = np.random.default_rng(0)
        for f in zips:
            z = open_remote_zip(repo, f["path"])
            names = [n for n in z.namelist() if n.endswith((".npz", ".npy"))]
            if source == "phuma":              # G1 only, everyday-motion categories, per-category quota
                names = [n for n in names if n.startswith("data/g1/") and n.split("/")[2] in PHUMA_QUOTA]
            rng.shuffle(names)                 # a uniform sample, not the first N of one category
            per_category = {}
            for n in names:
                if kept >= budget[source]:
                    break
                category = n.split("/")[2] if source == "phuma" else "carry"
                if source == "phuma" and per_category.get(category, 0) >= PHUMA_QUOTA[category]:
                    continue
                early = out / source / f"{source}__{Path(n).stem}.npz"
                if early.exists():                      # resume without re-downloading the member
                    from motion_io import load_clip
                    local = load_clip(early)
                    seen.add(dedupe_key(local)); per_category[category] = per_category.get(category, 0) + 1
                    manifest.append(dict(clip_id=early.stem, path=str(early.relative_to(out)), source=source,
                                         category=category, **summarize_clip(local)))
                    kept += 1
                    continue
                clip = parse_member(source, n, z.read(n))
                if clip is None:
                    continue
                duration = len(clip["dof_pos"]) / clip["fps"]
                if duration < args.min_seconds:
                    continue
                key = dedupe_key(clip)
                if key in seen:
                    continue
                seen.add(key)
                per_category[category] = per_category.get(category, 0) + 1
                clip_id = f"{source}__{Path(n).stem}"
                target = out / source / f"{clip_id}.npz"
                if target.exists():                     # resume: keep what an earlier run saved
                    manifest.append(dict(clip_id=clip_id, path=str(target.relative_to(out)), source=source,
                                         category=category, **summarize_clip(clip)))
                    kept += 1
                    continue
                path = save_clip(out / source / f"{clip_id}.npz", clip, source=source, source_member=f"{f['path']}::{n}",
                                 licence="Apache-2.0" if source == "phuma" else "MIT")
                manifest.append(dict(clip_id=clip_id, path=str(path.relative_to(out)), source=source,
                                     category=category, **summarize_clip(clip)))
                kept += 1
                if kept % 50 == 0:
                    print(f"{source}: {kept} clips", flush=True)
        print(f"{source}: kept {kept} unique clips", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(m["seconds"] for m in manifest)
    print(json.dumps(dict(clips=len(manifest), minutes=round(total / 60, 1), out=str(out)), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.add_argument("--members", action="store_true", help="also read zip central directories")
    s.set_defaults(func=cmd_list)
    s = sub.add_parser("fetch")
    s.add_argument("--out", default="data/motion_library/raw")
    s.add_argument("--phuma-max-clips", type=int, default=sum(PHUMA_QUOTA.values()), help="sampled per category quota")
    s.add_argument("--omni-max-clips", type=int, default=200, help="unique carry clips (after de-duplicating augmentations)")
    s.add_argument("--min-seconds", type=float, default=1.0, help="PHUMA ships ~1.6 s chunks")
    s.set_defaults(func=cmd_fetch)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
