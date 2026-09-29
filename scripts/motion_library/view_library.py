#!/usr/bin/env python3
"""Render library clips on the G1 with the per-frame skill labels burned in, to check the tags.

  MUJOCO_GL=egl .venv-mjlab/bin/python scripts/motion_library/view_library.py --per-group 2 --out outputs/library_preview
  ... --clips amass_g1:108_21   (substring match on clip_id)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")


def render_clip(m, qpos, labels, title, out_path, fps=50, size=(640, 480)):
    import mujoco
    from PIL import Image, ImageDraw
    import imageio_ffmpeg
    r = mujoco.Renderer(m, height=size[1], width=size[0]); d = mujoco.MjData(m)
    cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 3.0, -15, 120
    writer = imageio_ffmpeg.write_frames(str(out_path), size, fps=fps // 2, codec="libx264", quality=7)
    writer.send(None)
    for t in range(0, len(qpos), 2):
        d.qpos[:] = qpos[t]; mujoco.mj_forward(m, d)
        cam.lookat[:] = [qpos[t, 0], qpos[t, 1], 0.7]
        r.update_scene(d, camera=cam)
        img = Image.fromarray(r.render()); draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, size[0], 44], fill=(245, 248, 250))
        draw.text((10, 6), title[:90], fill=(20, 40, 60))
        draw.text((10, 24), f"t={t / fps:5.2f}s  skills: {labels[t] or '-'}", fill=(170, 40, 30))
        writer.send(np.asarray(img, dtype=np.uint8).tobytes())
    writer.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library", default="data/motion_library/library_v1")
    p.add_argument("--out", default="outputs/library_preview")
    p.add_argument("--per-group", type=int, default=2)
    p.add_argument("--clips", nargs="*", default=[])
    p.add_argument("--max-seconds", type=float, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    import mujoco
    from cat_mjlab.model import assemble_training_xml
    m = mujoco.MjModel.from_xml_string(assemble_training_xml())
    L = np.load(ROOT / (args.library + ".npz")); meta = json.loads((ROOT / (args.library + ".json")).read_text())
    table, skills, groups = meta["clips"], meta["summary"]["skills"], meta["summary"]["groups"]
    clip_of = L["clip"]; starts = np.searchsorted(clip_of, np.arange(len(table))); ends = np.append(starts[1:], len(clip_of))
    rng = np.random.default_rng(args.seed); chosen = []
    if args.clips:
        chosen = [i for i, c in enumerate(table) if any(s.split(":")[-1] in c["clip_id"] for s in args.clips)]
    else:
        for g in range(len(groups)):
            cands = [i for i in range(len(table)) if (L["group"][starts[i]:ends[i]] == g).mean() > 0.3]
            chosen += list(rng.choice(cands, min(args.per_group, len(cands)), replace=False)) if cands else []
        cands = [i for i, c in enumerate(table) if c["skill_seconds"]["carry"] > 3]
        chosen += list(rng.choice(cands, min(args.per_group, len(cands)), replace=False))
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    for i in chosen:
        a, b = starts[i], min(ends[i], starts[i] + int(args.max_seconds * 50))
        sk = L["skills"][a:b]; labels = [",".join(s for s, on in zip(skills, row) if on) for row in sk]
        c = table[i]; present = np.bincount(L["group"][a:b], minlength=len(groups)); present[0] = 0
        dominant = groups[int(present.argmax())] if present.any() else groups[0]   # name by the obstacle skill it shows
        name = f"{dominant}__{c['library']}__{c['clip_id'][:40]}.mp4"
        render_clip(m, L["qpos"][a:b].astype(np.float64), labels, f"{c['source']} | {c['clip_id']} | group={dominant}", out / name)
        print("wrote", out / name, flush=True)


if __name__ == "__main__":
    main()
