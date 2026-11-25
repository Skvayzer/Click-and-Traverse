#!/usr/bin/env python3
"""Contact sheet of random G0 events: per family N events, 4 frames each (0.5 s before, start, middle, end), side view."""
from __future__ import annotations

import argparse, json, os, random
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


FAMS = ["duck", "crawl", "high_step", "sideways", "arms_raised", "arms_tucked", "detour"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True); ap.add_argument("--library", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--per-family", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--families", nargs="*"); ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    import mujoco
    from PIL import Image
    from cat_mjlab.cos.render import model_with_boxes, label, fonts
    E = json.loads(Path(a.events).read_text())
    q = np.load(a.library + ".npz", mmap_mode="r")["qpos"]
    W, H = 320, 360
    model = model_with_boxes(None, width=W, height=H)
    data = mujoco.MjData(model)
    opt = mujoco.MjvOption(); opt.geomgroup[3:] = 0
    font, _ = fonts(13)
    rng = random.Random(a.seed)
    rows, picked = [], []
    fams = a.families or FAMS
    with mujoco.Renderer(model, width=W, height=H) as r:
        for fam in fams:
            ev = [x for x in E["events"] if x["family"] == fam and (not a.clean or x["clean"])]
            for x in rng.sample(ev, min(a.per_family, len(ev))):
                picked.append(x)
                f0 = x["frame0"]; i0, i1 = int(x["t0"] / 0.02), int(x["t1"] / 0.02)
                n = int(x["clip_seconds"] / 0.02)
                ts = [max(0, i0 - 25), i0, (i0 + i1) // 2, min(i1, n - 1)]
                cam = mujoco.MjvCamera(); cam.distance, cam.elevation = 3.0, -12
                p0 = np.asarray(q[f0 + ts[0], :3]); p1 = np.asarray(q[f0 + ts[-1], :3])
                d = p1[:2] - p0[:2]
                cam.azimuth = float(np.degrees(np.arctan2(d[1], d[0]))) + 90 if np.linalg.norm(d) > 0.2 else 90
                tiles = []
                for k, t in enumerate(ts):
                    data.qpos[:] = q[f0 + t]; mujoco.mj_forward(model, data)
                    cam.lookat[:] = [*(0.5 * (p0[:2] + p1[:2])), 0.6]
                    r.update_scene(data, camera=cam, scene_option=opt)
                    img = r.render().copy()
                    txt = [f"{fam} {x['source']} {x['clip_id'][:34]}", f"t={t * 0.02:.1f}s mag={x['mag']}"] if k == 0 else [f"t={t * 0.02:.1f}s"]
                    tiles.append(label(img, txt, font, xy=(6, 4)))
                rows.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(picked, indent=1))


if __name__ == "__main__":
    main()
