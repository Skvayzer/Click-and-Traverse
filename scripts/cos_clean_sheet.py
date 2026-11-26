#!/usr/bin/env python3
"""C0 contact sheet of removed windows (one tile per window at the frame that tripped its first removal reason).

  cluster: python scripts/cos_clean_sheet.py pick   --v2 /l/.../cos/scenes_v2 --v1 /l/.../cos/scenes_v1 --cache DIR --n 24
  local:   python scripts/cos_clean_sheet.py render --cache DIR --out sheet.png
"""
import argparse, collections, json, os, random, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "egl")


def pick(a):
    v2, v1, cache = Path(a.v2), Path(a.v1), Path(a.cache)
    (cache / "motions").mkdir(parents=True, exist_ok=True)
    rows = []
    for src in sorted(p for p in v2.iterdir() if (p / "window_flags.jsonl").exists()):
        F = [json.loads(l) for l in open(src / "window_flags.jsonl")]
        rows += [(src.name, f) for f in F if f["reasons"]]
    rng = random.Random(a.seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r[1]["reasons"][0]].append(r)
    # stratified: round-robin over first reasons (proportional floor of 2 each), then random fill
    out = []
    keys = sorted(by)
    for k in keys:
        rng.shuffle(by[k])
    while len(out) < a.n and any(by.values()):
        for k in keys:
            if by[k] and len(out) < a.n:
                out.append(by[k].pop())
    want = {f["window"] for _, f in out}
    scenes = {}
    for src in {s for s, _ in out}:
        for l in open(v1 / src / "scenes.jsonl"):
            s = json.loads(l)
            if s["window"] in want and s["window"] not in scenes:
                scenes[s["window"]] = s
    rec = []
    for src, f in out:
        shutil.copy(v1 / src / "motions" / f"{f['window']}.npz", cache / "motions" / f"{f['window']}.npz")
        rec.append(dict(flags=f, scene=scenes[f["window"]]))
    json.dump(rec, open(cache / "pick.json", "w"))
    print(len(rec), collections.Counter(r["flags"]["reasons"][0] for r in rec))


def render(a):
    import mujoco, torch
    from PIL import Image
    from cat_mjlab.cos.features import FrameFeatures, _extent
    from cat_mjlab.cos.render import fonts, label, model_with_boxes
    from cat_mjlab.cos.synth import boxes_arrays
    cache = Path(a.cache); rec = json.load(open(cache / "pick.json"))
    FF = FrameFeatures(); G = FF.g
    W, H = 400, 300; font, _ = fonts(12); tiles = []
    for r in rec:
        f, s = r["flags"], r["scene"]
        M = np.load(cache / "motions" / f"{s['window']}.npz")["M"].astype(np.float64)
        q = torch.as_tensor(M, dtype=torch.float32)
        world, xpos = G.primitives(q); lo = _extent(world, G.compiled)[0].numpy(); xpos = xpos.numpy()
        blo = lo[:, :, 2].min(1); fmin = lo[:, FF.left_foot + FF.right_foot, 2].min(1); floor = f["floor"]
        pel = xpos[:, G.body["pelvis"]]
        why = f["reasons"][0]
        if why in ("flight", "floating", "uneven_floor"):
            t = int(np.argmax(blo)); val = f"lowest body point {100 * (blo[t] - floor):.0f} cm above floor"
            if why != "uneven_floor":
                val += f", airborne {f['flight_frames'] * 0.02:.2f} s"
        elif why == "foot_penetration":
            t = int(np.argmin(fmin)); val = f"foot {100 * (floor - fmin[t]):.1f} cm below window floor"
        elif why == "pelvis_vz":
            vz = np.abs(np.gradient(pel[:, 2], 0.02)); t = int(np.argmax(vz)); val = f"pelvis vz {f['max_vz']:.1f} m/s"
        elif why == "acrobatic":
            t = int(np.argmin(np.asarray(G.fk(q)[1][:, G.body['pelvis'], 2, 2])))
            val = f"pelvis up-z {f['min_upz']:.2f}, roll-z {f['max_latz']:.2f}, ang {f['max_angvel']:.1f} rad/s"
        else:
            ef = s["event_frames"][0]; t = (ef[0] + ef[1]) // 2; val = "name keyword"
        boxes = s["boxes"]
        model = model_with_boxes(boxes_arrays(boxes), colors=[(0.8, 0.5, 0.2, 0.5)] * len(boxes), width=W, height=H)
        data = mujoco.MjData(model)
        data.qpos[:] = M[t]; mujoco.mj_forward(model, data)
        cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 3.0, -8, 90
        cam.lookat[:] = [M[t, 0], M[t, 1], 0.55]
        opt = mujoco.MjvOption(); opt.geomgroup[3:] = 0
        with mujoco.Renderer(model, width=W, height=H) as rr:
            rr.update_scene(data, camera=cam, scene_option=opt); img = rr.render().copy()
        tiles.append(label(img, ["+".join(f["reasons"])[:50], f"{s['source']} {s['clip'][-34:]}", val[:56],
                                 f"t={t * 0.02:.1f}s family {s['family']}"], font, xy=(6, 4)))
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    Image.fromarray(np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)).save(a.out)
    print(a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=("pick", "render"))
    p.add_argument("--v2"); p.add_argument("--v1"); p.add_argument("--cache", required=True); p.add_argument("--out")
    p.add_argument("--n", type=int, default=24); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); (pick if a.mode == "pick" else render)(a)
