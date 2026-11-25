#!/usr/bin/env python3
"""Videos and a contact sheet of COS scenes (docs/cos_obstacles.md, G2).

  video:  cos_render.py video --synth DIR --scene ID --out OUT_DIR [--name NAME]
  sheet:  cos_render.py sheet --synth DIR --out sheet.png [--n 24] [--seed 0]

Robot = the real motion M (opaque); translucent ghost = the default motion D(M), drawn red while it penetrates the
explaining obstacle; obstacle colours: explaining part orange, supports brown, distractors blue-grey, room walls light grey.
1280x720 H.264 yuv420p, faststart, plus a JPEG poster.
"""
from __future__ import annotations

import argparse, json, math, os, random, sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COL = dict(main=(0.95, 0.55, 0.15, 0.92), support=(0.55, 0.38, 0.22, 0.92), distractor=(0.45, 0.55, 0.7, 0.45),
           wall=(0.85, 0.85, 0.82, 0.18))
NAMES = dict(arms_in="arms in (sideways event)", duck="duck under", crawl="crawl", high_step="high step", sideways="turn sideways", arms_tucked="arms tucked",
             arms_raised="arms raised", detour="detour")


def load(synth, scene_id=None):
    S = [json.loads(l) for l in open(Path(synth) / "scenes.jsonl")]
    if scene_id:
        S = [s for s in S if s["scene"] == scene_id]
    return S


def setup(scene, synth, W, H):
    import mujoco
    from cat_mjlab.cos.render import model_with_boxes
    from cat_mjlab.cos.synth import boxes_arrays
    boxes = scene["boxes"]
    A = boxes_arrays(boxes)
    model = model_with_boxes(A, colors=[COL[b["role"]] for b in boxes], width=W, height=H)
    mot = np.load(Path(synth) / "motions" / f"{scene['window']}.npz")
    return model, mot["M"].astype(np.float64), mot["D"].astype(np.float64)


def d_touch(scene, D):
    """Per-frame: does D penetrate the main obstacle (exact geometry)?"""
    from cat_mjlab.exact_geometry import ExactGeometry
    from cat_mjlab.cos.synth import Body
    g = ExactGeometry()
    main = [b for b in scene["boxes"] if b["role"] == "main"]
    v, _ = Body(g, D).sep(main)
    vm, _ = Body(g, np.load if False else None) if False else (None, None)
    return v.min(1)


def m_clear(scene, M):
    from cat_mjlab.exact_geometry import ExactGeometry
    from cat_mjlab.cos.synth import Body
    g = ExactGeometry()
    v, _ = Body(g, M).sep(scene["boxes"])
    return v.min(1)


def camera_for(scene):
    fam = scene["family"]
    # side view perpendicular to the path for vertical families, oblique high view for lateral ones
    if fam in ("sideways", "arms_tucked", "detour", "arms_raised"):
        return dict(distance=4.2, elevation=-38., az_offset=-60.)
    return dict(distance=3.8, elevation=-16., az_offset=-55.)


def video(a):
    import mujoco
    from PIL import Image
    from cat_mjlab.cos.render import Writer, add_ghost, fonts, label
    scene = load(a.synth, a.scene)[0]
    W, H = 1280, 720
    model, M, D = setup(scene, a.synth, W, H)
    pen = d_touch(scene, D); clr = m_clear(scene, M)
    data, ghost = mujoco.MjData(model), mujoco.MjData(model)
    opt = mujoco.MjvOption(); opt.geomgroup[3:] = 0
    perturb = mujoco.MjvPerturb()
    cam = mujoco.MjvCamera(); c = camera_for(scene)
    cam.distance, cam.elevation, cam.azimuth = c["distance"], c["elevation"], c["az_offset"]
    font, big = fonts(19)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = a.name or scene["scene"]
    fps = 25; stride = 2
    frames = list(range(0, len(M), stride))
    # hold 1 s at the start and the end
    seq = [frames[0]] * fps + frames + [frames[-1]] * fps
    wr = Writer(out / f"{name}.mp4", W, H, fps)
    fa, fb = scene["event_frames"]; mid = (fa + fb) // 2
    poster = None
    look = M[0, :2].copy()
    kind = scene.get("kind") or "obstacle"
    title = f"COS: {NAMES[scene['family']]} -> {kind}  |  {scene['source']}  {scene['clip_id'][:48]}"
    with mujoco.Renderer(model, width=W, height=H) as r:
        for k, t in enumerate(seq):
            data.qpos[:] = M[t]; mujoco.mj_forward(model, data)
            look = 0.92 * look + 0.08 * M[t, :2] if k else M[t, :2].copy()
            cam.lookat[:] = [look[0], look[1], 0.6]
            r.update_scene(data, camera=cam, scene_option=opt)
            hit = pen[t] < 0
            add_ghost(r.scene, model, ghost, mujoco, opt, perturb, D[t],
                      (1.0, 0.15, 0.1, 0.45) if hit else (0.3, 0.6, 1.0, 0.22))
            img = r.render().copy()
            lines = [title, f"t = {t * 0.02:4.1f} s   real motion clearance {100 * clr[t]:5.1f} cm (min over scene "
                            f"{100 * scene['verify']['m_clear']:.1f} cm)",
                     "default motion (ghost): " + ("HITS the obstacle" if hit else "clear") +
                     f"   (max penetration {100 * -scene['verify']['d_main']:.1f} cm)"]
            img = label(img, lines, font)
            img = label(img, ["robot = real motion (AMASS/PHUMA on G1)   ghost = default upright straight walk, red while colliding",
                              "orange = explaining obstacle  brown = supports  blue-grey = distractors  grey = room walls"],
                        font, xy=(14, H - 70), fill=(255, 235, 170))
            wr.write(img)
            if poster is None and t >= mid:
                poster = img
    wr.close()
    Image.fromarray(poster).save(out / f"{name}.jpg", quality=90)
    print(out / f"{name}.mp4")


def sheet(a):
    import mujoco
    from PIL import Image
    from cat_mjlab.cos.render import add_ghost, fonts, label
    S = load(a.synth)
    S = [s for s in S if s["verify"]["ok"]]
    rng = random.Random(a.seed)
    pick = rng.sample(S, min(a.n, len(S)))
    W, H = 400, 300
    font, _ = fonts(12)
    tiles = []
    for s in pick:
        model, M, D = setup(s, a.synth, W, H)
        data, ghost = mujoco.MjData(model), mujoco.MjData(model)
        opt = mujoco.MjvOption(); opt.geomgroup[3:] = 0
        pert = mujoco.MjvPerturb()
        fa, fb = s["event_frames"]; t = (fa + fb) // 2
        cam = mujoco.MjvCamera(); c = camera_for(s)
        cam.distance, cam.elevation, cam.azimuth = c["distance"] - 0.6, c["elevation"], c["az_offset"]
        cam.lookat[:] = [M[t, 0], M[t, 1], 0.6]
        with mujoco.Renderer(model, width=W, height=H) as r:
            data.qpos[:] = M[t]; mujoco.mj_forward(model, data)
            r.update_scene(data, camera=cam, scene_option=opt)
            add_ghost(r.scene, model, ghost, mujoco, opt, pert, D[t], (1.0, 0.2, 0.1, 0.35))
            img = r.render().copy()
        tiles.append(label(img, [f"{NAMES[s['family']]} -> {s.get('kind')}", f"{s['source']} {s['clip_id'][:30]}"], font, xy=(6, 4)))
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    Image.fromarray(np.concatenate(rows, 0)).save(a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps([s["scene"] for s in pick]))
    print(a.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("video", "sheet"))
    p.add_argument("--synth", required=True); p.add_argument("--out", required=True)
    p.add_argument("--scene"); p.add_argument("--name"); p.add_argument("--n", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    (video if a.mode == "video" else sheet)(a)


if __name__ == "__main__":
    main()
