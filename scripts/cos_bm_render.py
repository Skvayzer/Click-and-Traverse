#!/usr/bin/env python3
"""B4: videos of BlinkMimic rollouts from tools/cos_obstacle_eval.py (with or without the COS obstacles): the simulated robot
(solid) and the human reference (blue ghost), 1280x720 H.264 yuv420p faststart, 25 fps real time.

  python scripts/cos_bm_render.py --run B3_DIR --scenes test200.jsonl --pick SCENE_ID [...] --out DIR [--no-boxes] [--tag T]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--scenes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--pick", nargs="+", required=True); ap.add_argument("--no-boxes", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import mujoco
    from cat_mjlab.cos.render import Writer, add_ghost, fonts, label, model_with_boxes
    from cat_mjlab.cos.synth import boxes_arrays
    run = Path(a.run); R = np.load(run / "rollouts.npz"); per = {r["scene"]: r for r in json.load(open(run / "per_scene.json"))}
    names = list(R["scenes"]); rows = {json.loads(l)["scene"]: json.loads(l) for l in open(a.scenes) if l.strip()}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    W, H, fps = 1280, 720, 25
    font, _ = fonts(19)
    made = []
    for sid in a.pick:
        k = names.index(sid); r = rows[sid]; res = per[sid]
        s = next(json.loads(l) for l in open(Path(r["folder"]) / "scenes.jsonl") if json.loads(l)["scene"] == sid)
        ref = np.load(Path(r["folder"]) / "motions" / f"{s['window']}.npz")["M"].astype(np.float64)
        alive = np.flatnonzero(R["alive"][:, k]); q = R["qpos"][alive, k].astype(np.float64)
        boxes = [] if a.no_boxes else s["boxes"]
        col = {"main": (0.95, 0.55, 0.2, 0.55), "support": (0.6, 0.45, 0.3, 0.55), "distractor": (0.5, 0.6, 0.75, 0.45),
               "wall": (0.6, 0.6, 0.6, 0.25)}
        if boxes:
            model = model_with_boxes(boxes_arrays(boxes), colors=[col.get(b["role"], (0.6, 0.6, 0.6, 0.4)) for b in boxes], width=W, height=H)
        else:
            from cat_mjlab.cos.render import model_with_boxes as mwb
            model = mwb(boxes_arrays([dict(center=[0, 0, -5], half=[.01, .01, .01], yaw=0., role="wall")]), width=W, height=H)
        data, ghost = mujoco.MjData(model), mujoco.MjData(model)
        opt = mujoco.MjvOption(); opt.geomgroup[3:] = 0; pert = mujoco.MjvPerturb()
        cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 4.2, -18, 120
        outcome = ("fell / tracking failure" if res["failed"] else ("CONTACT" + (" (hand)" if res["hand_contact"] else "")
                   if res["contact"] and not a.no_boxes else "clean"))
        fam = r["family"]
        name = f"{a.tag}{'motion' if a.no_boxes else 'scene'}_{fam}_{'fail' if outcome != 'clean' else 'clean'}_{sid[-12:]}"
        wr = Writer(out / f"{name}.mp4", W, H, fps)
        frames = list(range(0, len(q), 2)); seq = [frames[0]] * fps + frames + [frames[-1]] * fps
        look = q[0, :2].copy(); poster = None
        title = (f"BlinkMimic tracker (trained from scratch on COS motions), held-out {'motion' if a.no_boxes else 'COS scene'}: {fam}")
        with mujoco.Renderer(model, width=W, height=H) as rr:
            for i, t in enumerate(seq):
                data.qpos[:] = q[t]; mujoco.mj_forward(model, data)
                look = 0.92 * look + 0.08 * q[t, :2] if i else look
                cam.lookat[:] = [look[0], look[1], 0.6]
                rr.update_scene(data, camera=cam, scene_option=opt)
                add_ghost(rr.scene, model, ghost, mujoco, opt, pert, ref[min(t, len(ref) - 1)], (0.3, 0.6, 1.0, 0.25))
                img = rr.render().copy()
                err = float(np.linalg.norm(q[t, :2] - ref[min(t, len(ref) - 1), :2]))
                img = label(img, [title, f"t = {t * 0.02:4.1f} s   pelvis offset from the reference {100 * err:4.1f} cm",
                                  f"episode: {outcome}"], font)
                img = label(img, ["solid: simulated robot    blue ghost: human reference (retargeted to G1)"], font,
                            xy=(14, H - 40), fill=(255, 235, 170))
                wr.write(img)
                if poster is None and t >= len(q) // 2:
                    poster = img
        wr.close()
        from PIL import Image
        Image.fromarray(poster).save(out / f"{name}.jpg", quality=88)
        made.append(dict(video=f"{name}.mp4", scene=sid, family=fam, outcome=outcome))
        print(name, outcome, flush=True)
    json.dump(made, open(out / f"index{('_' + a.tag) if a.tag else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
