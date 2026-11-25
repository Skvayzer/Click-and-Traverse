#!/usr/bin/env python3
"""G1 feasibility flags at scale: frozen SONIC tracks motion windows in the lightweight harness
(cat_mjlab/cos/lite_harness.py), many processes in parallel (docs/cos_obstacles.md).

  python scripts/cos_feasibility.py --windows W.jsonl --sonic DIR --out OUT.jsonl [--workers 12] [--save-dir D]
  python scripts/cos_feasibility.py --post-full DIR --out OUT.jsonl     # metrics of full-harness runs (validation)

Window lines: {"id", "qpos_file", "a", "b"} (frames [a, b) of a 50 Hz qpos .npy). Metrics: end (complete / fell /
sank), joint error p50/p90 (rad), local body error p50/p90 (proxy primitive centres in the pelvis-heading frame, m),
heading drift at the end (deg), root drift per second, and tracked := complete and joint p90 <= 0.15 and local p90 <= 0.10
and |heading drift| <= 30 (pre-registered G1 rule).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
G = None


def yaw_np(q):
    w, x, y, z = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def post_metrics(tracked, ref, end):
    """tracked/ref [n, 36] aligned frame by frame."""
    global G
    if G is None:
        from cat_mjlab.exact_geometry import ExactGeometry
        G = ExactGeometry()
    n = min(len(tracked), len(ref))
    tr, rf = tracked[:n].astype(np.float64), ref[:n].astype(np.float64)
    je = np.abs(tr[:, 7:] - rf[:, 7:]).mean(1)

    def local(q):
        w, _ = G.primitives(torch.as_tensor(q, dtype=torch.float32))
        c = w["centers"].numpy() - q[:, None, :3]
        yw = yaw_np(q)
        cs, sn = np.cos(-yw)[:, None], np.sin(-yw)[:, None]
        return np.stack([cs * c[..., 0] - sn * c[..., 1], sn * c[..., 0] + cs * c[..., 1], c[..., 2]], -1)

    le = np.linalg.norm(local(tr) - local(rf), axis=-1).mean(1)
    dyaw = math.degrees(float((yaw_np(tr[-1]) - yaw_np(rf[-1]) + np.pi) % (2 * np.pi) - np.pi))
    root = np.linalg.norm(tr[:, :2] - rf[:, :2], axis=1)
    m = dict(end=end, seconds=round(n * .02, 2), joint_err_p50=float(np.median(je)), joint_err_p90=float(np.percentile(je, 90)),
             local_p50=float(np.median(le)), local_p90=float(np.percentile(le, 90)), yaw_drift_deg=round(dyaw, 1),
             root_drift_per_s=float(root[-1] / max(n * .02, .02)))
    m["tracked"] = bool(end == "complete" and m["joint_err_p90"] <= 0.15 and m["local_p90"] <= 0.10 and abs(dyaw) <= 30)
    return m


ARGS = LITE = TRACKER = HOLD = None


def init(a):
    global ARGS, LITE, TRACKER, HOLD
    torch.set_num_threads(1)
    from cat_mjlab.cos.lite_harness import LiteTask, make_tracker
    ARGS = a
    LITE = LiteTask(seed=0)
    HOLD = {}
    TRACKER = make_tracker(LITE, Path(a.sonic) / "model_encoder.onnx", Path(a.sonic) / "model_decoder.onnx",
                           reference=lambda: HOLD["ref"]())
    for s in (TRACKER.encoder, TRACKER.decoder):
        pass


def track(clip):
    """SONIC follows clip [T, 36] (50 Hz) open-loop in time from its first frame."""
    clip = torch.as_tensor(clip, dtype=torch.float32)
    last = len(clip) - 1
    jv = torch.zeros_like(clip[:, 7:]); jv[1:-1] = (clip[2:, 7:] - clip[:-2, 7:]) / .04
    v0 = torch.zeros(35); v0[:3] = (clip[1, :3] - clip[0, :3]) / .02; v0[6:] = jv[0]
    k = {"k": 0}

    def reference():
        idx = (k["k"] + torch.arange(10) * 5).clamp_max(last)
        return clip[idx][None], jv[idx][None]

    HOLD["ref"] = reference
    LITE.reset(clip[0].numpy(), v0.numpy())
    qs, end = [LITE.data.qpos.copy()], "complete"
    for step in range(last):
        k["k"] = step
        action, target = TRACKER()
        LITE.step(action, position_targets=target)
        q = LITE.data.qpos.copy(); ref = clip[step + 1]
        qs.append(q)
        if float(ref[2]) > .6 and q[2] < .5:
            end = "fell"; break
        if q[2] - float(ref[2]) < -.25:
            end = "sank"; break
    return np.array(qs, np.float32), end


def one(w):
    t = time.time()
    try:
        q = np.load(w["qpos_file"], mmap_mode="r")[w["a"]:w["b"]]
        tracked, end = track(np.asarray(q))
        m = post_metrics(tracked, np.asarray(q), end)
        if ARGS.save_dir:
            np.savez_compressed(Path(ARGS.save_dir) / f"{w['id']}.npz", tracked=tracked)
    except Exception as e:  # noqa: BLE001
        m = dict(end="error", error=repr(e)[:300], tracked=False)
    m.update(id=w["id"], wall_s=round(time.time() - t, 2))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows"); ap.add_argument("--sonic"); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--save-dir")
    ap.add_argument("--post-full", help="directory of full-harness outputs (cos_g1_track.py: <id>.json + <id>.npz)")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.post_full:
        with open(a.out, "w") as f:
            for j in sorted(Path(a.post_full).glob("*.json")):
                r = json.loads(j.read_text()); z = Path(a.post_full) / f"{j.stem}.npz"
                if not z.exists():
                    continue
                d = np.load(z)
                m = post_metrics(d["tracked"], d["ref"], r["end"]); m.update(id=j.stem)
                f.write(json.dumps(m) + "\n")
        return
    W = [json.loads(l) for l in open(a.windows)][:a.limit]
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["id"] for l in open(a.out)}
    W = [w for w in W if w["id"] not in done]
    if a.save_dir:
        Path(a.save_dir).mkdir(parents=True, exist_ok=True)
    print(f"{len(W)} windows, {a.workers} workers", flush=True)
    import multiprocessing as mp
    t = time.time(); c = {}
    with mp.get_context("fork").Pool(a.workers, initializer=init, initargs=(a,)) as pool, open(a.out, "a") as f:
        for i, m in enumerate(pool.imap_unordered(one, W, chunksize=4)):
            f.write(json.dumps(m) + "\n"); f.flush()
            c[m["end"]] = c.get(m["end"], 0) + 1; c["tracked"] = c.get("tracked", 0) + int(m.get("tracked", False))
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(W)} {c} {time.time() - t:.0f}s", flush=True)
    print("FEASIBILITY DONE", json.dumps(c), f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
