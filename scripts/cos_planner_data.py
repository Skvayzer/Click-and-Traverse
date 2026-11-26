#!/usr/bin/env python3
"""C2 planner data from COS scenes v2 in the planner v1 format (docs/cos_planner_reflex.md, C2).

  python scripts/cos_planner_data.py --manifest /l/.../cos/scenes_v2/manifest.json --splits /l/.../cos/splits_v2 \
      --out /l/.../cos/planner_data_v1 --jobs 12

Splits: train takes -> train, val takes -> test (the trainer's held-out split), test takes -> cos_test.
meta = (scene, bucket, world, t0, source, split): scene = 1,000,000 + COS scene index (scene_index.json), bucket = 100 +
family index, world = 0 levelled M / 1 SONIC-tracked rollout, source = 10.
"""
import argparse, collections, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
FAMS = ("duck", "crawl", "high_step", "sideways", "arms_in", "arms_tucked", "arms_raised", "detour", "multi")
SPLIT = {"train": "train", "val": "test", "test": "cos_test"}
SCENE0 = 1_000_000
A = None
MJ = None


def box_sdf(points, boxes):
    """Signed distance [P] of points [P, 3] to the union of yawed boxes (exact for boxes, floor excluded)."""
    d = np.full(len(points), 10.0)
    for b in boxes:
        c = np.asarray(b["center"]); h = np.asarray(b["half"]); y = float(b.get("yaw", 0.))
        p = points - c; cy, sy = math.cos(y), math.sin(y)
        lx = cy * p[:, 0] + sy * p[:, 1]; ly = -sy * p[:, 0] + cy * p[:, 1]
        q = np.abs(np.stack((lx, ly, p[:, 2]), 1)) - h
        out = np.linalg.norm(np.maximum(q, 0), axis=1); inside = np.minimum(q.max(1), 0)
        d = np.minimum(d, out + inside)
    return d


def qvel_of(q):
    import mujoco
    v = np.zeros((len(q), MJ.nv))
    for t in range(len(q) - 1):
        mujoco.mj_differentiatePos(MJ, v[t], 0.02, q[t], q[t + 1])
    v[-1] = v[-2]
    v[1:-1] = 0.5 * (v[1:-1] + v[:-2])           # centred (average of forward differences)
    return v


def init(a):
    global A, MJ
    import mujoco, torch
    torch.set_num_threads(1)
    from cat_mjlab.model import assemble_training_xml
    A = a; MJ = mujoco.MjModel.from_xml_string(assemble_training_xml())


def one(job):
    import torch
    from cat_mjlab.planner.data import features, H, STRIDE
    from cat_mjlab.planner.grid import cell_centres, quantize, to_world
    idx, folder, s, split, seed = job
    rng = np.random.default_rng(seed)
    d = np.load(Path(folder) / "motions" / f"{s['window']}.npz")
    trajs = [(0, d["M"].astype(np.float64))]
    if "tracked" in d.files and s.get("tracked_clear") is not None and s["tracked_clear"] >= 0.03:
        trajs.append((1, d["tracked"].astype(np.float64)))
    out = []
    span = H * STRIDE
    local = cell_centres().reshape(-1, 3)
    for world, q in trajs:
        T = len(q)
        t0s = np.arange(0, T - span - 1, 10)
        ok = [t for t in t0s if any(t <= b + 50 and t + span >= a - 50 for a, b in s["event_frames"])]
        if not ok:
            continue
        ok = rng.permutation(ok)[:A.per_traj] if split != "cos_test" else rng.permutation(ok)[:A.per_traj_test]
        qv = qvel_of(q)
        qt = torch.as_tensor(q[:, None], dtype=torch.float64); vt = torch.as_tensor(qv[:, None], dtype=torch.float64)
        st = np.stack((np.sort(ok), np.zeros(len(ok), int)), 1)
        state, chunk, goal, origin, yaw = features(qt, vt, st)
        grids = []
        for k in range(len(st)):
            w = to_world(local[None], origin[k:k + 1].float(), yaw[k:k + 1].float())[0].numpy()
            grids.append(quantize(torch.as_tensor(box_sdf(w.astype(np.float64), s["boxes"]), dtype=torch.float32)))
        grid = torch.stack(grids).reshape(len(st), 40, 32, 36).numpy()
        meta = np.stack([np.full(len(st), SCENE0 + idx), np.full(len(st), 100 + FAMS.index(s["family"])),
                         np.full(len(st), world), st[:, 0], np.full(len(st), 10),
                         np.full(len(st), 0 if split == "train" else 1)], 1)
        pose = torch.cat((origin, yaw[:, None]), 1).numpy()
        out.append(dict(grid=grid, state=state.numpy(), chunk=chunk.numpy(), goal=goal.numpy(), meta=meta, pose=pose))
    return SPLIT[split], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--splits", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=12); ap.add_argument("--per-traj", type=int, default=6)
    ap.add_argument("--per-traj-test", type=int, default=3); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from cat_mjlab.planner.data import ShardWriter, write_info
    man = json.load(open(a.manifest)); takes = json.load(open(Path(a.splits) / "take_split.json"))
    from cos_splits import take_of
    jobs, index = [], []
    for folder in man["scene_folders"]:
        for l in open(Path(folder) / "scenes.jsonl"):
            s = json.loads(l)
            if not s.get("verified"):
                continue
            sp = takes[take_of(s["clip"])]
            jobs.append((len(index), folder, s, sp, len(index)))
            index.append(dict(scene=s["scene"], folder=folder, family=s["family"], split=sp))
    if a.limit:
        jobs = jobs[:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(index, open(out / "scene_index.json", "w"))
    W = ShardWriter(out, size=20000); n = collections.Counter(); t = time.time()
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.jobs, initializer=init, initargs=(a,)) as pool:
        for k, (split, parts) in enumerate(pool.imap_unordered(one, jobs, chunksize=8)):
            for p in parts:
                W.add(split, **p); n[split] += len(p["state"])
            if k % 2000 == 0:
                print(f"{k}/{len(jobs)} scenes {dict(n)} {time.time() - t:.0f}s", flush=True)
    W.flush()
    write_info(out, dict(schema="wbp-planner-data-v2-cos", scene_types=[f"cos_{f}" for f in FAMS], families=FAMS,
                         bucket_offset=100, scene_offset=SCENE0, horizon_frames=20, frame_stride_steps=5, control_dt=0.02,
                         grid=dict(shape=[40, 32, 36], cell=0.05, origin=[-0.5, -0.8, 0.0], sdf_clip=[-0.1, 0.4]),
                         examples=dict(n), manifest=a.manifest, splits=a.splits))
    print("DONE", dict(n), f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
