#!/usr/bin/env python3
"""G3(a/b): planner-v1 training windows from COS scenes, plus the two ablation scene sources on the same windows.

    python scripts/cos_g3_build.py --synth G2DIR --out DATA_ROOT [--variants cos mob random] [--jobs 8]

For every accepted, verified scene of G2 the real motion M (chord frame) is cut into 2 s windows (20 frames x 5 steps at
50 Hz, cat_mjlab/planner/data.features) starting every 10 steps, kept when the window overlaps the event +- 0.5 s (at most
8 per scene). Each window gets a robot-centred SDF grid (cat_mjlab/planner/grid: 40x32x36 cells of 5 cm, clipped to
[-0.1, 0.4] m, uint8) of one scene source:
  cos     the COS scene: explaining obstacle, supports, distractors, room walls (analytic box SDF)
  mob     MOB-style complement: every voxel farther than the margin from the body's swept volume (sphere cover, all
          frames of the window's motion) is solid, z >= 0.10 m (the floor stays floor) -- distance transform
  random  the same number of boxes as the COS scene, placed at random anywhere clear of M by >= margin (no
          "the default would hit it" constraint), plus the same room walls
Splits are by take (PHUMA chunks / clip id), 20% test, identical for all variants. Output per variant: the shard layout
of cat_mjlab/planner/data (DATA_ROOT/<variant>/{train,test}/000_*.npy, info.json) and scenes.json (boxes by scene id).
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, re, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

FAMS = ["duck", "crawl", "high_step", "sideways", "arms_tucked", "arms_raised", "detour", "arms_in"]
VAR_ID = dict(cos=10, mob=11, random=12)


def take_of(clip_id):
    s = re.sub(r"_chunk_\d+$", "", clip_id)
    return s


def is_test(clip_id, frac=0.2):
    h = int(hashlib.md5(take_of(clip_id).encode()).hexdigest()[:8], 16)
    return (h % 1000) < frac * 1000


def qvel_from_qpos(q, dt=0.02):
    """MuJoCo qvel [T, 35]: world linear velocity, body-frame angular velocity, joint velocities (central differences)."""
    from cat_mjlab.cos.default_motion import qmul
    T = len(q)
    v = np.zeros((T, 35))
    v[:, :3] = np.gradient(q[:, :3], dt, axis=0)
    v[:, 6:] = np.gradient(q[:, 7:], dt, axis=0)
    quat = q[:, 3:7].copy()
    for i in range(1, T):
        if (quat[i] * quat[i - 1]).sum() < 0:
            quat[i] = -quat[i]
    dq = np.gradient(quat, dt, axis=0)
    conj = quat * np.array([1, -1, -1, -1])
    w = 2 * qmul(conj, dq)
    v[:, 3:6] = w[:, 1:]
    return v


def box_sdf(points, boxes):
    """Signed distance [P] from points [P, 3] to the union of boxes (center, half, yaw), floor excluded."""
    out = np.full(len(points), 10.)
    for b in boxes:
        c, s = math.cos(b.get("yaw", 0.)), math.sin(b.get("yaw", 0.))
        d = points - np.asarray(b["center"])
        local = np.stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]), -1)
        q = np.abs(local) - np.asarray(b["half"])
        dist = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
        out = np.minimum(out, dist)
    return out


def mob_field(qc, geom_spheres, margin=0.05, cell=0.05):
    """Scene-level SDF of the MOB complement in the chord frame: dict(origin, cell, sdf [nx, ny, nz])."""
    import torch
    from scipy import ndimage
    from scipy.spatial import cKDTree
    q = torch.as_tensor(qc[::2], dtype=torch.float32)
    centres = geom_spheres(q).reshape(-1, 3).numpy()
    radii = np.tile(geom_spheres.radii.numpy(), len(q))
    lo = np.array([qc[:, 0].min() - 1.0, -1.6, 0.]); hi = np.array([qc[:, 0].max() + 1.6, 1.6, 2.0])
    n = np.ceil((hi - lo) / cell).astype(int)
    g = np.stack(np.meshgrid(*[lo[i] + cell * (np.arange(n[i]) + .5) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    tree = cKDTree(centres)
    rmax = float(radii.max()) + margin
    free = np.zeros(len(g), bool)
    for i, idx in enumerate(tree.query_ball_point(g, rmax)):
        if idx:
            d = np.linalg.norm(centres[idx] - g[i], axis=1) - radii[idx]
            free[i] = (d <= margin).any()
    free = free.reshape(n)
    solid = ~free
    solid[:, :, :2] = False                                    # the floor stays floor (z < 0.10 m)
    out = ndimage.distance_transform_edt(~solid) * cell - ndimage.distance_transform_edt(solid) * cell
    return dict(origin=lo, cell=cell, sdf=out.astype(np.float32))


def sample_field(f, points):
    from scipy.ndimage import map_coordinates
    u = (points - f["origin"]) / f["cell"] - .5
    return map_coordinates(f["sdf"], u.T, order=1, mode="nearest")


def random_boxes(M, cos_boxes, rng, margin=0.05, tries=200):
    from cat_mjlab.cos.synth import MARGIN
    k = sum(1 for b in cos_boxes if b["role"] != "wall")
    frames = np.arange(0, len(M.pel), 2)
    out = []
    xr = (float(M.pel[:, 0].min()) - 1.0, float(M.pel[:, 0].max()) + 1.0)
    for _ in range(tries):
        if len(out) >= k:
            break
        t = rng.random()
        if t < 0.35:
            half = [rng.uniform(0.15, 0.4), rng.uniform(0.15, 0.4), rng.uniform(0.2, 0.6)]
        elif t < 0.6:
            r = rng.uniform(0.05, 0.2); half = [r, r, rng.uniform(0.5, 1.1)]
        elif t < 0.8:
            half = [rng.uniform(0.3, 1.2), rng.uniform(0.05, 0.12), rng.uniform(0.5, 1.1)]
        else:                                                  # overhead slab or beam, like COS beams / ceilings
            half = [rng.uniform(0.05, 0.6), rng.uniform(0.5, 1.2), rng.uniform(0.03, 0.12)]
        z = half[2] if t < 0.8 else rng.uniform(0.6, 1.6)
        b = dict(center=[rng.uniform(*xr), rng.uniform(-1.2, 1.2), z], half=half, yaw=rng.uniform(-math.pi, math.pi),
                 role="random")
        if M.sep([b], frames)[0].min() >= MARGIN:
            out.append(b)
    return out + [b for b in cos_boxes if b["role"] == "wall"]


ARGS = None


def init(a):
    global ARGS, G, SPH
    import torch, mujoco
    torch.set_num_threads(1)
    ARGS = a
    from cat_mjlab.exact_geometry import ExactGeometry
    from cat_mjlab.planner.fk import BodySpheres
    G = ExactGeometry()
    SPH = BodySpheres(G.model)


def one(job):
    """All windows of one G2 motion file (several scenes share it) for all variants."""
    import torch
    from cat_mjlab.cos.synth import Body
    from cat_mjlab.planner.data import features
    from cat_mjlab.planner.grid import cell_centres, quantize, to_world
    a = ARGS
    wid, scenes = job
    mot = np.load(Path(a.synth) / "motions" / f"{wid}.npz")
    qc = mot["M"].astype(np.float64)
    T = len(qc)
    if T < 101:
        return []
    qv = qvel_from_qpos(qc)
    qt, vt = torch.as_tensor(qc[:, None]), torch.as_tensor(qv[:, None])
    M = Body(G, qc)
    rng = np.random.default_rng([int(hashlib.md5(wid.encode()).hexdigest()[:8], 16), 7])
    local = cell_centres().reshape(1, -1, 3)
    mob = None
    out = []
    for sc in scenes:
        fa, fb = sc["event_frames"]
        t0s = [t for t in range(0, T - 100, 10) if t + 100 >= fa - 25 and t <= fb + 25]
        if len(t0s) > a.per_scene:
            t0s = sorted(rng.choice(t0s, a.per_scene, replace=False).tolist())
        if not t0s:
            continue
        starts = np.array([[t, 0] for t in t0s])
        state, chunk, goal, origin, yaw = features(qt, vt, starts)
        world = to_world(local.expand(len(t0s), -1, -1).double(), origin.double(), yaw.double()).numpy()     # [K, C, 3]
        rand = random_boxes(M, sc["boxes"], rng) if "random" in a.variants else None
        grids = {}
        for var in a.variants:
            if var == "cos":
                sdf = np.stack([box_sdf(w, sc["boxes"]) for w in world])
            elif var == "random":
                sdf = np.stack([box_sdf(w, rand) for w in world])
            else:
                if mob is None:
                    mob = mob_field(qc, SPH)
                sdf = np.stack([sample_field(mob, w) for w in world])
            grids[var] = quantize(torch.as_tensor(sdf, dtype=torch.float32)).reshape(len(t0s), 40, 32, 36).numpy()
        out.append(dict(scene=sc["scene"], family=sc["family"], test=is_test(sc["clip_id"]), t0=t0s,
                        state=state.numpy(), chunk=chunk.numpy(), goal=goal.numpy(),
                        pose=torch.cat((origin, yaw[:, None]), -1).numpy(), grids=grids,
                        random_boxes=rand))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variants", nargs="+", default=["cos", "mob", "random"])
    ap.add_argument("--per-scene", type=int, default=8); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    from cat_mjlab.planner.data import ShardWriter, write_info, H, STRIDE
    S = [json.loads(l) for l in open(Path(a.synth) / "scenes.jsonl")]
    S = [s for s in S if s["verify"]["ok"]]
    by = {}
    for s in S:
        by.setdefault(s["window"], []).append(s)
    jobs = sorted(by.items())[:a.limit]
    writers = {v: ShardWriter(a.out / v, size=20000) for v in a.variants}
    scene_ids, boxes = {}, {v: {} for v in a.variants}
    counts = {v: {"train": 0, "test": 0} for v in a.variants}
    fam_counts = {}
    t = time.time()
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.jobs, initializer=init, initargs=(a,)) as pool:
        for k, res in enumerate(pool.imap_unordered(one, jobs, chunksize=1)):
            for r in res:
                sid = scene_ids.setdefault(r["scene"], 100000 + len(scene_ids))
                split = "test" if r["test"] else "train"
                n = len(r["t0"])
                meta = np.stack([np.full(n, sid), np.full(n, 100 + FAMS.index(r["family"])), np.zeros(n, int),
                                 np.array(r["t0"]), np.zeros(n, int), np.full(n, int(split == "test"))], 1).astype(np.int32)
                for v in a.variants:
                    m = meta.copy(); m[:, 4] = VAR_ID[v]
                    writers[v].add(split, grid=r["grids"][v], state=r["state"], chunk=r["chunk"], goal=r["goal"], meta=m,
                                   pose=r["pose"])
                    counts[v][split] += n
                    sc = next(s for s in S if s["scene"] == r["scene"])
                    boxes[v][sid] = dict(scene=r["scene"], family=r["family"], window=sc["window"],
                                         boxes=r["random_boxes"] if v == "random" else sc["boxes"])
                fam_counts.setdefault(r["family"], {"train": 0, "test": 0})[split] += n
            if k % 50 == 0:
                print(f"{k}/{len(jobs)} motions {time.time() - t:.0f}s {counts}", flush=True)
    for v in a.variants:
        writers[v].flush()
        write_info(a.out / v, dict(schema="wbp-planner-data-v2", source=f"cos-{v}", synth=str(a.synth),
                                   scene_types=FAMS, examples=counts[v], horizon_frames=H, frame_stride_steps=STRIDE,
                                   family_examples=fam_counts, note="meta[:,0] = scene id (scenes.json), meta[:,1] = 100 + family"))
        (a.out / v / "scenes.json").write_text(json.dumps({str(k): x for k, x in boxes[v].items()}))
    print("BUILD DONE", json.dumps(dict(counts=counts, families=fam_counts, minutes=round((time.time() - t) / 60, 1))))


if __name__ == "__main__":
    main()
