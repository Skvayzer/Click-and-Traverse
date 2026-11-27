#!/usr/bin/env python3
"""B1-full: the C0 tests per whole clip; failing stretches cut out with 0.5 s margins; kept segments (>= 2 s) levelled to
their own floor and exported as BlinkMimic motions (docs/cos_blinkmimic.md, B1-full).

  python scripts/cos_full_segments.py --manifest /l/.../cos/scenes_v1/manifest.json --splits /l/.../cos/splits_v2 \
      --out /l/.../cos/tracker/motions_full --jobs 12

Per frame (50 Hz library qpos, proxy primitives as in C0): local floor F(t) = p5 of the min foot height within +-2 s;
flight / floating: whole-body lowest point > F + 0.05 m (runs >= 5 frames; > 30 frames = floating); pelvis vertical speed
> 1.0 m/s; acrobatic (pelvis up-z < 0, |lateral-axis z| > 0.87, head top below the pelvis, angular speed > 8 rad/s);
uneven floor: the whole-body support level of a 0.5 s block > F + 0.05 m; foot penetration: min foot < F - 0.02 m.
Clip names matching the C0 trick / terrain keywords are dropped whole; dropped sources (manifest) are skipped.
Bad frames are dilated by 0.5 s; good runs >= 2 s become segments; a segment whose floor F varies by > 0.05 m is dropped
(uneven floor); each kept segment is shifted by dz = -0.010 - p5(min foot) (the physical proxy floor, C0 amendment 1).
Split by take with the C1 rule (take_split.json, else the same sha1 rule). Output OUT/<split>/<clip>__s<k>.npz.
"""
import argparse, collections, json, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
FPS = 50; MARGIN = 25; MIN_LEN = 100; PHYS_FLOOR = -0.010
G = FF = None; A = None


def init(a):
    global G, FF, A
    import torch
    torch.set_num_threads(1)
    from cat_mjlab.cos.features import FrameFeatures
    FF = FrameFeatures(); G = FF.g; A = a


def runs(mask):
    """(start, end) of True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def frame_flags(q):
    import torch
    from cat_mjlab.cos.features import _extent
    T = len(q); bad = {}
    lo_l, hi_l, pel_l, R_l = [], [], [], []
    for s in range(0, T, 4000):
        qt = torch.as_tensor(q[s:s + 4000], dtype=torch.float32)
        world, xpos = G.primitives(qt); xmat = G.fk(qt)[1]
        lo, hi = _extent(world, G.compiled)
        lo_l.append(lo.numpy()); hi_l.append(hi.numpy()); pel_l.append(xpos[:, G.body["pelvis"]].numpy())
        R_l.append(xmat[:, G.body["pelvis"]].numpy())
    lo, hi, pel, R = (np.concatenate(x) for x in (lo_l, hi_l, pel_l, R_l))
    fmin = lo[:, FF.left_foot + FF.right_foot, 2].min(1); blo = lo[:, :, 2].min(1)
    W = 100
    pad = np.pad(fmin, W, mode="edge")
    # local floor: p5 over +-2 s (strided for speed, then interpolated)
    idx = np.arange(0, T, 10)
    Fs = np.array([np.percentile(pad[i:i + 2 * W + 1], 5) for i in idx])
    F = np.interp(np.arange(T), idx, Fs)
    air = blo > F + 0.05
    flight = np.zeros(T, bool); floating = np.zeros(T, bool)
    for a_, b_ in runs(air):
        if b_ - a_ > 30:
            floating[a_:b_] = True
        elif b_ - a_ >= 5:
            flight[a_:b_] = True
    k = 5
    vz = np.zeros(T)
    if T > 2 * k:
        vz[k:-k] = np.abs(pel[2 * k:, 2] - pel[:-2 * k, 2]) / (2 * k / FPS)
    pelvis_vz = vz > 1.0
    upz = R[:, 2, 2]; latz = np.abs(R[:, 2, 1])
    head = hi[:, FF.groups["head"], 2].max(1)
    ang = np.zeros(T)
    if T > 2 * k:
        rel = np.einsum("tji,tjk->tik", R[:-2 * k], R[2 * k:])
        ang[k:-k] = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)) / (2 * k / FPS)
    acrobatic = (upz < 0) | (latz > 0.87) | (head < pel[:, 2]) | (ang > 8.0)
    uneven = np.zeros(T, bool)
    for s in range(0, T, 25):
        if blo[s:s + 25].min() > F[s:s + 25].mean() + 0.05:
            uneven[s:s + 25] = True
    uneven &= ~flight & ~floating            # an airborne block is flight, not a floor
    pen = fmin < F - 0.02
    bad = dict(flight=flight, floating=floating, pelvis_vz=pelvis_vz, acrobatic=acrobatic, uneven_floor=uneven,
               foot_penetration=pen)
    return bad, fmin, F


def one(job):
    from cos_clean import TRICK, TERRAIN, STEP_OK, name_of
    c, split = job
    out = dict(clip=c["clip_id"], source=c["source"], split=split, frames=c["frames"], kept=0, segments=[],
               removed=collections.Counter())
    nm = name_of(c["clip_id"])
    tn = bool(TERRAIN.search(nm)) and not (STEP_OK.search(nm) and not re.search(r"stair|climb|ramp|slope|platform|ladder", nm, re.I))
    if TRICK.search(nm) or tn:
        out["removed"]["trick_name" if TRICK.search(nm) else "terrain_name"] += c["frames"]
        return out
    lib = Path(A.lib_root) / A.lib_of[c["source"]]
    q = np.load(lib / f"qpos_{c['shard']:03d}.npy", mmap_mode="r")[c["start"]:c["start"] + c["frames"]].astype(np.float64)
    if len(q) < MIN_LEN:
        out["removed"]["short_clip"] += len(q)
        return out
    bad, fmin, F = frame_flags(q)
    any_bad = np.zeros(len(q), bool)
    for v in bad.values():
        any_bad |= v
    dil = any_bad.copy()
    for a_, b_ in runs(any_bad):
        dil[max(0, a_ - MARGIN):b_ + MARGIN] = True
    for r, v in bad.items():                 # flagged frames per reason (non-exclusive, before the 0.5 s margins)
        out["removed"][r] += int(v.sum())
    outdir = Path(A.out) / split; outdir.mkdir(parents=True, exist_ok=True)
    k = 0
    for a_, b_ in runs(~dil):
        if b_ - a_ < MIN_LEN:
            out["removed"]["short_segment"] += int(b_ - a_)
            continue
        if F[a_:b_].max() - F[a_:b_].min() > 0.05:
            out["removed"]["segment_floor_drift"] += int(b_ - a_)
            continue
        seg = q[a_:b_].copy()
        dz = PHYS_FLOOR - float(np.percentile(fmin[a_:b_], 5))
        seg[:, 2] += dz
        name = f"{c['clip_id']}__s{k}"
        np.savez_compressed(outdir / f"{name}.npz", qpos=seg.astype(np.float32), fps=np.float32(FPS), dz=np.float32(dz),
                            take=A.take_of(c["clip_id"]), source=c["source"], start=np.int32(a_))
        out["segments"].append(dict(name=name, frames=int(b_ - a_), dz=round(dz, 4)))
        out["kept"] += int(b_ - a_); k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--splits", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=12); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from cos_splits import split_of, take_of
    man = json.load(open(a.manifest)); takes = json.load(open(Path(a.splits) / "take_split.json"))
    drop = set(man.get("dropped_sources", []))
    a.lib_root = str(Path(man["libraries"][0]).parent); a.lib_of = {}; a.take_of = take_of
    jobs = []
    for lib in man["libraries"]:
        for l in open(Path(lib) / "clips.jsonl"):
            c = json.loads(l)
            if c["source"] in drop:
                continue
            a.lib_of[c["source"]] = Path(lib).name
            t = take_of(c["clip_id"])
            jobs.append((c, takes.get(t) or split_of(t)))
    if a.limit:
        jobs = jobs[::max(1, len(jobs) // a.limit)][:a.limit]
    print(f"{len(jobs)} clips, {sum(c['frames'] for c, _ in jobs) / FPS / 3600:.1f} h", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp
    S = collections.defaultdict(lambda: collections.Counter()); R = collections.defaultdict(collections.Counter)
    t0 = time.time()
    fo = open(Path(a.out) / "clips.jsonl", "w")            # opened before the workers fork (EFAULT seen otherwise)
    with mp.get_context("spawn").Pool(a.jobs, initializer=init, initargs=(a,)) as pool:
        for i, o in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            fo.write(json.dumps(dict(o, removed=dict(o["removed"]))) + "\n")
            key = (o["split"], o["source"].split("_")[0] if o["source"].startswith("amass") else o["source"])
            src = o["source"] if not o["source"].startswith("amass") else ("amass_gmr" if "gmr" in o["source"] else "amass_ember")
            for sp in (o["split"], "all"):
                S[(sp, src)]["clips"] += 1; S[(sp, src)]["frames_in"] += o["frames"]
                S[(sp, src)]["frames_kept"] += o["kept"]; S[(sp, src)]["segments"] += len(o["segments"])
                R[sp].update(o["removed"])
            if i % 5000 == 0:
                print(f"{i}/{len(jobs)} clips {time.time() - t0:.0f}s", flush=True)
    fo.close()
    summ = {f"{sp}/{src}": dict(clips=v["clips"], hours_in=round(v["frames_in"] / FPS / 3600, 2),
                                hours_kept=round(v["frames_kept"] / FPS / 3600, 2), segments=v["segments"])
            for (sp, src), v in sorted(S.items())}
    for sp in ("train", "val", "test", "all"):
        tot = [v for (s_, _), v in S.items() if s_ == sp]
        summ[f"{sp}/TOTAL"] = dict(clips=sum(v["clips"] for v in tot), hours_in=round(sum(v["frames_in"] for v in tot) / FPS / 3600, 2),
                                   hours_kept=round(sum(v["frames_kept"] for v in tot) / FPS / 3600, 2),
                                   segments=sum(v["segments"] for v in tot))
    summ["removed_hours_by_reason"] = {sp: {r: round(n / FPS / 3600, 2) for r, n in R[sp].items()} for sp in R}
    json.dump(summ, open(Path(a.out) / "summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if "TOTAL" in k}), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
