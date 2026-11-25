#!/usr/bin/env python3
"""G2: counterfactual obstacle synthesis over the G0/G1 windows (docs/cos_obstacles.md).

  python scripts/cos_g2_synth.py --windows W.json --library LIB --gait-bank G.pkl --out DIR [--tracked G1DIR] [--jobs 8]

Writes DIR/scenes.jsonl (one accepted, verified scene per line), DIR/motions/<window>.npz (M and D in the chord frame),
DIR/attempts.jsonl (one line per event: candidates tried, accepted) and DIR/summary.json.
"""
from __future__ import annotations

import argparse, json, math, os, pickle, time
from pathlib import Path

import numpy as np

ARGS = None


def _j(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def init(a):
    global ARGS, G, FF, Q, BANK
    import torch
    torch.set_num_threads(1)
    ARGS = a
    from cat_mjlab.exact_geometry import ExactGeometry
    from cat_mjlab.cos.features import FrameFeatures
    G = ExactGeometry(); FF = FrameFeatures(G)
    Q = np.load(a.library + ".npz", mmap_mode="r")["qpos"]
    BANK = pickle.load(open(a.gait_bank, "rb"))


def tracked_ok(m):
    return (m.get("end") == "complete" and m.get("joint_err_p90", 9) <= 0.15 and m.get("local_p90", 9) <= 0.10
            and abs(m.get("yaw_drift_deg", 99)) <= 30)


def one(w):
    from cat_mjlab.cos.default_motion import default_motion, pick_cycle, to_frame
    from cat_mjlab.cos.synth import Body, distractors, obstacles_for_event, room_walls, verify
    a = ARGS
    rng = np.random.default_rng(abs(hash((w["id"], a.seed))) % 2**32)
    if a.tracked:
        f = Path(a.tracked) / f"{w['id']}.npz"; mj = Path(a.tracked) / f"{w['id']}_metrics.json"
        if not f.exists() or not mj.exists() or not tracked_ok(json.loads(mj.read_text())):
            return dict(window=w["id"], skip="not_tracked"), [], []
        qw = np.load(f)["tracked"].astype(np.float64)
    else:
        qw = np.asarray(Q[w["frame0"] + w["f0"]: w["frame0"] + w["f1"]], dtype=np.float64)
    T = len(qw)
    chord = qw[-1, :2] - qw[0, :2]; L = float(np.linalg.norm(chord))
    if L < 0.5:
        return dict(window=w["id"], skip="no_travel", chord=L), [], []
    yaw = math.atan2(chord[1], chord[0])
    qc = to_frame(qw, qw[0, :2], yaw)
    cyc = pick_cycle(BANK, L / (T * .02), rng)
    Dq = default_motion(qc, cyc, phase0=float(rng.random()))
    M, D = Body(G, qc), Body(G, Dq)
    st = FF(qc)
    scenes, attempts = [], []
    for e_i, (fam, t0, t1, mag) in enumerate(w["events"]):
        fa = max(0, int(round(t0 / .02)) - w["f0"]); fb = min(T, int(round(t1 / .02)) - w["f0"])
        if fb - fa < 2:
            attempts.append(dict(window=w["id"], event=e_i, family=fam, tried=0, accepted=0, skip="outside")); continue
        ev = dict(family=fam, fa=fa, fb=fb)
        acc, n = obstacles_for_event(ev, M, D, st, rng, tries=a.tries, want=a.variants)
        attempts.append(dict(window=w["id"], event=e_i, family=fam, tried=n, accepted=len(acc), source=w["source"]))
        for v_i, o in enumerate(acc):
            xr = (float(min(qc[:, 0].min(), 0.)) - 1.0, float(qc[:, 0].max()) + 1.0)
            dis = distractors(M, D, o["boxes"], rng, x_range=xr)
            walls = room_walls(M, D, rng, xr)
            boxes = o["boxes"] + dis + walls
            ver = verify(G, qc, Dq, boxes)
            scenes.append(dict(scene=f"{w['id']}_e{e_i}_v{v_i}", window=w["id"], clip_id=w["clip_id"], source=w["source"],
                               family=fam, event_frames=[fa, fb], kind=o["boxes"][0].get("kind"),
                               chord=dict(origin=qw[0, :2].tolist(), yaw=yaw, length=L), gait_src=cyc["src"],
                               search=dict(m_clear=o["m_clear"], m_clear_main=o["m_clear_main"], d_pen=o["d_pen"],
                                           d_region=o["d_region"]),
                               verify=ver, n_distractors=len(dis), boxes=boxes))
    if scenes:
        np.savez_compressed(Path(a.out) / "motions" / f"{w['id']}.npz", M=qc.astype(np.float32), D=Dq.astype(np.float32))
    return dict(window=w["id"], T=T, chord=L), scenes, attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True); ap.add_argument("--library", required=True)
    ap.add_argument("--gait-bank", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tracked"); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tries", type=int, default=60); ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    out = Path(a.out); (out / "motions").mkdir(parents=True, exist_ok=True)
    W = json.loads(Path(a.windows).read_text())[:a.limit]
    t = time.time()
    import multiprocessing as mp
    res = []
    with mp.get_context("fork").Pool(a.jobs, initializer=init, initargs=(a,)) as pool, \
            open(out / "scenes.jsonl", "w") as fs, open(out / "attempts.jsonl", "w") as fa, open(out / "windows.jsonl", "w") as fw:
        for k, (info, scenes, attempts) in enumerate(pool.imap_unordered(one, W, chunksize=2)):
            fw.write(json.dumps(info, default=_j) + "\n")
            for s in scenes:
                fs.write(json.dumps(s, default=_j) + "\n")
            for x in attempts:
                fa.write(json.dumps(x, default=_j) + "\n")
            if k % 50 == 0:
                print(f"{k}/{len(W)} windows {time.time() - t:.0f}s", flush=True)
    print(f"done {len(W)} windows in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
