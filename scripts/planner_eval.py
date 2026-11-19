#!/usr/bin/env python3
"""Kinematic evaluation of the whole-body planner on held-out scenes (no tracker, no physics).

For each held-out situation (test half of the held-out scenes, at most --per-type per scene type) it draws K chunks
with the scene and K blind (the learned "no scene" tokens), and scores every chunk with the sphere cover of the
collision body against the stored scene grid (cat_mjlab/planner/metrics.py). Methods compared:
    executed  the policy's real chunk (reference; contact-free in simulation by construction)
    scene     one sample of the scene-conditioned planner
    scene+16  best of K scene-conditioned samples (largest minimum clearance, then smallest goal error)
    blind     one sample of the blind planner
    blind+16  best of K blind samples, selected with the same scene check (sample-and-check with a motion prior)
Collision-free: minimum clearance >= -tol over all 20 frames and 497 spheres (default tol 1 cm). Floor penetration:
the lowest sphere more than 8 cm below the floor (the conservative foot spheres of the real motion reach about 6 cm).

    python scripts/planner_eval.py --data DATA --checkpoint TRAIN/planner.pt --out EVAL --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def best_of(scores):
    """Index [B] of the best sample per context: largest min clearance (capped at 2 cm), then smallest goal error."""
    import torch
    key = scores["min_clear"].clamp(max=0.02) * 100 - scores["goal_err"]
    return torch.argmax(key, -1)


def plot_examples(path, contexts, data_grid, gt, scene_samples, blind_samples, scene_scores, blind_scores, kinds):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:                     # plotting is optional
        print(json.dumps(dict(plot_skipped=str(error))), flush=True)
        return
    from cat_mjlab.planner.grid import CELL, X0, Y0, Z0, dequantize
    import torch
    n = len(contexts)
    fig, axes = plt.subplots(n, 2, figsize=(9.5, 3.0 * n), squeeze=False)
    for row, c in enumerate(contexts):
        sdf = dequantize(torch.as_tensor(data_grid[c])).numpy()                        # [NX, NY, NZ]
        occ_top = (sdf[:, :, 1:34] <= 0).any(-1)                                       # obstacles 5 cm - 1.7 m
        occ_side = (sdf[:, 10:22, :] <= 0).any(1)
        ax = axes[row, 0]
        ax.imshow(occ_top.T, origin="lower", extent=[X0, X0 + CELL * sdf.shape[0], Y0, Y0 + CELL * sdf.shape[1]],
                  cmap="Greys", vmin=0, vmax=1.6, alpha=0.9)
        for k in range(blind_samples.shape[1]):
            ax.plot(blind_samples[c, k, :, 0], blind_samples[c, k, :, 1], color="#999999", lw=0.8, alpha=0.6)
        for k in range(scene_samples.shape[1]):
            ok = float(scene_scores["min_clear"][c, k]) >= -0.01
            ax.plot(scene_samples[c, k, :, 0], scene_samples[c, k, :, 1], color="#2ca25f" if ok else "#de2d26", lw=1.0)
        ax.plot(gt[c, :, 0], gt[c, :, 1], color="black", lw=2.0)
        ax.plot([0], [0], "k^", ms=6)
        ax.set_xlim(X0, X0 + 2.0); ax.set_ylim(Y0, Y0 + 1.6); ax.set_aspect("equal")
        ax.set_title(f"{kinds[row]}: top view (pelvis paths)", fontsize=9)
        ax.set_xlabel("x ahead (m)", fontsize=8); ax.set_ylabel("y left (m)", fontsize=8)
        ax = axes[row, 1]
        ax.imshow(occ_side.T, origin="lower", extent=[X0, X0 + CELL * sdf.shape[0], Z0, Z0 + CELL * sdf.shape[2]],
                  cmap="Greys", vmin=0, vmax=1.6, alpha=0.9)
        for k in range(scene_samples.shape[1]):
            ok = float(scene_scores["min_clear"][c, k]) >= -0.01
            ax.plot(scene_samples[c, k, :, 0], scene_samples[c, k, :, 2], color="#2ca25f" if ok else "#de2d26", lw=1.0)
        ax.plot(gt[c, :, 0], gt[c, :, 2], color="black", lw=2.0)
        ax.set_xlim(X0, X0 + 2.0); ax.set_ylim(0, 1.8); ax.set_aspect("equal")
        ax.set_title("side view, |y| < 0.3 m (pelvis height)", fontsize=9)
        ax.set_xlabel("x ahead (m)", fontsize=8); ax.set_ylabel("z (m)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--ode-steps", type=int, default=16)
    p.add_argument("--per-type", type=int, default=300, help="Situations per scene type (random)")
    p.add_argument("--tol", type=float, default=0.01, help="Allowed overlap (m) before a chunk counts as colliding")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance", type=float, default=1.0, help="Classifier-free guidance weight for the scene samples (1 = none)")
    p.add_argument("--save-plans", type=int, default=0,
                   help="Situations per scene type whose full chunks (executed, all scene and blind samples) go to plans.npz")
    args = p.parse_args(argv)
    import torch
    import mujoco
    from cat_mjlab.model import assemble_training_xml
    from cat_mjlab.planner.data import load_split
    from cat_mjlab.planner.metrics import ChunkScorer, diversity
    from cat_mjlab.planner.model import FlowPlanner, Normalizer, sample
    from cat_mjlab.runner import SCENE_BUCKETS
    from planner_train import held_out_halves
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    held = load_split(args.data, "test")
    _, test_mask = held_out_halves(held["meta"], int(ckpt["args"].get("seed", 0)))
    test = {k: v[test_mask] for k, v in held.items()}
    rng = np.random.default_rng(args.seed)
    chosen = []
    for b in np.unique(test["meta"][:, 1]):
        idx = np.nonzero(test["meta"][:, 1] == b)[0]
        chosen += rng.choice(idx, min(args.per_type, len(idx)), replace=False).tolist()
    chosen = np.array(sorted(chosen))
    test = {k: v[chosen] for k, v in test.items()}
    n = len(chosen)
    model = FlowPlanner(**ckpt["config"]).to(device).eval()
    model.load_state_dict(ckpt["model"])
    norm = Normalizer(np.zeros((2, 67)), np.zeros((2, 5)), np.zeros((2, 20, 35))).to(device)
    norm.load_state_dict(ckpt["norm"])
    scorer = ChunkScorer(mujoco.MjModel.from_xml_string(assemble_training_xml()), device)
    keys = ("min_clear", "min_hand_clear", "min_low", "skate", "goal_err")
    gt = {k: [] for k in keys}; sc = {k: [] for k in keys}; bl = {k: [] for k in keys}
    div_scene, div_blind = [], []
    save = set()
    if args.save_plans:   # prefer situations whose executed motion passed within 15 cm of an obstacle
        near = []
        for s in range(0, n, args.batch):
            b = {k: torch.as_tensor(np.ascontiguousarray(test[k][s:s + args.batch]), device=device) for k in ("chunk", "grid", "goal")}
            near.append(scorer.score(b["chunk"][:, None], b["grid"], b["goal"])["min_clear"][:, 0].cpu().numpy() < 0.15)
        near = np.concatenate(near)
        pick_rng = np.random.default_rng(args.seed + 1)
        for b in np.unique(test["meta"][:, 1]):
            kind = test["meta"][:, 1] == b
            pool = np.nonzero(kind & near)[0]
            if len(pool) < args.save_plans:
                pool = np.concatenate((pool, pick_rng.permutation(np.nonzero(kind & ~near)[0])[:args.save_plans - len(pool)]))
            save.update(pick_rng.permutation(pool)[:args.save_plans].tolist())
    full_scene, full_blind = {}, {}
    keep_scene, keep_blind = [], []
    g = torch.Generator(device=device).manual_seed(args.seed)
    for s in range(0, n, args.batch):
        b = {k: torch.as_tensor(np.ascontiguousarray(v[s:s + args.batch]), device=device) for k, v in test.items()}
        st, gl = norm.norm("state", b["state"]), norm.norm("goal", b["goal"])
        xs = norm.denorm("chunk", sample(model, st, gl, b["grid"], use_scene=True, samples=args.samples, steps=args.ode_steps, guidance=args.guidance, generator=g))
        xb = norm.denorm("chunk", sample(model, st, gl, None, use_scene=False, samples=args.samples, steps=args.ode_steps, generator=g))
        for store, chunks in ((gt, b["chunk"][:, None]), (sc, xs), (bl, xb)):
            out = scorer.score(chunks, b["grid"], b["goal"])
            for k in keys:
                store[k].append(out[k].cpu())
        div_scene.append(diversity(xs).cpu()); div_blind.append(diversity(xb).cpu())
        keep_scene.append(xs[:, :, :, :3].cpu()); keep_blind.append(xb[:, :, :, :3].cpu())
        for j in range(len(xs)):
            if s + j in save:
                full_scene[s + j], full_blind[s + j] = xs[j].cpu().numpy(), xb[j].cpu().numpy()
    cat = lambda d: {k: torch.cat(v) for k, v in d.items()}
    gt, sc, bl = cat(gt), cat(sc), cat(bl)
    kinds = test["meta"][:, 1]
    tol = -args.tol

    def summarize(mask):
        res = {}
        free = lambda s: (s["min_clear"] >= tol)
        pick_s, pick_b = best_of(sc), best_of(bl)
        take = lambda s, i: {k: v[torch.arange(len(i)), i] for k, v in s.items()}
        steered_s, steered_b = take(sc, pick_s), take(bl, pick_b)
        m = torch.as_tensor(mask)
        for name, s in (("executed", {k: v[:, 0] for k, v in gt.items()}), ("scene", sc), ("scene+16", steered_s),
                        ("blind", bl), ("blind+16", steered_b)):
            f = free(s)
            res[name] = dict(collision_free=round(float(f[m].float().mean()), 3),
                             hand_clear_ok=round(float((s["min_hand_clear"] >= tol)[m].float().mean()), 3),
                             goal_err_m=round(float(s["goal_err"][m].mean()), 3),
                             floor_penetration=round(float((s["min_low"] < -0.08)[m].float().mean()), 3),
                             lowest_sphere_m=round(float(s["min_low"][m].mean()), 3),
                             foot_skate_m=round(float(s["skate"][m].mean()), 4))
        res["scene_any_of_16_free"] = round(float(free(sc).any(-1)[m].float().mean()), 3)
        res["blind_any_of_16_free"] = round(float(free(bl).any(-1)[m].float().mean()), 3)
        res["diversity_scene_m"] = round(float(torch.cat(div_scene)[m].mean()), 3)
        res["diversity_blind_m"] = round(float(torch.cat(div_blind)[m].mean()), 3)
        res["situations"] = int(mask.sum())
        return res

    summary = dict(all=summarize(np.ones(n, bool)))
    for b in np.unique(kinds):
        summary[SCENE_BUCKETS[b]] = summarize(kinds == b)
    summary["_info"] = dict(checkpoint=str(args.checkpoint), step=int(ckpt["step"]), samples=args.samples, ode_steps=args.ode_steps, guidance=args.guidance,
                            tol_m=args.tol, situations=n, seconds=round(time.time() - started, 1))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    np.savez_compressed(args.out / "scores.npz", kinds=kinds, meta=test["meta"],
                        **{f"{name}_{k}": v.numpy() for name, d in (("executed", gt), ("scene", sc), ("blind", bl)) for k, v in d.items()})
    # example figure: the first situation of up to 8 scene types where the planner had to avoid something
    scene_np, blind_np = torch.cat(keep_scene).numpy(), torch.cat(keep_blind).numpy()
    examples, labels = [], []
    for b in np.unique(kinds):
        idx = np.nonzero((kinds == b) & (gt["min_clear"][:, 0].numpy() < 0.15))[0]
        if len(idx):
            examples.append(int(idx[0])); labels.append(SCENE_BUCKETS[b])
        if len(examples) == 8:
            break
    if examples:
        plot_examples(args.out / "examples.png", examples, test["grid"], test["chunk"][:, :, :3], scene_np, blind_np,
                      {k: v for k, v in sc.items()}, {k: v for k, v in bl.items()}, labels)
    if save:
        idx = np.array(sorted(save))
        pick_s, pick_b = best_of(sc), best_of(bl)
        np.savez_compressed(args.out / "plans.npz", idx=idx, kinds=kinds[idx], kind_names=np.array(SCENE_BUCKETS),
                            meta=test["meta"][idx], pose=test["pose"][idx], state=test["state"][idx], goal=test["goal"][idx],
                            grid=test["grid"][idx], executed=test["chunk"][idx],
                            scene=np.stack([full_scene[i] for i in idx]), blind=np.stack([full_blind[i] for i in idx]),
                            scene_best=pick_s[idx].numpy(), blind_best=pick_b[idx].numpy(),
                            **{f"{name}_{k}": d[k][idx].numpy() for name, d in (("executed", gt), ("scene", sc), ("blind", bl))
                               for k in ("min_clear", "min_hand_clear", "min_low", "goal_err")})
    print("EVAL SUMMARY " + json.dumps(summary["all"]), flush=True)
    for b in np.unique(kinds):
        print("EVAL TYPE " + json.dumps(dict(type=SCENE_BUCKETS[b], **{k: v["collision_free"] for k, v in summary[SCENE_BUCKETS[b]].items() if isinstance(v, dict)})), flush=True)


if __name__ == "__main__":
    main()
