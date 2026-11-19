#!/usr/bin/env python3
"""Inserted-obstacle probe: does the planner plan around an obstacle that its goal does not reveal?

The held-out goal is where the policy's own motion ended, so it already encodes a free route. Here a vertical column
(radius --radius) is added to the local map beside the executed pelvis path at --frame (0.1 s per frame), --offset to
the left or to the right, where it cuts the executed body. Start and goal stay the same. Plans are sampled with the
edited map (scene) and without any map (blind) and scored against the edited map. Probes whose column does not touch
the executed motion, or comes within --keep of the start or goal pelvis, are skipped.

    python scripts/planner_probe.py --plans EVAL/plans.npz --checkpoint TRAIN/planner.pt --out PROBE --device cpu
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
import cat_mjlab  # noqa: E402,F401


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plans", type=Path, help="plans.npz from planner_eval --save-plans (a few situations per type)")
    p.add_argument("--data", type=Path, help="dataset root: probe its held-out test half instead (--per-type situations)")
    p.add_argument("--per-type", type=int, default=300)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--ode-steps", type=int, default=16)
    p.add_argument("--radius", type=float, default=0.10)
    p.add_argument("--offset", type=float, default=0.25)
    p.add_argument("--frame", type=int, default=10)
    p.add_argument("--keep", type=float, default=0.35)
    p.add_argument("--guidance", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    import torch
    import mujoco
    from cat_mjlab.model import assemble_training_xml
    from cat_mjlab.planner.grid import cell_centres, quantize, dequantize
    from cat_mjlab.planner.metrics import ChunkScorer
    from cat_mjlab.planner.model import FlowPlanner, Normalizer, sample
    started = time.time()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.data:
        from cat_mjlab.planner.data import load_split
        from cat_mjlab.runner import SCENE_BUCKETS
        from planner_train import held_out_halves
        held = load_split(args.data, "test")
        _, test_mask = held_out_halves(held["meta"], 0)
        test = {k: v[test_mask] for k, v in held.items()}
        rng = np.random.default_rng(args.seed)
        chosen = []
        for b in np.unique(test["meta"][:, 1]):
            idx = np.nonzero(test["meta"][:, 1] == b)[0]
            chosen += rng.choice(idx, min(args.per_type, len(idx)), replace=False).tolist()
        chosen = np.array(sorted(chosen))
        z = dict(state=test["state"][chosen], goal=test["goal"][chosen], grid=test["grid"][chosen], executed=test["chunk"][chosen],
                 meta=test["meta"][chosen], pose=test["pose"][chosen], kinds=test["meta"][chosen, 1], kind_names=np.array(SCENE_BUCKETS))
    else:
        z = dict(np.load(args.plans, allow_pickle=False))
    names = [str(n) for n in z["kind_names"]]
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FlowPlanner(**ckpt["config"]).to(device).eval(); model.load_state_dict(ckpt["model"])
    norm = Normalizer(np.zeros((2, 67)), np.zeros((2, 5)), np.zeros((2, 20, 35))).to(device); norm.load_state_dict(ckpt["norm"])
    scorer = ChunkScorer(mujoco.MjModel.from_xml_string(assemble_training_xml()), device)
    centres = cell_centres(device)                                    # [NX, NY, NZ, 3]
    probes = []
    for r in range(len(z["state"])):
        ex = z["executed"][r]; k = args.frame - 1
        pk = ex[k, :2]; tangent = ex[min(k + 1, 19), :2] - ex[max(k - 1, 0), :2]
        if np.linalg.norm(tangent) < 1e-3:                            # standing still: use the heading at that frame
            tangent = np.array([np.cos(ex[k, 3]), np.sin(ex[k, 3])])
        t = tangent / np.linalg.norm(tangent); n = np.array([-t[1], t[0]])
        for side in (1, -1):
            c = pk + side * args.offset * n
            if np.linalg.norm(c) - args.radius < args.keep or np.linalg.norm(z["goal"][r][:2] - c) - args.radius < args.keep:
                continue
            probes.append((r, side, c))
    rows = np.array([q[0] for q in probes])
    cols = torch.as_tensor(np.stack([q[2] for q in probes]), dtype=torch.float32, device=device)
    edited = []
    for s in range(0, len(rows), 256):
        grid = torch.as_tensor(z["grid"][rows[s:s + 256]], device=device)
        column = torch.linalg.vector_norm(centres[None, ..., :2] - cols[s:s + 256, None, None, None], dim=-1) - args.radius
        edited.append(quantize(torch.minimum(dequantize(grid), column)))
    edited = torch.cat(edited)
    state = torch.as_tensor(z["state"][rows], device=device); goal = torch.as_tensor(z["goal"][rows], device=device)
    executed = torch.as_tensor(z["executed"][rows], device=device)
    ex_clear = torch.cat([scorer.score(executed[s:s + 256, None], edited[s:s + 256], goal[s:s + 256])["min_clear"][:, 0]
                          for s in range(0, len(rows), 256)])
    cut = (ex_clear < -args.tol).cpu().numpy()
    g = torch.Generator(device=device).manual_seed(args.seed)
    st, gl = norm.norm("state", state), norm.norm("goal", goal)
    results, keep = {}, {}
    for name, use_scene, guid in [("blind", False, 1.0)] + [(f"scene_g{w:g}", True, w) for w in args.guidance]:
        clear, gerr, dev = [], [], []
        for s in range(0, len(rows), 32):
            sl = slice(s, s + 32)
            x = norm.denorm("chunk", sample(model, st[sl], gl[sl], edited[sl] if use_scene else None, use_scene=use_scene,
                                            samples=args.samples, steps=args.ode_steps, guidance=guid, generator=g))
            sc = scorer.score(x, edited[sl], goal[sl])
            clear.append(sc["min_clear"].cpu()); gerr.append(sc["goal_err"].cpu())
            # how far the plan's pelvis passes from the column centre at the probe frame, against the executed motion
            dev.append(torch.linalg.vector_norm(x[:, :, args.frame - 1, :2] - cols[sl, None], dim=-1).cpu())
            if name != "blind" and args.data is None:
                keep.setdefault(name, []).append(x.cpu())
        clear, gerr, dev = torch.cat(clear), torch.cat(gerr), torch.cat(dev)
        key = clear.clamp(max=0.02) * 100 - gerr
        best = key.argmax(-1); idx = torch.arange(len(best))
        free = clear >= -args.tol
        m = torch.as_tensor(cut)
        ex_dist = torch.linalg.vector_norm(executed[:, args.frame - 1, :2].cpu() - cols.cpu(), dim=-1)
        results[name] = dict(best_of_16_free=round(float(free[idx, best][m].float().mean()), 3),
                             one_sample_free=round(float(free[:, 0][m].float().mean()), 3),
                             any_free=round(float(free.any(-1)[m].float().mean()), 3),
                             best_goal_err_m=round(float(gerr[idx, best][m].mean()), 3),
                             pelvis_to_column_m=round(float(dev[idx, best][m].mean()), 3),
                             executed_pelvis_to_column_m=round(float(ex_dist[m].mean()), 3))
        per_type = {}
        for kind in np.unique(z["kinds"][rows]):
            mk = torch.as_tensor((z["kinds"][rows] == kind) & cut)
            if int(mk.sum()):
                per_type[names[int(kind)]] = dict(n=int(mk.sum()), best_free=round(float(free[idx, best][mk].float().mean()), 3))
        results[name]["per_type"] = per_type
        results[name]["_best"] = best.numpy()
        results[name]["_clear"] = clear.numpy()
    summary = dict(probes=len(rows), cutting_executed=int(cut.sum()), radius_m=args.radius, offset_m=args.offset,
                   frame=args.frame, seconds=round(time.time() - started, 1),
                   **{k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()})
    (args.out / "probe_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    save = dict(kinds=z["kinds"][rows], kind_names=z["kind_names"], meta=z["meta"][rows], pose=z["pose"][rows],
                state=z["state"][rows], goal=z["goal"][rows], executed=z["executed"][rows]) if args.data is None else {}
    np.savez_compressed(args.out / "probe.npz", **save, rows=rows, side=np.array([q[1] for q in probes]), column=cols.cpu().numpy(),
                        cut=cut, executed_clear=ex_clear.cpu().numpy(), **({} if args.data else dict(edited=edited.cpu().numpy())),
                        **{f"{k}_best": v["_best"] for k, v in results.items()}, **{f"{k}_clear": v["_clear"] for k, v in results.items()},
                        **{f"{k}_chunks": torch.cat(v).numpy() for k, v in keep.items()})
    print(json.dumps({k: v for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
