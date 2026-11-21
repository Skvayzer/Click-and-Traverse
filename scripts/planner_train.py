#!/usr/bin/env python3
"""Train the whole-body flow-matching planner on data from scripts/planner_collect.py.

Holds the training split in memory (grids as uint8), samples random batches, and trains with AdamW, a cosine
schedule and an EMA of the weights (the EMA is what is evaluated and saved). Held-out scenes are split by scene id
into a validation half (monitored here) and a test half (for scripts/planner_eval.py only). Every --eval-every
steps it logs the validation flow loss and a quick kinematic check of sampled chunks (collision-free share of 4
samples on 256 validation situations, with and without the scene).

    python scripts/planner_train.py --data DATA --out TRAIN --steps 60000 --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]


def held_out_halves(meta, seed=0):
    """Split held-out examples by scene id into validation and test halves (boolean masks)."""
    scenes = np.unique(meta[:, 0])
    if len(scenes) < 2:                                   # smoke tests: one held-out scene serves both roles
        everything = np.ones(len(meta), bool)
        return everything, everything
    rng = np.random.default_rng(seed)
    val_scenes = set(rng.permutation(scenes)[: len(scenes) // 2].tolist())
    val = np.array([s in val_scenes for s in meta[:, 0]])
    return val, ~val


def to_torch(arrays, device, idx=None):
    import torch
    sel = (lambda a: a) if idx is None else (lambda a: a[idx])
    return {k: torch.as_tensor(np.ascontiguousarray(sel(v)), device=device) for k, v in arrays.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--scene-dropout", type=float, default=0.15)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drop-stopped", type=float, metavar="SPEED",
                   help="drop windows slower than SPEED m/s in CAT courses, passages and table edges (planner v2)")
    args = p.parse_args(argv)
    import torch
    import mujoco
    from cat_mjlab.model import assemble_training_xml
    from cat_mjlab.planner.data import drop_stopped, load_split
    from cat_mjlab.planner.metrics import ChunkScorer
    from cat_mjlab.planner.model import FlowPlanner, Normalizer, flow_loss, sample
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    keep = None
    if args.drop_stopped is not None:
        keep = drop_stopped(json.loads((Path(args.data) / "info.json").read_text()), speed=args.drop_stopped)
    train = load_split(args.data, "train", keep=keep)
    held = load_split(args.data, "test", keep=keep)
    val_mask, _ = held_out_halves(held["meta"], args.seed)
    val = {k: v[val_mask] for k, v in held.items()}
    print(json.dumps(dict(train=len(train["state"]), val=len(val["state"]), load_s=round(time.time() - t0, 1),
                          grid_gb=round(train["grid"].nbytes / 2**30, 2))), flush=True)
    device = torch.device(args.device)
    norm = Normalizer(train["state"], train["goal"], train["chunk"]).to(device)
    cpu = {k: torch.as_tensor(v) for k, v in train.items()}       # grids stay uint8 in host memory
    model = FlowPlanner(args.width, args.layers).to(device)
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    sched = lambda step: min(1., (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1., step / args.steps)))
    scorer = ChunkScorer(mujoco.MjModel.from_xml_string(assemble_training_xml()), device)
    rng = np.random.default_rng(args.seed)
    vsel = rng.choice(len(val["state"]), min(256, len(val["state"])), replace=False)
    vb = to_torch(val, device, vsel)
    gt_clear = scorer.score(vb["chunk"][:, None], vb["grid"], vb["goal"])["min_clear"][:, 0]
    print(json.dumps(dict(params_m=round(params / 1e6, 2), val_gt_collision_free=round(float((gt_clear >= -0.01).float().mean()), 3))), flush=True)

    def batch(idx):
        return {k: v[idx].to(device, non_blocking=True) for k, v in cpu.items()}

    def val_loss(m, n_batches=8):
        losses = []
        for i in range(n_batches):
            sel = np.arange(i * 256, min((i + 1) * 256, len(val["state"])))
            if not len(sel):
                break
            b = to_torch(val, device, sel)
            torch.manual_seed(1000 + i)
            losses.append(float(flow_loss(m, norm.norm("chunk", b["chunk"]), norm.norm("state", b["state"]),
                                          norm.norm("goal", b["goal"]), b["grid"], scene_dropout=0.0)))
        return float(np.mean(losses))

    def quick_eval(m):
        out = {}
        for use_scene in (True, False):
            x = sample(m, norm.norm("state", vb["state"]), norm.norm("goal", vb["goal"]), vb["grid"], use_scene=use_scene, samples=4)
            s = scorer.score(norm.denorm("chunk", x), vb["grid"], vb["goal"])
            key = "scene" if use_scene else "blind"
            out[f"{key}_collision_free"] = round(float((s["min_clear"] >= -0.01).float().mean()), 3)
            out[f"{key}_goal_err_m"] = round(float(s["goal_err"].mean()), 3)
        return out

    model.train()
    t0, last = time.time(), time.time()
    running = []
    for step in range(args.steps + 1):
        for group in opt.param_groups:
            group["lr"] = args.lr * sched(step)
        idx = torch.as_tensor(rng.integers(0, len(train["state"]), args.batch))
        b = batch(idx)
        loss = flow_loss(model, norm.norm("chunk", b["chunk"]), norm.norm("state", b["state"]), norm.norm("goal", b["goal"]),
                         b["grid"], scene_dropout=args.scene_dropout)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
        running.append(float(loss))
        if step % args.log_every == 0:
            rate = args.log_every / max(1e-6, time.time() - last); last = time.time()
            print(json.dumps(dict(step=step, loss=round(float(np.mean(running)), 4), grad=round(float(grad), 3),
                                  lr=round(args.lr * sched(step), 6), steps_per_s=round(rate, 1))), flush=True)
            running = []
        if step % args.eval_every == 0 and step > 0 or step == args.steps:
            ema.eval()
            record = dict(step=step, val_loss=round(val_loss(ema), 4), **quick_eval(ema), minutes=round((time.time() - t0) / 60, 1))
            print("EVAL " + json.dumps(record), flush=True)
            torch.save(dict(schema="wbp-planner-v1", model=ema.state_dict(), norm=norm.state_dict(),
                            config=dict(width=args.width, layers=args.layers), step=step, args=vars(args) | dict(data=str(args.data), out=str(args.out))),
                       args.out / "planner.pt")
    print("TRAIN DONE " + json.dumps(dict(steps=args.steps, minutes=round((time.time() - t0) / 60, 1))), flush=True)


if __name__ == "__main__":
    main()
