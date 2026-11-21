#!/usr/bin/env python3
"""Do plans start moving? Planner checkpoints compared on held-out situations in hard scene types.

For held-out windows in CAT courses, passages and table edges whose recorded motion moved (mean speed >= --min-speed),
each planner samples plans from the same state, goal and map (guidance 2, 6 Euler steps, as in the training loop) and
the pelvis travel after 0.3, 0.6, 1.0 and 2.0 s is compared with the recorded motion, plus the share of plans whose
body-sphere clearance stays above -1 cm. Planner v1 learned from data in which 38-55% of the windows in these scenes
were the robot standing still; its plans for a standing robot covered 4 cm in the first 0.3 s, against 8 cm elsewhere.

    planner_hesitation.py --data RUN/all --checkpoint v1.pt v2.pt [--n 2000] [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HARD = ("procedural_cat", "original_cat", "published_cat", "narrow_passage", "protected_passage", "transition_passage",
        "table_edges", "open_passage")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--min-speed", type=float, default=.08)
    p.add_argument("--start-speed", type=float, default=.10, help="only situations starting below this pelvis speed")
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    import torch
    import mujoco
    from cat_mjlab.model import assemble_training_xml
    from cat_mjlab.planner.data import load_split
    from cat_mjlab.planner.metrics import ChunkScorer
    from cat_mjlab.planner.model import FlowPlanner, Normalizer, sample
    info = json.loads((args.data / "info.json").read_text())
    hard = [info["scene_types"].index(b) for b in HARD if b in info["scene_types"]]
    held = load_split(args.data, "test")
    speed = held["goal"][:, 4]
    start_speed = np.linalg.norm(held["state"][:, 61:63], axis=-1)          # pelvis planar velocity at the start
    pick = np.nonzero(np.isin(held["meta"][:, 1].astype(int), hard) & (speed >= args.min_speed) & (start_speed < args.start_speed))[0]
    rng = np.random.default_rng(0)
    pick = rng.choice(pick, min(args.n, len(pick)), replace=False)
    device = torch.device(args.device)
    scorer = ChunkScorer(mujoco.MjModel.from_xml_string(assemble_training_xml()), device)
    frames = (3, 6, 10, 20)

    def travel(chunks):            # chunks [n, 20, 35] in the chunk frame (start at the origin)
        return {f"{f / 10:.1f}s": float(np.linalg.norm(chunks[:, f - 1, :2], axis=-1).mean()) for f in frames}

    result = dict(situations=int(len(pick)), recorded=travel(held["chunk"][pick]),
                  recorded_speed_mps=round(float(speed[pick].mean()), 3))
    for path in args.checkpoint:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        model = FlowPlanner(**ck["config"]).to(device).eval(); model.load_state_dict(ck["model"])
        norm = Normalizer(np.zeros((2, 67)), np.zeros((2, 5)), np.zeros((2, 20, 35))).to(device); norm.load_state_dict(ck["norm"])
        out, clear = [], []
        for a in range(0, len(pick), 256):
            idx = pick[a:a + 256]
            st = norm.norm("state", torch.as_tensor(held["state"][idx], device=device))
            gl = norm.norm("goal", torch.as_tensor(held["goal"][idx], device=device))
            grid = torch.as_tensor(held["grid"][idx], device=device)
            x = sample(model, st, gl, grid, samples=1, steps=6, guidance=2.0)[:, 0]
            chunk = norm.denorm("chunk", x)
            clear.append(scorer.score(chunk[:, None], grid, torch.as_tensor(held["goal"][idx], device=device))["min_clear"][:, 0].cpu().numpy())
            out.append(chunk.cpu().numpy())
        chunks = np.concatenate(out); clear = np.concatenate(clear)
        result[str(path)] = dict(travel_m=travel(chunks), clear_share=round(float((clear >= -.01).mean()), 4))
        print(json.dumps({str(path): result[str(path)]}), flush=True)
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
