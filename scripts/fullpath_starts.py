#!/usr/bin/env python3
"""Start states of training scenes for the S3 oracle-plan bank (docs/fullpath_plan.md): the task's own reset (CPU, one world,
seed = 100 + k) for each listed scene, saved in the recording format scripts/fullpath_oracle.py reads (qpos[0], boxes, meta).

    fullpath_starts.py --scenes @list.txt --out /tmp/cat-fullpath/s3/starts --per-scene 1
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-scene", type=int, default=1)
    p.add_argument("--checkpoint", default="/tmp/cat-energy-current-videos/inputs/COMBO_FIX_315808_u150.pt")
    p.add_argument("--bank", default="/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab/data/furniture/tablegap_v1")
    a = p.parse_args()
    from cat_mjlab.runner import create_task
    from cat_mjlab.sim_cpu import CPUSimulation
    from planner_collect import install_subset_loader
    from fullpath_rollout import scene_boxes
    scenes = [l.strip() for l in open(a.scenes[1:]) if l.strip()] if a.scenes.startswith("@") else a.scenes.split(",")
    install_subset_loader(set(scenes))
    contract = torch.load(a.checkpoint, map_location="cpu", weights_only=True, mmap=True)["contract"]
    env = copy.deepcopy(contract["environment_config"])
    for k in ("side_gap_curriculum", "sgs", "robustness", "refusal_training", "tracker", "goal_seed"):
        env.pop(k, None)
    env["push_config"]["enable"] = False
    env["sdf_rate_obs"] = env.get("sdf_rate_obs", False)
    if "teleop" in env:
        env["teleop"].update(fraction=0., cat_fraction=0., cat_scenes=False)
    B = Path(a.bank)
    ns = SimpleNamespace(bank_manifest=Path(str(B) + "_packed/manifest.json"), body_collision_bank=Path(str(B) + "_collision/manifest.json"),
                         body_collision_resets=Path(str(B) + "_resets/manifest.json"), device="cpu", seed=100, num_envs=1,
                         compile_task=False, nconmax=contract["nconmax"], njmax=contract["njmax"], reactive_bank=None,
                         reactive_row=None, goal_seed_scene_ids=None)
    task, sim, _ = create_task(ns, environment_config=env, sim_class=CPUSimulation)
    task.raised_reset_fraction = 0.
    if getattr(task, "reactive_objects", None) is not None:
        task.reactive_objects = None
    names = [s["scene_id"] for s in task.bank.manifest["scenes"]]
    a.out.mkdir(parents=True, exist_ok=True)
    for sc in scenes:
        sid = names.index(sc)
        meta_scene = task.bank.manifest["scenes"][sid]
        for k in range(a.per_scene):
            out = a.out / f"{sc}_0.15_{k}.npz"
            if out.exists():
                continue
            task.generator.manual_seed(100 + k)
            task.reset(None, torch.tensor([sid]))
            q = task.data.qpos[0, :36].numpy().copy()
            meta = dict(scene=sc, speed=.15, trial=k, mix="start", outcome=dict(result="start"), start=list(map(float, meta_scene["start"])),
                        goal=list(map(float, meta_scene["goal"])), planning_benchmark=False)
            np.savez_compressed(out, meta=json.dumps(meta), qpos=q[None], **scene_boxes(task.collision, sid))
    print("starts written", len(list(a.out.glob("*.npz"))))


if __name__ == "__main__":
    main()
