#!/usr/bin/env python3
"""Per-scene deterministic evaluation of one policy: K episodes in every selected scene.

Why: distillation routes each scene to the expert that actually solves it (scripts/build_distill_routing.py).
The exclusion lists split scenes by geometry (openings >= 0.45 m -> traversal), which says nothing about who is
better on the borderline, and some scenes have no competent teacher at all (published-side-crouch2: narrow 0%).

Each env runs a queue of (scene, repeat) jobs: every scene gets exactly --episodes episodes (envs pick the next job
at reset; once the queue is empty, extra episodes are ignored). Field-driven (no keyboard) for CAT scenes; rooms
follow their route. Environment = the checkpoint's own training config (side-gap curriculum and pushes off), or
--env-from another checkpoint's (e.g. the student env).

Per scene: episodes, clean (goal and no touch), contact, refused (no goal, no touch, no fall), fall, hands.
Usage: eval_per_scene.py CHECKPOINT --scenes cat|room|all [--episodes 16] [--num-envs 16384] --out FILE.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--scenes", default="cat", help="cat | room | all | @file with scene ids")
    p.add_argument("--episodes", type=int, default=16, help="Episodes per scene")
    p.add_argument("--num-envs", type=int, default=16384)
    p.add_argument("--bank", default="data/furniture/sidegap_v2")
    p.add_argument("--env-from", help="Take the environment config from this checkpoint instead")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from record_mjlab_rollout import load_policy
    from cat_mjlab.runner import create_task
    dev = "cuda:0"
    contract = torch.load(args.env_from or args.checkpoint, map_location="cpu", weights_only=True, mmap=True)["contract"]
    env = copy.deepcopy(contract["environment_config"])
    env.pop("side_gap_curriculum", None)                 # standard CAT starts, every width
    env["push_config"]["enable"] = False
    if "teleop" in env:
        env["teleop"].update(fraction=0., cat_fraction=0., cat_scenes=False)
    B = ROOT / args.bank
    ns = SimpleNamespace(bank_manifest=Path(str(B) + "_packed/manifest.json"), body_collision_bank=Path(str(B) + "_collision/manifest.json"),
                         body_collision_resets=Path(str(B) + "_resets/manifest.json"), device=dev, seed=args.seed, num_envs=args.num_envs,
                         compile_task=False, nconmax=contract["nconmax"], njmax=contract["njmax"], reactive_bank=None, reactive_row=None)
    task, sim, _ = create_task(ns, environment_config=env)
    scenes = task.bank.manifest["scenes"]
    if args.scenes.startswith("@"):
        wanted = {l.strip() for l in open(args.scenes[1:]) if l.strip()}
        ids = [i for i, s in enumerate(scenes) if s["scene_id"] in wanted]
    else:
        ids = [i for i, s in enumerate(scenes) if args.scenes == "all" or s.get("task_kind") == args.scenes]
    S = len(ids)
    jobs = torch.tensor(ids, device=dev).repeat(args.episodes)
    jobs = jobs[torch.randperm(len(jobs), device=dev, generator=torch.Generator(device=dev).manual_seed(args.seed))]
    cursor = [0]
    env_scene = torch.full((task.num_envs,), -1, dtype=torch.long, device=dev)      # -1: no job (ignored)
    original_reset = task.reset

    def queued(env_ids=None, scene_ids=None):
        env_ids = task.all_ids if env_ids is None else env_ids.long()
        n = len(env_ids)
        take = min(n, len(jobs) - cursor[0])
        chosen = torch.full((n,), -1, dtype=torch.long, device=dev)
        if take > 0:
            chosen[:take] = jobs[cursor[0]:cursor[0] + take]; cursor[0] += take
        env_scene[env_ids] = chosen
        filler = torch.tensor(ids, device=dev)[torch.randint(S, (n,), device=dev)]
        return original_reset(env_ids, torch.where(chosen >= 0, chosen, filler))

    task.reset = queued
    task.reset()
    learner, _ = load_policy(args.checkpoint, device=dev, policy_id=0)
    cols = ("episodes", "clean", "contact", "refused", "fall", "hands", "length")
    stats = torch.zeros((len(scenes), len(cols)), dtype=torch.float64, device=dev)
    target = S * args.episodes
    start = time.time()
    for t in range(args.max_steps):
        before = env_scene.clone()
        with torch.no_grad():
            out = task.step(learner.act(task.obs, policy_ids=0, deterministic=not args.stochastic)["action"])
        done = out["done"].bool() & (before >= 0)
        if bool(done.any()):
            m = out["metrics"]; zero = torch.zeros_like(done)
            goal = m.get("episode/goal_reached", m["successful"]).bool()
            obstacle = m.get("episode/obstacle", zero).bool(); fall = m.get("episode/fall", zero).bool()
            hands = m.get("episode/body_collision_hands", zero).bool()
            flags = (done, done & goal & ~obstacle, done & obstacle, done & ~goal & ~obstacle & ~fall, done & fall, done & hands)
            d = torch.nonzero(done).flatten(); sc = before[d]
            for c, f in enumerate(flags):
                stats[:, c].index_add_(0, sc, f[d].double())
            stats[:, 6].index_add_(0, sc, m["episode_length"][d].double())
        finished = int(stats[:, 0].sum())
        if t % 500 == 0:
            print(f"step {t}: {finished}/{target} episodes, {time.time() - start:.0f}s", flush=True)
        if finished >= target:
            break
    result = {}
    for i in ids:
        e = float(stats[i, 0])
        if e == 0:
            continue
        result[scenes[i]["scene_id"]] = dict(episodes=int(e), **{c: round(float(stats[i, k]) / e, 4) for k, c in enumerate(cols) if k},)
    E = stats[ids, 0].sum().clamp_min(1)
    summary = {c: round(float(stats[ids, k].sum() / E), 4) for k, c in enumerate(cols) if k}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(checkpoint=str(args.checkpoint), env_from=args.env_from, scenes=args.scenes, episodes_per_scene=args.episodes,
                                        deterministic=not args.stochastic, finished=int(stats[:, 0].sum()), target=target,
                                        seconds=round(time.time() - start), summary=summary, per_scene=result), indent=1) + "\n")
    print(json.dumps(dict(out=str(args.out), finished=int(stats[:, 0].sum()), target=target, summary=summary)))


if __name__ == "__main__":
    main()
