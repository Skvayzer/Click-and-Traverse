#!/usr/bin/env python3
"""Measure how often room robots lose their route ("blocked") and whether they get stuck.

Runs one batch of worlds (one per listed scene, repeated) with a checkpoint's policy and
tracks, per world, its FIRST episode only:
  blocked_steps     steps where navigation['blocked'] is set (command forced to stand, guidance zeroed)
  longest_blocked   longest consecutive blocked stretch (steps)
  moving_blocked    blocked steps with planar speed > 0.05 m/s (what the stand_still cost prices)
  outcome           success / failure reason / still running at the step limit

Usage (CPU, safe while training occupies the GPU):
  .venv-mjlab/bin/python scripts/diagnose_route_blocking.py --checkpoint outputs/.../resume.pt \
      --bank-manifest ... --body-collision-bank ... --body-collision-resets ... \
      --scenes-file scenes.txt --repeats 2 --steps 2000 --device cpu --output out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bank-manifest", type=Path, required=True)
    p.add_argument("--body-collision-bank", type=Path, required=True)
    p.add_argument("--body-collision-resets", type=Path, required=True)
    p.add_argument("--scenes-file", type=Path, required=True, help="one scene_id per line")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                   help="Override a top-level environment_config key, e.g. goal_hold_seconds=1.0")
    args = p.parse_args(argv)

    import torch
    from record_mjlab_rollout import load_policy
    from cat_mjlab.runner import create_task
    torch.manual_seed(args.seed)
    learner, saved = load_policy(args.checkpoint, device=args.device, policy_id=0)
    contract = saved["contract"]
    manifest = json.loads(args.bank_manifest.read_text())
    index = {str(s["scene_id"]): i for i, s in enumerate(manifest["scenes"])}
    names = [l.strip() for l in args.scenes_file.read_text().splitlines() if l.strip()] * args.repeats
    scene_ids = torch.tensor([index[n] for n in names], device=args.device)
    n = len(names)
    factory = SimpleNamespace(checkpoint=args.checkpoint, bank_manifest=args.bank_manifest,
                              body_collision_bank=args.body_collision_bank, body_collision_resets=args.body_collision_resets,
                              device=args.device, seed=args.seed, num_envs=n, compile_task=False,
                              nconmax=contract["nconmax"], njmax=contract["njmax"],
                              reactive_bank=None, reactive_row=None)
    environment = json.loads(json.dumps(contract["environment_config"]))
    for item in args.set:
        key, value = item.split("=", 1); environment[key] = json.loads(value)
        print(f"override {key} = {environment[key]}")
    task, _, _ = create_task(factory, environment_config=environment)
    task.reset(scene_ids=scene_ids)

    live = torch.ones(n, dtype=torch.bool)
    blocked_steps = torch.zeros(n, dtype=torch.long); moving_blocked = torch.zeros(n, dtype=torch.long)
    run = torch.zeros(n, dtype=torch.long); longest = torch.zeros(n, dtype=torch.long)
    length = torch.zeros(n, dtype=torch.long); stand_steps = torch.zeros(n, dtype=torch.long); first_blocked = torch.full((n,), -1, dtype=torch.long)
    outcome = ["running"] * n
    goal_step = torch.full((n,), -1, dtype=torch.long)
    started = time.time()
    for step in range(args.steps):
        with torch.no_grad():
            action = learner.act(task.obs, policy_ids=0, deterministic=args.deterministic)["action"]
        # Read the navigation state the reward of THIS step is computed from, before autoreset.
        blocked_before = task.navigation["blocked"].cpu() & task.navigation["enabled"].cpu()
        speed = torch.linalg.vector_norm(task.data.qvel[:, :2], dim=-1).cpu()
        stand = (task.info["command"][:, 0] < .5).cpu()
        result = task.step(action)
        done = result["done"].cpu()
        b = blocked_before & live
        blocked_steps += b.long(); moving_blocked += (b & (speed > .05)).long()
        run = torch.where(b, run + 1, torch.zeros_like(run)); longest = torch.maximum(longest, run)
        first_blocked = torch.where(b & (first_blocked < 0), torch.full_like(first_blocked, step), first_blocked)
        length += live.long(); stand_steps += (stand & live).long()
        m = result["metrics"]
        hit = m["successful"].cpu().bool() & live & (goal_step < 0)
        goal_step = torch.where(hit, torch.full_like(goal_step, step), goal_step)
        for e in torch.nonzero(done & live).flatten().tolist():
            if goal_step[e] >= 0:
                outcome[e] = "success" + ("+timeout" if bool(result["truncated"][e]) else "")
            elif bool(result["truncated"][e]):
                outcome[e] = "timeout"
            else:
                flags = [k.removeprefix("episode/") for k, v in m.items()
                         if k.startswith("episode/") and v.dtype == torch.bool and bool(v[e])]
                outcome[e] = "fail:" + ",".join(flags)
        live &= ~done
        if step % 250 == 0 or not live.any():
            print(f"step {step:5d}  live {int(live.sum()):3d}/{n}  blocked-now {int((blocked_before & live).sum())}  "
                  f"{time.time() - started:6.0f}s", flush=True)
        if not live.any():
            break
    rows = [dict(scene=names[e], outcome=outcome[e], length=int(length[e]), blocked_steps=int(blocked_steps[e]), stand_command_steps=int(stand_steps[e]),
                 longest_blocked=int(longest[e]), goal_step=int(goal_step[e]), moving_blocked=int(moving_blocked[e]), first_blocked_step=int(first_blocked[e]))
            for e in range(n)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(checkpoint=str(args.checkpoint), steps=args.steps,
                                           deterministic=args.deterministic, rows=rows), indent=2) + "\n")
    total = sum(r["length"] for r in rows)
    print(f"\nblocked share of all steps: {sum(r['blocked_steps'] for r in rows) / max(total, 1):.3f}  "
          f"(moving while blocked: {sum(r['moving_blocked'] for r in rows) / max(total, 1):.3f}; "
          f"stand command: {sum(r['stand_command_steps'] for r in rows) / max(total, 1):.3f})")
    for r in rows:
        print(f"{r['scene'][:58]:58s} {r['outcome'][:40]:40s} len {r['length']:5d}  blocked {r['blocked_steps']:5d}  stand {r['stand_command_steps']:5d}  "
              f"longest {r['longest_blocked']:5d}  goal@{r['goal_step']}")


if __name__ == "__main__":
    main()
