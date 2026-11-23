#!/usr/bin/env python3
"""Record the planner-with-a-field-reflex policy for videos (render with scripts/render_plan_reflex.py).

One world per scene id, one episode each (until it ends or --steps). Per control step: the robot's qpos, the active
whole-body plan as 21 world-frame poses (start pose + 20 frames, 0.1 s apart; captured from the online planner),
the plan clock, which hands are inside the reflex band, whether the stall fallback has dropped the plan, and whether
the world still records. Per replan (event arrays ev_*): every candidate chunk as 21 world-frame poses with its
body-sphere clearance, score and goal-heading offset (the chosen one has the highest score). At the end: each
episode's outcome as the per-scene evaluation scores it (clean / contact / refused / fall, hand contact).

    record_plan_reflex.py CHECKPOINT --scenes ID [ID ...] --out rec.npz [--mix plan=1] [--speed 0.15] [--steps 3000]
                          [--fallback STALL_S HOLD_S]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--scenes", nargs="+", required=True, help="scene ids (one world each) or @file")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mix", default="plan=1", help="tracker sources, e.g. plan=1 (planner + reflex) or none=1 (reactive only)")
    p.add_argument("--speed", type=float, default=0.15, help="operator speed, m/s")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bank", default="data/furniture/tablegap_v1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--goal-seed-cache", type=Path,
                   help="local copy of the exact goal-column cache for the forced recording scenes")
    p.add_argument("--planner-checkpoint", type=Path, help="local copy of the checkpoint's online planner weights")
    p.add_argument("--fallback", type=float, nargs=2, metavar=("STALL_S", "HOLD_S"),
                   help="the stall rule: no progress along the route for STALL_S s drops the plan for HOLD_S s")
    args = p.parse_args()
    scenes = [l.strip() for l in open(args.scenes[0][1:]) if l.strip()] if args.scenes[0].startswith("@") else args.scenes
    from record_mjlab_rollout import load_policy
    from cat_mjlab.runner import create_task
    from planner_render import world_qpos
    dev = "cuda:0"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    contract = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)["contract"]
    env = copy.deepcopy(contract["environment_config"])
    env.pop("side_gap_curriculum", None); env.pop("sgs", None); env.pop("robustness", None)
    env["push_config"]["enable"] = False
    env["operator_speed"] = dict(min=args.speed, max=args.speed, nominal=.525, cutout=.05, near_sdf=.35, near_share=0.,
                                 resample=None, length_scale_max=.525 / .1, anchor_free_below=0., stall_fraction=.2,
                                 rest_start_share=0.)
    if "teleop" in env:
        env["teleop"].update(fraction=0., cat_fraction=0., cat_scenes=False)
    if args.goal_seed_cache:
        goal_seed = env.get("goal_seed")
        if goal_seed is None:
            p.error("--goal-seed-cache requires a checkpoint trained with a goal-column cache")
        goal_seed["cache"] = str(args.goal_seed_cache.resolve())
    tracker = env.get("tracker")
    if args.planner_checkpoint:
        if not tracker or not tracker.get("planner"):
            p.error("--planner-checkpoint requires an online-planner tracker")
        tracker["planner"]["path"] = str(args.planner_checkpoint.resolve())
    if tracker:
        tracker["mix"] = [[a, float(b)] for a, b in (item.split("=", 1) for item in args.mix.split(","))]
        if args.fallback:
            tracker["fallback"] = dict(stall_s=float(args.fallback[0]), hold_s=float(args.fallback[1]))
    elif args.mix != "none=1" or args.fallback:
        raise ValueError("A tracker-free checkpoint requires --mix none=1 and no fallback")
    B = ROOT / args.bank
    ns = SimpleNamespace(bank_manifest=Path(str(B) + "_packed/manifest.json"), body_collision_bank=Path(str(B) + "_collision/manifest.json"),
                         body_collision_resets=Path(str(B) + "_resets/manifest.json"), device=dev, seed=args.seed,
                         num_envs=len(scenes), compile_task=False, nconmax=contract["nconmax"], njmax=contract["njmax"],
                         reactive_bank=None, reactive_row=None,
                         goal_seed_scene_ids=scenes if args.goal_seed_cache else None)
    task, sim, _ = create_task(ns, environment_config=env)
    names = [s["scene_id"] for s in task.bank.manifest["scenes"]]
    missing = [s for s in scenes if s not in names]
    if missing:
        raise ValueError(f"unknown scenes: {missing}")
    sids = torch.tensor([names.index(s) for s in scenes], device=dev)
    n = len(scenes)
    plans = np.zeros((n, 21, 36), np.float32)
    events = dict(t=[], w=[], cands=[], clear=[], score=[], offset=[])
    now = [0]
    op = getattr(task, "online_planner", None)
    if op is not None:
        op.capture = True
        original = task._replan

        def replan(ids):
            ids_plan = ids[task.tracker.is_plan[ids]] if len(ids) else ids
            op.last = None
            original(ids)
            if op.last is not None and len(ids_plan):
                last = op.last
                for k, w in enumerate(ids_plan.tolist()):
                    st, pose = last["state"][k].numpy(), last["pose"][k].numpy()
                    plans[w] = world_qpos(st, last["chunk"][k].numpy(), pose)
                    if "chunks" in last:
                        events["t"].append(now[0]); events["w"].append(w)
                        events["cands"].append(np.stack([world_qpos(st, c, pose) for c in last["chunks"][k].numpy()]).astype(np.float16))
                        for key in ("clear", "score", "offset"):
                            events[key].append(last[key][k].float().numpy())

        task._replan = replan
    task.reset(None, sids)
    learner, _ = load_policy(args.checkpoint, device=dev, policy_id=0)
    reflex = getattr(task, "reflex", None)
    band = float(reflex.get("hand_band", .15)) if reflex and args.mix != "none=1" else 0.
    T = args.steps
    rec = dict(qpos=np.zeros((T + 1, n, 36), np.float32), plan=np.zeros((T + 1, n, 21, 36), np.float32),
               plan_t=np.zeros((T + 1, n), np.float32), released=np.zeros((T + 1, n, 2), bool),
               has_plan=np.zeros((T + 1, n), bool), hand_clear=np.zeros((T + 1, n, 2), np.float32),
               fallback=np.zeros((T + 1, n), bool), body_clear=np.zeros((T + 1, n), np.float32))
    live = np.ones(n, bool); end = np.full(n, T, np.int64)
    outcome = [None] * n
    t0 = time.time()
    for t in range(T + 1):
        now[0] = t
        rec["qpos"][t] = task.data.qpos[:, :36].float().cpu().numpy()
        rec["plan"][t] = plans
        if tracker:
            rec["plan_t"][t] = task.tracker.plan_t.cpu().numpy()
            rec["has_plan"][t] = (task.tracker.is_plan & task.tracker.has_plan & task.tracker.valid).cpu().numpy()
        clear = task.info["sdf"][:, 5:7, 0].float().cpu().numpy()
        rec["hand_clear"][t] = clear
        if band > 0.:
            rec["released"][t] = rec["has_plan"][t, :, None] & (clear < band)
        rec["body_clear"][t] = task.info["sdf"][..., 0].amin(-1).float().cpu().numpy()
        if tracker and hasattr(task.tracker, "fallback_t"):
            rec["fallback"][t] = (task.tracker.fallback_t > 0).cpu().numpy()
        if t == T or not live.any():
            end = np.where(live, t, end); break
        with torch.no_grad():
            out = task.step(learner.act(task.obs, policy_ids=0, deterministic=True)["action"])
        done = out["done"].bool().cpu().numpy()
        if done.any():
            m = out["metrics"]; zero = torch.zeros_like(out["done"]).bool()
            goal = m.get("episode/goal_reached", m["successful"]).bool().cpu().numpy()
            obstacle = m.get("episode/obstacle", zero).bool().cpu().numpy()
            fall = m.get("episode/fall", zero).bool().cpu().numpy()
            hands = m.get("episode/body_collision_hands", zero).bool().cpu().numpy()
            for w in np.nonzero(done & live)[0]:
                kind = "clean" if goal[w] and not obstacle[w] else "contact" if obstacle[w] else "fall" if fall[w] else "refused"
                outcome[w] = dict(result=kind, hand_contact=bool(hands[w]), steps=int(t + 1))
                # The next pose would be the automatically reset episode.
                live[w] = False; end[w] = t
        if t % 500 == 0:
            print(f"step {t}: {int(live.sum())} of {n} still running, {time.time() - t0:.0f} s", flush=True)
    meta = dict(checkpoint=str(args.checkpoint), scenes=scenes, mix=args.mix, speed=args.speed, band_m=band,
                outcome=outcome, end=end.tolist(), dt=float(task.dt), bank=args.bank, fallback=args.fallback)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ev = {}
    if events["t"]:
        ev = dict(ev_t=np.array(events["t"]), ev_w=np.array(events["w"]), ev_cands=np.stack(events["cands"]),
                  ev_clear=np.stack(events["clear"]), ev_score=np.stack(events["score"]), ev_offset=np.stack(events["offset"]))
    np.savez_compressed(args.out, meta=json.dumps(meta), **{k: v[:int(end.max()) + 1] for k, v in rec.items()}, **ev)
    print("RECORDED", json.dumps(dict(out=str(args.out), outcome=outcome)), flush=True)


if __name__ == "__main__":
    main()
