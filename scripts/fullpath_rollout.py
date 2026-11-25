#!/usr/bin/env python3
"""One deterministic episode of a policy in one scene on CPU MuJoCo, recorded for the full-path study (docs/fullpath_plan.md).

Records per control step the robot qpos, the plan clock / validity / reflex release, and per replan the chosen 21-frame
whole-body chunk and every candidate (world qpos, as scripts/planner_render.world_qpos builds them) with their sphere-cover
clearance; the first body contact (step, proxy regions, the obstacle boxes that touched); the episode outcome as
eval_per_scene.py scores it; and the scene's collision boxes (for the exact-geometry audit, scripts/fullpath_audit.py).

Modes: --mix plan=1 (planner v1 in the loop), --mix none=1 (same network, no reference), a tracker-free checkpoint with
--mix none=1, or --oracle FILE (S2: a precomputed full start->goal plan fed as distance-indexed pelvis waypoints and
upper-body targets through the tracker's 'plan' source; see cat_mjlab/fullpath.py).

    fullpath_rollout.py CHECKPOINT --scene ID --speed 0.15 --trial 0 --out rec.npz [--planning-benchmark] [--mix plan=1]
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


def scene_boxes(checker, scene):
    b = checker.bank
    a = int(b["scene_box_offsets"][scene]); n = int(b["scene_box_counts"][scene])
    return dict(box_center=b["centers"][a:a + n].cpu().numpy(), box_rot=b["rotations"][a:a + n].cpu().numpy(),
                box_half=b["half_sizes"][a:a + n].cpu().numpy(), box_first=np.int64(a))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--scene", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mix", default="plan=1")
    p.add_argument("--speed", type=float, default=0.15)
    p.add_argument("--trial", type=int, default=0, help="episode index (0/1): task-reset seed, or the certified start")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=9000)
    p.add_argument("--bank", default="/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab/data/furniture/tablegap_v1")
    p.add_argument("--planning-benchmark", action="store_true")
    p.add_argument("--goal-seed-cache", type=Path)
    p.add_argument("--planner-checkpoint", type=Path)
    p.add_argument("--env-from", help="task config from this checkpoint (tracker-free policies keep their own by default)")
    p.add_argument("--oracle", type=Path, help="S2 full-path plan (npz from scripts/fullpath_oracle.py)")
    p.add_argument("--oracle-lead", type=float, default=0.0, help="oracle: waypoint clock offset (s) ahead of the robot")
    p.add_argument("--oracle-library", type=Path, help="S2: plan the oracle at the episode start with this clip library")
    p.add_argument("--oracle-margins", default="0.08,0.06,0.04,0.02", help="margins tried in order (m)")
    p.add_argument("--oracle-budget", type=float, default=150., help="search seconds per margin")
    p.add_argument("--no-candidates", action="store_true")
    args = p.parse_args()
    from record_mjlab_rollout import load_policy
    from cat_mjlab.runner import create_task
    from cat_mjlab.sim_cpu import CPUSimulation
    from planner_collect import install_subset_loader
    from planner_render import world_qpos
    from eval_per_scene import benchmark_start
    torch.set_num_threads(max(1, torch.get_num_threads()))
    install_subset_loader({args.scene})
    seed = 1 + args.trial
    torch.manual_seed(seed)
    contract = torch.load(args.env_from or args.checkpoint, map_location="cpu", weights_only=True, mmap=True)["contract"]
    env = copy.deepcopy(contract["environment_config"])
    env.pop("side_gap_curriculum", None); env.pop("sgs", None); env.pop("robustness", None); env.pop("refusal_training", None)
    env["push_config"]["enable"] = False
    env["operator_speed"] = dict(min=args.speed, max=args.speed, nominal=.525, cutout=.05, near_sdf=.35, near_share=0.,
                                 resample=None, length_scale_max=.525 / .1, anchor_free_below=0., stall_fraction=.2,
                                 rest_start_share=0.)
    if args.planning_benchmark:
        env["eval_goal_stop"] = True
    if "teleop" in env:
        env["teleop"].update(fraction=0., cat_fraction=0., cat_scenes=False)
    if args.goal_seed_cache and env.get("goal_seed"):
        env["goal_seed"]["cache"] = str(args.goal_seed_cache.resolve())
    tracker = env.get("tracker")
    use_oracle = bool(args.oracle or args.oracle_library)
    if use_oracle and not tracker:
        p.error("--oracle needs a tracker checkpoint")
    if tracker:
        mix = "plan=1" if use_oracle else args.mix
        tracker["mix"] = [[a, float(b)] for a, b in (item.split("=", 1) for item in mix.split(","))]
        if args.planner_checkpoint and tracker.get("planner"):
            tracker["planner"]["path"] = str(args.planner_checkpoint.resolve())
    elif args.mix != "none=1":
        p.error("a tracker-free checkpoint runs with --mix none=1")
    B = Path(args.bank)
    ns = SimpleNamespace(bank_manifest=Path(str(B) + "_packed/manifest.json"), body_collision_bank=Path(str(B) + "_collision/manifest.json"),
                         body_collision_resets=Path(str(B) + "_resets/manifest.json"), device="cpu", seed=seed, num_envs=1,
                         compile_task=False, nconmax=contract["nconmax"], njmax=contract["njmax"], reactive_bank=None,
                         reactive_row=None, goal_seed_scene_ids={args.scene} if args.goal_seed_cache else None)
    task, sim, _ = create_task(ns, environment_config=env, sim_class=CPUSimulation)
    task.collision_every_substep = False
    scenes = task.bank.manifest["scenes"]
    names = [s["scene_id"] for s in scenes]
    sid = names.index(args.scene)
    meta_scene = scenes[sid]
    if args.planning_benchmark:
        if meta_scene["source"]["planning_benchmark"]["family"] != "table-gap":
            task.bank.reset_is_cat[sid] = False; task.bank.crossed_is_plane[sid] = False
            task.bank.reset_xy_scale[sid] = .08; task.bank.episode_lengths[sid] = 4000
        task.raised_reset_fraction = 0.
        task.generator.manual_seed(seed)
    sids = torch.tensor([sid])
    # ---- first contact: wrap the episode-ending collision check
    from cat_mjlab.diagnose import colliding_boxes
    checker = task.collision
    first = dict(step=-1)
    now = [0]

    def collision(scene_ids, data):
        flags = checker(scene_ids, data)
        if first["step"] < 0 and bool(flags.any()) and now[0] > 0:
            boxes, _ = colliding_boxes(checker.compiled, checker.bank, scene_ids, data.xpos, data.xmat,
                                       checker.metadata["static_candidate_count"])
            first.update(step=now[0], regions=flags[0].cpu().numpy().copy(), shape_box=boxes[0].cpu().numpy().copy(),
                         qpos=task.data.qpos[0, :36].cpu().numpy().copy())
        return flags
    # ---- the first terminating obstacle cause (task._termination: body proxy, 13-point field < -4 cm, elbows, route corridor)
    original_term = task._termination
    cause = dict(step=-1)

    def termination(regions, contacts):
        out_ = original_term(regions, contacts)
        if cause["step"] < 0 and now[0] > 0:
            i = task.info; thr = -float(task.config.get("term_collision_threshold", .04)); grace = bool(i["step"][0] >= 50)
            fields = bool((i["sdf"][0] < thr).any()) and grace
            elbows = bool((i["elbow_clearance"][0] < thr).any()) and grace
            root = bool(task.navigation["enabled"][0] & task.navigation["violation"][0])
            body = bool(regions[0].any())
            if fields or elbows or root or body:
                cause.update(step=now[0], fields=fields, elbows=elbows, root=root, body=body,
                             sdf_point=int(i["sdf"][0, :, 0].argmin()), sdf_min=round(float(i["sdf"][0, :, 0].min()), 4),
                             qpos=task.data.qpos[0, :36].cpu().numpy().copy())
        return out_
    task._termination = termination
    # ---- plan capture
    events = dict(t=[], chosen=[], cands=[], sclear=[], score=[], cont=[], interval=[])
    op = getattr(task, "online_planner", None)
    oracle = [None]
    if use_oracle:
        task._replan = lambda ids, **kw: oracle[0].replan(ids, events, now[0]) if oracle[0] is not None else None
    elif op is not None and "plan" in args.mix:
        op.capture = True
        original = task._replan

        def replan(ids, **kw):
            ids_plan = ids[task.tracker.is_plan[ids]] if len(ids) else ids
            op.last = None
            original(ids, **kw)
            if op.last is not None and len(ids_plan):
                last = op.last
                st, pose = last["state"][0].numpy(), last["pose"][0].numpy()
                events["t"].append(now[0])
                events["chosen"].append(world_qpos(st, last["chunk"][0].numpy(), pose).astype(np.float32))
                if not args.no_candidates:
                    events["cands"].append(np.stack([world_qpos(st, c, pose) for c in last["chunks"][0].numpy()]).astype(np.float32))
                events["sclear"].append(last["clear"][0].float().numpy()); events["score"].append(last["score"][0].float().numpy())
                events["cont"].append(float(task.plan_cont[0])); events["interval"].append(float(task.tracker.replan_in[0]))
        task._replan = replan
    start = benchmark_start(task, sids, torch.tensor([args.trial]), args.trials) if args.planning_benchmark else {}
    task.reset(None, sids, **start)
    task.collision = collision
    oracle_meta, oracle_plan = None, None
    if use_oracle:
        from cat_mjlab.fullpath import OracleFeeder, ClipLibrary, plan_scene
        from cat_mjlab.exact_geometry import ExactGeometry
        from cat_mjlab.tracking import SOURCES
        if args.oracle:
            oracle_plan = dict(np.load(args.oracle)); oracle_meta = json.loads(str(oracle_plan.get("meta", "{}")))
        else:
            geo = ExactGeometry("cpu"); lib = ClipLibrary.load(args.oracle_library)
            sb = scene_boxes(checker, sid)
            q0 = task.data.qpos[0, :36].double().numpy()
            gxy = np.array(meta_scene["goal"][:2], dtype=np.float64)
            u = gxy - np.array(meta_scene["start"][:2], dtype=np.float64)
            radius = .3 if args.planning_benchmark else None
            tried = []
            for mg in [float(x) for x in args.oracle_margins.split(",")]:
                r = plan_scene(lib, geo, sb, q0, gxy, u, margin=mg, budget_s=args.oracle_budget, goal_radius=radius)
                tried.append(dict(margin=mg, ok=r["ok"], seconds=r["seconds"], expansions=r["expansions"], progress=r["progress"]))
                if r["ok"]:
                    break
            same = [lib.sources[c][0] == args.scene for c in r["clips"]]
            oracle_meta = dict(ok=r["ok"], margin=r["margin"] if r["ok"] else None, min_sep=r["min_sep"], tried=tried,
                               route=r["route"], progress=r["progress"], same_scene_clips=float(np.mean(same)) if same else None,
                               mode_share=dict(zip(("sideways", "crouch", "high_step", "hands_raised", "hands_tucked"),
                                                   np.asarray(r["modes"]).mean(0).round(3).tolist())) if len(r["modes"]) else None)
            oracle_plan = dict(qpos=r["qpos"]) if r["ok"] else None
        if oracle_plan is not None:
            oracle[0] = OracleFeeder(oracle_plan, task, lead=args.oracle_lead)
            task.tracker.has_plan[0] = False; task.tracker.replan_in[0] = 0.
            task._replan(task.tracker.due())
        else:                                        # no verified plan: the same network without a reference
            task.tracker.source[0] = SOURCES.index("none"); task.tracker.has_plan[0] = False
        print("ORACLE", json.dumps(oracle_meta), flush=True)
    learner, _ = load_policy(args.checkpoint, device="cpu", policy_id=0)
    reflex = getattr(task, "reflex", None)
    band = float(reflex.get("hand_band", .15)) if reflex and tracker and (args.mix != "none=1" or use_oracle) else 0.
    T = args.max_steps
    rec = dict(qpos=np.zeros((T + 1, 36), np.float32), plan_t=np.zeros(T + 1, np.float32), valid=np.zeros(T + 1, bool),
               released=np.zeros((T + 1, 2), bool), body_clear=np.zeros(T + 1, np.float32))
    outcome = None; end = T; t0 = time.time()
    for t in range(T + 1):
        now[0] = t
        rec["qpos"][t] = task.data.qpos[0, :36].float().numpy()
        if tracker:
            rec["plan_t"][t] = float(task.tracker.plan_t[0]); rec["valid"][t] = bool(task.tracker.valid[0])
        clear = task.info["sdf"][0, :, 0].float()
        rec["body_clear"][t] = float(clear[:11].amin())
        if band > 0.:
            rec["released"][t] = bool(rec["valid"][t]) & (clear[5:7] < band).numpy()
        if t == T:
            break
        with torch.no_grad():
            out = task.step(learner.act(task.obs, policy_ids=0, deterministic=True)["action"])
        if bool(out["done"][0]):
            m = out["metrics"]; zero = torch.zeros_like(out["done"]).bool()
            goal = bool(m.get("episode/goal_reached", m["successful"])[0]); obstacle = bool(m.get("episode/obstacle", zero)[0])
            fall = bool(m.get("episode/fall", zero)[0]); hands = bool(m.get("episode/body_collision_hands", zero)[0])
            kind = "clean" if goal and not obstacle else "contact" if obstacle else "fall" if fall else "refused"
            outcome = dict(result=kind, hand_contact=hands, steps=int(t + 1), length=int(m["episode_length"][0]))
            end = t + 1
            break
    if outcome is None:
        outcome = dict(result="refused", hand_contact=False, steps=T, length=T, capped=True)
    meta = dict(checkpoint=str(args.checkpoint), scene=args.scene, mix=args.mix, oracle=oracle_meta,
                speed=args.speed, trial=args.trial, band_m=band, outcome=outcome, dt=float(task.dt), bank=str(args.bank),
                start=list(map(float, meta_scene["start"])), goal=list(map(float, meta_scene["goal"])),
                planning_benchmark=bool(args.planning_benchmark), first_contact_step=int(first["step"]),
                wall_s=round(time.time() - t0, 1))
    if cause["step"] >= 0:
        meta["term_cause"] = {k: v for k, v in cause.items() if k != "qpos"}
    if first["step"] >= 0:
        from cat_mjlab.collision import REGIONS
        meta["first_contact_regions"] = [r for r, f in zip(REGIONS, first["regions"]) if f]
    out = dict(meta=json.dumps(meta), **{k: v[:end + 1] for k, v in rec.items()}, **scene_boxes(checker, sid))
    if first["step"] >= 0:
        out.update(contact_regions=first["regions"], contact_shape_box=first["shape_box"], contact_qpos=first["qpos"])
    if cause["step"] >= 0:
        out.update(term_qpos=cause["qpos"])
    if oracle_plan is not None:
        out.update(oracle_qpos=np.asarray(oracle_plan["qpos"], np.float32))
    if events["t"]:
        out.update(ev_t=np.array(events["t"]), ev_chosen=np.stack(events["chosen"]), ev_cont=np.array(events["cont"]),
                   ev_interval=np.array(events["interval"]))
        if events["sclear"]:
            out.update(ev_sclear=np.stack(events["sclear"]), ev_score=np.stack(events["score"]))
        if events["cands"]:
            out.update(ev_cands=np.stack(events["cands"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("RECORDED", json.dumps(dict(out=str(args.out), **outcome, wall_s=meta["wall_s"])), flush=True)


if __name__ == "__main__":
    main()
