#!/usr/bin/env python3
"""Per-scene deterministic evaluation of one policy: K episodes in every selected scene.

Why: distillation routes each scene to the expert that actually solves it (scripts/build_distill_routing.py).
The exclusion lists split scenes by geometry (openings >= 0.45 m -> traversal), which says nothing about who is
better on the borderline, and some scenes have no competent teacher at all (published-side-crouch2: narrow 0%).

Each env runs a queue of (scene, repeat) jobs: every scene gets exactly --episodes episodes (envs pick the next job
at reset; once the queue is empty, extra episodes are ignored). Field-driven (no keyboard) for CAT scenes; rooms
follow their route. Environment = the checkpoint's own training config (side-gap curriculum and pushes off), or
--env-from another checkpoint's (e.g. the student env).

Per scene: episodes, clean (goal and no touch), contact, refused (no goal, no touch, no fall), fall, hands, and the
energy used: joint mechanical work sum_j |tau_j qdot_j| dt per episode (J, the training's energy term), the root's path
length (m), mean power (W) and cost of transport (energy / (robot mass * g * path length), episodes that moved >= 0.5 m).
Usage: eval_per_scene.py CHECKPOINT --scenes cat|room|all [--episodes 16] [--num-envs 16384] --out FILE.json
Speed sweep in one run: --operator-speeds 0.1 0.2 0.35 0.525 --out-dir DIR (every scene gets --episodes episodes at each
speed, DIR/speed_<s>.json per speed, same format as --operator-speed S --out FILE.json).
Baselines from an array archive (the released CAT generalist, outputs/mjlab_migration_20260919/original-cat-expanded.npz):
CHECKPOINT ending in .npz needs --env-from (a .pt checkpoint for the task); the task is then built with the native 222/310
inputs (no SDF-rate or tracker block), which is the archive's contract.
"""
from __future__ import annotations

import argparse
import hashlib
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


from cat_mjlab.energy import SHAPE, report_energy


def benchmark_start(task, scene_ids, trial, trials):
    """Shared narrow start envelope in the geometry's declared facing frame."""
    pose = task.init_q.expand(len(scene_ids), -1).clone()
    pose[:, :3] = task.bank.starts[scene_ids]
    spread = 2. * trial / max(trials - 1, 1) - float(trials > 1)
    yaw = (task.bank.reset_yaws[scene_ids] + .05 * spread) / 2
    zeros = torch.zeros_like(yaw)
    orientation = torch.stack((yaw.cos(), zeros, zeros, yaw.sin()), dim=1)
    pose[:, 3:7] = task.math.quat_mul(pose[:, 3:7], orientation)
    velocity = pose.new_zeros(len(scene_ids), 35)
    velocity[:, 0] = .015 * spread * (2 * yaw).cos()
    velocity[:, 1] = .015 * spread * (2 * yaw).sin()
    return dict(initial_pose=pose, initial_velocity=velocity)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--scenes", default="cat", help="cat | room | all | @file with scene ids")
    p.add_argument("--episodes", type=int, default=16, help="Episodes per scene")
    p.add_argument("--num-envs", type=int, default=16384)
    p.add_argument("--bank", default="data/furniture/sidegap_v2")
    p.add_argument("--env-from", help="Take the environment config from this checkpoint instead")
    p.add_argument("--brace-config", type=Path, help="published BRACE model-sidecar YAML")
    p.add_argument("--brace-robot", type=Path, help="published BRACE G1 MJCF for reference kinematics")
    p.add_argument("--brace-provider", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-compile", action="store_true", help="Skip torch.compile of the task (about 4x slower at 16k envs)")
    p.add_argument("--nominal", action="store_true", help="Drop the robustness perturbations (delay, pose/mount error, friction/mass/COM/torque DR): skill only")
    p.add_argument("--operator-speed", type=float, help="Fixed operator speed (m/s): field direction at this speed, no stop latch, "
                   "cut-out 0.05 m/s, episode time scaled by 0.525/speed (the slow-traversal command; works for any checkpoint)")
    p.add_argument("--operator-speeds", type=float, nargs="+", metavar="S",
                   help="Several fixed operator speeds in one run (needs --out-dir): episodes at every speed in parallel")
    p.add_argument("--tracker-mix", help="Override the reference sources of a tracker checkpoint, e.g. plan=1 (planner with a "
                                         "field reflex, every episode) or none=1 (the same policy with no reference)")
    p.add_argument("--tracker-hold-plan", type=float, help="Override the share of reactive standing worlds on a stand-still plan")
    p.add_argument("--tracker-fallback", type=float, nargs=2, metavar=("STALL_S", "HOLD_S"),
                   help="The monitor's stall rule: a plan world without progress along its route for STALL_S s acts without "
                        "its plan for HOLD_S s, then replans")
    p.add_argument("--stall-audit", action="store_true", help="log before/during-stall terms for refused scenes")
    p.add_argument("--refusal-fix", choices=("fallback", "margin", "both"))
    p.add_argument("--planning-benchmark", action="store_true", help="use certified benchmark starts and report traversal metrics")
    p.add_argument("--plan-only", action="store_true", help="hide actor obstacle inputs while retaining plan references")
    p.add_argument("--refused-rewards", action="store_true",
                   help="Aggregate named scaled terms per transition over eventually refused episodes")
    p.add_argument("--goal-seed-cache", type=Path, help="exact goal-column cache for the selected scene IDs")
    p.add_argument("--planner-checkpoint", type=Path, help="relocate the trained online planner checkpoint")
    p.add_argument("--blind-variant", choices=("static", "moving", "throw"))
    p.add_argument("--pushes", action="store_true", help="keep the checkpoint's random push disturbances (A2 robustness); default: pushes off")
    guide_mode = p.add_mutually_exclusive_group()
    guide_mode.add_argument("--write-blind-guide", type=Path, help="write matched episodes and obstacle schedule without evaluation")
    guide_mode.add_argument("--blind-guide", type=Path, help="precomputed obstacle schedules shared by all policy checkpoints")
    p.add_argument("--blind-record", type=Path, help="save short traces of both predefined guide episodes")
    p.add_argument("--blind-replay-starts", action="store_true", help="replay guide starts for a checkpoint with a different reset")
    p.add_argument("--out", type=Path)
    p.add_argument("--out-dir", type=Path)
    args = p.parse_args()
    blind = args.write_blind_guide is not None or args.blind_guide is not None
    if blind != (args.blind_variant is not None) or (args.blind_record and not args.blind_guide):
        p.error("Blind guide generation/evaluation needs --blind-variant, and tracing needs an existing guide")
    if args.blind_variant == "throw" and args.write_blind_guide:
        p.error("Throw guides come from scripts/make_throw_guide.py")
    if args.blind_replay_starts and (not args.blind_guide or args.planning_benchmark):
        p.error("Blind start replay requires an existing blind guide")
    if blind and (args.operator_speed is None or args.operator_speeds or args.episodes < 1 or args.stochastic or not args.nominal):
        p.error("Plan-blind evaluation requires nominal deterministic episodes and one fixed operator speed")
    if args.operator_speeds and (args.operator_speed is not None or args.out_dir is None or args.out is not None):
        p.error("--operator-speeds needs --out-dir (and neither --operator-speed nor --out)")
    if not args.operator_speeds and args.out is None and not args.write_blind_guide:
        p.error("--out is required")
    from record_mjlab_rollout import load_policy
    from cat_mjlab.runner import create_task
    dev = "cuda:0"
    archive = str(args.checkpoint).endswith(".npz")
    brace = str(args.checkpoint).endswith(".onnx")
    if brace and not (args.env_from and args.brace_config and args.brace_robot and args.tracker_mix == "plan=1"
                      and args.nominal and not args.stochastic):
        p.error("BRACE requires --env-from, its robot and config, --tracker-mix plan=1 and --nominal")
    if brace and (blind or args.refusal_fix or args.plan_only):
        p.error("BRACE evaluates direct plan tracking on a static scene bank")
    if archive and not args.env_from:
        p.error("an .npz policy needs --env-from <a .pt checkpoint> for the task")
    contract = torch.load(args.env_from or args.checkpoint, map_location="cpu", weights_only=True, mmap=True)["contract"]
    env = copy.deepcopy(contract["environment_config"])
    if archive:
        env["sdf_rate_obs"] = False; env.pop("tracker", None)
    env.pop("side_gap_curriculum", None)                 # standard CAT starts, every width
    env["push_config"]["enable"] = bool(args.pushes)
    env.pop("sgs", None)                                 # every scene gets its K jobs, no success-guided resampling
    env.pop("refusal_training", None)                    # compare policies with the same physical deadline
    if args.planning_benchmark:
        env["eval_goal_stop"] = True
    if brace:
        env["dm_rand_config"]["enable_pd"] = False
    speeds = [float(x) for x in args.operator_speeds] if args.operator_speeds else (
        [float(args.operator_speed)] if args.operator_speed is not None else [None])
    if speeds[0] is not None:
        # One operator dict for the process; with several speeds each episode's speed is set at its reset (queued below).
        env["operator_speed"] = dict(min=min(speeds), max=max(speeds), nominal=.525, cutout=.05, near_sdf=.35, near_share=0.,
                                     resample=None, length_scale_max=.525 / .1, anchor_free_below=0., stall_fraction=.2,
                                     rest_start_share=0.)
    if args.nominal:
        env.pop("robustness", None)
    if args.tracker_mix or args.tracker_hold_plan is not None or args.tracker_fallback:
        if not env.get("tracker"):
            p.error("--tracker-mix / --tracker-hold-plan need a checkpoint trained with the tracker")
        if args.tracker_mix:
            env["tracker"]["mix"] = [[a, float(b)] for a, b in (item.split("=", 1) for item in args.tracker_mix.split(","))]
        if args.tracker_hold_plan is not None:
            env["tracker"]["hold_plan"] = float(args.tracker_hold_plan)
        if args.tracker_fallback:
            env["tracker"]["fallback"] = dict(stall_s=float(args.tracker_fallback[0]), hold_s=float(args.tracker_fallback[1]))
    if args.refusal_fix:
        if not env.get("tracker", {}).get("planner") or args.tracker_mix != "plan=1":
            p.error("Refusal fixes require an online planner and --tracker-mix plan=1")
        if args.refusal_fix in ("fallback", "both"):
            env["tracker"]["fallback"] = dict(stall_s=2., until_progress=True, progress_m=.05)
        if args.refusal_fix in ("margin", "both"):
            env["tracker"]["planner"]["clearance_margins"] = [.05, .03]
    if "teleop" in env:
        env["teleop"].update(fraction=0., cat_fraction=0., cat_scenes=False)
    wanted = None
    if args.scenes.startswith("@"):
        wanted = {line.strip() for line in Path(args.scenes[1:]).read_text().splitlines() if line.strip()}
    if args.goal_seed_cache:
        if not env.get("goal_seed") or not wanted:
            p.error("A relocated goal cache needs a column-goal policy and an explicit @scene list")
        env["goal_seed"]["cache"] = str(args.goal_seed_cache.resolve())
    if args.plan_only:
        if args.tracker_mix != "plan=1" or args.refusal_fix:
            p.error("Plan-only needs --tracker-mix plan=1 without a recovery variant")
        env["eval_hide_actor_fields"] = True
    if args.planner_checkpoint:
        if not env.get("tracker", {}).get("planner"):
            p.error("A relocated planner needs an online-planner tracker")
        env["tracker"]["planner"]["path"] = str(args.planner_checkpoint.resolve())
    if args.write_blind_guide and (not env.get("tracker", {}).get("planner") or args.tracker_mix != "plan=1"):
        p.error("Generating a blind guide needs an online-planner tracker and --tracker-mix plan=1")
    if args.planning_benchmark and not wanted:
        p.error("The planning benchmark requires an explicit @scene list")
    B = ROOT / args.bank
    ns = SimpleNamespace(bank_manifest=Path(str(B) + "_packed/manifest.json"), body_collision_bank=Path(str(B) + "_collision/manifest.json"),
                         body_collision_resets=Path(str(B) + "_resets/manifest.json"), device=dev, seed=args.seed, num_envs=args.num_envs,
                         compile_task=not args.no_compile and not args.write_blind_guide, nconmax=contract["nconmax"],
                         njmax=contract["njmax"], reactive_bank=None, reactive_row=None,
                         goal_seed_scene_ids=wanted if args.goal_seed_cache else None)
    sim_class = None
    if blind:
        from cat_mjlab.plan_blind import BlindObstacle, BlindSimulation, BlindTrace, candidate_placements, replay_assignments
        sim_class = BlindSimulation
    task, sim, _ = create_task(ns, environment_config=env, sim_class=sim_class)
    scenes = task.bank.manifest["scenes"]
    if wanted is not None:
        ids = [i for i, s in enumerate(scenes) if s["scene_id"] in wanted]
        if len(ids) != len(wanted):
            p.error("The selected scene list includes an unknown bank scene ID")
    else:
        ids = [i for i, s in enumerate(scenes) if args.scenes == "all" or s.get("task_kind") == args.scenes]
    if args.planning_benchmark:
        if not all("planning_benchmark" in scenes[i]["source"] for i in ids):
            p.error("Planning benchmark selection contains unrelated scenes")
        field_worlds = [i for i in ids if scenes[i]["source"]["planning_benchmark"]["family"] != "table-gap"]
        # These layouts retain native field guidance, with the same tight start region
        # and goal-radius success rule as the table worlds.
        task.bank.reset_is_cat[field_worlds] = False
        task.bank.crossed_is_plane[field_worlds] = False
        task.bank.reset_xy_scale[field_worlds] = .08
        task.bank.episode_lengths[field_worlds] = 4000
        task.raised_reset_fraction = 0.
        task.generator.manual_seed(args.seed)
    S = len(ids); NS = len(scenes); K = len(speeds)
    if args.planning_benchmark and args.num_envs != S * K * args.episodes:
        p.error("The planning benchmark needs one simulator world per matched episode")
    if blind and args.num_envs != S * args.episodes:
        p.error("Plan-blind evaluation uses exactly one initial simulator world per matched episode")
    # A job is scene + NS * (speed index); statistics are kept per job id.
    jobs = torch.tensor([i + NS * k for k in range(K) for i in ids], device=dev).repeat(args.episodes)
    speed_of = torch.tensor([x if x is not None else 0. for x in speeds], device=dev)
    order = torch.randperm(len(jobs), device=dev, generator=torch.Generator(device=dev).manual_seed(args.seed))
    jobs = jobs[order]
    guide_starts = None
    if blind:
        meta = dict(bank=task.bank.manifest["manifest_sha256"], seed=args.seed,
                    speed=args.operator_speed, variant=args.blind_variant, episodes=args.episodes)
    if args.blind_guide:
        with np.load(args.blind_guide, allow_pickle=False) as source:
            schedule = dict(source)
        if str(schedule["meta"]) != json.dumps(meta, sort_keys=True):
            raise ValueError("Blind guide metadata differs from this protocol")
        if args.blind_replay_starts:
            assigned = replay_assignments(schedule, {scenes[i]["scene_id"]: i for i in ids}, args.episodes)
            jobs = torch.as_tensor(assigned, dtype=torch.long, device=dev)
            guide_starts = {
                name: torch.as_tensor(schedule[key], dtype=task.data.qpos.dtype, device=dev)
                for name, key in (("initial_pose", "pose"), ("initial_velocity", "velocity"))
            }
    cursor = [0]
    env_scene = torch.full((task.num_envs,), -1, dtype=torch.long, device=dev)      # -1: no job (ignored)
    original_reset = task.reset

    def queued(env_ids=None, scene_ids=None):
        env_ids = task.all_ids if env_ids is None else env_ids.long()
        n = len(env_ids)
        take = min(n, len(jobs) - cursor[0])
        trial = torch.zeros(n, dtype=torch.long, device=dev) if args.planning_benchmark else None
        chosen = torch.full((n,), -1, dtype=torch.long, device=dev)
        if take > 0:
            batch = slice(cursor[0], cursor[0] + take)
            chosen[:take] = jobs[batch]
            if trial is not None:
                trial[:take] = order[batch] // (S * K)
            cursor[0] += take
        env_scene[env_ids] = chosen
        collision = task.collision
        if blind and sim.obstacle is not None:
            sim.obstacle.enabled[env_ids] = sim.obstacle.valid[env_ids] & (chosen >= 0)
            # Initial-pose certification works on selected rows and static geometry.
            task.collision = static_collision
        filler = torch.tensor(ids, device=dev)[torch.randint(S, (n,), device=dev)]
        try:
            selected = torch.where(chosen >= 0, chosen % NS, filler)
            start = benchmark_start(task, selected, trial, args.episodes) if trial is not None else {}
            if guide_starts is not None and take:
                if take != task.num_envs or not torch.equal(env_ids, task.all_ids):
                    raise RuntimeError("Blind initial states must be replayed together in guide order")
                start = guide_starts
            out_ = original_reset(env_ids, selected, **start)
        finally:
            task.collision = collision
        if K > 1:
            s_ = speed_of[torch.where(chosen >= 0, chosen // NS, 0)]
            for name in ("op_far", "op_near", "op_speed", "op_speed_ema"):
                getattr(task, name)[env_ids] = s_
            task.op_use_near[env_ids] = False; task.anchor_free[env_ids] = False
            task.op_length_scale[env_ids] = (.525 / s_).clamp(1., .525 / .1)
        return out_

    task.reset = queued
    task.reset()
    trace = None
    if args.planning_benchmark:
        if bool(task.acceptance_reset_regions.any() | task.episode["reset_replaced"].any()):
            raise RuntimeError("An authored planning benchmark start touched an obstacle")
        start_hash = hashlib.sha256(task.data.qpos.cpu().numpy().tobytes() +
                                    task.data.qvel.cpu().numpy().tobytes()).hexdigest()
        print(json.dumps(dict(initial_state_sha256=start_hash)), flush=True)
    if blind:
        pose = task.data.qpos.cpu().numpy()
        velocity = task.data.qvel.cpu().numpy()
        ordered_scenes = np.array([scenes[i]["scene_id"] for i in env_scene.tolist()])
        if args.write_blind_guide:
            schedule = candidate_placements(task, task.all_ids, args.blind_variant)
            valid_ids = np.flatnonzero(schedule["valid"])
            chosen = []
            for prefix in ("sidegap", "contrast", "random"):
                found = [i for i in valid_ids if ordered_scenes[i].startswith(prefix)]
                if found:
                    chosen.append(min(found, key=lambda i: abs(schedule["planned_hand_gap"][i] + .018)))
            if not chosen and len(valid_ids):
                chosen = [valid_ids[0]]
            args.write_blind_guide.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.write_blind_guide, meta=json.dumps(meta, sort_keys=True), scenes=ordered_scenes,
                                pose=pose, velocity=velocity, clip_rows=np.array(chosen, dtype=np.int64), **schedule)
            print(json.dumps(dict(guide=str(args.write_blind_guide), safe=int(len(valid_ids)), total=len(pose),
                                  clips=ordered_scenes[chosen].tolist())), flush=True)
            return
        if not np.array_equal(ordered_scenes, schedule["scenes"]):
            raise ValueError("Blind guide episode assignment differs")
        if (not np.allclose(pose, schedule["pose"], rtol=0., atol=2.e-6)
                or not np.allclose(velocity, schedule["velocity"], rtol=0., atol=2.e-6)):
            raise ValueError("Blind guide initial robot state differs")
        if args.blind_replay_starts:
            start_hash = hashlib.sha256(pose.tobytes() + velocity.tobytes()).hexdigest()
            print(json.dumps(dict(blind_initial_state_sha256=start_hash)), flush=True)
        if args.blind_variant == "throw":
            from cat_mjlab.threats import ThrownObstacle
            sim.obstacle = ThrownObstacle(schedule, task, dev)
        else:
            sim.obstacle = BlindObstacle(schedule, args.blind_variant, dev)
        task.analytic_objects = sim.obstacle
        static_collision = task.collision
        task.collision = lambda scene, data: static_collision(scene, data) | sim.hits
        split_stats = torch.zeros((NS * K, 4), device=dev)
        throw_stats = torch.zeros((NS * K, 4, 3), device=dev)    # launched, skipped, hand hit, any hit per speed tier
        inserted_hits = torch.zeros_like(sim.hits)
        if args.blind_record:
            trace = BlindTrace(task, sim.obstacle, schedule["clip_rows"], ordered_scenes, meta, args.checkpoint)
    if archive:
        from cat_mjlab.learning import Learner, LearnerConfig
        from cat_mjlab.conversion import load_array_archive
        learner = Learner(LearnerConfig(algorithm="ppo"), device=dev)
        load_array_archive(learner, args.checkpoint, restore_optimizer=False)
        if task.observation_size != learner.model.config.actor_obs:
            raise ValueError(f"task gives {task.observation_size} actor inputs, the archive expects {learner.model.config.actor_obs}")
    elif brace:
        from cat_mjlab.brace_tracker import BraceTracker
        external = BraceTracker(task, args.checkpoint, args.brace_config, args.brace_robot,
                                provider=args.brace_provider, seed=args.seed)
    else:
        learner, _ = load_policy(args.checkpoint, device=dev, policy_id=0)
    cols = ("episodes", "clean", "contact", "refused", "fall", "hands", "length", "energy_j", "path_m", "cot", "cot_episodes")
    # Near obstacles (any body point within 15 cm): actual vs commanded pelvis speed, both feet off the ground.
    near_steps = torch.zeros(NS * K, dtype=torch.float64, device=dev); near_v = torch.zeros_like(near_steps)
    near_cmd = torch.zeros_like(near_steps); near_flight = torch.zeros_like(near_steps); overspeed = torch.zeros_like(near_steps)
    # The same on the 0.4 s mean pelvis velocity (the gait's sideways sway, large at 0.1 m/s, averages out).
    near_vbar = torch.zeros_like(near_steps); overspeed_bar = torch.zeros_like(near_steps)
    v_bar = torch.zeros((task.num_envs, 2), device=dev)
    stats = torch.zeros((NS * K, len(cols)), dtype=torch.float64, device=dev)
    group_energy = torch.zeros((NS * K, *SHAPE), dtype=torch.float64, device=dev)
    path = torch.zeros(task.num_envs, dtype=torch.float64, device=dev)
    try:
        mass = float(task.model.body_subtreemass[task.pelvis_body])      # whole robot (pelvis = root of its tree)
    except Exception:
        mass = float("nan")
    if args.refused_rewards:
        refusal_terms = ("track_pos", "track_lead", "track_upper", "handsdf", "wholebody_hand_clearance",
                         "wholebody_arm_clearance", "stall", "goal_progress", "global_power", "table_power")
        refusal_acc = torch.zeros((task.num_envs, len(refusal_terms)), device=dev)
        refusal_totals = torch.zeros((NS * K, len(refusal_terms)), device=dev)
        refusal_steps = torch.zeros(NS * K, device=dev)
    audit = None
    if args.stall_audit:
        if task.tracker is None or args.tracker_mix != "plan=1":
            p.error("Stall audit needs --tracker-mix plan=1")
        from stall_audit import StallAudit
        task.measure_track_progress = True
        audit = StallAudit(task.num_envs, NS * K, dev, task.dt)
    target = S * args.episodes * K
    if args.planning_benchmark:
        seconds_to_goal = torch.full_like(path, -1.)
        min_palm = task.info["sdf"][:, 5:7, 0].amin(-1).clone()
        traversal = torch.zeros((NS * K, 3), dtype=torch.float64, device=dev)
    start = time.time()
    for t in range(args.max_steps):
        before = env_scene.clone(); root0 = task.data.qpos[:, :2].clone()
        if blind:
            sim.hits.zero_()
            if args.blind_variant == "throw":
                sim.obstacle.update(task)
            if trace is not None:
                trace.record()
        with torch.no_grad():
            if brace:
                action, target_positions = external()
                out = task.step(action, position_targets=target_positions)
            else:
                out = task.step(learner.act(task.obs, policy_ids=0, deterministic=not args.stochastic)["action"])
        m = out["metrics"]
        if brace and bool(m["episode/numerical"].any()):
            raise RuntimeError("Non-finite BRACE physics state")
        done = out["done"].bool() & (before >= 0)
        if args.planning_benchmark:
            first = (before >= 0) & (seconds_to_goal < 0.) & m["episode/goal_reached"].bool()
            seconds_to_goal = torch.where(first, m["episode_length"] * task.dt, seconds_to_goal)
            min_palm = torch.minimum(min_palm, m["acceptance/hand_clearance"].amin(-1))
        if blind:
            inserted_hits |= sim.hits
            if args.blind_variant == "throw":
                sim.obstacle.observe(sim.hits)
            if trace is not None:
                trace.finish(done, sim.hits)
        if audit is not None:
            refused = done & ~m["episode/goal_reached"].bool() & ~m["episode/obstacle"].bool() & ~m["episode/fall"].bool()
            audit.observe(m, before, done, refused)
        # Energy and root position metrics include the terminal transition.
        moving = (before >= 0) & ~done
        v_bar = torch.where(done[:, None], 0., .95 * v_bar + .05 * task.data.qvel[:, :2])
        path += torch.where(before >= 0, torch.linalg.vector_norm(m['acceptance/root_xy'] - root0, dim=-1).double(), 0.)
        near = moving & (task.info['sdf'][:, :11, 0].amin(-1) < .15)
        if bool(near.any()):
            sc_ = before.clamp_min(0); w_ = near.double()
            v_ = torch.linalg.vector_norm(task.data.qvel[:, :2], dim=-1).double()
            c_ = torch.linalg.vector_norm(task.info['command'][:, 1:3], dim=-1).double()
            feet = task.sim.contact_flags(task.contact_pairs)[:, :2]
            near_steps.index_add_(0, sc_, w_); near_v.index_add_(0, sc_, w_ * v_); near_cmd.index_add_(0, sc_, w_ * c_)
            near_flight.index_add_(0, sc_, w_ * (~feet.any(-1)).double())
            overspeed.index_add_(0, sc_, w_ * ((c_ > 1e-3) & (v_ > 1.5 * c_)).double())
            vb_ = torch.linalg.vector_norm(v_bar, dim=-1).double()
            near_vbar.index_add_(0, sc_, w_ * vb_); overspeed_bar.index_add_(0, sc_, w_ * ((c_ > 1e-3) & (vb_ > 1.5 * c_)).double())
        if args.refused_rewards:
            zero_term = torch.zeros_like(path, dtype=torch.float32)
            terms = torch.stack([m.get("reward/" + term, zero_term) for term in refusal_terms], 1)
            refusal_acc += torch.where((before >= 0)[:, None], terms, 0.)
        if bool(done.any()):
            zero = torch.zeros_like(done)
            goal = m.get("episode/goal_reached", m["successful"]).bool()
            obstacle = m.get("episode/obstacle", zero).bool(); fall = m.get("episode/fall", zero).bool()
            hands = m.get("episode/body_collision_hands", zero).bool()
            flags = (done, done & goal & ~obstacle, done & obstacle, done & ~goal & ~obstacle & ~fall, done & fall, done & hands)
            d = torch.nonzero(done).flatten(); sc = before[d]
            for c, f in enumerate(flags):
                stats[:, c].index_add_(0, sc, f[d].double())
            if blind:
                flags_blind = (sim.obstacle.valid, sim.revealed, inserted_hits.any(-1), inserted_hits[:, 5])
                split_stats.index_add_(0, sc, torch.stack(flags_blind, -1)[d].float())
                if args.blind_variant == "throw":
                    episode_throws = sim.obstacle.per_episode()
                    throw_stats.index_add_(0, sc, torch.stack([episode_throws[x] for x in ("launched", "skipped", "hand", "any")], 1)[d].float())
            if args.refused_rewards:
                refused = flags[3][d]
                refusal_totals.index_add_(0, sc, torch.where(refused[:, None], refusal_acc[d], 0.))
                refusal_steps.index_add_(0, sc, torch.where(refused, m["episode_length"][d], 0).float())
                refusal_acc[d] = 0.
            if args.planning_benchmark:
                arrived = flags[1][d]
                traversal[:, 0].index_add_(0, sc, arrived.double())
                traversal[:, 1].index_add_(0, sc, torch.where(arrived, seconds_to_goal[d], 0.))
                traversal[:, 2].index_add_(0, sc, min_palm[d].double())
                seconds_to_goal[d] = -1.
                min_palm[d] = task.info["sdf"][d, 5:7, 0].amin(-1)
            stats[:, 6].index_add_(0, sc, m["episode_length"][d].double())
            episode_j = m['energy/episode_j'][d].double()
            group_energy.index_add_(0, sc, episode_j)
            energy = episode_j[:, 0, 0].sum(-1)
            stats[:, 7].index_add_(0, sc, energy); stats[:, 8].index_add_(0, sc, path[d])
            far = path[d] >= .5
            stats[:, 9].index_add_(0, sc, torch.where(far, energy / (mass * 9.81 * path[d].clamp_min(1e-6)), 0.))
            stats[:, 10].index_add_(0, sc, far.double())
        path = torch.where(done, 0., path)
        finished = int(stats[:, 0].sum())
        if t % 500 == 0:
            print(f"step {t}: {finished}/{target} episodes, {time.time() - start:.0f}s", flush=True)
        if finished >= target:
            break
    if blind:
        sim.capacity_report()
        if trace is not None:
            trace.save(args.blind_record)
        if finished != target:
            raise RuntimeError(f"Plan-blind benchmark stopped at {finished}/{target} completed episodes. "
                               f"Revealed {int(sim.revealed.sum())}, with {int(inserted_hits[:,5].sum())} "
                               "physical inserted hand contacts. No aggregate scores were saved." +
                               (f" Throws launched {int(sim.obstacle.launched.sum())}, skipped {int(sim.obstacle.skipped.sum())}, "
                                f"hand hits {int((sim.obstacle.slot_hand & sim.obstacle.launched).sum())}." if args.blind_variant == "throw" else ""))
        blind_names = ("inserted_scheduled", "inserted_revealed", "inserted_contact", "inserted_hand_contact")
    if args.planning_benchmark:
        if finished != target:
            raise RuntimeError(f"Planning benchmark completed only {finished}/{target} episodes")
        traversal = traversal.cpu()

    def traversal_summary(totals, count):
        goals, time_sum, clearance_sum = totals.tolist()
        return dict(clean_goals=int(goals), mean_time_to_clean_goal_s=time_sum / goals if goals else None,
                    mean_episode_min_palm_field_m=clearance_sum / count)

    group_energy = group_energy.cpu()
    for k, speed in enumerate(speeds):
        jid = [i + NS * k for i in ids]
        result = {}
        for i, j in zip(ids, jid):
            e = float(stats[j, 0])
            if e == 0:
                continue
            row = dict(episodes=int(e), **{c: round(float(stats[j, c_]) / e, 4) for c_, c in enumerate(cols) if 0 < c_ < 9},)
            if blind:
                row.update(dict(zip(blind_names, (split_stats[j] / e).cpu().tolist())))
                if args.blind_variant == "throw":
                    row["throws"] = throw_stats[j].cpu().tolist()
            if audit is not None:
                row["stall_windows"] = audit.summary([j])
            row["energy"] = report_energy(group_energy[j], e, float(stats[j, 6]) * task.dt)
            row["power_w"] = round(float(stats[j, 7] / (stats[j, 6] * task.dt).clamp_min(1e-9)), 2)
            row["cot"] = round(float(stats[j, 9] / stats[j, 10]), 4) if float(stats[j, 10]) else None
            if float(near_steps[j]) > 0:
                row["near_speed_ratio"] = round(float(near_v[j] / near_cmd[j].clamp_min(1e-9)), 3)
                row["near_mean_speed_ratio"] = round(float(near_vbar[j] / near_cmd[j].clamp_min(1e-9)), 3)
            if args.planning_benchmark:
                row.update(traversal_summary(traversal[j], e))
                row["layout_family"] = scenes[i]["source"]["planning_benchmark"]["family"]
            result[scenes[i]["scene_id"]] = row
        E = stats[jid, 0].sum().clamp_min(1)
        summary = {c: round(float(stats[jid, c_].sum() / E), 4) for c_, c in enumerate(cols) if 0 < c_ < 9}
        if blind:
            summary.update(dict(zip(blind_names, (split_stats[jid].sum(0) / E).cpu().tolist())))
            if args.blind_variant == "throw":
                from cat_mjlab.threats import SPEEDS
                counts = throw_stats[jid].sum(0).cpu()
                summary["throws"] = {f"{speed:g}": dict(launched=int(counts[0, i]), skipped=int(counts[1, i]), hand_hits=int(counts[2, i]),
                                                        any_hits=int(counts[3, i])) for i, speed in enumerate(SPEEDS)}
        if audit is not None:
            summary["stall_windows"] = audit.summary(jid)
        if args.planning_benchmark:
            summary.update(traversal_summary(traversal[jid].sum(0), float(E)))
        if args.refused_rewards:
            count = float(refusal_steps[jid].sum())
            summary["refused_reward_terms_per_step"] = {
                "steps": int(count),
                "terms": dict(zip(refusal_terms, (refusal_totals[jid].sum(0) / count).tolist())) if count else {},
            }
        summary["energy"] = report_energy(group_energy[jid].sum(0), int(stats[jid, 0].sum()),
                                          float(stats[jid, 6].sum()) * task.dt)
        summary["power_w"] = round(float(stats[jid, 7].sum() / (stats[jid, 6].sum() * task.dt).clamp_min(1e-9)), 2)
        summary["cot"] = round(float(stats[jid, 9].sum() / stats[jid, 10].sum().clamp_min(1)), 4)
        summary["robot_mass_kg"] = round(mass, 2)
        n_ = near_steps[jid].sum().clamp_min(1); c_sum = near_cmd[jid].sum().clamp_min(1e-9)
        summary["near_obstacle"] = dict(steps=int(near_steps[jid].sum()), speed_ratio=round(float(near_v[jid].sum() / c_sum), 3),
                                        mean_speed=round(float(near_v[jid].sum() / n_), 3), flight_fraction=round(float(near_flight[jid].sum() / n_), 4),
                                        overspeed_1p5x_fraction=round(float(overspeed[jid].sum() / n_), 4),
                                        smoothed_speed_ratio=round(float(near_vbar[jid].sum() / c_sum), 3),
                                        smoothed_mean_speed=round(float(near_vbar[jid].sum() / n_), 3),
                                        smoothed_overspeed_1p5x_fraction=round(float(overspeed_bar[jid].sum() / n_), 4))
        out = args.out if args.out is not None else args.out_dir / f"speed_{speed:g}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(checkpoint=str(args.checkpoint), env_from=args.env_from, scenes=args.scenes, episodes_per_scene=args.episodes,
                                       operator_speed=speed, deterministic=not args.stochastic, finished=int(stats[jid, 0].sum()),
                                       target=S * args.episodes, seconds=round(time.time() - start), summary=summary, per_scene=result,
                                       **({"blind_variant": args.blind_variant, "blind_guide": str(args.blind_guide)} if blind else {}),
                                       **({"blind_initial_state_sha256": start_hash} if args.blind_replay_starts else {}),
                                       **({"external_tracker": "BRACE", "brace_provider": args.brace_provider} if brace else {}),
                                       **({"planning_benchmark": True, "plan_only": args.plan_only,
                                           "refusal_fix": args.refusal_fix, "initial_state_sha256": start_hash}
                                          if args.planning_benchmark else {})),
                                  indent=1) + "\n")
        print(json.dumps(dict(out=str(out), finished=int(stats[jid, 0].sum()), target=S * args.episodes, summary=summary)))

if __name__ == "__main__":
    main()
