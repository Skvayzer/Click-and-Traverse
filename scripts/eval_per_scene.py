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
    p.add_argument("--no-compile", action="store_true", help="Skip torch.compile of the task (about 4x slower at 16k envs)")
    p.add_argument("--nominal", action="store_true", help="Drop the robustness perturbations (delay, pose/mount error, friction/mass/COM/torque DR): skill only")
    p.add_argument("--operator-speed", type=float, help="Fixed operator speed (m/s): field direction at this speed, no stop latch, "
                   "cut-out 0.05 m/s, episode time scaled by 0.525/speed (the slow-traversal command; works for any checkpoint)")
    p.add_argument("--operator-speeds", type=float, nargs="+", metavar="S",
                   help="Several fixed operator speeds in one run (needs --out-dir): episodes at every speed in parallel")
    p.add_argument("--tracker-mix", help="Override the reference sources of a tracker checkpoint, e.g. plan=1 (planner with a "
                                         "field reflex, every episode) or none=1 (the same policy with no reference)")
    p.add_argument("--tracker-hold-plan", type=float, help="Override the share of reactive standing worlds on a stand-still plan")
    p.add_argument("--out", type=Path)
    p.add_argument("--out-dir", type=Path)
    args = p.parse_args()
    if args.operator_speeds and (args.operator_speed is not None or args.out_dir is None or args.out is not None):
        p.error("--operator-speeds needs --out-dir (and neither --operator-speed nor --out)")
    if not args.operator_speeds and args.out is None:
        p.error("--out is required")
    from record_mjlab_rollout import load_policy
    from cat_mjlab.runner import create_task
    dev = "cuda:0"
    contract = torch.load(args.env_from or args.checkpoint, map_location="cpu", weights_only=True, mmap=True)["contract"]
    env = copy.deepcopy(contract["environment_config"])
    env.pop("side_gap_curriculum", None)                 # standard CAT starts, every width
    env["push_config"]["enable"] = False
    env.pop("sgs", None)                                 # every scene gets its K jobs, no success-guided resampling
    speeds = [float(x) for x in args.operator_speeds] if args.operator_speeds else (
        [float(args.operator_speed)] if args.operator_speed is not None else [None])
    if speeds[0] is not None:
        # One operator dict for the process; with several speeds each episode's speed is set at its reset (queued below).
        env["operator_speed"] = dict(min=min(speeds), max=max(speeds), nominal=.525, cutout=.05, near_sdf=.35, near_share=0.,
                                     resample=None, length_scale_max=.525 / .1, anchor_free_below=0., stall_fraction=.2,
                                     rest_start_share=0.)
    if args.nominal:
        env.pop("robustness", None)
    if args.tracker_mix or args.tracker_hold_plan is not None:
        if not env.get("tracker"):
            p.error("--tracker-mix / --tracker-hold-plan need a checkpoint trained with the tracker")
        if args.tracker_mix:
            env["tracker"]["mix"] = [[a, float(b)] for a, b in (item.split("=", 1) for item in args.tracker_mix.split(","))]
        if args.tracker_hold_plan is not None:
            env["tracker"]["hold_plan"] = float(args.tracker_hold_plan)
    if "teleop" in env:
        env["teleop"].update(fraction=0., cat_fraction=0., cat_scenes=False)
    B = ROOT / args.bank
    ns = SimpleNamespace(bank_manifest=Path(str(B) + "_packed/manifest.json"), body_collision_bank=Path(str(B) + "_collision/manifest.json"),
                         body_collision_resets=Path(str(B) + "_resets/manifest.json"), device=dev, seed=args.seed, num_envs=args.num_envs,
                         compile_task=not args.no_compile, nconmax=contract["nconmax"], njmax=contract["njmax"], reactive_bank=None, reactive_row=None)
    task, sim, _ = create_task(ns, environment_config=env)
    scenes = task.bank.manifest["scenes"]
    if args.scenes.startswith("@"):
        wanted = {l.strip() for l in open(args.scenes[1:]) if l.strip()}
        ids = [i for i, s in enumerate(scenes) if s["scene_id"] in wanted]
    else:
        ids = [i for i, s in enumerate(scenes) if args.scenes == "all" or s.get("task_kind") == args.scenes]
    S = len(ids); NS = len(scenes); K = len(speeds)
    # A job is scene + NS * (speed index); statistics are kept per job id.
    jobs = torch.tensor([i + NS * k for k in range(K) for i in ids], device=dev).repeat(args.episodes)
    speed_of = torch.tensor([x if x is not None else 0. for x in speeds], device=dev)
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
        out_ = original_reset(env_ids, torch.where(chosen >= 0, chosen % NS, filler))
        if K > 1:
            s_ = speed_of[torch.where(chosen >= 0, chosen // NS, 0)]
            for name in ("op_far", "op_near", "op_speed", "op_speed_ema"):
                getattr(task, name)[env_ids] = s_
            task.op_use_near[env_ids] = False; task.anchor_free[env_ids] = False
            task.op_length_scale[env_ids] = (.525 / s_).clamp(1., .525 / .1)
        return out_

    task.reset = queued
    task.reset()
    learner, _ = load_policy(args.checkpoint, device=dev, policy_id=0)
    cols = ("episodes", "clean", "contact", "refused", "fall", "hands", "length", "energy_j", "path_m", "cot", "cot_episodes")
    # Near obstacles (any body point within 15 cm): actual vs commanded pelvis speed, both feet off the ground.
    near_steps = torch.zeros(NS * K, dtype=torch.float64, device=dev); near_v = torch.zeros_like(near_steps)
    near_cmd = torch.zeros_like(near_steps); near_flight = torch.zeros_like(near_steps); overspeed = torch.zeros_like(near_steps)
    # The same on the 0.4 s mean pelvis velocity (the gait's sideways sway, large at 0.1 m/s, averages out).
    near_vbar = torch.zeros_like(near_steps); overspeed_bar = torch.zeros_like(near_steps)
    v_bar = torch.zeros((task.num_envs, 2), device=dev)
    stats = torch.zeros((NS * K, len(cols)), dtype=torch.float64, device=dev)
    energy = torch.zeros(task.num_envs, dtype=torch.float64, device=dev); path = torch.zeros_like(energy)
    try:
        mass = float(task.model.body_subtreemass[task.pelvis_body])      # whole robot (pelvis = root of its tree)
    except Exception:
        mass = float("nan")
    target = S * args.episodes * K
    start = time.time()
    for t in range(args.max_steps):
        before = env_scene.clone(); root0 = task.data.qpos[:, :2].clone()
        with torch.no_grad():
            out = task.step(learner.act(task.obs, policy_ids=0, deterministic=not args.stochastic)["action"])
        done = out["done"].bool() & (before >= 0)
        # Energy as the training prices it (PD torque of the last substep x joint velocity); a finished env has already
        # been reset (ctrl holds reset targets), so its terminal step is left out of both sums.
        moving = (before >= 0) & ~done
        v_bar = torch.where(done[:, None], 0., .95 * v_bar + .05 * task.data.qvel[:, :2])
        power = (task.data.ctrl * task.data.qvel[:, 6:]).abs().sum(-1).double()
        energy += torch.where(moving, power * task.dt, 0.)
        path += torch.where(moving, torch.linalg.vector_norm(task.data.qpos[:, :2] - root0, dim=-1).double(), 0.)
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
            stats[:, 7].index_add_(0, sc, energy[d]); stats[:, 8].index_add_(0, sc, path[d])
            far = path[d] >= .5
            stats[:, 9].index_add_(0, sc, torch.where(far, energy[d] / (mass * 9.81 * path[d].clamp_min(1e-6)), 0.))
            stats[:, 10].index_add_(0, sc, far.double())
        energy = torch.where(done, 0., energy); path = torch.where(done, 0., path)
        finished = int(stats[:, 0].sum())
        if t % 500 == 0:
            print(f"step {t}: {finished}/{target} episodes, {time.time() - start:.0f}s", flush=True)
        if finished >= target:
            break
    for k, speed in enumerate(speeds):
        jid = [i + NS * k for i in ids]
        result = {}
        for i, j in zip(ids, jid):
            e = float(stats[j, 0])
            if e == 0:
                continue
            row = dict(episodes=int(e), **{c: round(float(stats[j, c_]) / e, 4) for c_, c in enumerate(cols) if 0 < c_ < 9},)
            row["power_w"] = round(float(stats[j, 7] / (stats[j, 6] * task.dt).clamp_min(1e-9)), 2)
            row["cot"] = round(float(stats[j, 9] / stats[j, 10]), 4) if float(stats[j, 10]) else None
            if float(near_steps[j]) > 0:
                row["near_speed_ratio"] = round(float(near_v[j] / near_cmd[j].clamp_min(1e-9)), 3)
                row["near_mean_speed_ratio"] = round(float(near_vbar[j] / near_cmd[j].clamp_min(1e-9)), 3)
            result[scenes[i]["scene_id"]] = row
        E = stats[jid, 0].sum().clamp_min(1)
        summary = {c: round(float(stats[jid, c_].sum() / E), 4) for c_, c in enumerate(cols) if 0 < c_ < 9}
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
                                       target=S * args.episodes, seconds=round(time.time() - start), summary=summary, per_scene=result),
                                  indent=1) + "\n")
        print(json.dumps(dict(out=str(out), finished=int(stats[jid, 0].sum()), target=S * args.episodes, summary=summary)))

if __name__ == "__main__":
    main()
