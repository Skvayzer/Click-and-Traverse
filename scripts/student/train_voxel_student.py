#!/usr/bin/env python3
"""DAgger: distil field-based experts into the deployable voxel student (head Mid-360 only).

Experts (frozen, privileged scene fields) label every state; the student sees proprioception + command +
the simulated LiDAR voxel grid (cat_mjlab.lidar). Per episode the expert drives with probability beta
(decaying), otherwise the student drives; either way the expert's action is the label (DAgger: the
student learns on the states its own mistakes lead to). Regression: MSE of the student's pre-squash
action mean onto the expert's, over a replay of recent rollouts.

The task, rewards and scene sampling are the experts' training setup (their saved environment config),
so success/contact metrics of student-driven episodes are directly comparable with the experts'.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CAT_BUCKETS = ("procedural_cat", "original_cat", "published_cat")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rooms-expert", default="outputs/saved_checkpoints/cat_g1_rooms_keyboard_v9_u492_full.pt")
    p.add_argument("--cat-expert", help="CAT expert checkpoint (CAT scenes); omitted: rooms expert labels them too")
    p.add_argument("--bank", default="data/furniture/table_edges_v1")
    p.add_argument("--experience-masses", type=float, nargs=8, default=[.10, .25, .15, .25, 0, .02, .13, 0],
                   help="original CAT, procedural CAT, furniture, clutter, narrow, flat, table edges, passages")
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--unroll", type=int, default=32)
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--beta-start", type=float, default=1.)
    p.add_argument("--beta-end", type=float, default=.1)
    p.add_argument("--beta-decay", type=int, default=150, help="updates to go from beta-start to beta-end")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--replay", type=int, default=3, help="rollouts kept for regression")
    p.add_argument("--rays", type=int, default=2048)
    p.add_argument("--memory-scans", type=int, default=10)
    p.add_argument("--student-noise", type=float, default=.05)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", default="outputs/voxel_student")
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse()
    import torch
    import torch.nn.functional as F
    from cat_mjlab.distill import load_expert
    from cat_mjlab.learning import gaussian_parameters
    from cat_mjlab.runner import create_task, SCENE_BUCKETS, _scene_buckets
    from cat_mjlab.lidar import Mid360, VoxelMemory, sensor_pose_noise
    from cat_mjlab.student import VoxelStudent, pack_voxels, unpack_voxels, student_proprio, PROPRIO
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)
    rooms_contract = torch.load(args.rooms_expert, map_location="cpu", weights_only=True, mmap=True)["contract"]
    env = copy.deepcopy(rooms_contract["environment_config"])
    ns = SimpleNamespace(bank_manifest=Path(args.bank + "_packed/manifest.json"), body_collision_bank=Path(args.bank + "_collision/manifest.json"),
                         body_collision_resets=Path(args.bank + "_resets/manifest.json"), device=args.device, seed=args.seed,
                         num_envs=args.num_envs, compile_task=False, nconmax=rooms_contract["nconmax"], njmax=rooms_contract["njmax"],
                         reactive_bank=None, reactive_row=None, experience_masses=args.experience_masses,
                         scene_group_override=["table_edges=6", "protected_passage=7", "transition_passage=7", "open_passage=7", "narrow_passage=7"],
                         narrow_sampling_group=4)
    task, sim, _ = create_task(ns, environment_config=env)
    experts = {"rooms": load_expert(args.rooms_expert, device)}
    if args.cat_expert:
        experts["cat"] = load_expert(args.cat_expert, device)
    bucket_of_scene = torch.as_tensor(_scene_buckets(task.bank.manifest), device=device)
    cat_ids = torch.tensor([SCENE_BUCKETS.index(b) for b in CAT_BUCKETS], device=device)
    lidar = Mid360(sim.model, device, rays=args.rays)
    memory = VoxelMemory(args.num_envs, args.rays, device, scans=args.memory_scans)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    pose = sensor_pose_noise(args.num_envs, device, gen)
    latency = torch.randint(1, 3, (args.num_envs,), device=device, generator=gen)       # 100-200 ms: 1-2 scans withheld
    student = VoxelStudent().to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    start = 0
    if args.resume and (run / "student.pt").exists():
        state = torch.load(run / "student.pt", map_location=device, weights_only=True)
        student.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); start = state["update"]
    (run / "config.json").write_text(json.dumps(dict(vars(args), student=student.config, environment_config_from=args.rooms_expert), indent=1))
    task.reset()
    drives = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    replay = []; control_step = 0
    log = (run / "metrics.jsonl").open("a")

    def expert_labels(state):
        mean, _ = gaussian_parameters(experts["rooms"].logits(state[:, :experts["rooms"].config.actor_obs], 0))
        if "cat" in experts:
            cat = torch.isin(bucket_of_scene[task.scene_ids], cat_ids)
            if bool(cat.any()):
                m, _ = gaussian_parameters(experts["cat"].logits(state[cat][:, :experts["cat"].config.actor_obs], 0))
                mean = mean.clone(); mean[cat] = m
        return mean

    def yaw_now():
        q = sim.data.qpos[:, 3:7]
        return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    for update in range(start, args.updates):
        beta = max(args.beta_end, args.beta_start - (args.beta_start - args.beta_end) * update / max(1, args.beta_decay))
        t0 = time.time()
        buffer = dict(proprio=[], voxels=[], label=[], expert=[])
        counts = torch.zeros((len(SCENE_BUCKETS), 3), device=device)        # student-driven: ended, success, obstacle
        steps = 0
        for t in range(args.unroll):
            with torch.no_grad():
                fresh = task.info["step"] == 0
                if bool(fresh.any()):
                    ids = torch.nonzero(fresh).flatten()
                    memory.clear(ids)
                    drives[ids] = torch.rand(len(ids), device=device, generator=gen) < beta
                    off, rot = sensor_pose_noise(len(ids), device, gen); pose[0][ids] = off; pose[1][ids] = rot
                    latency[ids] = torch.randint(1, 3, (len(ids),), device=device, generator=gen)
                if control_step % 5 == 0:                                       # 10 Hz scans at 50 Hz control
                    d = sim.data
                    points, valid = lidar.scan(d.site_xpos[:, lidar.site], d.site_xmat[:, lidar.site], task.scene_ids, task.bank,
                                               d.xpos, d.xmat, pose_noise=pose, generator=gen)
                    memory.add(points, valid)
                voxels = memory.grid(sim.data.qpos[:, :3], yaw_now(), latency_scans=latency, generator=gen)
                proprio = student_proprio(task)
                label = expert_labels(task.obs["state"])
                mean = student(proprio, voxels)
                own = mean + args.student_noise * torch.randn(mean.shape, device=device, generator=gen)
                raw = torch.where(drives[:, None], label, own)
                buffer["proprio"].append(proprio); buffer["voxels"].append(pack_voxels(voxels)); buffer["label"].append(label)
                buffer["expert"].append(torch.isin(bucket_of_scene[task.scene_ids], cat_ids))
                result = task.step(raw.tanh()); control_step += 1
                done = result["done"].bool(); m = result["metrics"]
                ended = done & ~drives
                bucket = bucket_of_scene[task.scene_ids] if "teleop/active" not in m else torch.where(
                    m["teleop/active"].bool(), SCENE_BUCKETS.index("teleop"), bucket_of_scene[task.scene_ids])
                for col, flag in enumerate((ended, ended & m["successful"].bool(), ended & m.get("episode/obstacle", torch.zeros_like(done)).bool())):
                    counts[:, col].scatter_add_(0, bucket, flag.float())
                steps += args.num_envs
        rollout_s = time.time() - t0
        batch = {k: torch.cat(v) for k, v in buffer.items()}
        replay.append(batch); replay = replay[-args.replay:]
        data = {k: torch.cat([b[k] for b in replay]) for k in batch}
        rows = len(data["label"]); losses = []
        for _ in range(args.epochs):
            perm = torch.randperm(rows, device=device)
            for s in range(0, rows - args.minibatch + 1, args.minibatch):
                idx = perm[s:s + args.minibatch]
                pred = student(data["proprio"][idx], unpack_voxels(data["voxels"][idx]))
                loss = F.mse_loss(pred, data["label"][idx])
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.); optimizer.step(); losses.append(float(loss))
        with torch.no_grad():
            pred = torch.cat([student(batch["proprio"][i:i + 8192], unpack_voxels(batch["voxels"][i:i + 8192])) for i in range(0, len(batch["label"]), 8192)])
            err = (pred - batch["label"]).square().mean(-1)
        metrics = dict(update=update + 1, beta=round(beta, 3), loss=round(sum(losses) / max(1, len(losses)), 4),
                       mse_rooms=round(float(err[~batch["expert"]].mean()), 4) if bool((~batch["expert"]).any()) else None,
                       mse_cat=round(float(err[batch["expert"]].mean()), 4) if bool(batch["expert"].any()) else None,
                       rollout_s=round(rollout_s, 1), update_s=round(time.time() - t0, 1), steps_per_s=round(steps / max(rollout_s, 1e-6)))
        c = counts.cpu()
        for i, name in enumerate(SCENE_BUCKETS):
            if c[i, 0] >= 1:
                metrics[f"student/{name}/ended"] = int(c[i, 0])
                metrics[f"student/{name}/success_rate"] = round(float(c[i, 1] / c[i, 0]), 3)
                metrics[f"student/{name}/obstacle_rate"] = round(float(c[i, 2] / c[i, 0]), 3)
        print(json.dumps(metrics), flush=True); log.write(json.dumps(metrics) + "\n"); log.flush()
        if (update + 1) % args.save_every == 0 or update + 1 == args.updates:
            torch.save(dict(model=student.state_dict(), optimizer=optimizer.state_dict(), update=update + 1, config=student.config,
                            args=vars(args)), run / "student.pt")


if __name__ == "__main__":
    main()
