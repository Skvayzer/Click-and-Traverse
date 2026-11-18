#!/usr/bin/env python3
"""Collect whole-body planner examples from a policy's own rollouts (cat_mjlab/planner/data.py has the format).

The policy (default: the deployed u150) runs deterministically in a fixed subset of the scene bank: per scene type,
training scenes and disjoint held-out scenes. Observation noise, pushes, PD randomisation, action delay and the
adaptive sampler are off; the training command mix (route following and teleop speeds) is kept. Field data is
loaded only for the chosen scenes and all other scenes get zero sampling weight.

Every 2 s window that stays inside one episode and has no contact or fall up to 0.5 s after its end becomes one
example: state, hindsight goal, the robot-centred 5 cm grid of the scene's signed distance, and the chunk.

    python scripts/planner_collect.py --checkpoint CKPT --bank data/furniture/sidegap_v2 --out DATA \
        --num-envs 4096 --steps 1500 --rounds 2 --device cuda:0
    # local smoke test: --device cpu --num-envs 1 --steps 300 --train-per-type 1 --test-per-type 1
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import cat_mjlab  # noqa: E402,F401  (installs CAT_PATH_MAP before any manifest is read)

# per scene type: (training scenes, held-out scenes) -- about 6 GB of fields on the GPU
DEFAULT_TYPES = dict(procedural_cat=(60, 15), original_cat=(28, 9), published_cat=(20, 7), side_gap=(60, 15),
                     clutter_dense=(16, 5), clutter_pilot=(24, 6), clutter_legacy=(8, 3), furniture_dense=(12, 4),
                     furniture_pilot=(16, 5), furniture_legacy=(8, 3), narrow_passage=(30, 10),
                     protected_passage=(24, 8), transition_passage=(24, 8), table_edges=(24, 8), open_passage=(20, 6))
BAD_FLAGS = ("fall", "obstacle", "body_collision", "hand_violation", "elbow_violation", "self_contact", "numerical",
             "outside_bounds")


def choose_scenes(manifest, types, scale, seed):
    from cat_mjlab.runner import SCENE_BUCKETS, _scene_buckets
    buckets = np.asarray(_scene_buckets(manifest))
    rng = np.random.default_rng(seed)
    train, test = [], []
    for name, (n_train, n_test) in types.items():
        idx = [int(i) for i in np.nonzero(buckets == SCENE_BUCKETS.index(name))[0]
               if manifest["scenes"][int(i)].get("sampling_weight", 1.0) > 0]
        rng.shuffle(idx)
        n_test = min(max(1, round(n_test * scale)), len(idx) // 3)
        n_train = min(max(1, round(n_train * scale)), len(idx) - n_test)
        test += idx[:n_test]; train += idx[n_test:n_test + n_train]
    return sorted(train), sorted(test), buckets


def install_subset_loader(keep_ids):
    """Load field data only for the chosen scenes (others: a 4x4x45 zero grid that is never sampled)."""
    import cat_ppo.furniture.generalist_fields as generalist_fields
    original_loader, original_np_load = generalist_fields.load_generalist_manifest, np.load
    keep_dirs = set()

    def loader(path, verify_files=True, **kw):
        result = original_loader(path, verify_files=False, **kw)
        if sys._getframe(1).f_code.co_filename.endswith("scene_bank.py"):
            for s in result["scenes"]:
                if s["scene_id"] in keep_ids:
                    keep_dirs.add(Path(s["path"]).name)
                else:
                    s["shape"] = [4, 4, 45]
        return result

    def np_load(path, *a, **kw):
        # Bank fields live in <bank>/<scene dir>/, the goal-seed cache (cat_mjlab/goal_seed.py) in <cache>/<scene_id>/:
        # a file is real if its folder names a chosen scene either way.
        p = Path(str(path))
        if (p.name in ("sdf.npy", "bf.npy", "gf.npy", "gf_valid.npy") and keep_dirs
                and p.parent.name not in keep_dirs and p.parent.name not in keep_ids):
            head = original_np_load(path, mmap_mode="r", allow_pickle=False)
            if head.ndim == 2:          # flat [cells, channels]
                return np.zeros((4 * 4 * 45, head.shape[1]), dtype=head.dtype)
            return np.zeros((4, 4, 45, *head.shape[3:]), dtype=head.dtype)
        return original_np_load(path, *a, **kw)
    generalist_fields.load_generalist_manifest = loader
    np.load = np_load


def rollout(task, learner, scenes, steps, rng, torch):
    """Run all worlds for `steps` control steps from fresh episodes in `scenes`; returns the recorded arrays."""
    n = task.num_envs
    first = torch.as_tensor(rng.choice(scenes, size=n, replace=len(scenes) < n), device=task.device)
    task.reset(None, first)
    q = np.empty((steps + 1, n, task.data.qpos.shape[-1]), np.float32)
    v = np.empty((steps + 1, n, task.data.qvel.shape[-1]), np.float32)
    scene = np.empty((steps + 1, n), np.int32); epstep = np.empty((steps + 1, n), np.int32)
    ended = np.zeros((steps, n), bool); bad = np.zeros((steps, n), bool)
    for t in range(steps):
        q[t] = task.data.qpos.float().cpu().numpy(); v[t] = task.data.qvel.float().cpu().numpy()
        scene[t] = task.scene_ids.cpu().numpy(); epstep[t] = task.info["step"].cpu().numpy()
        with torch.no_grad():
            action = learner.act(task.obs, policy_ids=0, deterministic=True)["action"]
        result = task.step(action)
        done, metrics = result["done"], result["metrics"]
        flags = torch.zeros_like(done)
        for name in BAD_FLAGS:
            value = metrics.get("episode/" + name)
            if value is not None:
                flags = flags | value.bool()
        ended[t] = done.cpu().numpy(); bad[t] = (done & flags).cpu().numpy()
    q[steps] = task.data.qpos.float().cpu().numpy(); v[steps] = task.data.qvel.float().cpu().numpy()
    scene[steps] = task.scene_ids.cpu().numpy(); epstep[steps] = task.info["step"].cpu().numpy()
    return q, v, scene, epstep, ended, bad


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bank", default="data/furniture/sidegap_v2", help="Prefix of <bank>_packed/_collision/_resets")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=1500, help="Control steps per round (50 Hz)")
    p.add_argument("--rounds", type=int, default=2, help="Training-scene rounds (fresh episodes each)")
    p.add_argument("--test-rounds", type=int, default=1)
    p.add_argument("--test-envs", type=int, default=None, help="Worlds for the held-out rounds (default: num-envs)")
    p.add_argument("--window-every", type=int, default=10, help="Window start spacing in control steps")
    p.add_argument("--max-train", type=int, default=300000, help="Cap on training examples (random subset)")
    p.add_argument("--max-test", type=int, default=30000)
    p.add_argument("--scene-scale", type=float, default=1.0, help="Multiply the per-type scene counts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source-id", type=int, default=0,
                   help="policy id written to meta[:, 4] when datasets from several policies are mixed (e.g. 0 u150, 1 E2, 2 V7, 3 T2)")
    args = p.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"{args.out} is not empty; refusing to mix datasets")
    import torch
    from types import SimpleNamespace
    from cat_mjlab.planner.data import ShardWriter, features, valid_starts, write_info, H, STRIDE
    from cat_mjlab.planner.grid import scene_grid, SHAPE, CELL, X0, Y0, Z0, SDF_LO, SDF_HI
    started = time.time()
    manifest = json.loads(Path(args.bank + "_packed/manifest.json").read_text())
    train, test, buckets = choose_scenes(manifest, DEFAULT_TYPES, args.scene_scale, args.seed)
    keep = {manifest["scenes"][i]["scene_id"] for i in train + test}
    install_subset_loader(keep)
    from record_mjlab_rollout import load_policy
    from cat_mjlab.runner import create_task, SCENE_BUCKETS
    learner, saved = load_policy(args.checkpoint, device=args.device, policy_id=0)
    env = copy.deepcopy(saved["contract"]["environment_config"])
    env["push_config"]["enable"] = False
    env["noise_config"]["level"] = 0.
    env["dm_rand_config"].update(enable_pd=False, enable_rfi=False)
    for key in ("sgs", "robustness"):
        env.pop(key, None)
    sim_class = None
    if args.device == "cpu":
        from cat_mjlab.sim_cpu import CPUSimulation as sim_class
        if args.num_envs != 1:
            raise ValueError("The CPU simulation runs one world: use --num-envs 1")
    ns = SimpleNamespace(bank_manifest=Path(args.bank + "_packed/manifest.json"), body_collision_bank=Path(args.bank + "_collision/manifest.json"),
                         body_collision_resets=Path(args.bank + "_resets/manifest.json"), device=args.device, seed=args.seed,
                         num_envs=args.num_envs, compile_task=False, nconmax=saved["contract"]["nconmax"],
                         njmax=saved["contract"]["njmax"], reactive_bank=None, reactive_row=None)
    task, sim, _ = create_task(ns, environment_config=env, sim_class=sim_class)
    if args.device == "cpu":
        task.collision_every_substep = False
    bank = task.bank
    base_weights = bank.weights.clone()
    if torch.cuda.is_available() and args.device != "cpu":
        free, total = torch.cuda.mem_get_info()
        print(json.dumps(dict(gpu_free_gb=round(free / 2**30, 1), gpu_total_gb=round(total / 2**30, 1))), flush=True)
    print(json.dumps(dict(checkpoint=args.checkpoint, bank=args.bank, train_scenes=len(train), test_scenes=len(test),
                          num_envs=args.num_envs, steps=args.steps, rounds=args.rounds, setup_s=round(time.time() - started, 1))), flush=True)
    writer = ShardWriter(args.out)
    rng = np.random.default_rng(args.seed)
    totals = {}
    for split, scenes, rounds, cap in (("train", train, args.rounds, args.max_train), ("test", test, args.test_rounds, args.max_test)):
        mask = torch.zeros_like(base_weights); mask[torch.as_tensor(scenes, device=mask.device)] = 1.
        bank.weights = base_weights * mask
        task.probabilities = bank.probabilities(bank.weights, stage=getattr(task, "curriculum_stage", None))
        examples, per_round = [], max(1, cap // rounds)
        for r in range(rounds):
            t_start = time.time()
            q, v, scene, epstep, ended, bad = rollout(task, learner, scenes, args.steps, rng, torch)
            starts = valid_starts(ended, bad, epstep, every=args.window_every)
            if len(starts) > per_round:
                starts = starts[np.sort(rng.choice(len(starts), per_round, replace=False))]
            device = args.device
            qt, vt = torch.as_tensor(q, device=device), torch.as_tensor(v, device=device)
            state, chunk, goal, origin, yaw = features(qt, vt, starts)
            sid = torch.as_tensor(scene[starts[:, 0], starts[:, 1]], device=device, dtype=torch.long)
            grid = scene_grid(bank, sid, origin, yaw).cpu().numpy()
            meta = np.stack((sid.cpu().numpy(), buckets[sid.cpu().numpy()], starts[:, 1], starts[:, 0],
                             np.full(len(starts), args.source_id, np.int64), np.full(len(starts), 0 if split == "train" else 1)), 1).astype(np.int32)
            for s in range(0, len(starts), 5000):
                pose = torch.cat((origin, yaw[:, None]), -1).cpu().numpy()
                writer.add(split, grid=grid[s:s + 5000], state=state[s:s + 5000].cpu().numpy(), chunk=chunk[s:s + 5000].cpu().numpy(),
                           goal=goal[s:s + 5000].cpu().numpy(), meta=meta[s:s + 5000], pose=pose[s:s + 5000])
            episodes = int(ended.sum()); bad_n = int(bad.sum())
            counts = np.bincount(meta[:, 1], minlength=len(SCENE_BUCKETS))
            print(json.dumps(dict(split=split, round=r, windows=int(len(starts)), episodes_ended=episodes, contact_or_fall=bad_n,
                                  mean_speed=round(float(goal[:, 4].mean()), 3), seconds=round(time.time() - t_start, 1),
                                  per_type={SCENE_BUCKETS[i]: int(c) for i, c in enumerate(counts) if c})), flush=True)
            totals[split] = totals.get(split, 0) + int(len(starts))
            del q, v, qt, vt, grid
    writer.flush()
    write_info(args.out, dict(schema="wbp-planner-data-v2", checkpoint=str(args.checkpoint), bank=args.bank, source_id=args.source_id,
                              train_scenes=[manifest["scenes"][i]["scene_id"] for i in train],
                              test_scenes=[manifest["scenes"][i]["scene_id"] for i in test],
                              scene_types=list(SCENE_BUCKETS), examples=totals, horizon_frames=H, frame_stride_steps=STRIDE,
                              control_dt=0.02, grid=dict(shape=list(SHAPE), cell=CELL, origin=[X0, Y0, Z0], sdf_clip=[SDF_LO, SDF_HI]),
                              seconds=round(time.time() - started, 1)))
    print("COLLECT DONE", json.dumps(dict(examples=totals, seconds=round(time.time() - started, 1))), flush=True)


if __name__ == "__main__":
    main()
