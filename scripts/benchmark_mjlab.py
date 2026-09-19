#!/usr/bin/env python3
"""Bounded full-task/learner capacity probe: no W&B, evaluation or checkpoints.

The default probe allocates the real 32-step rollout at one trajectory per
environment. --update additionally performs one complete native PPO/SAPG
update (64 minibatches, four passes from the checkpoint contract). Changes
exist only in this process. Throughput is aggregate physical control steps,
not SAPG's virtual relabeled transitions. This is an engineering check, not
an estimate of policy success or a stability guarantee for a long run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cat_mjlab.checkpoint_arrays import read_array_archive, sampling_state_from_archive
from cat_mjlab.conversion import load_array_archive
from cat_mjlab.learning import Learner
from cat_mjlab.runner import (_file_hash, collect_rollout, create_task,
                              environment_config_from_archive, learner_config_from_archive)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-npz", required=True, type=Path)
    p.add_argument("--bank-manifest", required=True, type=Path)
    p.add_argument("--body-collision-bank", required=True, type=Path)
    p.add_argument("--body-collision-resets", required=True, type=Path)
    p.add_argument("--num-envs", type=int, default=18432)
    p.add_argument("--algorithm", choices=("ppo", "sapg"))
    p.add_argument("--batch-size", type=int, help="Defaults to num_envs / checkpoint num_minibatches")
    p.add_argument("--unroll-length", type=int, default=32)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--update", action="store_true", help="Perform exactly one full learner update in memory")
    p.add_argument("--compile-task", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nconmax", type=int, default=64)
    p.add_argument("--njmax", type=int, default=256)
    return p


def geometry(args, config):
    if args.num_envs <= 0 or args.unroll_length <= 0 or args.warmup_steps < 0:
        raise ValueError("Environment/rollout counts must be positive; warmup must be nonnegative")
    if args.batch_size is None and args.num_envs % config.num_minibatches:
        raise ValueError("Default batch geometry requires num_envs divisible by num_minibatches")
    batch = args.batch_size if args.batch_size is not None else args.num_envs // config.num_minibatches
    trajectories = batch * config.num_minibatches
    if batch <= 0 or trajectories % args.num_envs:
        raise ValueError("batch_size*num_minibatches must be divisible by num_envs")
    if config.algorithm == "sapg" and (args.num_envs % config.num_policies or batch % config.num_policies):
        raise ValueError("SAPG environment and batch counts must be divisible by num_policies")
    return batch, trajectories


def memory(device):
    free, total = torch.cuda.mem_get_info(device)
    return dict(device_total_gib=total / 2**30, device_free_gib=free / 2**30,
                device_used_gib=(total-free) / 2**30,
                torch_allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                torch_reserved_gib=torch.cuda.memory_reserved(device) / 2**30,
                torch_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                torch_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)


def ensure_finite(values, label):
    bad = [name for name, value in values.items()
           if torch.is_floating_point(value) and not bool(torch.isfinite(value).all())]
    if bad:
        raise FloatingPointError(f"Nonfinite {label}: {bad}")


def benchmark(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The capacity benchmark requires the actual CUDA training device")
    torch.cuda.set_device(device)
    torch.cuda.init()
    started = time.monotonic()
    metadata, arrays = read_array_archive(args.checkpoint_npz)
    config = learner_config_from_archive(metadata, algorithm=args.algorithm)
    batch, trajectories = geometry(args, config)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = memory(device)
    learner = Learner(config, device=device)
    migration = load_array_archive(learner, args.checkpoint_npz, restore_optimizer=True)
    task, sim, env_config = create_task(args, environment_config=environment_config_from_archive(metadata))
    sampling, sampling_report = sampling_state_from_archive(metadata, arrays, _file_hash(args.bank_manifest))
    if sampling is not None:
        task.restore_sampling_state(sampling, resample=True)
    del arrays
    policies = config.num_policies if config.algorithm == "sapg" else 1
    ids = torch.arange(policies, device=device).repeat_interleave(args.num_envs // policies)
    task.set_policy_ids(ids)
    torch.cuda.synchronize(device)
    startup = time.monotonic() - started
    print(json.dumps(dict(phase="initialized", seconds=startup, num_envs=args.num_envs,
                          sampling=sampling_report, memory=memory(device))), flush=True)
    began = time.monotonic()
    for _ in range(args.warmup_steps):
        step = task.step(learner.act(task.obs, policy_ids=ids)["action"])
        ensure_finite(dict(reward=step["reward"], **step["obs"]), "warmup task")
    capacity = sim.capacity_report()
    torch.cuda.synchronize(device)
    warmup = time.monotonic() - began
    print(json.dumps(dict(phase="warmup_complete", seconds=warmup, capacity=capacity,
                          memory=memory(device))), flush=True)
    began = time.monotonic()
    rollout, _ = collect_rollout(task, learner, unroll_length=args.unroll_length,
                                 trajectories=trajectories, policy_ids=ids)
    torch.cuda.synchronize(device)
    rollout_seconds = time.monotonic() - began
    capacity = sim.capacity_report()
    ensure_finite(rollout, "rollout")
    ensure_finite(dict(qpos=sim.data.qpos, qvel=sim.data.qvel, **task.obs), "final physics/task state")
    rollout_memory = memory(device)
    print(json.dumps(dict(phase="rollout_complete", seconds=rollout_seconds,
                          capacity=capacity, memory=rollout_memory)), flush=True)
    update_seconds = None
    if args.update:
        began = time.monotonic()
        learner.update(rollout)
        torch.cuda.synchronize(device)
        update_seconds = time.monotonic() - began
        ensure_finite(learner.model.state_dict(), "updated model")
    physical_steps = trajectories * args.unroll_length
    return dict(schema="cat-mjlab-capacity-benchmark-v1", num_envs=args.num_envs,
        batch_size=batch, trajectories=trajectories, unroll_length=args.unroll_length,
        compile_task=args.compile_task, warmup_steps=args.warmup_steps, learner_config=asdict(config),
        source_step=metadata["step"], source_sha256=metadata.get("source_sha256"),
        source_optimizer=migration["optimizer"], sampling=sampling_report,
        bank_sha256=_file_hash(args.bank_manifest), physics_options=sim.option_contract(),
        initialization_seconds=startup, warmup_seconds=warmup, rollout_seconds=rollout_seconds,
        update_seconds=update_seconds, full_update_performed=args.update, physical_control_steps=physical_steps,
        rollout_control_steps_per_second=physical_steps/rollout_seconds,
        training_control_steps_per_second=physical_steps/(rollout_seconds+update_seconds) if args.update else None,
        physics_substeps_per_control=int(round(env_config["ctrl_dt"]/env_config["sim_dt"])),
        finite_rollout=True, finite_task_state=True, capacity=capacity,
        memory_before=initial_memory, memory_after_rollout=rollout_memory, memory_final=memory(device),
        memory_note="Device used includes Warp and other processes; Torch peaks exclude Warp allocations.",
        writes_checkpoints=False, wandb=False, evaluation=False,
        limitation="Bounded capacity/throughput probe; does not establish overnight stability or policy success.")


def main(argv=None):
    args = parser().parse_args(argv)
    print(json.dumps(benchmark(args), indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
