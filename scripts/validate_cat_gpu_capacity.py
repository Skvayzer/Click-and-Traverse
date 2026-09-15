"""Measure one disposable native PPO update's GPU capacity; never start W&B.

The update starts from released CAT weights and is discarded. No learner/model
checkpoint is saved. The only output is a capacity report outside the field bank.
This measures resource use, not whether the policy learns useful behavior.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def gpu_sample(index):
    output = subprocess.check_output([
        "nvidia-smi", f"--id={index}",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    name, total, used, free, utilization = next(csv.reader(output.strip().splitlines()))
    return dict(monotonic_seconds=time.monotonic(), name=name.strip(), total_mib=int(total),
                used_mib=int(used), free_mib=int(free), utilization_percent=int(utilization))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--reference-kl-coefficient", type=float, default=.05)
    parser.add_argument("--allocator-fraction", type=float, default=.92)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest_path, output = args.bank_manifest.resolve(), args.output.resolve()
    if not manifest_path.is_file() or output.is_relative_to(manifest_path.parent):
        parser.error("Need an existing manifest and an output path outside its bank")
    if not 0 < args.allocator_fraction < 1:
        parser.error("allocator-fraction must lie strictly between zero and one")
    if args.learning_rate <= 0 or args.reference_kl_coefficient < 0:
        parser.error("learning-rate must be positive and reference-KL coefficient nonnegative")
    # Preserve CPU callbacks alongside CUDA, as required by the native learner.
    # Nothing that can initialize JAX is imported before these assignments.
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.allocator_fraction)
    os.environ["WANDB_MODE"] = "disabled"

    initial_gpu = gpu_sample(args.gpu_index)
    report = dict(schema="cat-native-ppo-gpu-capacity-v1", status="starting", pid=os.getpid(),
        bank_manifest=str(manifest_path), parallel_environments=args.num_envs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, reference_kl_coefficient=args.reference_kl_coefficient,
        requested_updates=1, wandb_initialized=False, checkpoints_written=False,
        production_training_metrics_enabled=True,
        capacity_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initialization="released CAT checkpoint adapted to the compact 222/310 observation contract",
        scope="one disposable complete PPO update; resource-capacity evidence only",
        limitations=["No continuous-training stability or policy-improvement claim",
                     "No checkpoint serialization; production runtime snapshot transfer is not measured",
                     "NVIDIA polling may miss short peaks; JAX allocator peak is reported separately"],
        environment={name: os.environ.get(name) for name in (
            "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_FLAGS", "WANDB_MODE")},
        initial_gpu=initial_gpu, gpu_samples=[initial_gpu], updates=[])
    stopped = threading.Event()

    def monitor():
        while not stopped.wait(.5):
            try:
                report["gpu_samples"].append(gpu_sample(args.gpu_index))
            except Exception as error:
                report.setdefault("gpu_monitor_errors", []).append(str(error))

    thread = threading.Thread(target=monitor, name="capacity-gpu-monitor", daemon=True)
    thread.start()
    started = time.monotonic()
    device = None
    try:
        import jax
        import numpy as np
        import train_cat_wholebody as launcher
        from cat_ppo.furniture.generalist_config import ppo_kwargs
        from cat_ppo.furniture.generalist_training import wrap_for_cat_wholebody_training
        from cat_ppo.learning.policy.ppo import train as native_ppo

        if jax.devices()[0].platform != "gpu":
            raise RuntimeError(f"Capacity validation requires CUDA; got {jax.devices()}")
        device = jax.devices()[0]
        report.update(jax_version=jax.__version__, devices=[str(item) for item in jax.devices()])
        if args.reference_kl_coefficient and "reference_kl_config" not in inspect.signature(native_ppo.train).parameters:
            raise RuntimeError("Install the final reference-KL training code before this capacity check")
        launch_args = launcher.parser().parse_args([
            "validate", "--bank-manifest", str(manifest_path), "--profile", "single_gpu_32gb",
            "--num-envs", str(args.num_envs), "--batch-size", str(args.batch_size),
            "--seed", str(args.seed), "--wandb-mode", "disabled"])
        specification = launcher.plan(launch_args)
        specification["config"]["policy_config"]["learning_rate"] = args.learning_rate
        print("Preparing the complete compact field bank and released CAT warm-start...", flush=True)
        env, factory, target, preparation = launcher.prepare(launch_args, specification)
        report.update(code=preparation["code"], bank_sha256=preparation["bank_sha256"],
            warmstart=preparation["warmstart"],
            batch_geometry=specification["config"]["fine_tuning"]["effective_batch_geometry"],
            observation_dimensions={key: len(preparation["observation_contract"][field])
                                    for key, field in (("actor", "actor_features"), ("critic", "critic_features"))},
            field_array_bytes=sum(getattr(env, name).nbytes for name in ("sdf", "bf", "gf")),
            scene_count=env.num_pf_scenes, allocator_after_prepare=device.memory_stats(),
            preparation_seconds=time.monotonic()-started)
        options = ppo_kwargs(specification["config"])
        report["training_parameters"] = {name: options[name] for name in (
            "learning_rate", "clipping_epsilon", "entropy_cost", "unroll_length",
            "batch_size", "num_minibatches", "num_updates_per_batch", "num_envs")}
        if args.reference_kl_coefficient:
            options["reference_kl_config"] = dict(coefficient=args.reference_kl_coefficient,
                action_indices=list(range(12)),
                scene_mask=[scene["task_kind"] == "cat" for scene in env.field_bank_manifest["scenes"]])
        elif "reference_kl_config" in inspect.signature(native_ppo.train).parameters:
            options["reference_kl_config"] = None

        def progress(step, metrics):
            # The production metric aggregator also emits callbacks during a
            # rollout. Only the completed-update callback ends this preflight.
            if "training/rollout_reward_mean" not in metrics:
                return
            record = dict(step=int(step), elapsed_seconds=time.monotonic()-started,
                          metrics={name: float(np.asarray(value)) for name, value in metrics.items()},
                          allocator=device.memory_stats(), gpu=gpu_sample(args.gpu_index))
            if not all(np.isfinite(value) for value in record["metrics"].values()):
                raise ValueError("Nonfinite native PPO metrics during capacity validation")
            report["updates"].append(record)
            print(json.dumps(dict(completed_capacity_update=len(report["updates"]), **record)), flush=True)

        options.update(environment=env, num_timesteps=0, continuous=True, training_steps_per_epoch=1,
            wrap_env_fn=wrap_for_cat_wholebody_training, network_factory=factory,
            restore_params=target, restore_value_fn=True,
            runtime_checkpoint_fn=None, scored_checkpoint_fn=None, save_checkpoint_path=None,
            log_training_metrics=True, training_metrics_buffer_size=1000,
            training_metrics_steps=report["batch_geometry"]["transitions_per_update"], progress_fn=progress,
            should_stop_fn=lambda: len(report["updates"]) >= 1)
        print("Compiling and executing exactly one complete disposable PPO update...", flush=True)
        _, parameters, metrics = native_ppo.train(**options)
        jax.block_until_ready(parameters)
        expected = report["batch_geometry"]["transitions_per_update"]
        if len(report["updates"]) != 1 or int(metrics["training/completed_steps"]) != expected:
            raise ValueError("Capacity check did not complete exactly one full PPO update")
        report.update(status="passed", completed_steps=int(metrics["training/completed_steps"]),
                      final_allocator=device.memory_stats())
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stopped.set()
        thread.join(timeout=11)
        report["elapsed_seconds"] = time.monotonic()-started
        try:
            report["gpu_samples"].append(gpu_sample(args.gpu_index))
        except Exception:
            pass
        if device is not None:
            report["final_allocator"] = device.memory_stats()
        report["observed_gpu_used_peak_mib"] = max(item["used_mib"] for item in report["gpu_samples"])
        report["observed_gpu_free_min_mib"] = min(item["free_mib"] for item in report["gpu_samples"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({name: report[name] for name in (
            "status", "parallel_environments", "batch_size", "observed_gpu_used_peak_mib",
            "observed_gpu_free_min_mib", "elapsed_seconds")}, indent=2), flush=True)
        print(f"Capacity report: {output}", flush=True)


if __name__ == "__main__":
    main()
