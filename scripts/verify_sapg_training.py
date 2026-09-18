"""Bounded real-MJX implementation check, not a policy performance evaluation.

Run on a Slurm GPU allocation. Defaults to a tiny implementation check; use
--production-batch with explicit environment/batch counts to measure complete
production-sized updates. Scene and collision banks may be overridden together
for the hand specialist. Starts from the original checkpoint. No W&B, evaluation
episodes, retention, or changes to existing runs.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--production-batch", action="store_true",
                        help="Use native unroll=32, minibatches=64 and optimizer passes=4")
    parser.add_argument("--bank-manifest", type=Path,
                        default=ROOT / "data/furniture/cat_hand_protection_v1_20260917/manifest.json")
    parser.add_argument("--body-collision-bank", type=Path,
                        default=ROOT / "data/furniture/body_collision_hand_v1_20260917/manifest.json")
    parser.add_argument("--body-collision-resets", type=Path,
                        default=ROOT / "data/furniture/body_collision_hand_resets_v1_20260917/manifest.json")
    parser.add_argument("--min-free-vram-mib", type=float, default=0.,
                        help="Require this much assigned-GPU free VRAM after every completed update")
    return parser


def resource_settings(args, num_policies=6):
    """Validate actual checker geometry, including its augmented SAPG batch."""
    if args.updates < 2:
        raise ValueError("At least two updates are needed to exercise continuing training")
    if args.num_envs <= 0 or args.batch_size <= 0:
        raise ValueError("num-envs and batch-size must be positive")
    if args.num_envs % num_policies or args.batch_size % num_policies:
        raise ValueError("num-envs and batch-size must divide equally among six SAPG policies")
    if not math.isfinite(args.min_free_vram_mib) or args.min_free_vram_mib < 0:
        raise ValueError("min-free-vram-mib must be finite and nonnegative")
    unroll, minibatches, passes = (32, 64, 4) if args.production_batch else (4, 2, 2)
    if (args.batch_size * minibatches) % args.num_envs:
        raise ValueError("num-envs must divide batch-size * actual num-minibatches")
    return dict(num_envs=args.num_envs, batch_size=args.batch_size,
                unroll_length=unroll, num_minibatches=minibatches, num_updates_per_batch=passes)


def assigned_gpu_sample(pid=None):
    """Resolve this process's CUDA GPU UUID before querying device-wide memory.

    Slurm may remap CUDA device 0 to any physical GPU. GPU selection is never
    changed here, and a machine-wide/default-index memory sample is not accepted.
    """
    pid = os.getpid() if pid is None else pid
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    records = []
    for row in csv.reader(apps.splitlines()):
        if len(row) == 3 and row[0].strip() == str(pid):
            used = row[2].strip()
            records.append(dict(uuid=row[1].strip(),
                                process_used_mib=float(used) if used.replace(".", "", 1).isdigit() else None))
    uuids = {record["uuid"] for record in records}
    if len(uuids) != 1:
        raise RuntimeError(f"Expected this PID {pid} on one assigned GPU, found {sorted(uuids)}")
    uuid = next(iter(uuids))
    output = subprocess.check_output([
        "nvidia-smi", f"--id={uuid}",
        "--query-gpu=uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    rows = list(csv.reader(output.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 6 or rows[0][0].strip() != uuid:
        raise RuntimeError("Assigned GPU UUID could not be verified in NVIDIA memory query")
    _, name, total, used, free, utilization = [value.strip() for value in rows[0]]
    numeric = dict(total_mib=float(total), used_mib=float(used), free_mib=float(free),
                   utilization_percent=float(utilization))
    if not all(math.isfinite(value) and value >= 0 for value in numeric.values()):
        raise RuntimeError("Invalid NVIDIA GPU memory sample")
    return dict(pid=pid, uuid=uuid, name=name, monotonic_seconds=time.monotonic(),
                process_memory_records=records, **numeric)


def completed_update_record(step, metrics, device, *, started, min_free_vram_mib):
    """Ignore rollout callbacks; collect a synchronized completed-update sample."""
    if "training/sps" not in metrics:
        return None
    raw_values = {key: float(value) for key, value in metrics.items()}
    nonfinite = {key: repr(value) for key, value in raw_values.items() if not math.isfinite(value)}
    values = {key: value if math.isfinite(value) else None for key, value in raw_values.items()}
    reward = raw_values.get("training/rollout_reward_mean")
    reward_finite = reward is not None and math.isfinite(reward)
    gpu = assigned_gpu_sample()
    return dict(step=int(step), elapsed_seconds=time.monotonic() - started,
                metrics=values, metrics_finite=not nonfinite, nonfinite_metrics=nonfinite,
                reward_finite=reward_finite,
                sps_positive=math.isfinite(raw_values["training/sps"]) and raw_values["training/sps"] > 0,
                jax_memory={key: int(value) for key, value in (device.memory_stats() or {}).items()},
                gpu=gpu, min_free_vram_mib=min_free_vram_mib,
                free_vram_passes=gpu["free_mib"] >= min_free_vram_mib)


def main():
    arguments = parser()
    args = arguments.parse_args()
    args.report = args.report.expanduser().resolve()
    try:
        settings = resource_settings(args)
    except ValueError as error:
        arguments.error(str(error))
    if not os.environ.get("SLURM_JOB_ID"):
        arguments.error("Run this GPU check through Slurm")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["WANDB_MODE"] = "disabled"
    from cat_ppo.furniture.generalist_logging import atomic_json

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema="cat-sapg-gpu-training-check-v2", status="starting", pid=os.getpid(),
        slurm_job_id=os.environ["SLURM_JOB_ID"], production_batch=bool(args.production_batch),
        requested_updates=args.updates, min_free_vram_mib=args.min_free_vram_mib,
        validation_resources=settings, updates=[], wandb_initialized=False, evaluation_enabled=False,
        bank_manifest=str(args.bank_manifest.expanduser().resolve()),
        body_collision_bank=str(args.body_collision_bank.expanduser().resolve()),
        body_collision_resets=str(args.body_collision_resets.expanduser().resolve()),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment={key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
            "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")},
        limitations=["Bounded complete-update capacity check, not sustained stability or task performance",
                     "NVIDIA samples are completed-update snapshots, not continuous peak measurements",
                     "JAX allocator peak is cumulative; runtime state is captured in RAM, not written to disk"])
    atomic_json(args.report, report)
    started = time.monotonic()
    try:
        run_check(args, report, settings, started)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.monotonic() - started)
        atomic_json(args.report, report)
        raise
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


def run_check(args, report, settings, start):

    import jax
    import jax.numpy as jnp
    import numpy as np
    from brax.training.agents.ppo import checkpoint
    from cat_ppo.furniture.generalist_training import wrap_for_cat_wholebody_training
    from cat_ppo.furniture.generalist_config import ppo_kwargs
    from cat_ppo.furniture.generalist_logging import atomic_json
    from cat_ppo.learning.policy.ppo import train
    from cat_ppo.learning.policy.sapg.networks import collapse_policy_params
    import train_cat_wholebody as launcher

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError(f"Expected one Slurm-assigned GPU, got {jax.devices()}")
    device = jax.devices()[0]
    launch = launcher.parser().parse_args([
        "validate", "--algorithm", "sapg", "--num-envs", str(args.num_envs), "--batch-size", str(args.batch_size),
        "--bank-manifest", str(args.bank_manifest.expanduser().resolve()),
        "--body-collision-bank", str(args.body_collision_bank.expanduser().resolve()),
        "--body-collision-resets", str(args.body_collision_resets.expanduser().resolve()),
        "--wandb-mode", "disabled",
    ])
    environment, factory, initial, record = launcher.prepare(launch, launcher.plan(launch))
    print("Prepared requested scene bank and original checkpoint; starting bounded SAPG training check", flush=True)
    options = ppo_kwargs(record["config"])
    options.update(settings)
    transitions_per_update = (options["batch_size"] * options["num_minibatches"]
                              * options["unroll_length"] * options["action_repeat"])
    report.update(
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        code=record["code"], devices=[str(item) for item in jax.devices()],
        jax_version=jax.__version__, scene_count=record["field_bank"]["scene_count"],
        field_bank=record["field_bank"], bank_sha256=record["bank_sha256"],
        specialist_provenance=environment.field_bank_manifest.get("specialist"),
        scene_ids=[scene["scene_id"] for scene in environment.field_bank_manifest["scenes"]],
        scene_families={scene["scene_id"]: scene["family"] for scene in environment.field_bank_manifest["scenes"]},
        warmstart=record["warmstart"], sapg=record["config"]["sapg"],
        field_array_bytes=sum(getattr(environment, key).nbytes for key in ("sdf", "bf", "gf")),
        transitions_per_update=transitions_per_update,
        augmented_transitions_per_update=(transitions_per_update * (record["config"]["sapg"]["num_policies"] + 1)
                                           // record["config"]["sapg"]["num_policies"]),
        training_metrics_steps=transitions_per_update,
        training_parameters={key: options[key] for key in ("learning_rate", "entropy_cost", "clipping_epsilon",
            "discounting", "gae_lambda", "reward_scaling", "action_repeat", "num_evals", "num_eval_envs", "num_resets_per_eval")},
        preparation_seconds=time.monotonic() - start,
        jax_memory_after_prepare={key: int(value) for key, value in (device.memory_stats() or {}).items()},
    )
    if getattr(environment, "body_collision_enabled", False):
        report["body_collision"] = environment.body_collision_contract
        report["body_collision_array_bytes"] = sum(value.nbytes for value in environment._body_collision_bank.values())
    atomic_json(args.report, report)
    snapshots, events, exports = [], [], []

    def progress(step, metrics):
        update = completed_update_record(step, metrics, device, started=start,
                                         min_free_vram_mib=args.min_free_vram_mib)
        if update is None:
            return
        report["updates"].append(update)
        atomic_json(args.report, report)
        print(json.dumps(dict(completed_capacity_update=len(report["updates"]), **update)), flush=True)
        if not update["metrics_finite"] or not update["reward_finite"] or not update["sps_positive"]:
            raise AssertionError("Nonfinite SAPG update metrics/reward or nonpositive SPS")
        if not update["free_vram_passes"]:
            raise RuntimeError(f"Assigned GPU has {update['gpu']['free_mib']} MiB free, "
                               f"below requested {args.min_free_vram_mib} MiB")

    def save_runtime(step, snapshot):
        # Retain only the most recent complete snapshot, as production does.
        snapshots[:] = [snapshot]
        if step:
            events.append(int(step))
            print(f"Completed SAPG training update at {step} physical transitions", flush=True)

    def scored(step, make_policy, params, network_config, metrics, source):
        del make_policy, source
        exports[:] = [(int(step), collapse_policy_params(params), network_config)]

    make_policy, final, metrics = train.train(
        environment=environment, num_timesteps=0, continuous=True,
        should_stop_fn=lambda: len(events) >= args.updates,
        training_steps_per_epoch=1, network_factory=factory,
        wrap_env_fn=wrap_for_cat_wholebody_training,
        restore_params=initial, restore_value_fn=True,
        sapg_config=record["config"]["sapg"],
        runtime_checkpoint_fn=save_runtime, scored_checkpoint_fn=scored,
        progress_fn=progress, log_training_metrics=True, training_metrics_steps=transitions_per_update,
        **options,
    )
    if events != [transitions_per_update * (index + 1) for index in range(args.updates)]:
        raise AssertionError(f"Wrong physical transition accounting: {events}")
    if [update["step"] for update in report["updates"]] != events:
        raise AssertionError("Completed-update metric samples do not match completed runtime snapshots")
    if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(final)):
        raise AssertionError("Nonfinite SAPG parameters")
    change = float(jnp.max(jnp.abs(final[1]["params"]["hidden_0"]["kernel"] -
                                  initial[1]["params"]["hidden_0"]["kernel"])))
    if change <= 0:
        raise AssertionError("Actor did not update")
    step, exported, export_config = exports[-1]
    export_root = args.report.parent / "leader_export"
    checkpoint.save(export_root, step, exported, export_config)
    deployed = checkpoint.load_policy(export_root / f"{step:012d}", deterministic=True)
    observations = {"state": jnp.zeros((2, 222)), "privileged_state": jnp.zeros((2, 310))}
    # Compare the mathematical export using full float32 products. This scope
    # changes only verification: training retains native default GPU precision.
    with jax.default_matmul_precision("highest"):
        leader_action = make_policy(final, deterministic=True)(observations, jax.random.PRNGKey(0))[0]
        saved_action = deployed(observations, jax.random.PRNGKey(0))[0]
    np.testing.assert_allclose(leader_action, saved_action, rtol=1e-5, atol=1e-5)
    report.update(
        status="passed", kind="bounded actual-MJX SAPG training and GPU capacity check; not task evaluation",
        completed_updates=len(events), physical_transitions=events[-1], actor_max_abs_change=change,
        folded_export_max_abs_error=float(jnp.max(jnp.abs(leader_action - saved_action))),
        runtime_keeps_embeddings=any("policy_embeddings" in path for path in snapshots[-1]["training_state"]["paths"]),
        metrics={key: float(np.asarray(value)) for key, value in metrics.items()},
        elapsed_seconds=time.monotonic() - start,
        minimum_completed_update_free_mib=min(update["gpu"]["free_mib"] for update in report["updates"]),
        warm_update_seconds=report["updates"][-1]["elapsed_seconds"] - report["updates"][-2]["elapsed_seconds"],
    )
    if not report["runtime_keeps_embeddings"]:
        raise AssertionError("Full runtime lost SAPG embeddings")


if __name__ == "__main__":
    main()
