"""Bounded real-MJX implementation check, not a policy performance evaluation.

Run on a Slurm GPU allocation. Uses the complete scene/collision banks and
original checkpoint, but a tiny explicitly recorded training batch. No W&B,
evaluation episodes, retention, or changes to existing runs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()
    args.report = args.report.expanduser().resolve()
    if args.updates < 2:
        parser.error("At least two updates are needed to exercise continuing training")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this GPU check through Slurm")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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
    args.report.parent.mkdir(parents=True, exist_ok=True)
    launch = launcher.parser().parse_args([
        "validate", "--algorithm", "sapg", "--num-envs", "12", "--batch-size", "6",
        "--bank-manifest", str(ROOT / "data/furniture/cat_hand_protection_v1_20260917/manifest.json"),
        "--body-collision-bank", str(ROOT / "data/furniture/body_collision_hand_v1_20260917/manifest.json"),
        "--body-collision-resets", str(ROOT / "data/furniture/body_collision_hand_resets_v1_20260917/manifest.json"),
        "--wandb-mode", "disabled",
    ])
    start = time.monotonic()
    environment, factory, initial, record = launcher.prepare(launch, launcher.plan(launch))
    print("Prepared full scene bank and original checkpoint; starting bounded SAPG training check", flush=True)
    options = ppo_kwargs(record["config"])
    options.update(unroll_length=4, num_minibatches=2, num_updates_per_batch=2)
    snapshots, events, exports = [], [], []

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
        log_training_metrics=True, training_metrics_steps=48,
        **options,
    )
    if len(events) != args.updates or events[-1] != args.updates * 48:
        raise AssertionError(f"Wrong physical transition accounting: {events}")
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
    report = dict(
        status="passed", kind="bounded actual-MJX SAPG training correctness; not task evaluation or capacity proof",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        slurm_job_id=os.environ["SLURM_JOB_ID"], devices=[str(device) for device in jax.devices()],
        scene_count=record["field_bank"]["scene_count"], bank_sha256=record["bank_sha256"],
        warmstart=record["warmstart"], sapg=record["config"]["sapg"],
        validation_resources={key: options[key] for key in ("num_envs", "batch_size", "num_minibatches",
                                                            "unroll_length", "num_updates_per_batch")},
        completed_updates=len(events), physical_transitions=events[-1], actor_max_abs_change=change,
        folded_export_max_abs_error=float(jnp.max(jnp.abs(leader_action - saved_action))),
        runtime_keeps_embeddings=any("policy_embeddings" in path for path in snapshots[-1]["training_state"]["paths"]),
        metrics={key: float(np.asarray(value)) for key, value in metrics.items()},
        elapsed_seconds=time.monotonic() - start,
    )
    if not report["runtime_keeps_embeddings"]:
        raise AssertionError("Full runtime lost SAPG embeddings")
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
