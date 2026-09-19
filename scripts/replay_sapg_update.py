"""Replay bounded SAPG updates from a complete runtime snapshot, through Slurm.

Reads the archived run and snapshot without modifying either. No W&B,
evaluation, model export or runtime persistence is performed. Diagnostics are
optional because adding JAX callbacks/reductions can change compiler fusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", type=Path, required=True)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--runtime", type=Path, help="Defaults to run-dir/resume.msgpack")
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--updates", type=int, choices=range(1, 5), default=1)
    result.add_argument("--expected-start-step", type=int, required=True)
    result.add_argument("--allow-source-change", action="store_true",
                        help="Explicitly permit only source-hash migration in the in-memory runtime contract")
    result.add_argument("--diagnostics", action="store_true",
                        help="Instrument minibatches; preserves formulas/RNG but can change XLA fusion")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def runtime_for_source(runtime, old_record, new_record, *, allow_source_change):
    """Rebind metadata only; retain references to every large learner array."""
    identity = dict(config=old_record["config"], bank_sha256=old_record["bank_sha256"],
                    source_sha256=old_record["code"]["source_sha256"],
                    contract=old_record["observation_contract"])
    if runtime["contract"]["metadata"] != identity:
        raise ValueError("Snapshot metadata does not match the original run record")
    for key in ("config", "bank_sha256", "observation_contract"):
        if old_record[key] != new_record[key]:
            raise ValueError(f"Replay preparation changed {key}")
    source = new_record["code"]["source_sha256"]
    if source != identity["source_sha256"] and not allow_source_change:
        raise ValueError("Source differs; explicit --allow-source-change is required")
    metadata = dict(identity, source_sha256=source)
    result = dict(runtime, contract=dict(runtime["contract"], metadata=metadata))
    return result, metadata


def host_finiteness(tree):
    import jax
    import numpy as np
    leaves = jax.tree_util.tree_leaves(tree)
    arrays = [np.asarray(value) for value in leaves]
    floating = [value for value in arrays if np.issubdtype(value.dtype, np.inexact)]
    invalid = sum(int(np.count_nonzero(~np.isfinite(value))) for value in floating)
    return dict(finite=invalid == 0, nonfinite_elements=invalid,
                floating_elements=sum(value.size for value in floating), leaves=len(leaves))


def numerical_assessment(report):
    """Execution completion alone is insufficient: require finite learner updates."""
    failures = []
    loss_names = ("total_loss", "policy_loss", "v_loss", "entropy_loss")

    def finite(value):
        try:
            return math.isfinite(float(value))
        except (ValueError, TypeError):
            return False

    start, stride = report.get("start_step"), report.get("transitions_per_update")
    expected = ([] if start is None or stride is None else
                [start + stride * (index + 1) for index in range(report["requested_updates"])])
    if not report.get("execution_completed", False):
        failures.append("native replay did not complete")
    if not expected or [item["step"] for item in report["boundaries"]] != expected:
        failures.append("requested update boundaries were not reached")
    if not expected or [item["step"] for item in report["updates"]] != expected:
        failures.append("completed-update loss records are missing")
    if expected and report.get("final_step") != expected[-1]:
        failures.append("final physical step differs from requested target")
    if not report.get("initial_training_state", {}).get("finite", False):
        failures.append("starting learner state is missing or nonfinite")
    if not report.get("final_params", {}).get("finite", False):
        failures.append("final parameters are missing or nonfinite")
    for boundary in report["boundaries"]:
        if not boundary["training_state"]["finite"]:
            failures.append(f"learner/optimizer state is nonfinite at {boundary['step']}")
    for update in report["updates"]:
        for name in loss_names:
            if not finite(update["metrics"].get("training/" + name)):
                failures.append(f"missing/nonfinite {name} at {update['step']}")
    # The instrumented fixed implementation deliberately computes the OLD
    # unsafe expression for comparison. Its replay_legacy_* diagnostics do not
    # determine whether the actual fixed loss or gradients are numerically valid.
    for event in report["diagnostic_events"]:
        if event["kind"] != "minibatch":
            continue
        invalid = [name for name in loss_names if not finite(event.get(name))]
        invalid += [name for name in ("gradient_nonfinite", "updated_params_nonfinite", "optimizer_nonfinite")
                    if event.get(name) != 0]
        if invalid:
            failures.append(f"invalid minibatch {event.get('adam_update_index')}: {', '.join(invalid)}")
            break
    if not report.get("runtime_unchanged", False):
        failures.append("source runtime integrity was not verified")
    return dict(numerical_pass=not failures, numerical_failures=failures,
                requested_end_step=expected[-1] if expected else None)


def install_diagnostics(events):
    """Observe original loss and gradient update without changing their outputs."""
    from brax.training import gradients
    from cat_ppo.learning.policy.sapg import losses
    import jax
    import jax.numpy as jnp
    import optax

    original_loss = losses.compute_sapg_loss
    original_prepare = losses.prepare_rollout
    original_gradient_update = gradients.gradient_update_fn

    def stats(name, values):
        values = jax.lax.stop_gradient(values)
        finite = jnp.isfinite(values)
        return {name + "_nonfinite": jnp.sum(~finite),
                name + "_min_finite": jnp.min(jnp.where(finite, values, jnp.inf)),
                name + "_max_finite": jnp.max(jnp.where(finite, values, -jnp.inf))}

    def receive(kind, values):
        record = dict(kind=kind, **clean_json(values))
        events.append(record)
        bad = any(value != 0 for key, value in record.items()
                  if key.endswith("_nonfinite") and isinstance(value, (int, float)))
        if bad and not any(event.get("first_nonfinite_reported") for event in events):
            record["first_nonfinite_reported"] = True
            print(json.dumps(dict(first_nonfinite_callback=record)), flush=True)

    def prepared(*args, **kwargs):
        data = original_prepare(*args, **kwargs)
        extras = data.extras["policy_extras"]
        values = {}
        for name in (losses.TARGET_VALUE, losses.ADVANTAGE, losses.OLD_TARGET_LOG_PROB):
            values.update(stats(name, extras[name]))
        log_mu = extras[losses.OLD_TARGET_LOG_PROB] - extras["log_prob"]
        values.update(stats("log_importance", log_mu))
        if "sapg_importance_weight" in extras:
            values.update(stats("stored_importance", extras["sapg_importance_weight"]))
        jax.debug.callback(lambda values: receive("prepared_rollout", values), values)
        return data

    def diagnosed_loss(params, normalizer, data, rng, *args, **kwargs):
        value, metrics = original_loss(params, normalizer, data, rng, *args, **kwargs)
        extras = data.extras["policy_extras"]
        network = kwargs["ppo_network"]
        logits = network.policy_network.apply(
            normalizer, params.policy, data.observation, extras[losses.TARGET_POLICY_ID])
        log_prob = network.parametric_action_distribution.log_prob(logits, extras["raw_action"])
        old_log_prob = extras[losses.OLD_TARGET_LOG_PROB]
        log_mu = old_log_prob - extras["log_prob"]
        log_q = log_prob - old_log_prob
        advantages = extras[losses.ADVANTAGE]
        # These diagnostics deliberately expose the old expression, even when
        # replaying the fixed loss. They never contribute to its gradients.
        mu, q = jnp.exp(jax.lax.stop_gradient(log_mu)), jnp.exp(jax.lax.stop_gradient(log_q))
        epsilon = kwargs.get("clipping_epsilon", 0.3)
        legacy_surrogate = mu * jnp.minimum(q * advantages,
            jnp.clip(q, 1 - epsilon, 1 + epsilon) * advantages)
        diagnostics = {}
        for name, item in (("log_q", log_q), ("log_importance", log_mu),
                           ("combined_log_ratio", log_prob - extras["log_prob"]),
                           ("legacy_q", q), ("legacy_mu", mu),
                           ("legacy_surrogate", legacy_surrogate)):
            diagnostics.update(stats(name, item))
        diagnostics["legacy_zero_mu_infinite_q"] = jnp.sum((mu == 0) & jnp.isinf(q))
        return value, dict(metrics, **{"replay_" + key: item for key, item in diagnostics.items()})

    def gradient_update(loss_fn, optimizer, pmap_axis_name, has_aux=False):
        evaluate = gradients.loss_and_pgrad(loss_fn, pmap_axis_name, has_aux)

        def update(*args, optimizer_state):
            value, grads = evaluate(*args)
            updates, new_optimizer = optimizer.update(grads, optimizer_state)
            params = optax.apply_updates(args[0], updates)
            metrics = dict(value[1])
            metrics["gradient_norm"] = optax.global_norm(grads)
            for label, tree in (("gradient", grads), ("updated_params", params), ("optimizer", new_optimizer)):
                leaves = jax.tree_util.tree_leaves(tree)
                metrics[label + "_nonfinite"] = sum(jnp.sum(~jnp.isfinite(leaf)) for leaf in leaves)
                metrics[label + "_max_abs_finite"] = jnp.max(jnp.stack([
                    jnp.max(jnp.where(jnp.isfinite(leaf), jnp.abs(leaf), 0.)) for leaf in leaves]))
            counts = [leaf for path, leaf in jax.tree_util.tree_flatten_with_path(optimizer_state)[0]
                      if any(getattr(key, "name", None) == "count" for key in path)]
            if counts:
                metrics["adam_update_index"] = counts[0]
            jax.debug.callback(lambda metrics: receive("minibatch", metrics), metrics)
            return value, params, new_optimizer

        return update

    losses.prepare_rollout = prepared
    losses.compute_sapg_loss = diagnosed_loss
    gradients.gradient_update_fn = gradient_update

    def restore():
        losses.prepare_rollout = original_prepare
        losses.compute_sapg_loss = original_loss
        gradients.gradient_update_fn = original_gradient_update

    return restore


def replay(args, report, write):
    import jax
    import train_cat_wholebody as launcher
    from cat_ppo.furniture.generalist_config import ppo_kwargs
    from cat_ppo.furniture.generalist_runtime import load_runtime
    from cat_ppo.furniture.generalist_training import wrap_for_cat_wholebody_training
    from cat_ppo.learning.policy.ppo import train

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"Expected one Slurm-assigned GPU; got {devices}")
    old = json.loads((args.run_dir / "run.json").read_text())
    specification = json.loads((args.run_dir / "launch.json").read_text())["specification"]
    config = specification["config"]
    if config.get("algorithm") != "sapg" or config["fine_tuning"]["mode"] != "cat_train_only":
        raise ValueError("Replay requires SAPG training-only configuration")
    runtime = load_runtime(args.runtime)
    if runtime["step"] != args.expected_start_step:
        raise ValueError(f"Unexpected starting step {runtime['step']}")
    args_prepare = argparse.Namespace(bank_manifest=Path(specification["bank_manifest"]),
                                     seed=config["policy_config"]["seed"])
    environment, factory, target, record = launcher.prepare(args_prepare, specification, restore_model=False)
    if target is not None:
        raise AssertionError("Replay must not warm-start model parameters")
    runtime, identity = runtime_for_source(runtime, old, record, allow_source_change=args.allow_source_change)
    options = ppo_kwargs(config)
    if any(options[key] != 0 for key in ("num_evals", "num_eval_envs", "num_resets_per_eval")):
        raise ValueError("Replay cannot run evaluation or reset environments between updates")
    transitions = options["batch_size"] * options["num_minibatches"] * options["unroll_length"] * options["action_repeat"]
    report.update(start_step=int(runtime["step"]), transitions_per_update=transitions,
                  original_code=old["code"], replay_code=record["code"],
                  source_migration=old["code"] != record["code"],
                  configuration=config, bank_sha256=record["bank_sha256"],
                  initial_training_state=host_finiteness(runtime["training_state"]),
                  preserved_state=["actor", "critic", "all policy embeddings", "Adam", "normalizer", "RNG", "environment", "metric windows"],
                  devices=[str(device) for device in devices])
    if not report["initial_training_state"]["finite"]:
        raise ValueError("Starting learner state is already nonfinite")
    write()
    events, boundaries = report["diagnostic_events"], report["boundaries"]
    restore = install_diagnostics(events) if args.diagnostics else lambda: None

    def progress(step, metrics):
        if "training/sps" in metrics:
            report["updates"].append(dict(step=int(step), metrics=clean_json(metrics)))
            write()
            print(json.dumps(dict(replay_update=int(step), metrics=clean_json(metrics))), flush=True)

    def boundary(step, snapshot):
        if step <= args.expected_start_step:
            raise AssertionError("Unexpected pre-update snapshot during exact resume")
        boundaries.append(dict(step=int(step), training_state=host_finiteness(snapshot["training_state"]),
                               jax_memory=clean_json(devices[0].memory_stats() or {})))
        write()

    try:
        _, params, metrics = train.train(
            environment=environment, num_timesteps=0, continuous=True,
            training_steps_per_epoch=1, network_factory=factory,
            wrap_env_fn=wrap_for_cat_wholebody_training,
            restore_params=None, restore_value_fn=True,
            restore_runtime_state=runtime, runtime_metadata=identity,
            sapg_config=config["sapg"], runtime_checkpoint_fn=boundary,
            save_checkpoint_path=None, log_training_metrics=True,
            training_metrics_buffer_size=1000, training_metrics_steps=transitions,
            progress_fn=progress, should_stop_fn=lambda: len(boundaries) >= args.updates,
            **options)
        jax.effects_barrier()
        expected = [args.expected_start_step + transitions * (index + 1) for index in range(args.updates)]
        if [item["step"] for item in boundaries] != expected:
            raise AssertionError("Replay did not complete precisely the requested updates")
        report.update(status="completed", execution_completed=True,
                      final_step=int(metrics["training/completed_steps"]),
                      final_params=host_finiteness(params))
    finally:
        restore()
        events.sort(key=lambda item: (item.get("adam_update_index", -1), item["kind"]))


def main():
    args = parser().parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run this GPU replay through Slurm")
    args.source_root = args.source_root.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.runtime = (args.runtime or args.run_dir / "resume.msgpack").expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    if args.report == args.runtime or args.report.is_relative_to(args.run_dir):
        raise SystemExit("Report must be outside the original run directory")
    if args.report.exists():
        raise SystemExit("Refusing to overwrite an existing diagnostic report")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root))
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from cat_ppo.furniture.generalist_logging import atomic_json
    started = time.monotonic()
    report = dict(schema="cat-sapg-runtime-replay-v1", status="starting",
                  execution_completed=False, numerical_pass=False,
                  source_root=str(args.source_root), run_dir=str(args.run_dir), runtime=str(args.runtime),
                  runtime_sha256=sha256(args.runtime), requested_updates=args.updates,
                  expected_start_step=args.expected_start_step, diagnostics=args.diagnostics,
                  diagnostic_equivalence="same restored state and formulas; diagnostics may change compiler fusion",
                  slurm_job_id=os.environ["SLURM_JOB_ID"], pid=os.getpid(),
                  wandb_enabled=False, evaluation_enabled=False, runtime_writes=False,
                  updates=[], boundaries=[], diagnostic_events=[],
                  environment={name: os.environ.get(name) for name in (
                      "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")})

    def write():
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.report, clean_json(report))

    write()
    try:
        replay(args, report, write)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        write()
        raise
    finally:
        report["runtime_sha256_after"] = sha256(args.runtime)
        report["runtime_unchanged"] = report["runtime_sha256_after"] == report["runtime_sha256"]
        report.update(numerical_assessment(report))
        write()
    print(json.dumps(clean_json({key: value for key, value in report.items()
                                if key not in ("configuration", "diagnostic_events")})), flush=True)
    if not report.get("runtime_unchanged"):
        raise SystemExit("Original runtime changed during replay")
    if not report["numerical_pass"]:
        raise SystemExit("Replay completed but numerical validation failed; see report")


if __name__ == "__main__":
    main()
