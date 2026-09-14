"""Fine-tune native CAT for furniture traversal with explicit compute budgets."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-dir", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--steps", required=True, type=int)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unroll-length", type=int, default=16)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--updates-per-batch", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--num-evals", type=int, default=0)
    p.add_argument("--num-eval-envs", type=int, default=8)
    p.add_argument("--validation-scene-dir", type=Path)
    p.add_argument("--checkpoint-epochs", type=int, default=1,
                   help="Training chunks/checkpoint candidates when evaluation is disabled")
    p.add_argument("--max-fall-rate", type=float, default=0.0)
    p.add_argument("--max-contact-rate", type=float, default=0.0)
    p.add_argument("--action-dofs", type=int, choices=(12, 23, 29), default=29)
    p.add_argument("--env-config-json", "--config-json", type=Path, help="Explicit environment configuration overrides")
    p.add_argument("--warmstart-checkpoint", "--warmstart", type=Path)
    p.add_argument("--native-manifest", type=Path)
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--new-action-std", type=float, default=.05)
    p.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    p.add_argument("--wandb-project", default="CAT-furniture")
    p.add_argument("--export-onnx", action=argparse.BooleanOptionalAction, default=True)
    return p


def _validate(args):
    if min(args.steps, args.num_envs, args.unroll_length, args.num_minibatches, args.updates_per_batch, args.checkpoint_epochs) < 1:
        raise ValueError("Compute budgets and counts must be positive")
    if args.num_envs % args.num_minibatches:
        raise ValueError("num-envs must be divisible by num-minibatches")
    epochs = max(args.num_evals - 1, 1) if args.num_evals else args.checkpoint_epochs
    quantum = args.num_envs * args.unroll_length * epochs
    if args.steps % quantum:
        raise ValueError(f"steps must be divisible by {quantum}; refusing silent budget rounding")
    if args.num_evals < 0 or (args.num_evals and args.num_eval_envs < 1):
        raise ValueError("Invalid evaluation counts")
    if args.num_evals and args.validation_scene_dir is None:
        raise ValueError("Evaluation requires an explicit designated validation scene")
    if args.num_evals and args.validation_scene_dir.resolve() == args.scene_dir.resolve():
        raise ValueError("Validation scene must differ from the training scene")
    if args.from_scratch and args.warmstart_checkpoint is not None:
        raise ValueError("Choose native warm-start or from-scratch, not both")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    for rate in (args.max_fall_rate, args.max_contact_rate):
        if not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("Safety selection limits must be rates in [0,1]")


def _update_config(config, overrides):
    for key, value in overrides.items():
        if key not in config:
            raise ValueError(f"Unknown environment configuration field: {key}")
        if isinstance(value, dict):
            _update_config(config[key], value)
        else:
            config[key] = value


def _scene_provenance(path, expected_split):
    from cat_ppo.furniture.scenes import load_scene
    path = Path(path).resolve()
    scene = load_scene(path)
    if scene["split"] != expected_split:
        raise ValueError(f"Expected a {expected_split} scene, got {scene['split']!r}; sealed tests cannot select checkpoints")
    def digest_file(file):
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    files = {"scene.json": digest_file(path / "scene.json")}
    for field in scene["fields"].values():
        field_path = path / field["file"]
        digest = digest_file(field_path)
        if digest != field["sha256"]:
            raise ValueError(f"Scene field hash mismatch: {field_path}")
        files[field["file"]] = digest
    return {"path": str(path), "scene_id": scene["scene_id"], "split": scene["split"],
            "geometry_hash": scene["geometry_hash"], "files": files}


def _code_provenance():
    root = Path(__file__).resolve().parent
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True).stdout.decode().strip()
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root, check=True, capture_output=True).stdout
    paths = sorted((root / "cat_ppo").rglob("*.py")) + [Path(__file__).resolve()]
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return {"git_commit": commit, "tracked_dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "source_files": sources, "source_snapshot_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()}


def _wilson_interval(rate, count):
    z = 1.959963984540054
    denominator = 1 + z*z/count
    center = (rate + z*z/(2*count)) / denominator
    radius = z * math.sqrt(rate*(1-rate)/count + z*z/(4*count*count)) / denominator
    return max(0.0, center-radius), min(1.0, center+radius)


def train(args):
    _validate(args)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import jax
    # Match exported float32 inference rather than backend-dependent reduced
    # precision GEMM. This setting is explicit and recorded in every new run.
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp
    import numpy as np
    from brax.training.acme import running_statistics, specs
    from brax.training.agents.ppo import networks
    from cat_ppo.envs.g1.env_furniture import G1FurnitureEnv, default_config
    from cat_ppo.furniture.control import legacy_observation_contract
    from cat_ppo.furniture.learning import (DEFAULT_MANIFEST, adapt_native_params,
        fetch_native_checkpoint, load_native, mlp_hidden_sizes, verify_warmstart_parity)
    from cat_ppo.furniture.checkpoint import BestCheckpointStore, native_writer
    from cat_ppo.furniture.training import wrap_for_furniture_training
    from cat_ppo.learning.policy.ppo import train as native_ppo

    args.run_dir = args.run_dir.absolute()
    if args.run_dir.is_symlink() or (args.run_dir.exists() and any(args.run_dir.iterdir())):
        raise ValueError("run-dir must be new or empty; existing runs are never overwritten")
    training_scene = _scene_provenance(args.scene_dir, "train")
    validation_scene = _scene_provenance(args.validation_scene_dir, "validation") if args.num_evals else None
    if validation_scene and training_scene["geometry_hash"] == validation_scene["geometry_hash"]:
        raise ValueError("Training and validation geometry hashes must differ")
    code_provenance = _code_provenance()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = default_config()
    config.action_dofs = args.action_dofs
    if args.env_config_json:
        _update_config(config, json.loads(args.env_config_json.read_text()))
    if config.action_dofs != args.action_dofs:
        raise ValueError("action_dofs override conflicts with --action-dofs")
    environment = G1FurnitureEnv(args.scene_dir, config=config)
    contract = environment.observation_contract()
    contract["action_semantics"] = {
        "legs": "increment previous target by action_scale*action",
        "upper_body": "nominal posture offsets, bounded and slew limited",
        "leg_action_scale": float(config.action_scale),
        "upper_action_scale": float(config.upper_action_scale),
        "upper_target_rate_rad_s": float(config.upper_target_rate),
        "control_dt": float(config.ctrl_dt), "joint_command_units": "radians",
        "upper_body_mode": config.get("upper_body_mode", "learned"),
    }
    source, source_config = None, None
    source_contract = legacy_observation_contract()
    source_provenance = {"kind": "from_scratch"}
    if not args.from_scratch:
        source_path = args.warmstart_checkpoint
        if source_path is None:
            manifest_path = args.native_manifest or DEFAULT_MANIFEST
            manifest_digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
            source_path, manifest = fetch_native_checkpoint(
                Path(__file__).parent / "data/furniture/native_generalist_v1", manifest_path)
            source_provenance = {"kind": "pinned_public_native_checkpoint", "manifest_sha256": manifest_digest,
                                 "repo_id": manifest["repo_id"], "revision": manifest["revision"]}
        else:
            source_provenance = {"kind": "explicit_local_native_checkpoint", "path": str(source_path.resolve())}
        if (source_path / "native").is_dir():
            source_path = source_path / "native"
        stored_contract = source_path / "observation_contract.json"
        if stored_contract.is_file():
            source_contract = json.loads(stored_contract.read_text())
        source_provenance["files"] = {str(path.relative_to(source_path)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_path.rglob("*")) if path.is_file()}
        source = load_native(source_path)
        source_config = json.loads((source_path / "ppo_network_config.json").read_text())
        source_provenance["network_config"] = source_config
    policy_sizes = mlp_hidden_sizes(source[1]) if source else (256, 128, 64)
    value_sizes = mlp_hidden_sizes(source[2]) if source else (512, 256, 128)
    normalize = bool(source_config["normalize_observations"]) if source_config else False
    if source_config:
        kwargs = source_config.get("network_factory_kwargs", {})
        if kwargs.get("policy_obs_key", "state") != "state" or kwargs.get("value_obs_key", "privileged_state") != "privileged_state":
            raise ValueError("Released warm-start expects actor state and critic privileged_state")
    factory = functools.partial(networks.make_ppo_networks, policy_hidden_layer_sizes=policy_sizes,
        value_hidden_layer_sizes=value_sizes, policy_obs_key="state", value_obs_key="privileged_state")
    sizes = {"state": (len(contract["actor_features"]),), "privileged_state": (len(contract["critic_features"]),)}
    network = factory(sizes, environment.action_size,
                      preprocess_observations_fn=running_statistics.normalize if normalize else lambda x, _: x)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    target = (running_statistics.init_state({key: specs.Array(size, jnp.dtype("float32")) for key, size in sizes.items()}),
              network.policy_network.init(keys[0]), network.value_network.init(keys[1]))
    warmstart = {"kind": "from_scratch", "optimizer": "fresh", "critic_restored": False}
    if source:
        target, warmstart = adapt_native_params(source, target, source_contract, contract,
                                                new_action_std=args.new_action_std)
        warmstart["parity"] = verify_warmstart_parity(source, target, source_contract, contract,
                                                     normalize_observations=normalize, seed=args.seed)
    target = jax.tree.map(jnp.asarray, target)
    environment_sample = environment.reset(jax.random.PRNGKey(args.seed))
    for key, shape in sizes.items():
        if environment_sample.obs[key].shape != shape:
            raise ValueError(f"Actual environment observation disagrees with contract: {key}")
    validation_env = G1FurnitureEnv(args.validation_scene_dir, config=config) if args.num_evals else None
    store = BestCheckpointStore(args.run_dir, max_fall_rate=args.max_fall_rate, max_contact_rate=args.max_contact_rate)
    record = {"schema_version": 1, "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment_config": config.to_dict(), "observation_contract": contract, "warmstart": warmstart,
        "source": source_provenance, "normalization_enabled": normalize,
        "code": code_provenance, "training_scene": training_scene, "validation_scene": validation_scene,
        "wandb_mode": args.wandb_mode, "selection_source": "validation" if args.num_evals else "training_proxy",
        "training_telemetry": {"enabled": True, "episode_buffer_size": 100,
            "logging_interval_transitions": max(args.num_envs * args.unroll_length, min(4096, args.steps)),
            "outcomes_and_minimum_clearance": "completed-episode snapshots",
            "cross_track_map_age_unknown_fraction": "within-episode means",
            "selection": "unchanged; training telemetry does not run validation or select by success"},
        "python": sys.version, "jax_version": jax.__version__,
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        "devices": [str(device) for device in jax.devices()]}
    (args.run_dir / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, mode=args.wandb_mode,
                              name=args.run_dir.name, dir=str(args.run_dir), config=record)

    def progress(step, metrics):
        scalars = {key: float(np.asarray(value)) for key, value in metrics.items() if np.asarray(value).ndim == 0}
        finite = {key: value for key, value in scalars.items() if math.isfinite(value)}
        with (args.run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": int(step), "metrics": finite}) + "\n")
        if wandb_run:
            wandb_run.log(finite, step=int(step))

    def scored(step, make_policy, params, network_config, metrics, score_source):
        del make_policy
        if score_source == "training_proxy":
            selected_metrics = {"proxy_score": float(metrics["training/rollout_reward_mean"])}
        else:
            success = float(metrics["eval/episode_success"])
            selected_metrics = {"strict_success_rate": success,
                "fall_rate": float(metrics["eval/episode_fall"]),
                "contact_rate": float(metrics["eval/episode_forbidden_contact"]),
                "completion_time": float(metrics["eval/episode_completion_time"])}
            selected_metrics["validation_episodes"] = args.num_eval_envs
            for rate in ("strict_success_rate", "fall_rate", "contact_rate"):
                lo, hi = _wilson_interval(selected_metrics[rate], args.num_eval_envs)
                selected_metrics[f"{rate}_wilson95_lower"] = lo
                selected_metrics[f"{rate}_wilson95_upper"] = hi
        progress(step, {**metrics, **{f"selection/{key}": value for key, value in selected_metrics.items()}})
        store.consider(step=int(step), metrics=selected_metrics, source=score_source,
                       write_checkpoint=native_writer(params, network_config, int(step)), contract=contract,
                       provenance={"warmstart": warmstart, "source": source_provenance,
                                   "code": code_provenance, "training_scene": training_scene,
                                   "validation_scene": validation_scene})

    try:
        _, final_params, _ = native_ppo.train(environment=environment, num_timesteps=args.steps,
            num_envs=args.num_envs, episode_length=environment.episode_length, action_repeat=1,
            randomize_initial_episode_steps=False, wrap_env_fn=wrap_for_furniture_training,
            batch_size=args.num_envs // args.num_minibatches, num_minibatches=args.num_minibatches,
            unroll_length=args.unroll_length, num_updates_per_batch=args.updates_per_batch,
            learning_rate=args.learning_rate, entropy_cost=.01, discounting=.97, gae_lambda=.95,
            clipping_epsilon=.2, max_grad_norm=1.0, normalize_observations=normalize,
            network_factory=factory, seed=args.seed, num_evals=args.num_evals,
            num_eval_envs=args.num_eval_envs, eval_env=validation_env, deterministic_eval=True,
            eval_episode_length=validation_env.episode_length if validation_env is not None else None,
            log_training_metrics=True, training_metrics_buffer_size=100,
            training_metrics_steps=record["training_telemetry"]["logging_interval_transitions"],
            progress_fn=progress, restore_params=target,
            restore_value_fn=True, save_checkpoint_path=None, scored_checkpoint_fn=scored,
            num_training_epochs=args.checkpoint_epochs if args.num_evals == 0 else None)
        selected = store.selected(verify=True)
        if selected is None:
            raise RuntimeError("Training completed without a finite scored checkpoint")
        actor_delta = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                          for a, b in zip(jax.tree.leaves(target[1]), jax.tree.leaves(final_params[1])))
        summary = {"requested_steps": args.steps, "selected_step": selected["step"],
                   "selection_source": selected["selection_source"], "selected_score": selected["score"],
                   "actor_max_abs_parameter_update": actor_delta, "native_checkpoint": selected["path"]}
        if args.export_onnx:
            from cat_ppo.furniture.export import export_selected
            summary["export"] = export_selected(args.run_dir)
        (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        display = {**summary}
        if "export" in display:
            display["export"] = {key: value for key, value in display["export"].items() if key != "contract"}
        print(json.dumps(display, indent=2))
        return summary
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    train(parser().parse_args())
