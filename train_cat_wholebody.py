"""Fine-tune the released CAT generalist in one persistent mixed-scene learner."""
from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

from cat_ppo.furniture.generalist_config import ROOT, ppo_kwargs, training_config
from cat_ppo.furniture.generalist_logging import GeneralistLogger, atomic_json


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("plan", "validate", "run", "stop"))
    result.add_argument("--bank-manifest", type=Path, default=ROOT / "data/furniture/cat_diversity_v2_20260916/manifest.json")
    result.add_argument("--run-dir", type=Path, default=ROOT / "outputs/cat_wholebody_diversity_v2")
    result.add_argument("--profile", choices=("single_gpu_32gb", "released"), default="single_gpu_32gb")
    result.add_argument("--finetuning", choices=("stabilized", "gentle", "released", "hand_protection"), default="stabilized",
                        help="Stabilized: gentle PPO, bounded upper exploration, physical target penalties and retention validation")
    result.add_argument("--num-envs", type=int, help="Simulator parallelism; leaves PPO batch geometry unchanged")
    result.add_argument("--batch-size", type=int, help="Explicit trajectories/minibatch resource override")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--warmstart-best", type=Path,
                        help="Verified permanent selected-best archive; load weights with a fresh optimizer")
    result.add_argument("--body-collision-bank", type=Path,
                        help="Enable approved full-body primitive collision checks using this immutable bank")
    result.add_argument("--body-collision-resets", type=Path,
                        help="Validated clear reset fallback manifest for the same collision bank")
    result.add_argument("--resume", action="store_true", help="Restore the complete learner and same online W&B run")
    result.add_argument("--wandb-mode", choices=("online", "disabled"), default="online")
    result.add_argument("--wandb-project", default="CAT-wholebody")
    result.add_argument("--wandb-entity", default="skvayzer")
    return result


def plan(args):
    from cat_ppo.furniture.control import wholebody_observation_contract
    contract = wholebody_observation_contract()
    config = training_config(profile=args.profile, num_envs=args.num_envs,
                             batch_size=args.batch_size, seed=args.seed, finetuning=args.finetuning)
    warmstart_best = getattr(args, "warmstart_best", None)
    warmstart_provenance = None
    if warmstart_best:
        if args.finetuning != "hand_protection":
            raise ValueError("Selected-best warm start is only supported by the hand_protection profile")
        from cat_ppo.furniture.protected_warmstart import verified_best_archive
        _, _, warmstart_provenance = verified_best_archive(
            warmstart_best, target_contract=contract,
            network_config=config["policy_config"]["network_factory"])
        config["fine_tuning"]["initialization"] = "verified selected best actor and critic; fresh Adam"
        config["fine_tuning"]["reference_kl"]["reference"] = (
            "frozen selected-best actor on unchanged CAT observations; leg actions only")
    collision_bank = getattr(args, "body_collision_bank", None)
    collision_resets = getattr(args, "body_collision_resets", None)
    if bool(collision_bank) != bool(collision_resets):
        raise ValueError("Body collision training requires both geometry bank and validated reset manifests")
    result = {
        "config": config,
        "bank_manifest": str(args.bank_manifest.resolve()),
        "task": "G1CatWholeBodyEnv: released CAT + 29 body actions, one sphere/hand and one site/elbow",
        "observations": {"schema": contract["schema"], "actor": len(contract["actor_features"]),
                         "critic": len(contract["critic_features"]), "field_sample_count": 13},
        "physics": "CAT flat floor/feet contact model; field-only obstacles",
        "termination": "CAT body rule with hand surface clearances; elbow spheres use the same zero threshold and 50-step grace",
        "checkpoint_compatibility": "new compact run from released CAT; 406-input v1 runtime cannot resume into this schema",
        "logging": "one persisted W&B ID with aggregate episode metrics and scene diagnostics",
        "stop": str(args.run_dir.resolve() / "STOP"),
        "checkpoint_policy": "one selected best model and one atomically overwritten full resume.msgpack",
    }
    if collision_bank:
        result.update(
            body_collision_bank=str(collision_bank.resolve()),
            body_collision_resets=str(collision_resets.resolve()),
            physics="CAT floor/self contacts; 35 primitive-volume obstacle checks at 500 Hz without obstacle impulses",
            termination="Native CAT causes plus full-body obstacle collision without grace; -1 terminal event reward after clipping",
            checkpoint_compatibility="New collision/route contract, original released CAT parameters and fresh optimizer")
    if warmstart_provenance:
        result.update(warmstart_best=warmstart_provenance,
                      checkpoint_compatibility="Same 222/310 feature layout and 29 actions; selected-best weights, fresh optimizer")
    return result


def code_identity():
    paths = sorted((ROOT / "cat_ppo").rglob("*.py")) + [Path(__file__).resolve()]
    files = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }


def field_bank_summary(manifest, manifest_sha256):
    from collections import Counter
    scenes = [dict(scene_id=scene["scene_id"], family=scene["family"],
                   source_kind=scene["source"].get("kind", "released-original" if scene["family"] == "original_cat" else "generated-clutter"),
                   arrays_unchanged=bool(scene["source"].get("arrays_unchanged", False)))
              for scene in manifest["scenes"]]
    return dict(scene_count=len(scenes),
                schema=manifest.get("schema"),
                family_counts=dict(Counter(scene["family"] for scene in scenes)),
                fields_bytes=manifest.get("fields_bytes"),
                sampling_group_masses=manifest.get("sampling_group_masses"),
                sampling_mass_meaning="probability at episode reset, not fraction of training transitions",
                byte_verified_original_count=sum(scene["family"] == "original_cat" and scene["arrays_unchanged"] for scene in scenes),
                reconstructed_original_count=sum(scene["source_kind"] == "reconstructed-missing-original" for scene in scenes),
                manifest_sha256=manifest_sha256, scenes=scenes)


def reference_kl_config(environment, configuration):
    settings = configuration["fine_tuning"]["reference_kl"]
    if settings is None:
        return None
    scenes = environment.field_bank_manifest["scenes"]
    return {"coefficient": settings["coefficient"], "action_indices": settings["action_indices"],
            "scene_mask": [scene.get("task_kind", "cat" if scene["family"] == "original_cat" else "room") == "cat"
                           for scene in scenes]}


def prepare(args, specification, *, restore_model=True):
    # All accelerator imports follow resource environment configuration.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    from brax.training.acme import running_statistics, specs
    from brax.training.agents.ppo import networks
    from ml_collections import ConfigDict
    from cat_ppo.envs.g1.env_cat_wholebody import G1CatWholeBodyEnv, wholebody_config
    from cat_ppo.furniture.control import legacy_observation_contract
    from cat_ppo.furniture.learning import (adapt_native_params, fetch_native_checkpoint,
                                          load_native, verify_warmstart_parity)

    config = specification["config"]
    env_config = wholebody_config(ConfigDict(config["env_config"]), bank_manifest=args.bank_manifest.resolve(),
                                 stabilization=config["fine_tuning"].get("upper_stabilization", False),
                                 hand_protection=config["fine_tuning"].get("hand_protection", False))
    if specification.get("body_collision_bank"):
        env_config.wholebody.body_collision.update(dict(
            enabled=True, bank_manifest=specification["body_collision_bank"],
            reset_manifest=specification["body_collision_resets"]))
    environment = G1CatWholeBodyEnv(config=env_config)
    if (config["fine_tuning"].get("hand_protection")
            and not environment.field_bank_manifest.get("hand_protection_curriculum")):
        raise ValueError("Hand-protection profile requires a certified hand-curriculum scene bank")
    contract = environment.observation_contract()
    net_config = config["policy_config"]["network_factory"]
    distribution_settings = config["fine_tuning"].get("action_distribution")
    network_builder = networks.make_ppo_networks
    if distribution_settings is not None:
        from cat_ppo.learning.policy.ppo.wholebody_distribution import make_ppo_networks
        network_builder = make_ppo_networks
    factory = functools.partial(network_builder,
        policy_hidden_layer_sizes=tuple(net_config["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=tuple(net_config["value_hidden_layer_sizes"]),
        policy_obs_key="state", value_obs_key="privileged_state", **(distribution_settings or {}))
    shapes = {"state": (len(contract["actor_features"]),),
              "privileged_state": (len(contract["critic_features"]),)}
    actual = jax.eval_shape(environment.reset, jax.random.PRNGKey(args.seed))
    for key, shape in shapes.items():
        if actual.obs[key].shape != shape:
            raise ValueError(f"Observation contract mismatch: {key}")
    warmstart, target = None, None
    if restore_model:
        source_contract = legacy_observation_contract()
        if specification.get("warmstart_best"):
            from cat_ppo.furniture.protected_warmstart import verified_best_archive
            source_path, source_contract, provenance = verified_best_archive(
                specification["warmstart_best"]["archive"], target_contract=contract,
                network_config=net_config)
            if provenance != specification["warmstart_best"]:
                raise ValueError("Protected warm-start archive changed after planning")
        else:
            source_path, manifest = fetch_native_checkpoint(ROOT / "data/furniture/native_generalist_v1")
            provenance = {"model_revision": manifest["revision"]}
        source = load_native(source_path)
        network = factory(shapes, environment.action_size)
        keys = jax.random.split(jax.random.PRNGKey(args.seed), 2)
        target = (running_statistics.init_state({k: specs.Array(v, jnp.dtype("float32")) for k, v in shapes.items()}),
                  network.policy_network.init(keys[0]), network.value_network.init(keys[1]))
        target, warmstart = adapt_native_params(source, target, source_contract, contract, new_action_std=.05)
        warmstart["parity"] = verify_warmstart_parity(source, target, source_contract, contract,
                                                     normalize_observations=False, seed=args.seed)
        warmstart.update(provenance)
        warmstart["parity_scope"] = "raw actor MLP and critic; arm conditional distribution intentionally changes"
        target = jax.tree.map(jnp.asarray, target)
    record = dict(specification, environment_config=env_config.to_dict(), observation_contract=contract,
                  warmstart=warmstart, code=code_identity(),
                  bank_sha256=hashlib.sha256(args.bank_manifest.read_bytes()).hexdigest(),
                  devices=[str(device) for device in jax.devices()], jax_version=jax.__version__)
    record["field_bank"] = field_bank_summary(environment.field_bank_manifest, record["bank_sha256"])
    return environment, factory, target, record


def _read_metadata(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular metadata: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected metadata object: {path}")
    return value


def _assert_zero_progress(directory, status):
    """Permit an explicit startup retry only with positive evidence of no work."""
    for name in ("completed_steps", "resume_state_steps", "observed_steps"):
        value = status.get(name, 0)
        if type(value) is not int or value != 0:
            raise ValueError("Missing runtime snapshot after training progress; cold restart refused")
    if status.get("initial_runtime_written", False) or status.get("runtime_present_on_entry", False):
        raise ValueError("A previously written runtime snapshot is missing; cold restart refused")
    if status.get("phase") not in ("creating", "preparing", "checkpoint_store", "logger", "learner_initialization"):
        raise ValueError("Cannot prove a zero-progress startup failure")
    for marker in (directory / "checkpoints" / "best", directory / "resume.msgpack"):
        if marker.exists() or marker.is_symlink():
            raise ValueError("Runtime/best checkpoint evidence prevents a cold startup retry")
    generations = directory / ".checkpoint-generations"
    if generations.is_symlink() or (generations.exists() and not generations.is_dir()):
        raise ValueError("Unexpected checkpoint generation storage")
    if generations.exists() and any(path.name != "owner.json" for path in generations.iterdir()):
        raise ValueError("Checkpoint candidate evidence prevents a cold startup retry")
    checkpoints = directory / "checkpoints"
    if checkpoints.is_symlink() or (checkpoints.exists() and not checkpoints.is_dir()):
        raise ValueError("Unexpected checkpoint storage")
    if checkpoints.exists() and any(path.name != ".selection.lock" for path in checkpoints.iterdir()):
        raise ValueError("Checkpoint artifacts prevent a cold startup retry")
    metrics = directory / "metrics.jsonl"
    if metrics.is_symlink() or (metrics.exists() and (not metrics.is_file() or metrics.stat().st_size)):
        raise ValueError("Observed training metrics prevent a cold startup retry")
    identity = directory / "wandb.json"
    if identity.exists() or identity.is_symlink():
        if _read_metadata(identity).get("last_global_step") != -1:
            raise ValueError("W&B progress prevents a cold startup retry")
    if any(path.name.startswith(".resume.msgpack") for path in directory.iterdir()):
        raise ValueError("Incomplete runtime snapshot requires inspection; cold restart refused")


def _recover_startup_store(directory, attempt):
    """Preserve a partial owner marker from failed store creation, with no models."""
    generations = directory / ".checkpoint-generations"
    if not generations.exists():
        return
    try:
        owner = _read_metadata(generations / "owner.json").get("owner")
        if isinstance(owner, str) and owner:
            return
    except (ValueError, OSError):
        pass
    # The caller already checked that no generation/candidate/best exists.
    archive = directory / "startup-artifacts"
    if archive.is_symlink():
        raise ValueError("Startup archive cannot be a symlink")
    archive.mkdir(exist_ok=True)
    target = archive / f"checkpoint-owner-before-attempt-{attempt:04d}"
    if target.exists() or target.is_symlink():
        raise ValueError("Startup checkpoint-owner archive already exists")
    generations.rename(target)


def run(args, specification):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    directory = args.run_dir.resolve()
    if args.run_dir.is_symlink():
        raise ValueError("run-dir cannot be a symlink")
    if (directory / "STOP").exists():
        raise ValueError("STOP is present; remove it deliberately before launching this run")
    if not args.resume and directory.exists() and any(directory.iterdir()):
        raise ValueError("Existing run directory; use --resume for exact recovery")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".learner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch_path, status_path = directory / "launch.json", directory / "status.json"
        spec_hash = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
        previous_status, previous = {}, None
        destination = dict(project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode)
        launch_missing = not (launch_path.exists() or launch_path.is_symlink())
        if args.resume:
            # Validate existing evidence before writing any replacement status.
            if launch_missing:
                # A failed first metadata write can still leave a failed status.
                # No prepared run/identity is allowed in this narrow recovery.
                previous_status = _read_metadata(status_path)
                if (previous_status.get("schema") != "cat-generalist-status-v1" or
                        previous_status.get("specification_sha256") != spec_hash or
                        previous_status.get("phase") != "creating" or
                        (directory / "run.json").exists() or (directory / "wandb.json").exists()):
                    raise ValueError("Missing launch record cannot be recovered safely")
                launch = dict(schema="cat-generalist-launch-v1", owner=previous_status["owner"],
                              specification_sha256=spec_hash, specification=specification, wandb=destination)
            else:
                launch = _read_metadata(launch_path)
            if launch.get("schema") != "cat-generalist-launch-v1" or launch["specification_sha256"] != spec_hash:
                raise ValueError("Resume would change the original launch specification")
            if status_path.exists() or status_path.is_symlink():
                previous_status = _read_metadata(status_path)
                if previous_status.get("owner") != launch["owner"]:
                    raise ValueError("Startup status ownership differs")
            else:
                previous_status = dict(phase="creating")
            if (directory / "run.json").exists() or (directory / "run.json").is_symlink():
                previous = _read_metadata(directory / "run.json")
        else:
            launch = dict(schema="cat-generalist-launch-v1", owner=str(uuid.uuid4()),
                          specification_sha256=spec_hash, specification=specification,
                          wandb=destination)
        if launch["wandb"] != destination:
            raise ValueError("Resume would change the original logging destination")
        runtime_path = directory / "resume.msgpack"
        has_runtime = runtime_path.exists() or runtime_path.is_symlink()
        if args.resume and not has_runtime:
            _assert_zero_progress(directory, previous_status)
        if args.resume and has_runtime and (previous is None or not (directory / "wandb.json").is_file()):
            raise ValueError("Exact runtime resume requires its original run and W&B identity")
        status = dict(previous_status, schema="cat-generalist-status-v1", owner=launch["owner"],
                      specification_sha256=spec_hash,
                      status="starting", phase="creating", pid=os.getpid(), resume=args.resume,
                      startup_retry=bool(args.resume and not has_runtime),
                      runtime_present_on_entry=has_runtime or previous_status.get("runtime_present_on_entry", False),
                      attempt=int(previous_status.get("attempt", 0)) + 1)
        if previous_status.get("status") == "failed":
            status["last_failure"] = {key: previous_status.get(key) for key in ("attempt", "phase", "error_type", "error")}
        status.pop("error", None)
        status.pop("error_type", None)
        logger = None
        try:
            if launch_missing:
                atomic_json(launch_path, launch)
            atomic_json(status_path, status)
            GeneralistLogger.reserve_identity(directory, **destination, resume=args.resume)
            from cat_ppo.furniture.checkpoint import BestCheckpointStore, native_writer
            from cat_ppo.furniture.generalist_runtime import atomic_save_runtime, load_runtime
            from cat_ppo.furniture.run_control import StopRequest
            from cat_ppo.learning.policy.ppo import train as native_ppo
            from cat_ppo.furniture.generalist_training import wrap_for_cat_wholebody_training

            # A present but corrupt snapshot always fails; it never becomes a
            # fresh initialization, even if no metrics happened to be logged.
            runtime = load_runtime(runtime_path) if has_runtime else None
            status["phase"] = "preparing"
            atomic_json(status_path, status)
            environment, factory, target, record = prepare(args, specification, restore_model=runtime is None)
            identity = {"config": specification["config"], "bank_sha256": record["bank_sha256"],
                        "source_sha256": record["code"]["source_sha256"],
                        "contract": record["observation_contract"]}
            if previous is not None:
                if (previous["bank_sha256"] != record["bank_sha256"] or previous["config"] != record["config"]
                        or previous["code"]["source_sha256"] != record["code"]["source_sha256"]
                        or previous["observation_contract"] != record["observation_contract"]):
                    raise ValueError("Resume would change scene bank, training configuration or source")
                record = previous  # Keep the original warm-start provenance.
            else:
                atomic_json(directory / "run.json", record)
            status["phase"] = "checkpoint_store"
            if status["startup_retry"]:
                _recover_startup_store(directory, status["attempt"])
            store = BestCheckpointStore(directory)
            status["phase"] = "logger"
            logger = GeneralistLogger(directory, **destination, resume=True, config=record)
            status.update(status="running", phase="learner_initialization")
            atomic_json(status_path, status)

            validation_settings = specification["config"]["fine_tuning"].get("retention_validation")
            validator, baseline = None, None
            source_selection = None
            if specification.get("warmstart_best"):
                source_selection = _read_metadata(Path(specification["warmstart_best"]["archive"])
                                                  / "checkpoint/selection.json")

            def source_guard(result):
                if source_selection is None:
                    return None
                from cat_ppo.furniture.hand_retention_guard import source_best_retention_guard
                return source_best_retention_guard(result, source_selection,
                    source_archive=specification["warmstart_best"]["archive"])

            if validation_settings is not None:
                from cat_ppo.furniture.retention_validation import RetentionValidator, validation_scene_ids
                validator = RetentionValidator(environment, factory,
                    seeds=range(validation_settings["seeds_per_scene"]),
                    scene_ids=validation_scene_ids(environment.field_bank_manifest,
                        hand_protection=specification["config"]["fine_tuning"].get("hand_protection", False)))
                baseline_path = directory / "validation_baseline.json"
                if baseline_path.exists():
                    saved = _read_metadata(baseline_path)
                    if saved["identity"] != identity:
                        raise ValueError("Validation baseline belongs to a different training setup")
                    baseline = saved["result"]
                else:
                    if runtime is not None or target is None:
                        raise ValueError("Original-checkpoint validation baseline is missing")
                    print("Evaluating initial policy baseline on fixed scenes (both action modes)", flush=True)
                    baseline_result = validator.evaluate(target, step=0)
                    baseline = baseline_result.as_dict()
                    atomic_json(baseline_path, dict(identity=identity, result=baseline))
                    print("Initial policy validation baseline saved; starting continuous PPO", flush=True)
                guard = source_guard(baseline)
                if guard is not None:
                    atomic_json(directory / "source_best_startup_guard.json", guard)
                    print(f"Protected source-best startup comparison: {guard['comparisons']}", flush=True)
                    if not guard["eligible"]:
                        raise ValueError("Changed hand profile failed protected source-best retention: "
                                         + "; ".join(guard["reasons"]))
            transitions_per_update = specification["config"]["fine_tuning"]["effective_batch_geometry"]["transitions_per_update"]

            def progress(step, metrics):
                if int(step) > status.get("observed_steps", 0):
                    status["observed_steps"] = int(step)
                    atomic_json(status_path, status)
                logger.log(step, metrics)

            def scored(step, make_policy, params, network_config, metrics, source):
                del make_policy
                if validator is not None:
                    interval = validation_settings["interval_updates"] * transitions_per_update
                    if int(step) % interval:
                        return
                    print(f"Fixed-scene retention validation at {int(step)} transitions", flush=True)
                    result = validator.evaluate(params, step=int(step), baseline=baseline["modes"])
                    guard = source_guard(result)
                    if guard is not None:
                        result.selection["source_best_guard"] = guard
                        result.selection["eligible"] &= guard["eligible"]
                        result.selection["reasons"].extend(guard["reasons"])
                        result.metrics["validation/source_best_retention_eligible"] = int(guard["eligible"])
                        result.metrics["validation/retention_eligible"] = int(result.selection["eligible"])
                    atomic_json(directory / "validation_latest.json", result.as_dict())
                    selected = store.consider(step=int(step),
                        metrics={"selection": result.selection, "validation": result.metrics},
                        source="retention_validation", write_checkpoint=native_writer(params, network_config, int(step)),
                        contract=record["observation_contract"],
                        provenance={"code": record["code"], "bank_sha256": record["bank_sha256"],
                                    "action_distribution": specification["config"]["fine_tuning"]["action_distribution"],
                                    "validation": result.metadata,
                                    "selection": "CAT retention gates; clutter success and hand protection"})
                    logger.log(step, result.metrics | {"selection/best_updated": int(selected)})
                    print(f"Validation finished: {result.selection}", flush=True)
                    return
                selected = store.consider(step=int(step),
                    metrics={"proxy_score": float(metrics["training/rollout_reward_mean"])},
                    source=source, write_checkpoint=native_writer(params, network_config, int(step)),
                    contract=record["observation_contract"],
                    provenance={"code": record["code"], "bank_sha256": record["bank_sha256"],
                                "selection": "training reward proxy; no held-out evaluation"})
                logger.log(step, {"selection/best_updated": int(selected)})

            def save_runtime(step, snapshot):
                atomic_save_runtime(runtime_path, snapshot)
                status.update(completed_steps=int(step), resume_state_steps=int(step),
                              initial_runtime_written=True, phase="training")
                atomic_json(status_path, status)
                # Keep zero-progress startup retries possible until a durable
                # learner snapshot exists. Baseline evaluation is not training.
                if int(step) == 0 and baseline is not None:
                    logger.log(0, baseline["metrics"] | {
                        "baseline/" + key.removeprefix("validation/"): value
                        for key, value in baseline["metrics"].items()})

            with StopRequest(directory / "STOP") as stop:
                _, _, metrics = native_ppo.train(
                    environment=environment, num_timesteps=0, continuous=True,
                    training_steps_per_epoch=1, **ppo_kwargs(specification["config"]),
                    wrap_env_fn=wrap_for_cat_wholebody_training, network_factory=factory,
                    restore_params=target, restore_value_fn=True,
                    reference_kl_config=reference_kl_config(environment, specification["config"]),
                    restore_runtime_state=runtime, runtime_metadata=identity,
                    runtime_checkpoint_fn=save_runtime, save_checkpoint_path=None,
                    log_training_metrics=True, training_metrics_buffer_size=1000,
                    training_metrics_steps=transitions_per_update,
                    progress_fn=progress, scored_checkpoint_fn=scored, should_stop_fn=stop.requested)
                status.update(status="stopped", completed_steps=int(metrics["training/completed_steps"]),
                              stop_reason=stop.reason)
            atomic_json(status_path, status)
            logger.finish()
        except BaseException as error:
            status.update(status="failed", error_type=type(error).__name__, error=str(error))
            try:
                atomic_json(status_path, status)
            except BaseException as cleanup_error:
                error.add_note(f"Failed-status persistence also failed: {cleanup_error}")
            if logger is not None:
                try:
                    logger.finish(exit_code=1)
                except BaseException as cleanup_error:
                    error.add_note(f"W&B failure cleanup also failed: {cleanup_error}")
            raise
        return status


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "stop":
        if not (args.run_dir / "run.json").is_file():
            raise ValueError("Expected an existing CAT generalist run")
        (args.run_dir / "STOP").touch()
        print("Stop requested at the next completed PPO update; best and resume state retained.")
        return
    specification = plan(args)
    if args.command == "run":
        result = run(args, specification)
    elif args.command == "validate":
        _, _, _, result = prepare(args, specification)
        result["validation"] = "field integrity, reset tracing, observation shapes and native weight mapping; no training"
    else:
        result = specification
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
