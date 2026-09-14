"""Explicit, unwrapped furniture evaluation with strict contact accounting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def _configure_numerics():
    import jax
    # Match the native actor/export parity checks on GPUs whose default dot
    # products can otherwise use reduced-precision TF32 accumulation.
    jax.config.update("jax_default_matmul_precision", "highest")
    return {"jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision)}


def _resolve_checkpoint(path):
    from cat_ppo.furniture.checkpoint import BestCheckpointStore

    path = Path(path).absolute()
    run = None
    if (path / "checkpoints" / "best").is_symlink():
        run = path
    elif path.is_symlink() and path.name == "best" and path.parent.name == "checkpoints":
        run = path.parent.parent
    selection = BestCheckpointStore.open_existing(run).selected(verify=True) if run is not None else None
    native = Path(selection["path"]) if selection is not None else path.resolve()
    if selection is None and (native / "native").is_dir():
        native = native / "native"
    if not (native / "ppo_network_config.json").is_file():
        raise ValueError("checkpoint must include its native network configuration")
    return native, selection


def _apply_environment_overrides(config, overrides):
    # This module-level import is intentionally local; train_furniture's config
    # helper imports no simulation modules and rejects unknown nested keys.
    from train_furniture import _update_config
    _update_config(config, overrides)


def _code_provenance():
    """Bind results to actual local sources, including uncommitted Python files."""
    root = Path(__file__).resolve().parent
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                            check=True).stdout.decode().strip()
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root, capture_output=True,
                          check=True).stdout
    paths = sorted((root / "cat_ppo").rglob("*.py")) + [Path(__file__).resolve()]
    paths.extend(path for path in (root / "pyproject.toml", root / "uv.lock") if path.is_file())
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(paths)}
    snapshot = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"git_commit": commit, "tracked_dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "source_files": sources, "source_snapshot_sha256": snapshot}


def load_controller(checkpoint_path, environment):
    import jax
    import jax.numpy as jnp
    from brax.training.agents.ppo import checkpoint as native_checkpoint
    from cat_ppo.furniture.control import legacy_observation_contract
    from cat_ppo.furniture.learning import index_mapping, feature_names

    path = Path(checkpoint_path).resolve()
    contract_path = path / "observation_contract.json"
    source_contract = json.loads(contract_path.read_text()) if contract_path.exists() else legacy_observation_contract()
    target_contract = environment.observation_contract()
    maps = {key: jnp.asarray(index_mapping(feature_names(source_contract, key), feature_names(target_contract, key)))
            for key in ("state", "privileged_state")}
    actions = jnp.asarray(index_mapping(source_contract["action_names"], target_contract["action_names"]))
    inference = native_checkpoint.load_policy(path, deterministic=True)

    def policy(observation, key):
        source_obs = {name: observation[name][indices] for name, indices in maps.items()}
        value, _ = inference(source_obs, key)
        return jnp.zeros(len(target_contract["action_names"])).at[actions].set(value)

    return jax.jit(policy)


def run_episode(environment, policy, *, scene, case_id, training_seed, episode_seed,
                controller, provenance=None):
    import jax
    from cat_ppo.furniture.benchmark import EpisodeTracker

    key = jax.random.PRNGKey(episode_seed % (2**32))
    state = jax.jit(environment.reset)(key)
    step = jax.jit(environment.step)
    tracker = EpisodeTracker(scene, case_id=case_id, training_seed=training_seed,
        episode_seed=episode_seed, controller=controller,
        goal_tolerance=float(environment._config.goal_radius), provenance=provenance)
    count = int(np.ceil(float(scene["time_budget"]) / environment.dt))
    taxonomy = {0: "none", 1: "other_body", 2: "hand", 3: "arm"}
    for index in range(count):
        key, action_key = jax.random.split(key)
        state = step(state, policy(state.obs, action_key))
        info = jax.device_get(state.info)
        position = np.asarray(jax.device_get(state.data.qpos[:2]))
        finite_position = np.isfinite(position).all()
        numerical_failure = bool(info.get("numerical_failure", False)) or not finite_position
        contact_time = float(info.get("first_contact_time", -1.0))
        tracker.update(position=position if finite_position else tracker.last_position,
            elapsed=(index+1)*environment.dt,
            furniture_contact=bool(info["furniture_contact"]), hand_contact=bool(info["hand_contact"]),
            fall=bool(info["fall"]), self_collision=bool(info["self_collision"]),
            nonfoot_floor_contact=bool(info["nonfoot_floor_contact"]),
            goal_reached=bool(info["goal_reached"]) and not numerical_failure,
            numerical_failure=numerical_failure,
            first_contact_time=contact_time if contact_time >= 0 else None,
            minimum_clearance=float(info["minimum_clearance"]),
            contact_part=taxonomy.get(int(info["first_contact_part"]), "unclassified"))
        if numerical_failure:
            return tracker.finish(termination_reason="numerical_failure")
        if bool(jax.device_get(state.done)):
            record = tracker.finish()
            # Unknown termination is a failure, never an implicit success.
            if not record.strict_success and record.termination_reason == "incomplete":
                record.termination_reason = "environment_termination"
            return record
    return tracker.finish(termination_reason="timeout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Native checkpoint directory, or run directory with checkpoints/best")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-cache", type=Path, default=Path("generated_scenes/benchmark_cache"))
    parser.add_argument("--training-seed", type=int, required=True,
                        help="The seed of this specific trained checkpoint, not a claim of multiple trained policies")
    parser.add_argument("--controller", default="P")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--max-cases", type=int, default=None, help="Explicit pilot subset; recorded in provenance")
    parser.add_argument("--voxel-size", type=float, default=0.10)
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    numerics = _configure_numerics()
    from cat_ppo.envs.g1.env_furniture import G1FurnitureEnv, default_config
    from cat_ppo.furniture.benchmark import summarize
    from cat_ppo.furniture.manifest import load_manifest, materialize_case

    manifest = load_manifest(args.manifest)
    if args.training_seed not in manifest["training_seeds"]:
        raise ValueError("training seed is absent from benchmark manifest")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("max-cases must be positive")
    checkpoint, selection = _resolve_checkpoint(args.checkpoint)
    config = default_config()
    if args.env_config:
        _apply_environment_overrides(config, json.loads(args.env_config.read_text()))
    source_code = _code_provenance()
    hashes = {str(p.relative_to(checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(checkpoint.rglob("*")) if p.is_file()}
    cases = manifest["cases"][:args.max_cases]
    provenance = {"source_commit": source_code["git_commit"], "source_code": source_code,
        "manifest_sha256": manifest["manifest_sha256"],
        "split": manifest["split"], "full_manifest_cases": len(manifest["cases"]),
        "cases_run": len(cases), "pilot_subset": len(cases) != len(manifest["cases"]),
        "training_seed": args.training_seed, "checkpoint_files": hashes,
        "checkpoint_selection": selection, "env_config": config.to_dict(),
        "numerics": numerics,
        "perception": "simulated map experiments; no real sensor reconstruction"}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n")
    records = []
    with (args.output / "episodes.jsonl").open("x") as stream:
        for case in cases:
            bundle = materialize_case(case, args.scene_cache, voxel_size=args.voxel_size)
            env = G1FurnitureEnv(bundle, config=config)
            policy = load_controller(checkpoint, env)
            record = run_episode(env, policy, scene=env.scene, case_id=case["case_id"],
                training_seed=args.training_seed, episode_seed=case["episode_seed"],
                controller=args.controller, provenance={"geometry_hash": env.scene.get("geometry_hash"),
                                                        "goal_index": case["goal_index"]})
            records.append(record)
            stream.write(json.dumps(record.to_dict(), allow_nan=False)+"\n")
            stream.flush(); os.fsync(stream.fileno())
            print(json.dumps({"case": case["case_id"], "strict_success": record.strict_success,
                              "reason": record.termination_reason}), flush=True)
            del policy, env
            # Each room has different static MJCF topology. Bound compilation
            # cache growth over an explicitly requested many-room benchmark.
            import jax
            jax.clear_caches()
    result = summarize(records)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
