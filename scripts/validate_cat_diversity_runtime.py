"""CPU-only full-bank JIT reset and one-step check of five CAT scene families.

This loads every field array, but simulates only five representative scenes.
It runs no learner, policy checkpoint, optimizer, W&B session, or GPU work.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

# Set these before importing anything that can initialize JAX/CUDA.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAMILIES = ("original_cat", "published_cat", "procedural_cat", "furniture", "generic_clutter")


def progress(message):
    print(message, file=sys.stderr, flush=True)


def hlo_sizes(lowered):
    return {
        "serialized_hlo_bytes": len(lowered.compiler_ir("hlo").as_serialized_hlo_module_proto()),
        "stablehlo_text_bytes": len(str(lowered.compiler_ir("stablehlo")).encode()),
    }


def state_summary(state):
    import numpy as np

    checked = {
        "actor_observations": state.obs["state"],
        "critic_observations": state.obs["privileged_state"],
        "qpos": state.data.qpos, "qvel": state.data.qvel,
        "reward": state.reward, "done": state.done,
        "motor_targets": state.info["motor_targets"],
        "wholebody_clearances": state.info["wholebody_clearances"],
        "sampling_logits": state.info["pf_sampling_logits"],
    }
    finite = {name: bool(np.isfinite(np.asarray(value)).all()) for name, value in checked.items()}
    if not all(finite.values()):
        raise ValueError(f"Nonfinite runtime output: {finite}")
    return dict(finite=finite, actor_shape=list(state.obs["state"].shape),
                critic_shape=list(state.obs["privileged_state"].shape),
                scene_ids=np.asarray(state.info["pf_id"]).tolist(),
                reward=np.asarray(state.reward).tolist(), done=np.asarray(state.done).tolist(),
                hand_and_elbow_clearance_m=np.asarray(state.info["wholebody_clearances"]).tolist())


def validate(manifest_path, seed):
    import jax
    import jax.numpy as jp
    import numpy as np
    from ml_collections import ConfigDict

    from cat_ppo.envs.g1.env_cat_wholebody import G1CatWholeBodyEnv, wholebody_config
    from cat_ppo.furniture.generalist_config import released_config
    from cat_ppo.furniture.generalist_fields import EXPANDED_SCHEMA, sha256
    from cat_ppo.furniture.generalist_training import wrap_for_cat_wholebody_training
    from cat_ppo.learning.policy.ppo.field_arguments import FieldArguments

    devices = jax.devices()
    if any(device.platform != "cpu" for device in devices):
        raise RuntimeError(f"CPU-only validation refused non-CPU devices: {devices}")
    started = time.monotonic()
    progress("Verifying and loading the complete field bank on CPU...")
    config = wholebody_config(ConfigDict(released_config()["env_config"]), bank_manifest=manifest_path)
    env = G1CatWholeBodyEnv(config=config)
    manifest = env.field_bank_manifest
    if manifest["schema"] != EXPANDED_SCHEMA:
        raise ValueError("This validation requires the expanded v2 field bank")
    representatives = []
    for family in FAMILIES:
        found = next(((index, scene) for index, scene in enumerate(manifest["scenes"])
                      if scene["family"] == family), None)
        if found is None:
            raise ValueError(f"Field bank has no representative for {family}")
        index, scene = found
        representatives.append(dict(index=index, scene_id=scene["scene_id"], family=family,
            task_kind=scene["task_kind"], episode_length=scene["episode_length"],
            reset_mode=scene["reset_mode"], crossed_mode=scene["crossed_mode"],
            sampling_group=scene["sampling_group"]))
    wrapped = wrap_for_cat_wholebody_training(env)
    binding = FieldArguments(env)
    values = binding.values
    if binding.names != ("sdf", "bf", "gf"):
        raise ValueError("Full-bank dynamic field arguments were not enabled")
    field_arrays = {name: dict(shape=list(value.shape), dtype=str(value.dtype), bytes=int(value.nbytes))
                    for name, value in zip(binding.names, values)}
    keys = jax.random.split(jax.random.PRNGKey(seed), len(representatives))
    scene_ids = jp.asarray([item["index"] for item in representatives], dtype=jp.int32)
    loaded = time.monotonic()

    def reset(keys, ids, fields):
        with binding.bind(fields):
            return wrapped._reset_with_pf_id(keys, ids)

    def step(state, actions, fields):
        with binding.bind(fields):
            return wrapped.step(state, actions)

    progress("Lowering and compiling dynamic-field reset for five scene families...")
    reset_lowered = jax.jit(reset).lower(keys, scene_ids, values)
    reset_hlo = hlo_sizes(reset_lowered)
    reset_fn = reset_lowered.compile()
    state = reset_fn(keys, scene_ids, values)
    jax.block_until_ready(state)
    reset_report = state_summary(state)
    np.testing.assert_array_equal(np.asarray(state.info["pf_id"]), np.asarray(scene_ids))
    if state.obs["state"].shape != (5, 222) or state.obs["privileged_state"].shape != (5, 310):
        raise ValueError("Compact observation dimensions differ from 222/310")
    if env.action_size != 29:
        raise ValueError("Compact action count differs from 29")
    expected_limits = np.asarray([item["episode_length"] for item in representatives])
    np.testing.assert_array_equal(np.asarray(env._pf_scene_episode_lengths)[np.asarray(scene_ids)], expected_limits)
    reset_finished = time.monotonic()

    progress("Lowering and compiling one real wrapped physics step on CPU...")
    actions = jp.zeros((5, env.action_size), dtype=jp.float32)
    step_lowered = jax.jit(step).lower(state, actions, values)
    step_hlo = hlo_sizes(step_lowered)
    step_fn = step_lowered.compile()
    result = step_fn(state, actions, values)
    jax.block_until_ready(result)
    step_report = state_summary(result)
    if any(getattr(env, name) is not value for name, value in zip(binding.names, values)):
        raise ValueError("Field argument binding failed to restore environment arrays")
    finished = time.monotonic()
    return dict(schema="cat-full-bank-cpu-runtime-validation-v1", status="passed",
        scope="complete bank loaded; five representative scenes; JIT reset and one zero-action wrapped step",
        limitations=["CPU functional validation only; GPU capacity and training throughput are not established",
                     "Five scenes exercised dynamically; all scene files validated by loader fingerprints",
                     "No learned-policy retention, navigation success, or dynamic-feasibility evaluation"],
        seed=seed, jax_version=jax.__version__, devices=[str(device) for device in devices],
        bank_manifest=str(manifest_path), bank_manifest_file_sha256=sha256(manifest_path),
        bank_manifest_payload_sha256=manifest["manifest_sha256"], scene_count=manifest["scene_count"],
        family_counts=dict(Counter(scene["family"] for scene in manifest["scenes"])),
        representatives=representatives, field_arrays=field_arrays,
        total_field_array_bytes=sum(item["bytes"] for item in field_arrays.values()),
        field_argument_axes="shared unbatched fields; five environment states",
        action_size=env.action_size, reset=reset_report, step=step_report,
        compiler_ir=dict(reset=reset_hlo, step=step_hlo),
        seconds=dict(load_and_verify=loaded-started, reset_lower_compile_execute=reset_finished-loaded,
                     step_lower_compile_execute=finished-reset_finished, total=finished-started),
        source_sha256={name: sha256(ROOT / name) for name in (
            "scripts/validate_cat_diversity_runtime.py", "cat_ppo/learning/policy/ppo/field_arguments.py",
            "cat_ppo/furniture/generalist_fields.py", "cat_ppo/envs/g1/env_cat_wholebody.py")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON report outside the field bank")
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    manifest_path, output = args.bank_manifest.resolve(), args.output.resolve()
    if not manifest_path.is_file():
        parser.error(f"Manifest does not exist: {manifest_path}")
    if output.is_relative_to(manifest_path.parent):
        parser.error("Output must be outside the field bank; validation never writes into it")
    report = validate(manifest_path, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(output), status=report["status"],
                         scene_count=report["scene_count"], field_array_bytes=report["total_field_array_bytes"],
                         actor_shape=report["reset"]["actor_shape"], critic_shape=report["reset"]["critic_shape"],
                         compiler_ir=report["compiler_ir"], seconds=report["seconds"]), indent=2))


if __name__ == "__main__":
    main()
