#!/usr/bin/env python3
"""Check a local contrastive bank through the real training wrapper.

Runs shape validation plus one small CPU JIT reset/step with zero actions.
It does not load a policy, run learning, evaluate success, or contact W&B.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--collision-bank", type=Path, required=True)
    parser.add_argument("--reset-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shape-only", action="store_true")
    parser.add_argument("--scene-indices", type=int, nargs="+", help="Explicit scene slots to exercise, one per environment")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 8:
        parser.error("This runtime verification uses 1..8 environments")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jp
    import numpy as np
    from ml_collections import ConfigDict
    from cat_ppo.envs.g1.env_cat_wholebody import G1CatWholeBodyEnv, wholebody_config
    from cat_ppo.furniture.generalist_config import released_config
    from cat_ppo.learning.policy.ppo.field_arguments import FieldArguments
    from cat_ppo.learning.train.pf_utils import wrap_for_brax_training_reset

    paths = {name: getattr(args, name).resolve() for name in ("bank", "collision_bank", "reset_bank")}
    config = wholebody_config(ConfigDict(released_config()["env_config"]),
        bank_manifest=paths["bank"], stabilization=True, hand_protection=True, hand_contrast=True)
    config.wholebody_first_outcome_metrics = True
    config.wholebody.body_collision.update(dict(enabled=True,
        bank_manifest=str(paths["collision_bank"]), reset_manifest=str(paths["reset_bank"])))
    env = G1CatWholeBodyEnv(config=config)
    if args.scene_indices is not None and (len(args.scene_indices) != args.batch_size
            or any(index < 0 or index >= env.num_pf_scenes for index in args.scene_indices)):
        parser.error("--scene-indices must contain one valid scene index per environment")
    wrapper = wrap_for_brax_training_reset(env, episode_length=4000)
    fields = FieldArguments(wrapper)
    keys = jax.random.split(jax.random.PRNGKey(811), args.batch_size)

    def reset(rng, arrays):
        with fields.bind(arrays):
            if args.scene_indices is not None:
                return wrapper._reset_with_pf_id(rng, jp.asarray(args.scene_indices, jp.int32))
            return wrapper.reset(rng)

    def step(state, action, arrays):
        with fields.bind(arrays):
            return wrapper.step(state, action)

    start = time.monotonic()
    shape = jax.eval_shape(reset, keys, fields.values)
    next_shape = jax.eval_shape(step, shape,
        jax.ShapeDtypeStruct((args.batch_size, 29), jp.float32), fields.values)
    assert shape.obs["state"].shape == (args.batch_size, 222)
    assert shape.obs["privileged_state"].shape == (args.batch_size, 310)
    assert env.action_size == 29
    assert jax.tree.structure(shape) == jax.tree.structure(next_shape)
    assert next_shape.info["pf_contrast_outcome_counts"].shape == (args.batch_size, 3, 2)
    assert next_shape.info["wholebody_contrast_zone_steps"].shape == (args.batch_size, 6, 3)
    report = dict(schema="contrastive-training-runtime-verification-v1",
        learner_or_policy_evaluation=False, scene_count=env.num_pf_scenes,
        actor_observations=222, critic_observations=310, actions=29,
        batch_size=args.batch_size, platform="cpu", jax_version=jax.__version__,
        shape_check_passed=True, shape_seconds=time.monotonic() - start,
        banks={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
               for name, path in paths.items()},
        body_collision_contract=env.body_collision_contract)
    if not args.shape_only:
        start = time.monotonic()
        state = jax.jit(reset)(keys, fields.values)
        jax.block_until_ready(state.reward)
        report["jit_reset_seconds"] = time.monotonic() - start
        report["sampled_scene_indices"] = np.asarray(state.info["pf_id"]).tolist()
        if args.scene_indices is not None:
            assert report["sampled_scene_indices"] == args.scene_indices
        start = time.monotonic()
        result = jax.jit(step)(state, jp.zeros((args.batch_size, 29)), fields.values)
        jax.block_until_ready(result.reward)
        for value in (result.reward, result.obs["state"], result.obs["privileged_state"]):
            assert np.isfinite(np.asarray(value)).all()
        report.update(jit_reset_passed=True, jit_step_passed=True,
            jit_step_seconds=time.monotonic() - start,
            zero_action_transition_rewards=np.asarray(result.reward).tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
