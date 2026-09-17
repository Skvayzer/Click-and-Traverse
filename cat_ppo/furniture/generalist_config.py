"""Pinned released CAT configuration and explicit single-GPU resource profile."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASED_CONFIG = ROOT / "configs/cat_generalist_released.json"
RELEASED_SHA256 = "0e38cac262a3c1c95bacdf859193340056fef53b8d2f8414293789cee5d45d2d"
MODEL_REVISION = "46ce4b57ba0639168d51741b661ff62f7ce6f045"
UPSTREAM_COMMIT = "866ba392f1c1e84b92ad75fa66550f26e8af8e48"

# These are passed verbatim; the launcher owns callbacks, network construction,
# scene paths, storage and run control. DAgger has already trained this model.
PPO_KEYS = (
    "learning_rate", "entropy_cost", "discounting", "unroll_length",
    "batch_size", "num_minibatches", "num_updates_per_batch",
    "normalize_observations", "reward_scaling", "clipping_epsilon",
    "gae_lambda", "max_grad_norm", "normalize_advantage",
    "episode_length", "action_repeat", "randomize_initial_episode_steps",
)


def released_config():
    payload = RELEASED_CONFIG.read_bytes()
    if hashlib.sha256(payload).hexdigest() != RELEASED_SHA256:
        raise ValueError("Released CAT config checksum mismatch")
    return json.loads(payload)


def batch_geometry(policy):
    envs = policy["num_envs"]
    trajectories = policy["batch_size"] * policy["num_minibatches"]
    if envs <= 0 or trajectories <= 0 or trajectories % envs:
        raise ValueError("num_envs must divide batch_size * num_minibatches")
    return {
        "parallel_environments": envs,
        "rollout_chunks_per_update": trajectories // envs,
        "transitions_per_update": trajectories * policy["unroll_length"],
        "transitions_per_minibatch": policy["batch_size"] * policy["unroll_length"],
        "optimizer_steps_per_update": policy["num_minibatches"] * policy["num_updates_per_batch"],
    }


def training_config(*, profile="single_gpu_32gb", num_envs=None, batch_size=None, seed=0,
                    finetuning="released"):
    config = copy.deepcopy(released_config())
    policy = config["policy_config"]
    original = batch_geometry(policy)
    if profile == "single_gpu_32gb":
        # Whole-body observations make the released 4.19M-transition rollout
        # exceed this GPU's budget before simulator/gradient memory. Parallelism
        # is independent: reducing envs collects more chunks of the SAME batch.
        policy["num_envs"] = 2048
        policy["batch_size"] = 256
    elif profile != "released":
        raise ValueError(f"Unknown resource profile: {profile}")
    if num_envs is not None:
        policy["num_envs"] = int(num_envs)
    if batch_size is not None:
        policy["batch_size"] = int(batch_size)
    policy["seed"] = int(seed)
    if finetuning in ("gentle", "stabilized", "hand_protection"):
        policy["learning_rate"] = 3e-5
        policy["clipping_epsilon"] = .1
    elif finetuning != "released":
        raise ValueError(f"Unknown fine-tuning mode: {finetuning}")
    effective = batch_geometry(policy)
    policy.update(num_timesteps=0, continuous=True, num_evals=0, num_eval_envs=0,
                  num_resets_per_eval=0, max_devices_per_host=1)
    policy["dagger_config"]["enable"] = False
    policy["progress_fn"] = None
    config["fine_tuning"] = {
        "source_model_revision": MODEL_REVISION,
        "source_upstream_commit": UPSTREAM_COMMIT,
        "source_config_sha256": RELEASED_SHA256,
        "initialization": "released final generalist actor and critic; Adam initialized once",
        "dagger": "already completed in released generalist; direct PPO fine-tuning",
        "continuous": True,
        "automatic_evaluation": finetuning in ("stabilized", "hand_protection"),
        "upper_stabilization": finetuning in ("stabilized", "hand_protection"),
        "hand_protection": finetuning == "hand_protection",
        "action_distribution": ({"leg_action_count": 12, "upper_std_min": .02,
                                 "upper_std_max": .10, "upper_entropy_weight": 0.0}
                                if finetuning in ("stabilized", "hand_protection") else None),
        "retention_validation": ({"interval_updates": 50, "seeds_per_scene": 16,
                                   "cat_success_tolerance": .05,
                                   "per_scene_success_tolerance": .125,
                                   "modes": ["deterministic", "stochastic"],
                                   "outcome": "first clean goal, native failure, or native horizon",
                                   "scene_scope": "16 fixed training-bank layouts; retention monitor, not unseen generalization"}
                                  if finetuning in ("stabilized", "hand_protection") else None),
        "profile": profile,
        "mode": finetuning,
        "optimization_overrides": {k: {"released": released_config()["policy_config"][k], "effective": policy[k]}
                                   for k in ("learning_rate", "clipping_epsilon")
                                   if released_config()["policy_config"][k] != policy[k]},
        "reference_kl": ({"coefficient": .05, "action_indices": list(range(12)),
                          "scene_scope": "CAT task scenes only; rooms excluded",
                          "reference": "frozen initial actor mapped from released CAT, on the same compact observations"}
                         if finetuning in ("gentle", "stabilized", "hand_protection") else None),
        "released_batch_geometry": original,
        "effective_batch_geometry": effective,
        "resource_overrides": {k: {"released": released_config()["policy_config"][k], "effective": policy[k]}
                               for k in ("num_envs", "batch_size")
                               if released_config()["policy_config"][k] != policy[k]},
        "memory_validation": "profile is provisional until checked on the target GPU",
        "storage": "one best model plus one overwritten full learner resume state",
    }
    if finetuning == "hand_protection":
        from cat_ppo.furniture.control import JOINT_NAMES, wholebody_observation_contract
        features = wholebody_observation_contract()["actor_features"]
        # Existing previous-action observations make the conditional arm
        # distribution fully reproducible during PPO replay and deployment.
        config["fine_tuning"]["action_distribution"].update(
            exploration_version="arm_conditional_correlated_v2",
            arm_persistence=.95, arm_correlation=.8,
            arm_innovation_scale=(1. - .95 ** 2) ** .5,
            arm_last_action_indices=[features.index("last_action." + name)
                                     for name in JOINT_NAMES[15:]],
            arm_correlation_pattern="g1_raise_tuck_v1")
        config["fine_tuning"]["retention_validation"]["scene_scope"] = (
            "16 unchanged regression layouts plus one hand passage per kind/level; training layouts")
    return config


def ppo_kwargs(config):
    policy = config["policy_config"]
    return {key: policy[key] for key in PPO_KEYS} | {
        "num_envs": policy["num_envs"], "seed": policy["seed"],
        "max_devices_per_host": 1, "num_evals": 0, "num_eval_envs": 0,
        "num_resets_per_eval": 0, "randomization_fn": None,
    }
