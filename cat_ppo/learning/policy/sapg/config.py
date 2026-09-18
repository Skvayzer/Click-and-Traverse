"""Explicit SAPG configuration; ordinary PPO never enters this path."""
import numbers
from collections.abc import Mapping


def normalize_config(config):
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("SAPG config must be a mapping")
    unknown = set(config) - {"num_policies", "embedding_dim", "prepare_chunk_size"}
    if unknown:
        raise ValueError(f"Unknown SAPG settings: {sorted(unknown)}")
    result = {"num_policies": 6, "embedding_dim": 16, "prepare_chunk_size": 64}
    result.update(config)
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"SAPG {key} must be a positive integer")
        result[key] = int(value)
    if result["num_policies"] < 2:
        raise ValueError("SAPG requires a leader and at least one follower")
    return result


def configure_training(config, *, num_policies=6, embedding_dim=16):
    """Annotate a CAT training-only config without changing its PPO coefficients."""
    settings = normalize_config(dict(num_policies=num_policies, embedding_dim=embedding_dim))
    fine = config["fine_tuning"]
    if fine["mode"] != "cat_train_only":
        raise ValueError("SAPG currently supports --finetuning cat_train_only only")
    policy = config["policy_config"]
    count = settings["num_policies"]
    if policy["num_envs"] % count or policy["batch_size"] % count:
        raise ValueError("SAPG requires --num-envs and --batch-size divisible by --sapg-num-policies; "
                         "for six policies use e.g. 1536/192 or 24576/384")
    if policy["normalize_observations"]:
        raise ValueError("SAPG currently requires CAT's normalize_observations=False")
    config["algorithm"] = "sapg"
    config["sapg"] = settings
    geometry = fine["effective_batch_geometry"]
    geometry.update(
        policies=count,
        environments_per_policy=policy["num_envs"] // count,
        optimizer_transitions_per_update=geometry["transitions_per_update"] * (count + 1) // count,
        optimizer_transitions_per_minibatch=geometry["transitions_per_minibatch"] * (count + 1) // count,
    )
    fine.update(
        dagger="already completed in released generalist; direct SAPG fine-tuning",
        checkpoint_selection="highest leader training rollout reward proxy; no evaluation or retention gates",
        storage="one ordinary-format leader best model plus one overwritten full SAPG learner resume state",
    )
    fine["sapg_method"] = {
        "leader": 0,
        "conditioning": "shared learned embedding appended internally; robot observations unchanged",
        "initialization": "zero added input weights preserve every policy's pretrained output",
        "sharing": "one uniformly sampled follower block relabeled to leader per rollout",
        "off_policy_actor": "paper mu-centered clipping; old targets frozen for all optimizer passes",
        "off_policy_critic": "one-step leader bootstrap; no importance weighting",
        "advantages": "CAT GAE on-policy, one-step off-policy; normalize full augmented rollout once",
        "entropy": [float(policy["entropy_cost"])] * count,
        "success_rates": "first-outcome training counts pooled across all policies, not leader evaluation",
    }
    return config
