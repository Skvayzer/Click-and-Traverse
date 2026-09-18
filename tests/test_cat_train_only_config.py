"""The requested restart uses released PPO without evaluation or retention controls."""

import pytest

from cat_ppo.furniture.generalist_config import (
    MODEL_REVISION, PPO_KEYS, RELEASED_SHA256, UPSTREAM_COMMIT,
    ppo_kwargs, released_config, training_config,
)


@pytest.mark.parametrize("profile,num_envs,batch_size", [
    ("single_gpu_32gb", 16384, 256),
    ("single_gpu_32gb", 8192, 256),
    ("released", 65536, 2048),
])
def test_training_only_uses_released_ppo_except_explicit_resource_batch(profile, num_envs, batch_size):
    config = training_config(profile=profile, num_envs=num_envs, batch_size=batch_size,
                             finetuning="cat_train_only")
    native = released_config()["policy_config"]
    effective = ppo_kwargs(config)
    for key in PPO_KEYS:
        assert effective[key] == (batch_size if key == "batch_size" else native[key]), key
    assert effective["learning_rate"] == 3e-4
    assert effective["clipping_epsilon"] == .2
    assert effective["entropy_cost"] == .003
    assert effective["num_updates_per_batch"] == 4
    assert effective["num_minibatches"] == 64
    assert effective["unroll_length"] == 32
    assert effective["num_envs"] == num_envs
    assert config["fine_tuning"]["optimization_overrides"] == {}
    assert config["fine_tuning"]["effective_batch_geometry"] == {
        "parallel_environments": num_envs,
        "rollout_chunks_per_update": batch_size * 64 // num_envs,
        "transitions_per_update": batch_size * 64 * 32,
        "transitions_per_minibatch": batch_size * 32,
        "optimizer_steps_per_update": 256,
    }


def test_training_only_keeps_hand_task_but_has_no_evaluation_or_policy_retention():
    config = training_config(finetuning="cat_train_only")
    fine, policy = config["fine_tuning"], config["policy_config"]
    assert fine["upper_stabilization"] is True
    assert fine["hand_protection"] is True
    assert fine["training_only"] is True
    assert fine["automatic_evaluation"] is False
    assert fine["retention_validation"] is None
    assert fine["reference_kl"] is None
    assert "recovery" not in fine
    # The launcher selects native Brax networks for None. No bounded upper
    # sigma, AR arm smoothing/correlation, or frozen leg-scale reference.
    assert fine["action_distribution"] is None
    for mapping in (fine, policy["network_factory"], ppo_kwargs(config)):
        assert "leg_noise_reference" not in mapping
        assert "exploration_version" not in mapping
        assert "recovery_fn" not in mapping
    for key in ("num_evals", "num_eval_envs", "num_resets_per_eval"):
        assert policy[key] == ppo_kwargs(config)[key] == 0
    assert policy["continuous"] is True
    assert policy["num_timesteps"] == 0
    assert policy["dagger_config"]["enable"] is False
    assert fine["source_model_revision"] == MODEL_REVISION
    assert fine["source_upstream_commit"] == UPSTREAM_COMMIT
    assert fine["source_config_sha256"] == RELEASED_SHA256
    assert fine["initialization"].startswith("original released CAT final generalist actor and critic")
    assert "training rollout reward proxy" in fine["checkpoint_selection"]
    assert "one overwritten full learner resume state" in fine["storage"]


@pytest.mark.parametrize("mode,lr,clip,entropy,evaluation,reference,coherent,recovery", [
    ("released", 3e-4, .2, .003, False, False, False, False),
    ("gentle", 3e-5, .1, .003, False, True, False, False),
    ("stabilized", 3e-5, .1, .003, True, True, False, False),
    ("hand_protection", 3e-5, .1, .003, True, True, True, False),
    ("hand_recovery", 1e-5, .1, 0., True, True, True, True),
])
def test_legacy_profiles_keep_their_existing_explicit_semantics(
        mode, lr, clip, entropy, evaluation, reference, coherent, recovery):
    config = training_config(finetuning=mode)
    fine, policy = config["fine_tuning"], config["policy_config"]
    assert (policy["learning_rate"], policy["clipping_epsilon"], policy["entropy_cost"]) == (lr, clip, entropy)
    assert fine["automatic_evaluation"] is evaluation
    assert (fine["reference_kl"] is not None) is reference
    assert ("exploration_version" in (fine["action_distribution"] or {})) is coherent
    assert ("recovery" in fine) is recovery
    assert "training_only" not in fine


def test_training_only_does_not_mutate_released_or_legacy_config():
    released = released_config()
    training_config(finetuning="hand_recovery")
    first = training_config(finetuning="cat_train_only")
    first["policy_config"]["learning_rate"] = 999.
    first["fine_tuning"]["action_distribution"] = {"leg_noise_reference": "invalid"}
    second = training_config(finetuning="cat_train_only")
    assert second["policy_config"]["learning_rate"] == released["policy_config"]["learning_rate"]
    assert second["fine_tuning"]["action_distribution"] is None
    assert released_config() == released

