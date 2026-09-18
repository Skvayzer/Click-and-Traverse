"""The optional algorithm keeps the robot and native CAT coefficients intact."""
import pytest

from train_cat_wholebody import parser, plan
from cat_ppo.furniture.generalist_config import ppo_kwargs
from cat_ppo.learning.policy.sapg.config import normalize_config


def test_default_ppo_plan_is_unchanged_and_has_no_sapg_contract():
    default = plan(parser().parse_args(["plan"]))
    explicit = plan(parser().parse_args(["plan", "--algorithm", "ppo"]))
    assert default == explicit
    assert "sapg" not in default["config"]
    assert default["config"]["policy_config"]["num_envs"] == 2048


def test_sapg_plan_keeps_native_optimizer_robot_and_training_only_task():
    ppo = plan(parser().parse_args(["plan", "--num-envs", "24576", "--batch-size", "384"]))
    sapg = plan(parser().parse_args(["plan", "--algorithm", "sapg", "--num-envs", "24576", "--batch-size", "384"]))
    assert ppo_kwargs(sapg["config"]) == ppo_kwargs(ppo["config"])
    assert sapg["observations"] == ppo["observations"]
    fine = sapg["config"]["fine_tuning"]
    assert fine["reference_kl"] is None
    assert fine["retention_validation"] is None
    assert fine["action_distribution"] is None
    assert fine["training_only"] is True
    geometry = fine["effective_batch_geometry"]
    assert geometry["environments_per_policy"] == 4096
    assert geometry["transitions_per_update"] == 786432
    assert geometry["optimizer_transitions_per_update"] == 917504
    assert geometry["optimizer_transitions_per_minibatch"] == 14336
    assert fine["sapg_method"]["entropy"] == [.003] * 6
    assert "leader" in fine["checkpoint_selection"]


def test_default_sapg_resource_profile_is_divisible_and_explicit():
    result = plan(parser().parse_args(["plan", "--algorithm", "sapg"]))
    policy = result["config"]["policy_config"]
    assert policy["num_envs"] == 1536
    assert policy["batch_size"] == 192
    assert result["config"]["sapg"] == dict(num_policies=6, embedding_dim=16, prepare_chunk_size=64)


@pytest.mark.parametrize("arguments", [
    ["--num-envs", "2048", "--batch-size", "256"],
    ["--finetuning", "hand_recovery"],
    ["--sapg-num-policies", "1"],
    ["--sapg-embedding-dim", "0"],
    ["--warmstart-best", "/unused/finetuned/model"],
])
def test_incompatible_sapg_launch_rejected_before_loading_assets(arguments):
    with pytest.raises(ValueError):
        plan(parser().parse_args(["plan", "--algorithm", "sapg", *arguments]))


@pytest.mark.parametrize("config", [{"typo": 1}, {"num_policies": True}, {"embedding_dim": 1.5}])
def test_invalid_sapg_contract_rejected(config):
    with pytest.raises(ValueError):
        normalize_config(config)
