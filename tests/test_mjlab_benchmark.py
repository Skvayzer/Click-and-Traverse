"""The GPU benchmark is bounded and retains native trajectory minibatches."""
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from cat_mjlab.learning import LearnerConfig
from scripts.benchmark_mjlab import geometry, parser


@pytest.mark.parametrize("environments,batch", [(18432, 288), (36864, 576)])
def test_full_capacity_geometry_keeps_64_minibatches_and_six_policy_groups(environments, batch):
    args = SimpleNamespace(num_envs=environments, batch_size=None, unroll_length=32, warmup_steps=2)
    assert geometry(args, LearnerConfig()) == (batch, environments)


def test_capacity_probe_refuses_invalid_sapg_groups():
    args = SimpleNamespace(num_envs=16384, batch_size=None, unroll_length=32, warmup_steps=2)
    with pytest.raises(ValueError, match="num_policies"):
        geometry(args, LearnerConfig())
    args.num_envs = 36864
    args.unroll_length = 0
    with pytest.raises(ValueError, match="positive"):
        geometry(args, LearnerConfig())


def test_benchmark_has_no_continuous_training_or_checkpoint_logging_options():
    args = parser().parse_args(["--checkpoint-npz", "checkpoint.npz", "--bank-manifest", "fields.json",
        "--body-collision-bank", "collision.json", "--body-collision-resets", "resets.json"])
    assert args.unroll_length == 32 and args.warmup_steps == 2 and not args.update
    assert not hasattr(args, "wandb_mode") and not hasattr(args, "run_dir") and not hasattr(args, "max_updates")
