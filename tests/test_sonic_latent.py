"""Gate 1 of the SONIC latent adaptor: the torch port equals the released ONNX controller, a zero latent residual is
plain SONIC, and gradients reach only the adaptor (SONIC stays frozen)."""
from pathlib import Path

import pytest
import torch

SONIC = Path("/home/konstantinsmirnov/robotics/sonic/hf_gear_sonic")
needs_release = pytest.mark.skipif(not (SONIC / "model_encoder.onnx").exists(), reason="SONIC release not on disk")


@needs_release
def test_torch_port_matches_release():
    from cat_mjlab.sonic_torch import verify
    torch.backends.cuda.matmul.allow_tf32 = False
    r = verify(SONIC / "model_encoder.onnx", SONIC / "model_decoder.onnx", n=256, device="cpu")
    assert r["token_code_mismatch_fraction"] == 0.
    assert r["chain_max_abs"] <= 1e-5
    assert r["zero_residual_identical"]


@needs_release
def test_gradient_reaches_only_the_adaptor():
    from cat_mjlab.learning import Learner, LearnerConfig
    from cat_mjlab.sonic_residual import zero_residual_mean
    from cat_mjlab.sonic_torch import SonicTorch, fsq

    sonic = SonicTorch(SONIC / "model_encoder.onnx", SONIC / "model_decoder.onnx", device="cpu")
    assert not any(p.requires_grad for p in sonic.parameters())
    config = LearnerConfig(algorithm="ppo", num_policies=1, actor_obs=32, critic_obs=32, action_size=66)
    learner = Learner(config, device="cpu")
    zero_residual_mean(learner)
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(8, 32, generator=g)
    mean = learner.model.logits(obs, torch.zeros(8, dtype=torch.long)).chunk(2, -1)[0]
    assert torch.equal(mean, torch.zeros_like(mean))                         # starts as plain SONIC
    cmd, anchor, proprio = torch.randn(8, 580, generator=g), torch.randn(8, 60, generator=g), torch.randn(8, 930, generator=g)
    z = sonic.latent(cmd, anchor)
    # Straight-through FSQ: the decoded action responds to the latent residual, SONIC weights get no gradient.
    residual = torch.zeros(8, 64, requires_grad=True)
    action = sonic.decoder(torch.cat((fsq(z + .3 * residual), proprio), -1))
    action.square().sum().backward()
    assert residual.grad is not None and residual.grad.abs().sum() > 0
    assert all(p.grad is None for p in sonic.parameters())
    # PPO-style surrogate on the adaptor's own output reaches the actor's final layer (zero-initialized mean rows).
    learner.model.zero_grad()
    out = learner.model.logits(obs, torch.zeros(8, dtype=torch.long))
    (out[:, :66] * torch.randn(8, 66, generator=g)).sum().backward()
    assert learner.model.actor.layers[-1].weight.grad[:66].abs().sum() > 0
