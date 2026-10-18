"""Three-teacher distillation / RL fine-tune plumbing (2026-10-18): per-group reward config, commanded blocking-map
switch, CAT-only tight clearance, DAgger regulariser loss, critic warm-up."""
import pytest
import torch

from cat_mjlab.config import wholebody_config
from cat_mjlab.learning import Learner, LearnerConfig, compute_loss, gaussian_parameters


def test_reward_group_room_config():
    c = wholebody_config(reward_group_room=["wholebody_hand_clearance=-40", "heading_align=1.0"], commanded_blocking_map="off",
                         hand_clearance_tight=(.1, .4, .3), hand_tight_walls_only=True, hand_tight_cat_only=True,
                         heading_blocking_probes=True, heading_lookaheads=(.3, .6, 1.), heading_align_weight=.4)
    assert c["reward_group_scales"]["room"] == {"wholebody_hand_clearance": -40., "heading_align": 1.}
    assert c["blocking_map"] is True and c["blocking_map_commanded"] is False
    assert c["hand_clearance_tight"]["cat_only"] is True


def test_reward_group_rejects_unknown_term():
    with pytest.raises(ValueError):
        wholebody_config(reward_group_room=["no_such_term=1"])


def test_cat_only_requires_tight():
    with pytest.raises(ValueError):
        wholebody_config(hand_tight_cat_only=True)


def _rollout(config, T=4, N=8, label_shift=0.):
    torch.manual_seed(0)
    return dict(state=torch.randn(N, T, config.actor_obs), privileged_state=torch.randn(N, T, config.critic_obs),
                next_state=torch.randn(N, T, config.actor_obs), next_privileged_state=torch.randn(N, T, config.critic_obs),
                raw_action=torch.randn(N, T, 29) * .1, log_prob=torch.zeros(N, T), reward=torch.randn(N, T),
                discount=torch.ones(N, T), truncation=torch.zeros(N, T), policy_id=torch.zeros(N, T, dtype=torch.long),
                distill_label=torch.zeros(N, T, 29) + label_shift, distill_mask=torch.ones(N, T))


def test_distill_regulariser_adds_masked_mse():
    config = LearnerConfig(algorithm="ppo", num_policies=1)
    learner = Learner(config, device="cpu")
    data = _rollout(config)
    noise = torch.randn(8, 4, 29)          # the entropy estimate samples noise: fix it for an exact comparison
    base, _ = compute_loss(learner.model, {k: v for k, v in data.items() if not k.startswith("distill")}, config, entropy_noise=noise)
    total, m = compute_loss(learner.model, data, config, distill_lambda=2., entropy_noise=noise)
    mean, _ = gaussian_parameters(learner.model.logits(data["state"], data["policy_id"]))
    expected = mean.square().mean(-1).mean()
    assert torch.allclose(m["distill/reg_mse"], expected, atol=1e-6)
    assert torch.allclose(total, base + 2. * expected, atol=1e-5)
    data["distill_mask"] = torch.zeros_like(data["distill_mask"])
    total0, m0 = compute_loss(learner.model, data, config, distill_lambda=2., entropy_noise=noise)
    assert float(m0["distill/reg_mse"]) == 0. and torch.allclose(total0, base, atol=1e-6)


def test_critic_warmup_leaves_actor_unchanged():
    config = LearnerConfig(algorithm="ppo", num_policies=1, num_minibatches=2, num_updates_per_batch=1)
    learner = Learner(config, device="cpu")
    actor = {k: v.clone() for k, v in learner.model.actor.state_dict().items()}
    critic = {k: v.clone() for k, v in learner.model.critic.state_dict().items()}
    learner.actor_frozen = True; learner.distill_lambda = 1.
    learner.update(_rollout(config, label_shift=1.))
    assert all(torch.equal(actor[k], v) for k, v in learner.model.actor.state_dict().items())
    assert any(not torch.equal(critic[k], v) for k, v in learner.model.critic.state_dict().items())
    learner.actor_frozen = False
    learner.update(_rollout(config, label_shift=1.))
    assert any(not torch.equal(actor[k], v) for k, v in learner.model.actor.state_dict().items())
