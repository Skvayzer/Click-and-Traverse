"""Numerical parity of the PyTorch learner against the existing JAX learner."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp
from brax.training import types
from brax.training.agents.ppo import losses as brax_losses
from cat_ppo.learning.policy.sapg import losses as jax_losses
from cat_ppo.learning.policy.sapg.networks import make_sapg_networks
from cat_mjlab.conversion import load_native_params
from cat_mjlab.learning import (Learner, LearnerConfig, compute_gae, compute_loss,
    gaussian_parameters, importance_clipped_surrogate, log_probability,
    prepare_sapg_rollout, tanh_log_det, transformed_entropy)


def setup_pair(algorithm="sapg"):
    config = LearnerConfig(algorithm=algorithm, actor_obs=3, critic_obs=4, action_size=2,
        actor_hidden=(7, 5), critic_hidden=(9, 6), num_policies=3, embedding_dim=2,
        num_minibatches=2, num_updates_per_batch=2, prepare_chunk_size=2, entropy_cost=0.)
    if algorithm == "sapg":
        factory = make_sapg_networks
        extra = dict(num_policies=3, embedding_dim=2)
    else:
        from brax.training.agents.ppo.networks import make_ppo_networks
        factory, extra = make_ppo_networks, {}
    net = factory({"state": (3,), "privileged_state": (4,)}, 2,
        policy_hidden_layer_sizes=config.actor_hidden, value_hidden_layer_sizes=config.critic_hidden,
        policy_obs_key="state", value_obs_key="privileged_state", **extra)
    params = brax_losses.PPONetworkParams(policy=net.policy_network.init(jax.random.PRNGKey(1)),
                                         value=net.value_network.init(jax.random.PRNGKey(2)))
    if algorithm == "sapg":
        # Nonzero conditioning makes wrong policy IDs and critic table ownership visible.
        params.policy["params"]["hidden_0"]["kernel"] = params.policy["params"]["hidden_0"]["kernel"].at[-2:].set(.1)
        params.value["params"]["hidden_0"]["kernel"] = params.value["params"]["hidden_0"]["kernel"].at[-2:].set(.2)
    learner = Learner(config, device="cpu")
    load_native_params(learner, (None, params.policy, params.value))
    return learner, net, params


def make_rollout(learner):
    rng = np.random.default_rng(7)
    data = {key: torch.tensor(rng.normal(0, .3, (6, 4, width)), dtype=torch.float32)
            for key, width in (("state", 3), ("next_state", 3), ("privileged_state", 4), ("next_privileged_state", 4))}
    ids = np.repeat(np.arange(3), 2) if learner.config.algorithm == "sapg" else np.zeros(6, np.int64)
    data["policy_id"] = torch.tensor(np.broadcast_to(ids[:, None], (6, 4)).copy())
    with torch.no_grad():
        logits = learner.model.logits(data["state"], data["policy_id"])
        mean, std = gaussian_parameters(logits)
        data["raw_action"] = mean + std * torch.tensor(rng.normal(size=(6, 4, 2)), dtype=torch.float32)
        data["log_prob"] = log_probability(logits, data["raw_action"])
    data.update(reward=torch.tensor(rng.normal(size=(6, 4)), dtype=torch.float32),
                discount=torch.ones(6, 4), truncation=torch.zeros(6, 4))
    data["discount"][0, 1] = 0
    data["discount"][2, 2] = 0
    data["truncation"][2, 2] = 1
    return data


def as_jax(data):
    a = lambda name: jnp.asarray(data[name].numpy())
    return types.Transition(observation={"state": a("state"), "privileged_state": a("privileged_state")},
        next_observation={"state": a("next_state"), "privileged_state": a("next_privileged_state")},
        action=jnp.tanh(a("raw_action")), reward=a("reward"), discount=a("discount"),
        extras={"state_extras": {"truncation": a("truncation")},
                "policy_extras": {key: a(key) for key in ("raw_action", "log_prob", "policy_id")}})


def test_gaussian_distribution_and_entropy_preserve_brax_transformation():
    rng = np.random.default_rng(19)
    logits = torch.tensor(rng.normal(size=(7, 6)), dtype=torch.float32, requires_grad=True)
    raw = torch.tensor(rng.normal(size=(7, 3)), dtype=torch.float32)
    noise = torch.tensor(rng.normal(size=(7, 3)), dtype=torch.float32)
    from brax.training.distribution import NormalTanhDistribution
    distribution = NormalTanhDistribution(3)
    expected = distribution.log_prob(jnp.asarray(logits.detach().numpy()), jnp.asarray(raw.numpy()))
    np.testing.assert_allclose(log_probability(logits, raw).detach().numpy(), expected, atol=3e-6, rtol=3e-6)
    def entropy(x):
        d = distribution.create_dist(x)
        sample = d.loc + d.scale * jnp.asarray(noise.numpy())
        return jnp.sum(d.entropy() + 2 * (jnp.log(2.) - sample - jax.nn.softplus(-2 * sample)), axis=-1).mean()
    actual = transformed_entropy(logits, noise).mean()
    actual.backward()
    np.testing.assert_allclose(actual.detach(), entropy(jnp.asarray(logits.detach().numpy())), atol=2e-6)
    np.testing.assert_allclose(logits.grad, jax.grad(entropy)(jnp.asarray(logits.detach().numpy())), atol=2e-6)


def test_gae_terminal_and_truncation_matches_brax_not_standard_return_minus_value():
    rng = np.random.default_rng(3)
    arrays = [rng.normal(size=(5, 7)).astype(np.float32) for _ in range(2)]
    reward, values = map(torch.tensor, arrays)
    truncation, terminal = torch.zeros_like(values), torch.zeros_like(values)
    truncation[1, 3], terminal[2, 4] = 1, 1
    bootstrap = torch.tensor(rng.normal(size=5), dtype=torch.float32)
    target, advantage = compute_gae(truncation, terminal, reward, values, bootstrap)
    jtarget, jadvantage = brax_losses.compute_gae(*[jnp.asarray(a.numpy().T) for a in
        (truncation, terminal, reward, values)], jnp.asarray(bootstrap.numpy()), lambda_=.95, discount=.98)
    np.testing.assert_allclose(target, jtarget.T, atol=2e-6)
    np.testing.assert_allclose(advantage, jadvantage.T, atol=2e-6)


@pytest.mark.parametrize("algorithm", ["ppo", "sapg"])
def test_targets_loss_and_parameter_gradients_match_existing_jax(algorithm):
    learner, net, params = setup_pair(algorithm)
    data = make_rollout(learner)
    native = as_jax(data)
    if algorithm == "sapg":
        native = jax_losses.prepare_rollout(params, None, native, jax.random.PRNGKey(9), net,
            num_policies=3, discounting=.98, gae_lambda=.95, preparation_chunk_size=2)
        follower = int(np.asarray(native.extras["policy_extras"]["policy_id"])[-1, 0])
        data = prepare_sapg_rollout(learner.model, data, learner.config, follower_id=follower)
        for key, native_key in (("target_value", "sapg_target_value"), ("advantage", "sapg_advantage"),
            ("old_target_log_prob", "sapg_old_target_log_prob"), ("log_importance", "sapg_log_importance_weight"),
            ("target_policy_id", "sapg_target_policy_id"), ("action_std", "sapg_action_std")):
            np.testing.assert_allclose(data[key], native.extras["policy_extras"][native_key], atol=3e-6, rtol=3e-6)
        loss_fn = lambda p: jax_losses.compute_sapg_loss(p, None, native, jax.random.PRNGKey(3), net,
            entropy_cost=0., clipping_epsilon=.2, num_policies=3)
    else:
        loss_fn = lambda p: brax_losses.compute_ppo_loss(p, None, native, jax.random.PRNGKey(3), net,
            entropy_cost=0., clipping_epsilon=.2, discounting=.98, gae_lambda=.95)
    (jloss, jmetrics), jgrads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    loss, metrics = compute_loss(learner.model, data, learner.config)
    loss.backward()
    for key in metrics:
        np.testing.assert_allclose(metrics[key].detach(), jmetrics[key], rtol=3e-5, atol=3e-6)
    for trunk, native_grad in (("actor", jgrads.policy), ("critic", jgrads.value)):
        for index, layer in enumerate(getattr(learner.model, trunk).layers):
            np.testing.assert_allclose(layer.weight.grad.numpy(), np.asarray(native_grad["params"][f"hidden_{index}"]["kernel"]).T,
                                       rtol=5e-5, atol=3e-6)
            np.testing.assert_allclose(layer.bias.grad.numpy(), native_grad["params"][f"hidden_{index}"]["bias"], rtol=5e-5, atol=3e-6)
    if algorithm == "sapg":
        np.testing.assert_allclose(learner.model.policy_embeddings.grad.numpy(), jgrads.policy["policy_embeddings"], rtol=5e-5, atol=3e-6)


def test_fixed_log_domain_surrogate_retains_finite_values_and_gradients():
    current = torch.tensor([-120., 100., 0., 90.], requires_grad=True)
    old = torch.tensor([0., 0., 0., 0.])
    behavior = torch.tensor([-120., 100., -1000., 0.])
    importance = old - behavior
    advantages = torch.tensor([1., -1., 0., -1e-10])
    loss = importance_clipped_surrogate(current, old, behavior, importance, advantages, .2,
                                      log_normalizer=math_log4())
    fn = lambda x: jax_losses.importance_clipped_surrogate(x, jnp.asarray(old.numpy()),
        jnp.asarray(behavior.numpy()), jnp.asarray(importance.numpy()), jnp.asarray(advantages.numpy()), .2,
        log_normalizer=jnp.log(4.)).sum()
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(current.grad).all()
    np.testing.assert_allclose(loss.detach().sum(), fn(jnp.asarray(current.detach().numpy())), rtol=2e-5)
    np.testing.assert_allclose(current.grad, jax.grad(fn)(jnp.asarray(current.detach().numpy())), rtol=2e-5)


def math_log4():
    return torch.tensor(4.).log()


@pytest.mark.parametrize("algorithm", ["ppo", "sapg"])
def test_real_updates_count_physical_samples_once_and_resume_next_update_exactly(algorithm):
    learner, _, _ = setup_pair(algorithm)
    data = make_rollout(learner)
    initial = {key: value.clone() for key, value in learner.model.state_dict().items()}
    metrics = learner.update(data)
    assert metrics["physical_transitions"] == 24
    assert metrics["optimizer_transitions"] == (32 if algorithm == "sapg" else 24)
    assert metrics["optimizer_steps"] == 4
    assert any(not torch.equal(initial[key], value) for key, value in learner.model.state_dict().items())
    import copy
    state = copy.deepcopy(learner.state_dict())
    restored = Learner(learner.config, device="cpu")
    restored.load_state_dict(state)
    rng = torch.get_rng_state()
    metrics1 = learner.update(data)
    torch.set_rng_state(rng)
    metrics2 = restored.update(data)
    assert metrics1 == metrics2
    for a, b in zip(learner.model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_invalid_grouping_is_rejected_before_optimization():
    learner, _, _ = setup_pair()
    data = make_rollout(learner)
    data["policy_id"][0, 2] = 1
    with pytest.raises(ValueError, match="fixed ID"):
        learner.update(data)
