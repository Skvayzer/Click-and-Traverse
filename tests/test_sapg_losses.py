"""Numerical SAPG checks independent of the robot simulator and GPU trainer."""

import functools
from types import SimpleNamespace

from brax.training import types
from brax.training.agents.ppo import losses as ppo_losses
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cat_ppo.learning.policy.sapg import losses
from cat_ppo.learning.policy.sapg.networks import make_sapg_networks


class AnalyticDistribution:
    """Controlled log likelihood/entropy for exact surrogate checks."""

    def log_prob(self, logits, raw_action):
        del raw_action
        return logits[..., 0]

    def entropy(self, logits, key):
        del key
        return logits[..., 1]

    def create_dist(self, logits):
        return SimpleNamespace(scale=jnp.ones_like(logits[..., :1]))


def analytic_network(*, maximum_chunk=None):
    def features(observation):
        value = observation["state"]
        if maximum_chunk is not None:
            assert value.shape[0] <= maximum_chunk
        return value

    def actor(normalizer, params, observation, policy_ids=0):
        del normalizer
        obs = features(observation)
        first = jnp.broadcast_to(params["offsets"][policy_ids], obs.shape[:-1])
        return jnp.stack((first, jnp.ones_like(first)), axis=-1)

    def critic(normalizer, params, observation, policy_ids=0, *, policy_embeddings):
        del normalizer
        obs = features(observation)
        return params["slope"] * obs[..., 0] + params["embedding_scale"] * policy_embeddings[policy_ids, 0]

    return SimpleNamespace(
        policy_network=SimpleNamespace(apply=actor),
        value_network=SimpleNamespace(apply=critic),
        parametric_action_distribution=AnalyticDistribution(),
    )


def analytic_params(num_policies=3, *, slope=0., embedding_scale=0., offsets=None):
    return ppo_losses.PPONetworkParams(
        policy={
            "offsets": jnp.zeros(num_policies) if offsets is None else jnp.asarray(offsets),
            "policy_embeddings": jnp.arange(num_policies, dtype=jnp.float32)[:, None],
        },
        value={"slope": jnp.asarray(slope), "embedding_scale": jnp.asarray(embedding_scale)},
    )


def rollout(ids=(0, 0, 1, 1, 2, 2), time=3, params=None):
    ids = jnp.broadcast_to(jnp.asarray(ids, dtype=jnp.int32)[:, None], (len(ids), time))
    values = jnp.arange(ids.size, dtype=jnp.float32).reshape(ids.shape) / 10
    marker = jnp.broadcast_to(jnp.arange(len(ids))[:, None], ids.shape)
    obs = {"state": jnp.stack((values, marker), axis=-1)}
    following = {"state": obs["state"].at[..., 0].add(.75)}
    behavior_log_prob = jnp.zeros_like(values) if params is None else params.policy["offsets"][ids]
    return types.Transition(
        observation=obs, next_observation=following,
        action=jnp.zeros((*ids.shape, 1)), reward=jnp.ones_like(values),
        discount=jnp.ones_like(values), extras={
            "policy_extras": {"policy_id": ids, "raw_action": jnp.zeros((*ids.shape, 1)),
                              "log_prob": behavior_log_prob},
            "state_extras": {"truncation": jnp.zeros_like(values)},
        },
    )


def prepare(data, params=None, network=None, **kwargs):
    return losses.prepare_rollout(
        params if params is not None else analytic_params(), None, data,
        kwargs.pop("rng", jax.random.PRNGKey(8)),
        network if network is not None else analytic_network(),
        num_policies=kwargs.pop("num_policies", 3), **kwargs,
    )


def frozen_batch(advantage, importance=1., target_value=0., old_log_prob=0., target_id=0):
    data = rollout(ids=(target_id,), time=1)
    extras = {
        losses.ADVANTAGE: jnp.asarray([[advantage]]),
        losses.IMPORTANCE_WEIGHT: jnp.asarray([[importance]]),
        losses.TARGET_VALUE: jnp.asarray([[target_value]]),
        losses.OLD_TARGET_LOG_PROB: jnp.asarray([[old_log_prob]]),
        losses.TARGET_POLICY_ID: jnp.asarray([[target_id]], dtype=jnp.int32),
    }
    return data._replace(extras={**data.extras,
        "policy_extras": {**data.extras["policy_extras"], **extras}})


@pytest.mark.parametrize("advantage,ratio,expected", [
    (2., 1.5, -7.2), (2., .5, -3.), (-2., 1.5, 9.), (-2., .5, 4.8),
])
def test_importance_is_outside_positive_and_negative_clipped_surrogate(advantage, ratio, expected):
    params = analytic_params(2, offsets=[np.log(ratio), 0.])
    _, metrics = losses.compute_sapg_loss(
        params, None, frozen_batch(advantage, importance=3.), jax.random.PRNGKey(0),
        analytic_network(), num_policies=2, clipping_epsilon=.2, entropy_cost=0.,
    )
    np.testing.assert_allclose(metrics["policy_loss"], expected, rtol=1e-6)


def test_equal_blocks_and_critic_not_importance_weighted_and_uniform_cat_entropy():
    params = analytic_params(2, offsets=[np.log(2.), 0.])
    data = prepare(rollout((0, 1), time=1, params=params), params,
                   num_policies=2, normalize_advantage=False)
    extra = data.extras["policy_extras"]
    np.testing.assert_allclose(extra[losses.IMPORTANCE_WEIGHT][:, 0], [1., 1., 2.])
    _, metrics = losses.compute_sapg_loss(
        params, None, data, jax.random.PRNGKey(0), analytic_network(), num_policies=2,
    )
    np.testing.assert_allclose(metrics["policy_loss"], -4 / 3, rtol=1e-6)
    np.testing.assert_allclose(metrics["v_loss"], .25)
    np.testing.assert_allclose(metrics["entropy_loss"], -.003)


def test_one_complete_random_follower_block_handles_interleaved_rollout_segments():
    # Collection segment 0's groups followed by collection segment 1's groups.
    original = rollout((0, 0, 1, 1, 2, 2) * 2, time=2)
    selected_ids = set()
    for seed in range(8):
        augmented = prepare(original, rng=jax.random.PRNGKey(seed), preparation_chunk_size=5)
        assert augmented.reward.shape == (16, 2)
        for before, after in zip(jax.tree_util.tree_leaves(original),
                                 jax.tree_util.tree_leaves(augmented._replace(extras={
                                     **augmented.extras, "policy_extras": {
                                         key: value for key, value in augmented.extras["policy_extras"].items()
                                         if not key.startswith("sapg_")}}))):
            np.testing.assert_array_equal(before, after[:12])
        extra = augmented.extras["policy_extras"]
        selected = np.asarray(extra["policy_id"])[12:]
        assert np.unique(selected).size == 1 and selected[0, 0] in (1, 2)
        selected_ids.add(int(selected[0, 0]))
        expected = np.flatnonzero(np.asarray(original.extras["policy_extras"]["policy_id"][:, 0]) == selected[0, 0])
        np.testing.assert_array_equal(augmented.observation["state"][12:], original.observation["state"][expected])
        np.testing.assert_array_equal(extra[losses.TARGET_POLICY_ID][12:], 0)
        np.testing.assert_array_equal(extra[losses.TARGET_POLICY_ID][:12], extra["policy_id"][:12])
    assert selected_ids == {1, 2}


def test_brax_gae_and_one_step_leader_targets_for_terminal_and_truncation():
    params = analytic_params(slope=1., embedding_scale=.2)
    original = rollout(params=params)
    truncation = jnp.tile(jnp.asarray([0., 0., 1.]), (6, 1))
    original = original._replace(discount=jnp.tile(jnp.asarray([1., 0., 0.]), (6, 1)),
        extras={**original.extras, "state_extras": {"truncation": truncation}})
    data = prepare(original, params, discounting=.8, reward_scaling=2., gae_lambda=.7,
                   normalize_advantage=False)
    extras = data.extras["policy_extras"]
    network = analytic_network()
    ids = original.extras["policy_extras"]["policy_id"]
    values = network.value_network.apply(None, params.value, original.observation, ids,
                                        policy_embeddings=params.policy["policy_embeddings"])
    last_obs = jax.tree_util.tree_map(lambda value: value[:, -1], original.next_observation)
    bootstrap = network.value_network.apply(None, params.value, last_obs, ids[:, -1],
                                            policy_embeddings=params.policy["policy_embeddings"])
    expected_target, expected_advantage = ppo_losses.compute_gae(
        truncation.T, ((1 - original.discount) * (1 - truncation)).T,
        (original.reward * 2).T, values.T, bootstrap, lambda_=.7, discount=.8,
    )
    np.testing.assert_allclose(extras[losses.TARGET_VALUE][:6], expected_target.T)
    np.testing.assert_allclose(extras[losses.ADVANTAGE][:6], expected_advantage.T)
    off_baseline = data.observation["state"][6:, :, 0]
    expected_off_target = jnp.stack((
        2 + .8 * data.next_observation["state"][6:, 0, 0],
        jnp.full((2,), 2.), off_baseline[:, 2]), axis=-1)
    np.testing.assert_allclose(extras[losses.TARGET_VALUE][6:], expected_off_target)
    np.testing.assert_allclose(extras[losses.ADVANTAGE][6:], expected_off_target - off_baseline)
    np.testing.assert_array_equal(extras[losses.ADVANTAGE][:, 2], 0.)


def test_advantages_normalized_once_over_augmented_batch_then_truncation_masked():
    original = rollout()
    original = original._replace(reward=jnp.arange(18, dtype=jnp.float32).reshape(6, 3) / 10,
        extras={**original.extras, "state_extras": {"truncation": jnp.zeros((6, 3)).at[:, 2].set(1)}})
    raw = prepare(original, normalize_advantage=False)
    normalized = prepare(original, normalize_advantage=True)
    advantages = np.asarray(raw.extras["policy_extras"][losses.ADVANTAGE])
    expected = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    expected[:, 2] = 0
    np.testing.assert_allclose(normalized.extras["policy_extras"][losses.ADVANTAGE], expected, atol=1e-6)
    subset = jax.tree_util.tree_map(lambda value: value[:1], normalized)
    params = analytic_params()
    _, metrics = losses.compute_sapg_loss(params, None, subset, jax.random.PRNGKey(0),
        analytic_network(), num_policies=3, normalize_advantage=True, entropy_cost=0.)
    np.testing.assert_allclose(metrics["policy_loss"], -expected[0].mean(), atol=1e-6)


def test_all_prepared_targets_are_frozen_against_parameter_differentiation():
    params = analytic_params(slope=.5, embedding_scale=.2)
    original = rollout(params=params)
    frozen_keys = (losses.TARGET_VALUE, losses.ADVANTAGE, losses.OLD_TARGET_LOG_PROB,
                   losses.IMPORTANCE_WEIGHT, losses.ACTION_STD)

    def target_sum(parameters):
        prepared = prepare(original, parameters)
        return sum(jnp.sum(prepared.extras["policy_extras"][key]) for key in frozen_keys)

    gradients = jax.grad(target_sum)(params)
    assert all(np.count_nonzero(leaf) == 0 for leaf in jax.tree_util.tree_leaves(gradients))
    prepared = prepare(original, params)
    frozen_before = {key: np.array(prepared.extras["policy_extras"][key]) for key in frozen_keys}
    for delta in (0., .3):
        updated = params.replace(policy={**params.policy, "offsets": params.policy["offsets"] + delta})
        losses.compute_sapg_loss(updated, None, prepared, jax.random.PRNGKey(0),
                                 analytic_network(), num_policies=3)
    for key in frozen_keys:
        np.testing.assert_array_equal(prepared.extras["policy_extras"][key], frozen_before[key])


def test_chunked_preparation_matches_other_chunk_sizes_and_never_applies_full_rollout():
    params = analytic_params(slope=.3, embedding_scale=.7)
    original = rollout((0, 0, 1, 1, 2, 2) * 2, time=4, params=params)
    chunked = jax.jit(functools.partial(prepare, params=params,
        network=analytic_network(maximum_chunk=3), preparation_chunk_size=3))(original)
    padded = prepare(original, params, preparation_chunk_size=5)
    for first, second in zip(jax.tree_util.tree_leaves(chunked), jax.tree_util.tree_leaves(padded)):
        np.testing.assert_allclose(first, second, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(chunked.extras["policy_extras"][losses.ACTION_STD], 1.)


def test_critic_gradient_reaches_actor_owned_shared_embedding_with_no_actor_loss():
    network = make_sapg_networks(2, 1, num_policies=2, embedding_dim=1,
        policy_hidden_layer_sizes=(2,), value_hidden_layer_sizes=(2,), activation=lambda x: x)
    actor = network.policy_network.init(jax.random.PRNGKey(0))
    critic = network.value_network.init(jax.random.PRNGKey(1))
    # Isolate a nonzero path from the shared embedding through the critic.
    critic["params"]["hidden_0"]["kernel"] = jnp.asarray([[0., 0.], [0., 0.], [1., 0.]])
    critic["params"]["hidden_0"]["bias"] = jnp.zeros(2)
    critic["params"]["hidden_1"]["kernel"] = jnp.asarray([[1.], [0.]])
    critic["params"]["hidden_1"]["bias"] = jnp.zeros(1)
    params = ppo_losses.PPONetworkParams(policy=actor, value=critic)
    data = frozen_batch(0., target_value=2., target_id=1)
    grad = jax.grad(lambda parameters: losses.compute_sapg_loss(
        parameters, None, data, jax.random.PRNGKey(4), network,
        num_policies=2, entropy_cost=0.)[0])(params)
    assert abs(float(grad.policy["policy_embeddings"][1, 0])) > .5
    np.testing.assert_array_equal(grad.policy["policy_embeddings"][0], 0.)
    assert "policy_embeddings" not in grad.value


@pytest.mark.parametrize("ids", [(0, 0, 1, 1, 1, 2), (0, 0, 1, 1, 3, 3)])
def test_invalid_equal_group_contract_rejected(ids):
    with pytest.raises(ValueError, match="equal policy groups"):
        prepare(rollout(ids))


def test_traced_invalid_groups_do_not_silently_use_nonzero_padding():
    prepared = jax.jit(prepare)(rollout((0, 0, 1, 1, 1, 2)))
    assert np.all(np.isnan(prepared.extras["policy_extras"][losses.ADVANTAGE]))


def test_optional_entropy_coefficients_index_target_policy_not_behavior():
    params = analytic_params(2)
    data = prepare(rollout((0, 1), time=1), params, num_policies=2)
    _, metrics = losses.compute_sapg_loss(params, None, data, jax.random.PRNGKey(0),
        analytic_network(), num_policies=2, entropy_costs=(.01, .04))
    np.testing.assert_allclose(metrics["entropy_loss"], -(.01 + .04 + .01) / 3)
