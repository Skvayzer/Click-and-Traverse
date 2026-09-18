"""Conditioned-policy parity, shared gradients and ordinary CAT export."""

import copy

from brax.training import types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks
from flax import core, serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cat_ppo.learning.policy.sapg.networks import (
    collapse_policy_params,
    expand_ppo_params,
    make_inference_fn,
    make_sapg_networks,
)


def _fixture(normalize=False, frozen=False):
    sizes = {"state": (222,), "privileged_state": (310,)}
    kwargs = dict(policy_hidden_layer_sizes=(11, 7), value_hidden_layer_sizes=(13, 9),
                  policy_obs_key="state", value_obs_key="privileged_state")
    preprocessor = running_statistics.normalize if normalize else types.identity_observation_preprocessor
    kwargs["preprocess_observations_fn"] = preprocessor
    base = ppo_networks.make_ppo_networks(sizes, 29, **kwargs)
    conditioned = make_sapg_networks(sizes, 29, num_policies=6, embedding_dim=16, **kwargs)
    normalizer = running_statistics.init_state({
        name: specs.Array(shape, jnp.dtype("float32")) for name, shape in sizes.items()
    }).replace(mean={name: jnp.full(shape, .3) for name, shape in sizes.items()},
               std={name: jnp.full(shape, 1.7) for name, shape in sizes.items()})
    params = (normalizer, base.policy_network.init(jax.random.PRNGKey(1)),
              base.value_network.init(jax.random.PRNGKey(2)))
    if frozen:
        params = (normalizer, core.freeze(params[1]), core.freeze(params[2]))
    observations = {
        name: jax.random.normal(jax.random.PRNGKey(i + 3), (6, *shape))
        for i, (name, shape) in enumerate(sizes.items())
    }
    return base, conditioned, params, observations


def _learned_conditioning(params):
    """Make conditioning nonzero to test actual export/gradient semantics."""
    normalizer, actor, value = params
    actor, value = core.unfreeze(actor), core.unfreeze(value)
    for i, tree in enumerate((actor, value)):
        kernel = tree["params"]["hidden_0"]["kernel"]
        tree["params"]["hidden_0"]["kernel"] = kernel.at[-16:].set(
            .2 * jax.random.normal(jax.random.PRNGKey(30 + i), kernel[-16:].shape)
        )
    actor["policy_embeddings"] = jax.random.normal(jax.random.PRNGKey(32), (6, 16)) * .3
    return normalizer, actor, value


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_expansion_preserves_every_policy_mean_scale_and_critic(normalize, frozen):
    base, networks, original, obs = _fixture(normalize, frozen)
    expanded = expand_ppo_params(original, num_policies=6, embedding_dim=16, seed=14)
    norm, actor, value = expanded
    assert norm is original[0]
    assert isinstance(actor, core.FrozenDict) == frozen
    assert isinstance(value, core.FrozenDict) == frozen
    assert set(actor) == {"params", "policy_embeddings"}
    assert set(value) == {"params"}
    assert actor["params"]["hidden_0"]["kernel"].shape == (238, 11)
    assert value["params"]["hidden_0"]["kernel"].shape == (326, 13)
    assert original[1]["params"]["hidden_0"]["kernel"].shape == (222, 11)
    assert original[2]["params"]["hidden_0"]["kernel"].shape == (310, 13)
    ids = jnp.arange(6)
    np.testing.assert_allclose(
        networks.policy_network.apply(norm, actor, obs, ids),
        base.policy_network.apply(original[0], original[1], obs), rtol=2e-6, atol=2e-6,
    )
    np.testing.assert_allclose(
        networks.value_network.apply(norm, value, obs, ids, policy_embeddings=actor["policy_embeddings"]),
        base.value_network.apply(original[0], original[2], obs), rtol=2e-6, atol=2e-6,
    )
    assert np.unique(np.asarray(actor["policy_embeddings"]), axis=0).shape[0] == 6
    # The unchanged table/state must survive exact resume, with no re-expansion.
    resumed = expand_ppo_params(expanded, 6, 16, seed=999)
    assert resumed is expanded
    restored = serialization.from_bytes(expanded, serialization.to_bytes(expanded))
    assert jax.tree.structure(restored) == jax.tree.structure(expanded)
    for before, after in zip(jax.tree.leaves(expanded), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(before, after)
    with pytest.raises(ValueError, match="shape"):
        expand_ppo_params(expanded, num_policies=7)


def test_factory_initialization_matches_brax_with_same_seed_and_normalizer():
    base, networks, original, obs = _fixture(normalize=True)
    actor = networks.policy_network.init(jax.random.PRNGKey(1))
    value = networks.value_network.init(jax.random.PRNGKey(2))
    for tree, reference in ((actor, original[1]), (value, original[2])):
        np.testing.assert_array_equal(tree["params"]["hidden_0"]["kernel"][:-16],
                                      reference["params"]["hidden_0"]["kernel"])
        np.testing.assert_array_equal(tree["params"]["hidden_0"]["kernel"][-16:], 0)
    for policy_id in range(6):
        np.testing.assert_allclose(
            networks.policy_network.apply(original[0], actor, obs, policy_id),
            base.policy_network.apply(original[0], original[1], obs), rtol=2e-6, atol=2e-6,
        )
        np.testing.assert_allclose(
            networks.value_network.apply(original[0], value, obs, policy_id,
                                         policy_embeddings=actor["policy_embeddings"]),
            base.value_network.apply(original[0], original[2], obs), rtol=2e-6, atol=2e-6,
        )


def test_actor_and_critic_gradients_update_the_same_table_only():
    _, networks, original, obs = _fixture()
    norm, actor, value = _learned_conditioning(expand_ppo_params(original))

    def actor_loss(actor_params):
        return networks.policy_network.apply(norm, actor_params, obs, 2).sum()

    def value_loss(actor_params):
        return networks.value_network.apply(
            norm, value, obs, 2, policy_embeddings=actor_params["policy_embeddings"]
        ).sum()

    actor_grad = jax.grad(actor_loss)(actor)
    value_grad = jax.grad(value_loss)(actor)
    combined_grad = jax.grad(lambda p: actor_loss(p) + value_loss(p))(actor)
    for grad in (actor_grad, value_grad):
        assert np.linalg.norm(grad["policy_embeddings"][2]) > 0
        np.testing.assert_array_equal(np.delete(grad["policy_embeddings"], 2, axis=0), 0)
    np.testing.assert_allclose(combined_grad["policy_embeddings"],
                               actor_grad["policy_embeddings"] + value_grad["policy_embeddings"], atol=1e-6)
    for leaf in jax.tree.leaves(value_grad["params"]):
        np.testing.assert_array_equal(leaf, 0)
    assert "policy_embeddings" not in value
    with pytest.raises(ValueError, match="actor's policy_embeddings"):
        networks.value_network.apply(norm, value, obs)


def test_zero_init_preserves_outputs_but_allows_conditioning_rows_to_start_learning():
    _, networks, original, obs = _fixture()
    norm, actor, _ = expand_ppo_params(original)
    grad = jax.grad(lambda p: networks.policy_network.apply(norm, p, obs, 1).sum())(actor)
    np.testing.assert_array_equal(grad["policy_embeddings"], 0)
    assert np.linalg.norm(grad["params"]["hidden_0"]["kernel"][-16:]) > 0


@pytest.mark.parametrize("normalize", [False, True])
def test_collapse_preserves_learned_conditioning_for_all_policies_and_layers(normalize):
    base, networks, original, obs = _fixture(normalize)
    conditioned = _learned_conditioning(expand_ppo_params(original))
    snapshot = copy.deepcopy(conditioned)
    norm, actor, value = conditioned
    for policy_id in range(6):
        collapsed = collapse_policy_params(conditioned, policy_id)
        assert collapsed[0] is norm
        assert set(collapsed[1]) == {"params"}
        assert set(collapsed[2]) == {"params"}
        assert jax.tree.structure(collapsed) == jax.tree.structure(original)
        for before, after in zip(jax.tree.leaves(original), jax.tree.leaves(collapsed)):
            assert before.shape == after.shape
        np.testing.assert_allclose(
            base.policy_network.apply(collapsed[0], collapsed[1], obs),
            networks.policy_network.apply(norm, actor, obs, policy_id), rtol=3e-6, atol=3e-6,
        )
        np.testing.assert_allclose(
            base.value_network.apply(collapsed[0], collapsed[2], obs),
            networks.value_network.apply(norm, value, obs, policy_id,
                                         policy_embeddings=actor["policy_embeddings"]),
            rtol=3e-6, atol=3e-6,
        )
        for before, after in ((actor, collapsed[1]), (value, collapsed[2])):
            for name in before["params"]:
                if name != "hidden_0":
                    for leaf, exported in zip(jax.tree.leaves(before["params"][name]),
                                              jax.tree.leaves(after["params"][name])):
                        np.testing.assert_array_equal(leaf, exported)
    for before, after in zip(jax.tree.leaves(snapshot), jax.tree.leaves(conditioned)):
        np.testing.assert_array_equal(before, after)


def test_inference_rollout_ids_broadcast_across_time_and_match_ppo_likelihood():
    _, networks, original, obs = _fixture()
    params = _learned_conditioning(expand_ppo_params(original))
    time_obs = jax.tree.map(lambda x: jnp.stack((x, x + .1, x + .2)), obs)
    ids = jnp.arange(6)
    infer = make_inference_fn(networks, policy_ids=ids)(params)
    actions, extras = jax.jit(infer)(time_obs, jax.random.PRNGKey(51))
    assert actions.shape == (3, 6, 29)
    assert extras["raw_action"].shape == (3, 6, 29)
    assert extras["log_prob"].shape == (3, 6)
    assert extras["policy_id"].shape == (3, 6)
    np.testing.assert_array_equal(extras["policy_id"], np.broadcast_to(np.arange(6), (3, 6)))
    logits = networks.policy_network.apply(params[0], params[1], time_obs, policy_ids=ids)
    np.testing.assert_allclose(extras["log_prob"],
        networks.parametric_action_distribution.log_prob(logits, extras["raw_action"]), atol=1e-5)
    np.testing.assert_allclose(actions, jnp.tanh(extras["raw_action"]), atol=1e-6)
    deploy = make_inference_fn(networks)(params, deterministic=True)
    deterministic, deterministic_extras = deploy(time_obs, jax.random.PRNGKey(9))
    collapsed = collapse_policy_params(params)
    base, _, _, _ = _fixture()
    reference, _ = ppo_networks.make_inference_fn(base)(collapsed, deterministic=True)(
        time_obs, jax.random.PRNGKey(10))
    np.testing.assert_allclose(deterministic, reference, rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(deterministic_extras["policy_id"], 0)


def test_dynamic_jit_ids_and_single_vector_observations():
    _, networks, original, obs = _fixture()
    norm, actor, value = _learned_conditioning(expand_ppo_params(original))
    dynamic = jax.jit(lambda ids: networks.policy_network.apply(norm, actor, obs, ids))
    ids = jnp.arange(6, dtype=jnp.int32)
    np.testing.assert_allclose(dynamic(ids), networks.policy_network.apply(norm, actor, obs, ids), atol=1e-6)
    single_obs = jax.tree.map(lambda x: x[0], obs)
    assert networks.policy_network.apply(norm, actor, single_obs, 3).shape == (58,)
    assert networks.value_network.apply(norm, value, single_obs, 3,
                                         policy_embeddings=actor["policy_embeddings"]).shape == ()
    # Runtime invalid IDs cannot silently alias the last row through gather clipping.
    assert np.isnan(np.asarray(dynamic(ids.at[0].set(-1)))[0]).all()


@pytest.mark.parametrize("ids", [1.0, True, -1, 6, jnp.array([1., 2.]), jnp.array([True, False])])
def test_invalid_policy_ids_fail_before_network_evaluation(ids):
    _, networks, original, obs = _fixture()
    norm, actor, _ = expand_ppo_params(original)
    with pytest.raises(ValueError, match="policy_ids"):
        networks.policy_network.apply(norm, actor, obs, ids)


def test_wrong_batch_shape_table_shape_and_invalid_factory_dimensions_fail():
    _, networks, original, obs = _fixture()
    norm, actor, value = expand_ppo_params(original)
    with pytest.raises(ValueError, match="broadcast"):
        networks.policy_network.apply(norm, actor, obs, jnp.zeros((2, 6), jnp.int32))
    with pytest.raises(ValueError, match="shape"):
        networks.value_network.apply(norm, value, obs, policy_embeddings=jnp.zeros((7, 16)))
    for kwargs in ({"num_policies": 0}, {"num_policies": True}, {"embedding_dim": 1.5}):
        with pytest.raises(ValueError, match="positive integer"):
            make_sapg_networks(222, 29, **kwargs)
    with pytest.raises(ValueError, match="broadcast"):
        collapse_policy_params((norm, actor, value), jnp.array([0, 1]))
    with pytest.raises(ValueError, match="no SAPG"):
        collapse_policy_params(original)


def test_array_observation_factory_uses_default_brax_kwargs():
    networks = make_sapg_networks(5, 2, num_policies=3, embedding_dim=4)
    params = (None, networks.policy_network.init(jax.random.PRNGKey(1)),
              networks.value_network.init(jax.random.PRNGKey(2)))
    observations = jnp.ones((3, 5))
    actor = networks.policy_network.apply(None, params[1], observations, jnp.arange(3))
    critic = networks.value_network.apply(None, params[2], observations, jnp.arange(3),
                                           policy_embeddings=params[1]["policy_embeddings"])
    assert actor.shape == (3, 4)
    assert critic.shape == (3,)
