"""Probability and checkpoint-shape regressions for bounded added-joint noise."""

import json

from brax.training import distribution
from brax.training.agents.ppo import networks
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cat_ppo.learning.policy.ppo.wholebody_distribution import (
    DISTRIBUTION_KIND, WholeBodyNormalTanhDistribution, make_ppo_networks,
)


def _parameters():
    means = jnp.linspace(-1.5, 1.5, 3 * 29).reshape(3, 29)
    raw_scales = jnp.linspace(-20, 20, 3 * 29).reshape(3, 29)
    return jnp.concatenate([means, raw_scales], axis=-1)


def _leg_parameters(parameters):
    return jnp.concatenate([parameters[..., :12], parameters[..., 29:41]], axis=-1)


def test_bounded_scales_preserve_all_means_and_exact_native_leg_marginals():
    parameters = _parameters()
    repaired = WholeBodyNormalTanhDistribution(29)
    # Return arrays rather than a Python distribution from the compiled function.
    bounded = jax.jit(lambda p: (repaired.create_dist(p).loc, repaired.create_dist(p).scale))
    loc, scale = bounded(parameters)
    native = distribution.NormalTanhDistribution(29).create_dist(parameters)
    np.testing.assert_array_equal(loc, native.loc)
    np.testing.assert_array_equal(scale[..., :12], native.scale[..., :12])
    upper = np.asarray(scale[..., 12:])
    assert upper.min() == np.float32(.02)
    assert upper.max() == np.float32(.10)
    assert np.all((upper >= np.float32(.02)) & (upper <= np.float32(.10)))
    assert np.max(np.asarray(scale[..., :12])) > .1  # Legs were not silently clipped.


def test_existing_added_output_initialization_remains_point_zero_five():
    means = jnp.zeros((2, 29))
    raw_std = jnp.full_like(means, jnp.log(jnp.expm1(.05 - .001)))
    parameters = jnp.concatenate([means, raw_std], axis=-1)
    repaired = WholeBodyNormalTanhDistribution(29)
    np.testing.assert_allclose(repaired.create_dist(parameters).scale, .05, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(repaired.mode(parameters), 0)


def test_sampling_and_log_probability_use_same_bounded_gaussian():
    repaired = WholeBodyNormalTanhDistribution(29)
    parameters = _parameters()
    bounded = repaired.create_dist(parameters)
    key = jax.random.PRNGKey(81)
    raw_actions = repaired.sample_no_postprocessing(parameters, key)
    np.testing.assert_array_equal(raw_actions, bounded.sample(seed=key))
    np.testing.assert_array_equal(repaired.sample(parameters, key), jnp.tanh(raw_actions))
    expected_log_prob = jnp.sum(
        bounded.log_prob(raw_actions)
        - distribution.TanhBijector().forward_log_det_jacobian(raw_actions),
        axis=-1,
    )
    np.testing.assert_allclose(repaired.log_prob(parameters, raw_actions), expected_log_prob,
                               rtol=0, atol=0)
    # Recomputing PPO old/current log probabilities without a parameter update
    # yields ratio one even when raw upper scales greatly exceed the cap.
    recomputed = jax.jit(repaired.log_prob)(parameters, raw_actions)
    np.testing.assert_allclose(jnp.exp(recomputed - expected_log_prob), 1, atol=3e-5)


def test_entropy_is_exact_native_twelve_leg_entropy_with_zero_upper_gradients():
    parameters = _parameters()
    repaired = WholeBodyNormalTanhDistribution(29)
    native_legs = distribution.NormalTanhDistribution(12)
    key = jax.random.PRNGKey(7)
    expected = native_legs.entropy(_leg_parameters(parameters), key)
    np.testing.assert_array_equal(repaired.entropy(parameters, key), expected)
    gradient = jax.grad(lambda p: repaired.entropy(p, key).sum())(parameters)
    expected_gradient = jax.grad(
        lambda p: native_legs.entropy(_leg_parameters(p), key).sum(),
    )(parameters)
    np.testing.assert_array_equal(gradient, expected_gradient)
    np.testing.assert_array_equal(gradient[..., 12:29], 0)
    np.testing.assert_array_equal(gradient[..., 41:58], 0)
    assert np.any(np.asarray(gradient[..., :12]) != 0)
    assert np.any(np.asarray(gradient[..., 29:41]) != 0)


def test_factory_preserves_native_parameter_values_shapes_and_forward_outputs():
    kwargs = dict(
        observation_size={"state": (222,), "privileged_state": (310,)},
        action_size=29, policy_hidden_layer_sizes=(16, 8),
        value_hidden_layer_sizes=(32, 16),
        policy_obs_key="state", value_obs_key="privileged_state",
    )
    native, repaired = networks.make_ppo_networks(**kwargs), make_ppo_networks(**kwargs)
    observations = {"state": jnp.ones((3, 222)), "privileged_state": jnp.ones((3, 310))}
    for component in ("policy_network", "value_network"):
        native_network, repaired_network = getattr(native, component), getattr(repaired, component)
        native_params = native_network.init(jax.random.PRNGKey(5))
        repaired_params = repaired_network.init(jax.random.PRNGKey(5))
        assert jax.tree.structure(native_params) == jax.tree.structure(repaired_params)
        for left, right in zip(jax.tree.leaves(native_params), jax.tree.leaves(repaired_params)):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(
            native_network.apply(None, native_params, observations),
            repaired_network.apply(None, repaired_params, observations),
        )
    actor_params = repaired.policy_network.init(jax.random.PRNGKey(5))
    assert repaired.policy_network.apply(None, actor_params, observations).shape == (3, 58)
    assert repaired.parametric_action_distribution.param_size == native.parametric_action_distribution.param_size
    config = json.loads(json.dumps(repaired.parametric_action_distribution.config))
    assert config == {
        "kind": DISTRIBUTION_KIND, "event_size": 29, "leg_action_count": 12,
        "upper_std_min": .02, "upper_std_max": .10, "upper_entropy_weight": 0.,
        "native_min_std": .001, "native_var_scale": 1,
    }


@pytest.mark.parametrize("kwargs", [
    {"event_size": 12}, {"leg_action_count": 0}, {"leg_action_count": 29},
    {"upper_std_min": 0}, {"upper_std_max": .01}, {"upper_std_max": float("inf")},
    {"upper_entropy_weight": -1}, {"upper_entropy_weight": float("nan")},
])
def test_rejects_ambiguous_or_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        WholeBodyNormalTanhDistribution(**dict({"event_size": 29}, **kwargs))


def test_wrong_parameter_width_fails_before_sampling():
    repaired = WholeBodyNormalTanhDistribution(29)
    with pytest.raises(ValueError, match="58 distribution parameters"):
        repaired.create_dist(jnp.zeros((2, 24)))
