"""Shared-policy conditioning without adding robot sensor observations.

The actor owns the only ``policy_embeddings`` table, outside its Flax ``params``
collection. Both MLPs append the selected row *after* observation preprocessing.
The critic must receive that actor-owned table explicitly, so value gradients
can update the same embedding through the joint actor/critic loss.

This preserves CAT's separate actor/critic trunks and state-dependent Gaussian
scale outputs. Initial embedding input weights are zero; the original policy
therefore remains unchanged for every ID. The ordinary PPO export folds one
selected embedding into both first-layer biases, requiring no extra sensors or
context input in existing viewers. Full SAPG checkpoints retain the table.
"""

from collections.abc import Mapping, Sequence
import numbers

from brax.training import distribution, networks as brax_networks, types
from brax.training.agents.ppo import networks as ppo_networks
from flax import core, linen
import jax
import jax.numpy as jnp
import numpy as np


EMBEDDINGS_KEY = "policy_embeddings"
EMBEDDING_INITIAL_STD = 0.01


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _mutable_tree(params):
    if not isinstance(params, Mapping):
        raise ValueError("Expected a Flax parameter mapping")
    return core.unfreeze(params)


def _restore_tree_type(original, mutable):
    return core.freeze(mutable) if isinstance(original, core.FrozenDict) else mutable


def _first_layer(params):
    try:
        layer = params["params"]["hidden_0"]
        kernel, bias = layer["kernel"], layer["bias"]
    except (KeyError, TypeError) as error:
        raise ValueError("Expected Brax MLP params/hidden_0 kernel and bias") from error
    if kernel.ndim != 2 or bias.shape != (kernel.shape[1],):
        raise ValueError("Invalid Brax MLP first-layer shapes")
    return layer


def _extend_inputs(params, embedding_dim):
    mutable = _mutable_tree(params)
    layer = _first_layer(mutable)
    kernel = layer["kernel"]
    layer["kernel"] = jnp.concatenate(
        (kernel, jnp.zeros((embedding_dim, kernel.shape[1]), dtype=kernel.dtype)), axis=0
    )
    return _restore_tree_type(params, mutable)


def _new_embeddings(key, num_policies, embedding_dim, dtype):
    return EMBEDDING_INITIAL_STD * jax.random.normal(
        key, (num_policies, embedding_dim), dtype=dtype
    )


def _check_table(table, num_policies=None, embedding_dim=None):
    if not hasattr(table, "shape") or table.ndim != 2 or min(table.shape) <= 0:
        raise ValueError("policy_embeddings must be a nonempty rank-2 array")
    if num_policies is not None and table.shape != (num_policies, embedding_dim):
        raise ValueError(
            f"policy_embeddings shape must be {(num_policies, embedding_dim)}, got {table.shape}"
        )
    if not jnp.issubdtype(table.dtype, jnp.floating):
        raise ValueError("policy_embeddings must have floating-point dtype")
    return table


def _broadcast_policy_ids(policy_ids, leading_shape, num_policies):
    """Validate concrete IDs; traced integer IDs remain usable inside JIT."""
    if isinstance(policy_ids, numbers.Number):
        if isinstance(policy_ids, (bool, np.bool_)) or not isinstance(policy_ids, numbers.Integral):
            raise ValueError("policy_ids must contain integers, not floats or booleans")
        if not 0 <= policy_ids < num_policies:
            raise ValueError(f"policy_ids must lie in [0, {num_policies})")
    ids = jnp.asarray(policy_ids)
    if not jnp.issubdtype(ids.dtype, jnp.integer):
        raise ValueError("policy_ids must contain integers, not floats or booleans")
    if not isinstance(ids, jax.core.Tracer):
        concrete = np.asarray(ids)
        if np.any(concrete < 0) or np.any(concrete >= num_policies):
            raise ValueError(f"policy_ids must lie in [0, {num_policies})")
    try:
        return jnp.broadcast_to(ids, leading_shape).astype(jnp.int32)
    except ValueError as error:
        raise ValueError(
            f"policy_ids shape {ids.shape} cannot broadcast to observation batch {leading_shape}"
        ) from error


def _condition(features, table, policy_ids):
    ids = _broadcast_policy_ids(policy_ids, features.shape[:-1], table.shape[0])
    selected = table[jnp.clip(ids, 0, table.shape[0] - 1)]
    # A bad runtime ID must not silently select another policy through JAX's
    # out-of-bounds gather clipping. Concrete invalid IDs raise above; traced
    # invalid IDs produce nonfinite outputs for existing training health checks.
    valid = (ids >= 0) & (ids < table.shape[0])
    selected = jnp.where(valid[..., None], selected, jnp.nan)
    return jnp.concatenate((features, selected.astype(features.dtype)), axis=-1)


def expand_ppo_params(params, num_policies=6, embedding_dim=16, seed=0):
    """Add SAPG conditioning to an already adapted ``(normalizer, actor, value)``.

    Dense weights/biases and normalizer state are preserved. Added input rows
    are zero and the sole table is N(0, 0.01**2). Matching already-conditioned
    parameters are returned unchanged, making exact-resume calls idempotent;
    incompatible policy counts/embedding sizes fail rather than reshape state.
    This does not convert optimizer state or the original 12-action CAT model.
    """
    num_policies = _positive_integer(num_policies, "num_policies")
    embedding_dim = _positive_integer(embedding_dim, "embedding_dim")
    if not isinstance(params, (tuple, list)) or len(params) != 3:
        raise ValueError("Expected (normalizer, actor, value) parameters")
    normalizer, actor, value = params
    actor_first, value_first = _first_layer(actor), _first_layer(value)
    if EMBEDDINGS_KEY in value:
        raise ValueError("The critic must use the actor's shared policy_embeddings table")
    if EMBEDDINGS_KEY in actor:
        _check_table(actor[EMBEDDINGS_KEY], num_policies, embedding_dim)
        if min(actor_first["kernel"].shape[0], value_first["kernel"].shape[0]) <= embedding_dim:
            raise ValueError("Conditioned MLP inputs must include robot observations")
        return params
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")
    extended_actor = _mutable_tree(_extend_inputs(actor, embedding_dim))
    extended_actor[EMBEDDINGS_KEY] = _new_embeddings(
        jax.random.PRNGKey(int(seed)), num_policies, embedding_dim, actor_first["kernel"].dtype
    )
    return (
        normalizer,
        _restore_tree_type(actor, extended_actor),
        _extend_inputs(value, embedding_dim),
    )


def collapse_policy_params(params, policy_id=0):
    """Export one conditioned actor/critic as ordinary PPO parameters.

    For a first kernel ``[W_obs; W_phi]``, replace its bias by
    ``b + phi[policy_id] @ W_phi`` and retain ``W_obs``. No learned contribution
    is discarded, including after the embedding and conditioning rows change.
    The input tree is not mutated; the normalizer is returned unchanged.
    """
    if not isinstance(params, (tuple, list)) or len(params) != 3:
        raise ValueError("Expected (normalizer, actor, value) parameters")
    normalizer, actor, value = params
    if EMBEDDINGS_KEY not in actor:
        raise ValueError("Actor has no SAPG policy_embeddings table")
    if EMBEDDINGS_KEY in value:
        raise ValueError("The critic must use the actor's shared policy_embeddings table")
    table = _check_table(actor[EMBEDDINGS_KEY])
    ids = _broadcast_policy_ids(policy_id, (), table.shape[0])
    embedding = table[ids]
    width = table.shape[1]

    def collapse_one(original):
        mutable = _mutable_tree(original)
        layer = _first_layer(mutable)
        kernel = layer["kernel"]
        if kernel.shape[0] <= width:
            raise ValueError("Conditioned MLP inputs must include robot observations")
        layer["bias"] = layer["bias"] + embedding @ kernel[-width:]
        layer["kernel"] = kernel[:-width]
        mutable.pop(EMBEDDINGS_KEY, None)
        return _restore_tree_type(original, mutable)

    return normalizer, collapse_one(actor), collapse_one(value)


def make_sapg_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: brax_networks.ActivationFn = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    *,
    num_policies: int = 6,
    embedding_dim: int = 16,
) -> ppo_networks.PPONetworks:
    """Brax-compatible factory with one explicit shared conditioning table.

    ``policy_network.apply(norm, actor, obs, policy_ids=0)`` uses the actor's
    table. ``value_network.apply(norm, value, obs, policy_ids=0,
    policy_embeddings=actor['policy_embeddings'])`` requires that same table.
    IDs are scalar integers or arrays broadcastable to observation leading
    dimensions; e.g. an environment-ID vector broadcasts across rollout time.
    Normalizer and observation sizes remain the original robot-only sizes.
    """
    num_policies = _positive_integer(num_policies, "num_policies")
    embedding_dim = _positive_integer(embedding_dim, "embedding_dim")
    base = ppo_networks.make_ppo_networks(
        observation_size, action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation, policy_obs_key=policy_obs_key, value_obs_key=value_obs_key,
    )
    policy_module = brax_networks.MLP(
        layer_sizes=list(policy_hidden_layer_sizes) + [2 * action_size], activation=activation
    )
    value_module = brax_networks.MLP(
        layer_sizes=list(value_hidden_layer_sizes) + [1], activation=activation
    )

    def processed_features(obs, processor_params, key):
        processed = preprocess_observations_fn(obs, processor_params)
        return processed[key] if isinstance(processed, Mapping) else processed

    def init_policy(key):
        actor = _mutable_tree(_extend_inputs(base.policy_network.init(key), embedding_dim))
        actor[EMBEDDINGS_KEY] = _new_embeddings(
            jax.random.fold_in(key, 1), num_policies, embedding_dim,
            _first_layer(actor)["kernel"].dtype,
        )
        return actor

    def apply_policy(processor_params, policy_params, obs, policy_ids=0):
        table = _check_table(policy_params[EMBEDDINGS_KEY], num_policies, embedding_dim)
        features = processed_features(obs, processor_params, policy_obs_key)
        mlp_params = {key: value for key, value in policy_params.items() if key != EMBEDDINGS_KEY}
        return policy_module.apply(mlp_params, _condition(features, table, policy_ids))

    def apply_value(processor_params, value_params, obs, policy_ids=0, *, policy_embeddings=None):
        if policy_embeddings is None:
            raise ValueError("Pass the actor's policy_embeddings to value_network.apply")
        if EMBEDDINGS_KEY in value_params:
            raise ValueError("The critic must not own an independent policy_embeddings table")
        table = _check_table(policy_embeddings, num_policies, embedding_dim)
        features = processed_features(obs, processor_params, value_obs_key)
        output = value_module.apply(value_params, _condition(features, table, policy_ids))
        return jnp.squeeze(output, axis=-1)

    return ppo_networks.PPONetworks(
        policy_network=brax_networks.FeedForwardNetwork(init=init_policy, apply=apply_policy),
        value_network=brax_networks.FeedForwardNetwork(
            init=lambda key: _extend_inputs(base.value_network.init(key), embedding_dim),
            apply=apply_value,
        ),
        parametric_action_distribution=distribution.NormalTanhDistribution(event_size=action_size),
    )


def make_inference_fn(networks: ppo_networks.PPONetworks, policy_ids=0):
    """Brax policy factory; rollout extras also carry broadcast integer IDs.

    Stochastic extras are ``raw_action``, ``log_prob`` and ``policy_id``.
    Deterministic inference returns the distribution mode and ``policy_id``.
    The default is policy 0, the deployment/leader policy.
    """
    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        def policy(observations: types.Observation, key_sample: types.PRNGKey):
            logits = networks.policy_network.apply(params[0], params[1], observations, policy_ids=policy_ids)
            ids = _broadcast_policy_ids(
                policy_ids, logits.shape[:-1], params[1][EMBEDDINGS_KEY].shape[0]
            )
            action_distribution = networks.parametric_action_distribution
            if deterministic:
                return action_distribution.mode(logits), {"policy_id": ids}
            raw_actions = action_distribution.sample_no_postprocessing(logits, key_sample)
            return action_distribution.postprocess(raw_actions), {
                "raw_action": raw_actions,
                "log_prob": action_distribution.log_prob(logits, raw_actions),
                "policy_id": ids,
            }
        return policy
    return make_policy
