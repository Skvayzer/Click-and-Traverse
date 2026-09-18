"""Frozen-target SAPG updates using CAT's PPO networks and transition format.

Every update contains all M equally sized on-policy groups and one randomly
chosen follower group relabeled for the leader. The latter uses an old-policy
importance weight outside PPO's clipped surrogate, and a one-step leader value
target. A plain mean therefore gives each of the M+1 blocks equal weight.

Target preparation is separate from minibatch optimization: both advantages
and old target-policy likelihoods remain fixed across all optimizer passes.
Large rollout MLP evaluations are mapped over small trajectory chunks; only
scalar outputs survive each chunk, rather than full-rollout hidden activations.
"""

import numbers
from typing import Any

from brax.training import types
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jnp
import numpy as np


TARGET_VALUE = "sapg_target_value"
ADVANTAGE = "sapg_advantage"
OLD_TARGET_LOG_PROB = "sapg_old_target_log_prob"
IMPORTANCE_WEIGHT = "sapg_importance_weight"
TARGET_POLICY_ID = "sapg_target_policy_id"
ACTION_STD = "sapg_action_std"


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _chunked_map(function, inputs, chunk_size):
    """Map over padded, fixed-size trajectory blocks; retain scalar outputs."""
    leaves = jax.tree_util.tree_leaves(inputs)
    batch = leaves[0].shape[0]
    size = min(chunk_size, batch)
    padding = (-batch) % size

    def blocks(value):
        value = jnp.pad(value, ((0, padding),) + ((0, 0),) * (value.ndim - 1))
        return value.reshape((-1, size) + value.shape[1:])

    results = jax.lax.map(function, jax.tree_util.tree_map(blocks, inputs))
    return jax.tree_util.tree_map(
        lambda value: value.reshape((-1,) + value.shape[2:])[:batch], results
    )


def _validate_rollout(data, num_policies):
    if data.reward.ndim != 2 or min(data.reward.shape) < 1:
        raise ValueError("SAPG rollout must have nonempty [trajectory, time] rewards")
    shape = data.reward.shape
    if shape[0] % num_policies:
        raise ValueError("SAPG trajectory count must divide equally among policies")
    for value in jax.tree_util.tree_leaves(data):
        if value.shape[:2] != shape:
            raise ValueError("Every rollout leaf must start with [trajectory, time]")
    ids = data.extras["policy_extras"]["policy_id"]
    if ids.shape != shape or not jnp.issubdtype(ids.dtype, jnp.integer):
        raise ValueError("policy_id must be an integer array with shape [trajectory, time]")
    valid = jnp.all(ids == ids[:, :1]) & jnp.all((ids >= 0) & (ids < num_policies))
    counts = jnp.sum(ids[:, 0, None] == jnp.arange(num_policies), axis=0)
    valid &= jnp.all(counts == shape[0] // num_policies)
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError("SAPG requires equal policy groups with a fixed ID per trajectory")
    return ids, valid


def prepare_rollout(
    params: ppo_losses.PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jax.Array,
    ppo_network: ppo_networks.PPONetworks,
    *,
    num_policies: int,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    normalize_advantage: bool = True,
    preparation_chunk_size: int = 64,
) -> types.Transition:
    """Append one follower block and freeze all targets before optimization.

    ``data`` has leading dimensions [B, T], with equal fixed policy groups.
    Group rows may be interleaved across collected rollout segments. A random
    follower ID in [1, M) is selected once, and all B/M of its trajectories are
    copied to the final block targeting policy 0. Original behavior extras are
    preserved, including the behavior policy ID on duplicated records.

    On-policy targets use Brax GAE. Off-policy targets are one-step bootstrapped
    leader values, with no importance weighting in the critic loss. At a time
    truncation, the value target is the old baseline, as in Brax GAE. Advantages
    are globally normalized over the augmented batch, then truncations are
    masked again so normalization cannot create an artificial actor update.
    Physical episode termination and rollout rewards are never modified.
    """
    num_policies = _positive_int(num_policies, "num_policies", minimum=2)
    chunk_size = _positive_int(preparation_chunk_size, "preparation_chunk_size")
    ids, valid_groups = _validate_rollout(data, num_policies)
    table = params.policy["policy_embeddings"]
    if table.shape[0] != num_policies:
        raise ValueError("num_policies must match the actor embedding table")
    distribution = ppo_network.parametric_action_distribution
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply

    def old_on_policy(chunk):
        obs, next_obs, policy_ids = chunk
        values = value_apply(
            normalizer_params, params.value, obs, policy_ids,
            policy_embeddings=table,
        )
        terminal_obs = jax.tree_util.tree_map(lambda value: value[:, -1], next_obs)
        bootstrap = value_apply(
            normalizer_params, params.value, terminal_obs, policy_ids[:, -1],
            policy_embeddings=table,
        )
        logits = policy_apply(normalizer_params, params.policy, obs, policy_ids)
        action_std = jnp.mean(distribution.create_dist(logits).scale, axis=-1)
        return values, bootstrap, action_std

    baseline, bootstrap, action_std = _chunked_map(
        old_on_policy, (data.observation, data.next_observation, ids), chunk_size
    )
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)
    targets, advantages = ppo_losses.compute_gae(
        truncation=truncation.T,
        termination=termination.T,
        rewards=(data.reward * reward_scaling).T,
        values=baseline.T,
        bootstrap_value=bootstrap,
        lambda_=gae_lambda,
        discount=discounting,
    )
    targets, advantages = targets.T, advantages.T

    follower_id = jax.random.randint(rng, (), 1, num_policies)
    follower_indices = jnp.nonzero(
        ids[:, 0] == follower_id, size=data.reward.shape[0] // num_policies
    )[0]
    follower = jax.tree_util.tree_map(lambda value: value[follower_indices], data)
    leader_ids = jnp.zeros_like(follower.extras["policy_extras"]["policy_id"])

    def old_leader(chunk):
        obs, next_obs, raw_action, policy_ids = chunk
        values = value_apply(
            normalizer_params, params.value, obs, policy_ids,
            policy_embeddings=table,
        )
        next_values = value_apply(
            normalizer_params, params.value, next_obs, policy_ids,
            policy_embeddings=table,
        )
        logits = policy_apply(normalizer_params, params.policy, obs, policy_ids)
        return values, next_values, distribution.log_prob(logits, raw_action)

    leader_baseline, leader_next_value, leader_log_prob = _chunked_map(
        old_leader,
        (follower.observation, follower.next_observation,
         follower.extras["policy_extras"]["raw_action"], leader_ids),
        chunk_size,
    )
    follower_truncation = truncation[follower_indices]
    leader_target = (
        follower.reward * reward_scaling
        + discounting * (1 - termination[follower_indices]) * leader_next_value
    )
    leader_target = jnp.where(follower_truncation != 0, leader_baseline, leader_target)
    leader_advantage = (leader_target - leader_baseline) * (1 - follower_truncation)
    leader_importance = jnp.exp(
        leader_log_prob - follower.extras["policy_extras"]["log_prob"]
    )

    augmented = jax.tree_util.tree_map(
        lambda original, copied: jnp.concatenate((original, copied), axis=0), data, follower
    )
    advantages = jnp.concatenate((advantages, leader_advantage), axis=0)
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages *= 1 - augmented.extras["state_extras"]["truncation"]
    # A traced invalid grouping must fail health checks rather than silently
    # train on jnp.nonzero's padding/repeated first row.
    advantages = jnp.where(valid_groups, advantages, jnp.nan)
    frozen = {
        TARGET_VALUE: jnp.concatenate((targets, leader_target), axis=0),
        ADVANTAGE: advantages,
        OLD_TARGET_LOG_PROB: jnp.concatenate(
            (data.extras["policy_extras"]["log_prob"], leader_log_prob), axis=0
        ),
        IMPORTANCE_WEIGHT: jnp.concatenate((jnp.ones_like(data.reward), leader_importance), axis=0),
        TARGET_POLICY_ID: jnp.concatenate((ids, leader_ids), axis=0),
        ACTION_STD: jnp.concatenate((action_std, action_std[follower_indices]), axis=0),
    }
    frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, frozen)
    return augmented._replace(extras={
        **augmented.extras,
        "policy_extras": {**augmented.extras["policy_extras"], **frozen},
    })


def compute_sapg_loss(
    params: ppo_losses.PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jax.Array,
    ppo_network: ppo_networks.PPONetworks,
    entropy_cost: float = 0.003,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    *,
    num_policies: int,
    entropy_costs=None,
):
    """Joint actor/critic loss on minibatches from ``prepare_rollout``.

    The PPO-compatible target-construction arguments are accepted here for
    callers sharing the native PPO loop; only preparation uses them. In
    particular, advantages are never re-normalized per minibatch or pass.
    Entropy defaults to CAT's uniform coefficient for every policy. An optional
    explicit coefficient list can assign one coefficient per target policy.
    """
    del discounting, reward_scaling, gae_lambda, normalize_advantage
    num_policies = _positive_int(num_policies, "num_policies", minimum=2)
    extras = data.extras["policy_extras"]
    target_ids = extras[TARGET_POLICY_ID]
    table = params.policy["policy_embeddings"]
    if table.shape[0] != num_policies:
        raise ValueError("num_policies must match the actor embedding table")
    logits = ppo_network.policy_network.apply(
        normalizer_params, params.policy, data.observation, target_ids
    )
    baseline = ppo_network.value_network.apply(
        normalizer_params, params.value, data.observation, target_ids,
        policy_embeddings=table,
    )
    distribution = ppo_network.parametric_action_distribution
    log_prob = distribution.log_prob(logits, extras["raw_action"])
    old_log_prob = jax.lax.stop_gradient(extras[OLD_TARGET_LOG_PROB])
    advantages = jax.lax.stop_gradient(extras[ADVANTAGE])
    importance = jax.lax.stop_gradient(extras[IMPORTANCE_WEIGHT])
    ratio = jnp.exp(log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    policy_loss = -jnp.mean(importance * jnp.minimum(unclipped, clipped))

    value_error = jax.lax.stop_gradient(extras[TARGET_VALUE]) - baseline
    value_loss = 0.25 * jnp.mean(jnp.square(value_error))
    entropy = distribution.entropy(logits, rng)
    if entropy_costs is None:
        entropy_loss = -entropy_cost * jnp.mean(entropy)
    else:
        coefficients = jnp.asarray(entropy_costs)
        if coefficients.shape != (num_policies,):
            raise ValueError("entropy_costs must contain one coefficient per policy")
        if not isinstance(coefficients, jax.core.Tracer):
            if not np.all(np.isfinite(coefficients)) or np.any(np.asarray(coefficients) < 0):
                raise ValueError("entropy_costs must be finite and nonnegative")
        entropy_loss = -jnp.mean(coefficients[target_ids] * entropy)

    total_loss = policy_loss + value_loss + entropy_loss
    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": value_loss,
        "entropy_loss": entropy_loss,
    }
