"""Optional frozen warm-start action retention, additive to ordinary PPO."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np


def normalize_reference_config(config, action_size):
    """Validate the explicit scene/action mask and produce a portable contract."""
    if config is None:
        return None
    allowed = {"coefficient", "action_indices", "scene_mask"}
    if set(config) != allowed:
        raise ValueError(f"reference_kl_config must contain exactly {sorted(allowed)}")
    coefficient = float(config["coefficient"])
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("Reference KL coefficient must be finite and positive; use None to disable")
    indices = list(config["action_indices"])
    if (not indices or len(set(indices)) != len(indices)
            or any(type(index) is not int or not 0 <= index < action_size for index in indices)):
        raise ValueError("Reference action_indices must be distinct valid action dimensions")
    mask = list(config["scene_mask"])
    if not mask or not any(mask) or any(not isinstance(value, (bool, np.bool_)) for value in mask):
        raise ValueError("Reference scene_mask must be a nonempty boolean list containing CAT scenes")
    return dict(coefficient=coefficient, action_indices=indices, scene_mask=[bool(value) for value in mask],
                schema="frozen-warmstart-action-retention-v1", direction="reference-to-current",
                reduction="sum selected action KL dimensions, mean over selected CAT transitions",
                observations="same compact observations for frozen reference and current actor; original point geometry is not reconstructed")


def freeze_reference(policy, normalizer):
    """Copy the initial adapted actor and normalizer, independently of the optimizer."""
    return jax.tree_util.tree_map(
        lambda value: jax.lax.stop_gradient(jnp.array(value, copy=True)),
        {"policy": policy, "normalizer": normalizer})


def masked_gaussian_reference_kl(current_mean, current_std, reference_mean, reference_std,
                                  scene_ids, *, action_indices, scene_mask):
    """KL(reference || current), selected action marginal, selected transitions.

    CAT uses independent Gaussians followed by the same bijective tanh transform,
    so the analytic pre-tanh KL also describes this selected action marginal.
    Gradients never update the reference distribution or unselected outputs.
    """
    selected = jnp.asarray(action_indices, dtype=jnp.int32)
    current_mean = jnp.take(current_mean, selected, axis=-1)
    current_std = jnp.maximum(jnp.take(current_std, selected, axis=-1), 1e-6)
    reference_mean = jax.lax.stop_gradient(jnp.take(reference_mean, selected, axis=-1))
    reference_std = jax.lax.stop_gradient(jnp.maximum(jnp.take(reference_std, selected, axis=-1), 1e-6))
    terms = (jnp.log(current_std / reference_std)
             + (reference_std ** 2 + (reference_mean - current_mean) ** 2) / (2 * current_std ** 2) - .5)
    per_transition = jnp.sum(terms, axis=-1)
    table = jnp.asarray(scene_mask, dtype=jnp.bool_)
    scene_ids = jnp.asarray(scene_ids, dtype=jnp.int32)
    valid = (scene_ids >= 0) & (scene_ids < table.shape[0])
    mask = valid & table[jnp.clip(scene_ids, 0, table.shape[0] - 1)]
    if mask.shape != per_transition.shape:
        raise ValueError("Reference scene IDs must match the actor's batch/time dimensions")
    count = jnp.sum(mask)
    mean_kl = jnp.sum(jnp.where(mask, per_transition, 0.)) / jnp.maximum(count, 1)
    return mean_kl, jnp.mean(mask.astype(jnp.float32))


def reference_regularization(params, normalizer_params, data, *, network, reference, config):
    """Evaluate both actors on identical compact observations with frozen statistics."""
    current_logits = network.policy_network.apply(normalizer_params, params.policy, data.observation)
    frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, reference)
    reference_logits = network.policy_network.apply(frozen["normalizer"], frozen["policy"], data.observation)
    distribution = network.parametric_action_distribution
    current_dist, reference_dist = distribution.create_dist(current_logits), distribution.create_dist(reference_logits)
    if not all(hasattr(dist, name) for dist in (current_dist, reference_dist) for name in ("loc", "scale")):
        raise TypeError("Reference KL requires independent Gaussian action distributions")
    kl, fraction = masked_gaussian_reference_kl(current_dist.loc, current_dist.scale,
        reference_dist.loc, reference_dist.scale, data.extras["state_extras"]["pf_id"],
        action_indices=config["action_indices"], scene_mask=config["scene_mask"])
    weighted = config["coefficient"] * kl
    return weighted, {"reference_kl": kl, "reference_kl_loss": weighted,
                      "reference_cat_fraction": fraction}
