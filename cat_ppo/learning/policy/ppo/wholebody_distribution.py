"""Bound added-joint exploration while preserving released CAT's leg policy.

The network still emits the same means and raw scales in the same order.  Only
the Gaussian scales for the added actions are clipped, before sampling or any
probability calculation.  Consequently sampling, PPO log probabilities, and
reference KL all describe the same distribution.  Motor-target limits are a
separate part of the environment.
"""

import json
import math
from pathlib import Path
from typing import Callable, Sequence

from brax.training import distribution, types
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen
import jax
import jax.numpy as jnp


DISTRIBUTION_KIND = "wholebody_bounded_normal_tanh_v1"
_DISTRIBUTION_KWARGS = frozenset({
    "leg_action_count", "upper_std_min", "upper_std_max", "upper_entropy_weight",
})


class WholeBodyNormalTanhDistribution(distribution.NormalTanhDistribution):
    """Native CAT legs plus bounded Gaussian noise for added body actions.

    ``entropy`` is the weighted entropy objective used by PPO, not the full
    distribution's entropy when ``upper_entropy_weight != 1``.  With the repair
    default of zero it is exactly Brax's original 12-action leg entropy,
    including its Monte Carlo tanh correction and PRNG convention.
    """

    def __init__(
        self,
        event_size: int,
        *,
        leg_action_count: int = 12,
        upper_std_min: float = 0.02,
        upper_std_max: float = 0.10,
        upper_entropy_weight: float = 0.0,
    ):
        if (isinstance(event_size, bool) or not isinstance(event_size, int)
                or isinstance(leg_action_count, bool) or not isinstance(leg_action_count, int)
                or not 0 < leg_action_count < event_size):
            raise ValueError("Require integer 0 < leg_action_count < event_size")
        if (not math.isfinite(upper_std_min) or not math.isfinite(upper_std_max)
                or not 0 < upper_std_min < upper_std_max):
            raise ValueError("Require finite 0 < upper_std_min < upper_std_max")
        if not math.isfinite(upper_entropy_weight) or upper_entropy_weight < 0:
            raise ValueError("upper_entropy_weight must be finite and nonnegative")
        # Preserve released Brax's exact softplus(raw_scale) + 0.001 transform
        # for the original legs, and the existing 0.05 added-output bias.
        super().__init__(event_size=event_size)
        self.event_size = event_size
        self.leg_action_count = leg_action_count
        self.upper_std_min = float(upper_std_min)
        self.upper_std_max = float(upper_std_max)
        self.upper_entropy_weight = float(upper_entropy_weight)
        self._leg_distribution = distribution.NormalTanhDistribution(
            event_size=leg_action_count,
        )

    @property
    def config(self) -> dict:
        """Serializable semantics for checkpoint compatibility/fingerprints."""
        return {
            "kind": DISTRIBUTION_KIND,
            "event_size": self.event_size,
            "leg_action_count": self.leg_action_count,
            "upper_std_min": self.upper_std_min,
            "upper_std_max": self.upper_std_max,
            "upper_entropy_weight": self.upper_entropy_weight,
            "native_min_std": self._min_std,
            "native_var_scale": self._var_scale,
        }

    @classmethod
    def from_config(cls, config):
        """Restore explicit semantics without substituting missing defaults."""
        if not isinstance(config, dict) or config.get("kind") != DISTRIBUTION_KIND:
            raise ValueError("Unknown bounded action distribution configuration")
        required = _DISTRIBUTION_KWARGS | {
            "kind", "event_size", "native_min_std", "native_var_scale",
        }
        if set(config) != required:
            raise ValueError("Bounded action distribution configuration must include all settings")
        restored = cls(config["event_size"], **{key: config[key] for key in _DISTRIBUTION_KWARGS})
        if restored.config != config:
            raise ValueError("Bounded action distribution native semantics differ")
        return restored

    def create_dist(self, parameters):
        if parameters.shape[-1] != self.param_size:
            raise ValueError(
                f"Expected {self.param_size} distribution parameters, "
                f"got {parameters.shape[-1]}"
            )
        native = super().create_dist(parameters)
        upper_scale = jnp.clip(
            native.scale[..., self.leg_action_count:],
            self.upper_std_min,
            self.upper_std_max,
        )
        scale = jnp.concatenate(
            [native.scale[..., :self.leg_action_count], upper_scale], axis=-1,
        )
        return distribution.NormalDistribution(loc=native.loc, scale=scale)

    def entropy(self, parameters, seed):
        loc, raw_scale = jnp.split(parameters, 2, axis=-1)
        leg_parameters = jnp.concatenate(
            [loc[..., :self.leg_action_count], raw_scale[..., :self.leg_action_count]],
            axis=-1,
        )
        leg_entropy = self._leg_distribution.entropy(leg_parameters, seed)
        if self.upper_entropy_weight == 0:
            # Avoid evaluating any upper-body entropy term: its gradients are
            # exactly zero, while upper-body actions remain stochastic in PPO.
            return leg_entropy
        bounded = self.create_dist(parameters)
        upper = distribution.NormalDistribution(
            loc=bounded.loc[..., self.leg_action_count:],
            scale=bounded.scale[..., self.leg_action_count:],
        )
        upper_seed = jax.random.fold_in(seed, 1)
        upper_entropy = upper.entropy() + self._postprocessor.forward_log_det_jacobian(
            upper.sample(seed=upper_seed),
        )
        return leg_entropy + self.upper_entropy_weight * jnp.sum(upper_entropy, axis=-1)


def make_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32, 32, 32, 32),
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256, 256, 256),
    activation: Callable[[jax.Array], jax.Array] = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    *,
    leg_action_count: int = 12,
    upper_std_min: float = 0.02,
    upper_std_max: float = 0.10,
    upper_entropy_weight: float = 0.0,
) -> ppo_networks.PPONetworks:
    """Keep native policy/value networks and replace only their distribution.

    Defaults for native network arguments match Brax 0.12.3.  Production callers
    must retain the released CAT hidden sizes and observation keys, as before.
    Save the four additional keyword settings together with the factory identity
    when exporting a checkpoint; network weights alone cannot encode these bounds.
    """
    bounded = WholeBodyNormalTanhDistribution(
        action_size,
        leg_action_count=leg_action_count,
        upper_std_min=upper_std_min,
        upper_std_max=upper_std_max,
        upper_entropy_weight=upper_entropy_weight,
    )
    native = ppo_networks.make_ppo_networks(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        policy_obs_key=policy_obs_key,
        value_obs_key=value_obs_key,
    )
    return native.replace(parametric_action_distribution=bounded)


def checkpoint_distribution_config(network_config) -> dict | None:
    """Identify legacy versus bounded Brax checkpoints from explicit settings.

    Brax serializes partial-function arguments but not the factory identity.
    Repaired checkpoints therefore require all four distribution arguments;
    incomplete settings fail rather than silently filling current defaults.
    """
    kwargs = dict(network_config["network_factory_kwargs"])
    present = set(kwargs) & _DISTRIBUTION_KWARGS
    recorded = network_config.get("action_distribution")
    if not present:
        if recorded is not None:
            raise ValueError("Checkpoint distribution metadata lacks explicit factory settings")
        return None
    if present != _DISTRIBUTION_KWARGS:
        raise ValueError("Bounded checkpoint must explicitly record all distribution settings")
    restored = WholeBodyNormalTanhDistribution(
        network_config["action_size"], **{key: kwargs[key] for key in _DISTRIBUTION_KWARGS},
    )
    if recorded is not None:
        validated = WholeBodyNormalTanhDistribution.from_config(dict(recorded))
        if validated.config != restored.config:
            raise ValueError("Checkpoint distribution metadata disagrees with factory settings")
    return restored.config


def load_checkpoint_policy(path, *, deterministic=True):
    """Load native or bounded inference while honoring checkpoint normalization."""
    from brax.training.agents.ppo import checkpoint

    path = Path(path)
    config = json.loads((path / "ppo_network_config.json").read_text())
    bounded = checkpoint_distribution_config(config)
    factory = make_ppo_networks if bounded is not None else ppo_networks.make_ppo_networks
    return checkpoint.load_policy(path, network_factory=factory, deterministic=deterministic)
