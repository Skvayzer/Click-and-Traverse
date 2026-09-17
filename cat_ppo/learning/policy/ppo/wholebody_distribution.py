"""Versioned added-joint exploration preserving released CAT's leg policy.

The v1 distribution only bounds added-joint Gaussian scales. The opt-in v2
additionally conditions arm means on raw previous actions and correlates arm
innovations; policy parameter shapes and the first 15 actions remain unchanged.
Sampling and PPO likelihoods describe the same conditional distribution.
Motor-target limits remain a separate part of the environment.
"""

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Sequence

from brax.training import distribution, types
from brax.training import networks as brax_networks
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen
import jax
import jax.numpy as jnp
import numpy as np


DISTRIBUTION_KIND = "wholebody_bounded_normal_tanh_v1"
COHERENT_DISTRIBUTION_KIND = "wholebody_arm_conditional_correlated_tanh_v2"
COHERENT_EXPLORATION_VERSION = "arm_conditional_correlated_v2"
LEG_NOISE_REFERENCE_SCHEMA = "cat_frozen_leg_noise_reference_v1"
_DISTRIBUTION_KWARGS = frozenset({
    "leg_action_count", "upper_std_min", "upper_std_max", "upper_entropy_weight",
})
_COHERENT_KWARGS = frozenset({
    "exploration_version", "arm_persistence", "arm_correlation",
    "arm_last_action_indices", "arm_correlation_pattern",
})


def _policy_observation_width(observation_size, policy_obs_key):
    size = observation_size[policy_obs_key] if isinstance(observation_size, dict) else observation_size
    if isinstance(size, (tuple, list)):
        if len(size) != 1:
            raise ValueError("Frozen leg noise requires vector policy observations")
        size = size[0]
    if type(size) is not int or size <= 0:
        raise ValueError("Frozen leg noise requires a positive policy observation width")
    return size


def _reference_digest(payload):
    content = {key: value for key, value in payload.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_payload(value):
    value = np.asarray(value)
    if value.dtype != np.dtype("float32") or not np.isfinite(value).all():
        raise ValueError("Frozen leg noise requires finite float32 actor weights")
    value = np.ascontiguousarray(value, dtype="<f4")
    return {"shape": list(value.shape), "dtype": "<f4",
            "base64": base64.b64encode(value.tobytes()).decode("ascii")}


def _decode_array(payload, shape):
    if (not isinstance(payload, dict) or set(payload) != {"shape", "dtype", "base64"}
            or payload["shape"] != list(shape) or payload["dtype"] != "<f4"):
        raise ValueError("Frozen leg noise weight shape/dtype mismatch")
    try:
        raw = base64.b64decode(payload["base64"], validate=True)
        if len(raw) != math.prod(shape) * 4:
            raise ValueError("Frozen leg noise weight byte count mismatch")
        value = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid frozen leg noise encoded weights") from error
    if not np.isfinite(value).all():
        raise ValueError("Frozen leg noise requires finite actor weights")
    return value


def unpack_leg_noise_reference(payload):
    """Validate a self-contained immutable actor snapshot and decode its tree."""
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    fields = {"schema", "normalize_observations", "action_size", "leg_action_count",
              "policy_observation_size", "policy_obs_key", "policy_hidden_layer_sizes",
              "activation", "layers", "sha256"}
    if (not isinstance(payload, dict) or set(payload) != fields
            or payload["schema"] != LEG_NOISE_REFERENCE_SCHEMA
            or payload["normalize_observations"] is not False
            or payload["activation"] != "swish"):
        raise ValueError("Invalid frozen leg noise reference metadata")
    if payload["sha256"] != _reference_digest(payload):
        raise ValueError("Frozen leg noise reference sha256 mismatch")
    action_size, count = payload["action_size"], payload["leg_action_count"]
    width, hidden = payload["policy_observation_size"], payload["policy_hidden_layer_sizes"]
    if (type(action_size) is not int or type(count) is not int
            or not 0 < count < action_size or type(width) is not int or width <= 0
            or not isinstance(payload["policy_obs_key"], str)
            or not isinstance(hidden, (tuple, list))
            or any(type(size) is not int or size <= 0 for size in hidden)
            or not isinstance(payload["layers"], (tuple, list))
            or len(payload["layers"]) != len(hidden) + 1):
        raise ValueError("Invalid frozen leg noise actor architecture")
    dimensions = [width, *hidden, 2 * action_size]
    params = {}
    for i, layer in enumerate(payload["layers"]):
        if not isinstance(layer, dict) or set(layer) != {"kernel", "bias"}:
            raise ValueError("Invalid frozen leg noise actor layer")
        params[f"hidden_{i}"] = {
            "kernel": _decode_array(layer["kernel"], (dimensions[i], dimensions[i + 1])),
            "bias": _decode_array(layer["bias"], (dimensions[i + 1],)),
        }
    return {"params": params}


def pack_leg_noise_reference(policy_params, network_config):
    """Embed an unnormalized saved actor in JSON, without an external file path.

    The frozen actor evaluates the SAME raw state as the learner, but only its
    first ``leg_action_count`` raw scale outputs are used. Keeping its entire
    trunk is necessary: freezing only scale-head weights would still let shared
    trunk updates change the exploration distribution.
    """
    if hasattr(network_config, "to_dict"):
        network_config = network_config.to_dict()
    if network_config.get("normalize_observations") is not False:
        raise ValueError("Frozen leg noise reference requires unnormalized observations")
    kwargs = dict(network_config["network_factory_kwargs"])
    existing = kwargs.get("leg_noise_reference")
    if existing is not None:
        unpack_leg_noise_reference(existing)
        return json.loads(json.dumps(existing))
    if kwargs.get("activation") not in (None, "swish"):
        raise ValueError("Frozen leg noise reference requires default swish activation")
    key = kwargs.get("policy_obs_key", "state")
    hidden = list(kwargs.get("policy_hidden_layer_sizes", (32, 32, 32, 32)))
    if (set(policy_params) != {"params"}
            or set(policy_params["params"]) != {f"hidden_{i}" for i in range(len(hidden) + 1)}):
        raise ValueError("Frozen leg noise actor parameters do not match MLP architecture")
    payload = {
        "schema": LEG_NOISE_REFERENCE_SCHEMA, "normalize_observations": False,
        "action_size": network_config["action_size"],
        "leg_action_count": kwargs.get("leg_action_count", 12),
        "policy_observation_size": _policy_observation_width(network_config["observation_size"], key),
        "policy_obs_key": key, "policy_hidden_layer_sizes": hidden, "activation": "swish",
        "layers": [{name: _array_payload(value) for name, value in
                    policy_params["params"][f"hidden_{i}"].items()}
                   for i in range(len(hidden) + 1)],
    }
    payload["sha256"] = _reference_digest(payload)
    unpack_leg_noise_reference(payload)
    return payload


def _leg_noise_metadata(payload):
    return {"schema": LEG_NOISE_REFERENCE_SCHEMA, "sha256": payload["sha256"],
            "raw_scale_indices": list(range(payload["action_size"],
                                             payload["action_size"] + payload["leg_action_count"])),
            "conditioning": "frozen_actor_on_raw_observations",
            "gradient": "stop_gradient", "normalize_observations": False}


def _validated_leg_noise_metadata(metadata, action_size, leg_action_count):
    if hasattr(metadata, "to_dict"):
        metadata = metadata.to_dict()
    if not isinstance(metadata, dict):
        raise ValueError("Invalid frozen leg noise distribution metadata")
    digest = metadata.get("sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("Invalid frozen leg noise distribution sha256")
    expected = _leg_noise_metadata({"sha256": digest, "action_size": action_size,
                                    "leg_action_count": leg_action_count})
    if metadata != expected:
        raise ValueError("Frozen leg noise distribution semantics differ")
    return expected


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
        config = {
            "kind": DISTRIBUTION_KIND,
            "event_size": self.event_size,
            "leg_action_count": self.leg_action_count,
            "upper_std_min": self.upper_std_min,
            "upper_std_max": self.upper_std_max,
            "upper_entropy_weight": self.upper_entropy_weight,
            "native_min_std": self._min_std,
            "native_var_scale": self._var_scale,
        }
        if getattr(self, "leg_noise_reference_metadata", None) is not None:
            config["leg_noise_reference"] = dict(self.leg_noise_reference_metadata)
        return config

    @classmethod
    def from_config(cls, config):
        """Restore explicit semantics without substituting missing defaults."""
        if isinstance(config, dict) and config.get("kind") == COHERENT_DISTRIBUTION_KIND:
            return CoherentArmNormalTanhDistribution.from_config(config)
        if not isinstance(config, dict) or config.get("kind") != DISTRIBUTION_KIND:
            raise ValueError("Unknown bounded action distribution configuration")
        required = _DISTRIBUTION_KWARGS | {
            "kind", "event_size", "native_min_std", "native_var_scale",
        }
        if set(config) not in (required, required | {"leg_noise_reference"}):
            raise ValueError("Bounded action distribution configuration must include all settings")
        restored = cls(config["event_size"], **{key: config[key] for key in _DISTRIBUTION_KWARGS})
        if "leg_noise_reference" in config:
            restored.leg_noise_reference_metadata = _validated_leg_noise_metadata(
                config["leg_noise_reference"], restored.event_size, restored.leg_action_count)
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


def _arm_correlation_matrix(strength):
    """Fixed G1 raise/tuck directions, with an independent residual in every DOF.

    Each arm has independent raise and tuck factors. The signs match the
    FK-verified presets: raise=(-pitch, inward roll, -elbow), while tucking
    uses a smaller negative pitch and positive elbow. Shoulder yaw and wrists
    have independent noise. Normalize the diagonal to one so marginal scales
    retain their configured bounds; strength controls correlation, not variance.
    """
    factors = np.zeros((14, 4), dtype=np.float64)
    for side, roll_sign in ((0, -1.), (1, 1.)):
        start = side * 7
        factors[start, side * 2:side * 2 + 2] = [-1., -.25]
        factors[start + 1, side * 2:side * 2 + 2] = [.1875 * roll_sign] * 2
        factors[start + 3, side * 2:side * 2 + 2] = [-1., 1.]
    lengths = np.linalg.norm(factors, axis=-1)
    active = lengths > 0
    factors[active] /= lengths[active, None]
    return np.diag(1. - strength * active) + strength * (factors @ factors.T)


class _CorrelatedArmNormal:
    """Joint Gaussian with native independent legs/waist and a 14D arm block.

    ``scale`` exposes marginal standard deviations for existing leg KL and
    telemetry. ``log_prob`` and ``entropy`` return Cholesky factor contributions
    whose sum is the JOINT density/entropy, not independent arm marginals.
    """

    def __init__(self, loc, scale, cholesky, inverse_cholesky):
        self.loc, self.scale = loc, scale
        self._cholesky = jnp.asarray(cholesky, dtype=loc.dtype)
        self._inverse_cholesky = jnp.asarray(inverse_cholesky, dtype=loc.dtype)

    def sample(self, seed):
        # Identical draw shape/key/arithmetic for the first 15 actions as v1.
        noise = jax.random.normal(seed, shape=self.loc.shape)
        correlated = noise[..., 15:] @ self._cholesky.T
        noise = noise.at[..., 15:].set(correlated)
        return noise * self.scale + self.loc

    def mode(self):
        return self.loc

    def log_prob(self, x):
        result = distribution.NormalDistribution(self.loc, self.scale).log_prob(x)
        residual = (x[..., 15:] - self.loc[..., 15:]) / self.scale[..., 15:]
        whitened = residual @ self._inverse_cholesky.T
        arm_terms = (-.5 * whitened ** 2 - .5 * jnp.log(2. * jnp.pi)
                     - jnp.log(self.scale[..., 15:])
                     - jnp.log(jnp.diag(self._cholesky)))
        return result.at[..., 15:].set(arm_terms)

    def entropy(self):
        result = distribution.NormalDistribution(self.loc, self.scale).entropy()
        return result.at[..., 15:].add(jnp.log(jnp.diag(self._cholesky)))


class CoherentArmNormalTanhDistribution(WholeBodyNormalTanhDistribution):
    """PPO-visible conditional AR arm policy with fixed full-rank covariance.

    The factory conditions means on previous actions already in the observation:
    m_t=(1-rho)*network_mean_t + rho*atanh(previous_post_tanh_action).
    All likelihoods use that same conditional mean and covariance. No hidden
    noise state, extra actor inputs, external OU process, or detached gradients
    are introduced. Deterministic inference uses tanh(m_t), including smoothing.

    Bounds constrain nominal sigma; arm innovations multiply it by the explicit
    ``arm_innovation_scale``. For constant mean/scale before tanh, stationary
    variance is (nominal sigma * innovation scale) squared / (1-rho squared).
    Choosing sqrt(1-rho squared) therefore preserves nominal marginal variance
    while retaining temporal coherence. Older v2 checkpoints used scale 1.
    A near-bound previous action is clipped only for the atanh inverse,
    explicitly recorded as part of v2 semantics.
    """

    def __init__(self, event_size, *, exploration_version, arm_persistence,
                 arm_correlation, arm_last_action_indices, arm_correlation_pattern,
                 arm_innovation_scale=1., **kwargs):
        super().__init__(event_size, **kwargs)
        if event_size != 29 or self.leg_action_count != 12:
            raise ValueError("Coherent G1 arm exploration requires 29 actions and 12 legs")
        if exploration_version != COHERENT_EXPLORATION_VERSION:
            raise ValueError("Unknown coherent exploration version")
        if (not math.isfinite(arm_persistence) or not 0 <= arm_persistence < 1):
            raise ValueError("arm_persistence must be finite and in [0, 1)")
        if (not math.isfinite(arm_correlation) or not 0 <= arm_correlation < 1):
            raise ValueError("arm_correlation must be finite and in [0, 1)")
        if (isinstance(arm_innovation_scale, bool)
                or not math.isfinite(arm_innovation_scale)
                or not 0 < arm_innovation_scale <= 1):
            raise ValueError("arm_innovation_scale must be finite and in (0, 1]")
        if arm_correlation_pattern != "g1_raise_tuck_v1":
            raise ValueError("Unknown arm correlation pattern")
        if (not isinstance(arm_last_action_indices, (list, tuple))
                or len(arm_last_action_indices) != 14
                or len(set(arm_last_action_indices)) != 14
                or any(type(i) is not int or i < 0 for i in arm_last_action_indices)):
            raise ValueError("arm_last_action_indices must contain 14 distinct nonnegative integers")
        self.exploration_version = exploration_version
        self.arm_persistence = float(arm_persistence)
        self.arm_correlation = float(arm_correlation)
        self.arm_innovation_scale = float(arm_innovation_scale)
        self.arm_last_action_indices = tuple(arm_last_action_indices)
        self.arm_correlation_pattern = arm_correlation_pattern
        self.previous_action_clip = 1. - 1e-6
        self.arm_correlation_matrix = _arm_correlation_matrix(self.arm_correlation)
        self._arm_cholesky = np.linalg.cholesky(self.arm_correlation_matrix)
        self._arm_inverse_cholesky = np.linalg.inv(self._arm_cholesky)

    @property
    def config(self):
        return {**super().config, "kind": COHERENT_DISTRIBUTION_KIND,
                "exploration_version": self.exploration_version,
                "arm_persistence": self.arm_persistence,
                "arm_correlation": self.arm_correlation,
                "arm_innovation_scale": self.arm_innovation_scale,
                "arm_last_action_indices": list(self.arm_last_action_indices),
                "arm_correlation_pattern": self.arm_correlation_pattern,
                "previous_action_clip": self.previous_action_clip,
                "conditioned_action_indices": list(range(15, 29)),
                "mean_conditioning": "blend_raw_mean_with_atanh_raw_previous_action",
                "deterministic_output": "tanh(conditioned_mean)"}

    @classmethod
    def from_config(cls, config):
        required = (_DISTRIBUTION_KWARGS | _COHERENT_KWARGS | {
            "kind", "event_size", "native_min_std", "native_var_scale",
            "previous_action_clip", "conditioned_action_indices", "mean_conditioning",
            "deterministic_output"})
        if not isinstance(config, dict) or config.get("kind") != COHERENT_DISTRIBUTION_KIND:
            raise ValueError("Unknown coherent action distribution configuration")
        # Missing innovation scale is the exact historical v2 behavior. New
        # checkpoints always record it; never reinterpret an old v2 as scaled.
        optional = set(config) & {"arm_innovation_scale", "leg_noise_reference"}
        if set(config) != required | optional:
            raise ValueError("Coherent action distribution configuration must include all settings")
        restored = cls(config["event_size"], **{
            key: config[key] for key in _DISTRIBUTION_KWARGS | _COHERENT_KWARGS},
            arm_innovation_scale=config.get("arm_innovation_scale", 1.))
        if "leg_noise_reference" in config:
            restored.leg_noise_reference_metadata = _validated_leg_noise_metadata(
                config["leg_noise_reference"], restored.event_size, restored.leg_action_count)
        if restored.config != {"arm_innovation_scale": 1., **config}:
            raise ValueError("Coherent action distribution semantics differ")
        return restored

    def condition_parameters(self, parameters, raw_policy_observations):
        """Use raw observation history, independently of network normalization."""
        if max(self.arm_last_action_indices) >= raw_policy_observations.shape[-1]:
            raise ValueError("Previous-arm-action indices exceed observation width")
        if parameters.shape[:-1] != raw_policy_observations.shape[:-1]:
            raise ValueError("Observation and policy parameter batch/time dimensions differ")
        previous = jnp.take(raw_policy_observations,
                            jnp.asarray(self.arm_last_action_indices), axis=-1)
        previous = jnp.arctanh(jnp.clip(previous, -self.previous_action_clip,
                                      self.previous_action_clip))
        conditioned = ((1. - self.arm_persistence) * parameters[..., 15:29]
                       + self.arm_persistence * previous)
        return parameters.at[..., 15:29].set(conditioned)

    def create_dist(self, parameters):
        bounded = super().create_dist(parameters)
        # Modify the actual Gaussian, not just sampled actions: PPO log density,
        # entropy and exposed marginal scales must all use the same covariance.
        scale = bounded.scale.at[..., 15:].multiply(self.arm_innovation_scale)
        return _CorrelatedArmNormal(bounded.loc, scale,
                                    self._arm_cholesky, self._arm_inverse_cholesky)

    def entropy(self, parameters, seed):
        if self.upper_entropy_weight == 0:
            return super().entropy(parameters, seed)
        loc, raw_scale = jnp.split(parameters, 2, axis=-1)
        legs = jnp.concatenate([loc[..., :12], raw_scale[..., :12]], axis=-1)
        leg_entropy = self._leg_distribution.entropy(legs, seed)
        joint = self.create_dist(parameters)
        upper_seed = jax.random.fold_in(seed, 1)
        upper_entropy = (joint.entropy()[..., 12:]
                         + self._postprocessor.forward_log_det_jacobian(
                             joint.sample(upper_seed))[..., 12:])
        return leg_entropy + self.upper_entropy_weight * jnp.sum(upper_entropy, axis=-1)


def _make_distribution(action_size, base_kwargs, coherent_kwargs, arm_innovation_scale=None):
    present = {key for key, value in coherent_kwargs.items() if value is not None}
    if not present:
        if arm_innovation_scale is not None:
            raise ValueError("arm_innovation_scale requires coherent v2 exploration")
        return WholeBodyNormalTanhDistribution(action_size, **base_kwargs)
    if present != _COHERENT_KWARGS:
        raise ValueError("Coherent exploration requires all five explicit v2 settings")
    return CoherentArmNormalTanhDistribution(action_size, **base_kwargs, **coherent_kwargs,
        arm_innovation_scale=1. if arm_innovation_scale is None else arm_innovation_scale)


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
    exploration_version: str | None = None,
    arm_persistence: float | None = None,
    arm_correlation: float | None = None,
    arm_last_action_indices: Sequence[int] | None = None,
    arm_correlation_pattern: str | None = None,
    arm_innovation_scale: float | None = None,
    leg_noise_reference: dict | None = None,
) -> ppo_networks.PPONetworks:
    """Keep native parameter shapes and install versioned action semantics.

    Defaults for native network arguments match Brax 0.12.3.  Production callers
    must retain the released CAT hidden sizes and observation keys, as before.
    Save all four base distribution settings, and all five v2 settings when
    enabled, plus the innovation scale when overriding historical v2 scale 1.
    Network weights alone cannot encode bounds or arm conditioning. Recovery
    additionally embeds the frozen scale-reference actor in factory kwargs;
    sampling, likelihood and entropy all consume its replaced leg raw scales.
    """
    bounded = _make_distribution(action_size, {
        "leg_action_count": leg_action_count, "upper_std_min": upper_std_min,
        "upper_std_max": upper_std_max, "upper_entropy_weight": upper_entropy_weight,
    }, {"exploration_version": exploration_version, "arm_persistence": arm_persistence,
        "arm_correlation": arm_correlation, "arm_last_action_indices": arm_last_action_indices,
        "arm_correlation_pattern": arm_correlation_pattern}, arm_innovation_scale)
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
    reference_params = None
    if leg_noise_reference is not None:
        if hasattr(leg_noise_reference, "to_dict"):
            leg_noise_reference = leg_noise_reference.to_dict()
        reference_params = unpack_leg_noise_reference(leg_noise_reference)
        expected = {"action_size": action_size, "leg_action_count": leg_action_count,
                    "policy_observation_size": _policy_observation_width(observation_size, policy_obs_key),
                    "policy_obs_key": policy_obs_key,
                    "policy_hidden_layer_sizes": list(policy_hidden_layer_sizes)}
        if any(leg_noise_reference[key] != value for key, value in expected.items()):
            raise ValueError("Frozen leg noise reference architecture differs from learner")
        if activation is not linen.swish:
            raise ValueError("Frozen leg noise reference requires default swish activation")
        bounded.leg_noise_reference_metadata = _leg_noise_metadata(leg_noise_reference)
        # The reference always consumes raw state. It does not share the current
        # actor's trainable trunk OR a changing observation-normalization state.
        reference_actor = brax_networks.make_policy_network(
            param_size=2 * action_size, obs_size=observation_size,
            hidden_layer_sizes=policy_hidden_layer_sizes, activation=activation,
            obs_key=policy_obs_key)
        reference_params = jax.tree.map(jnp.asarray, reference_params)
    if isinstance(bounded, CoherentArmNormalTanhDistribution) or reference_params is not None:
        actor = native.policy_network

        def apply(normalizer_params, policy_params, observations):
            parameters = actor.apply(normalizer_params, policy_params, observations)
            if reference_params is not None:
                reference = reference_actor.apply(None, reference_params, observations)
                parameters = parameters.at[..., action_size:action_size + leg_action_count].set(
                    jax.lax.stop_gradient(reference[..., action_size:action_size + leg_action_count]))
            if isinstance(bounded, CoherentArmNormalTanhDistribution):
                raw_state = observations[policy_obs_key] if isinstance(observations, dict) else observations
                parameters = bounded.condition_parameters(parameters, raw_state)
            return parameters

        native = native.replace(policy_network=brax_networks.FeedForwardNetwork(
            init=actor.init, apply=apply))
    return native.replace(parametric_action_distribution=bounded)


def checkpoint_distribution_config(network_config) -> dict | None:
    """Identify legacy versus bounded Brax checkpoints from explicit settings.

    Brax serializes partial-function arguments but not the factory identity.
    Repaired checkpoints therefore require all four distribution arguments;
    incomplete settings fail rather than silently filling current defaults.
    """
    kwargs = dict(network_config["network_factory_kwargs"])
    present = set(kwargs) & _DISTRIBUTION_KWARGS
    coherent_present = set(kwargs) & _COHERENT_KWARGS
    recorded = network_config.get("action_distribution")
    reference = kwargs.get("leg_noise_reference")
    if not present and not coherent_present:
        if reference is not None:
            raise ValueError("Frozen leg noise requires explicit bounded distribution settings")
        if kwargs.get("arm_innovation_scale") is not None:
            raise ValueError("arm_innovation_scale requires coherent v2 exploration")
        if recorded is not None:
            raise ValueError("Checkpoint distribution metadata lacks explicit factory settings")
        return None
    if present != _DISTRIBUTION_KWARGS:
        raise ValueError("Bounded checkpoint must explicitly record all distribution settings")
    if coherent_present and coherent_present != _COHERENT_KWARGS:
        raise ValueError("Coherent checkpoint must explicitly record all five v2 settings")
    # Brax saves signature defaults, including all five None values for a v1
    # factory created by this version. All-absent (old v1) and all-None (new v1)
    # are equivalent; a partially populated v2 remains an error. If explicit
    # v2 metadata accompanies all-None settings, the agreement check below
    # rejects it instead of silently downgrading the policy.
    if (coherent_present
            and any(kwargs[key] is not None for key in _COHERENT_KWARGS)
            and any(kwargs[key] is None for key in _COHERENT_KWARGS)):
        raise ValueError("Coherent checkpoint v2 settings cannot be null")
    restored = _make_distribution(network_config["action_size"],
        {key: kwargs[key] for key in _DISTRIBUTION_KWARGS},
        {key: kwargs.get(key) for key in _COHERENT_KWARGS},
        kwargs.get("arm_innovation_scale"))
    if reference is not None:
        if hasattr(reference, "to_dict"):
            reference = reference.to_dict()
        unpack_leg_noise_reference(reference)
        if network_config.get("normalize_observations") is not False:
            raise ValueError("Frozen leg noise checkpoints require unnormalized observations")
        expected = {"action_size": network_config["action_size"],
                    "leg_action_count": kwargs["leg_action_count"],
                    "policy_obs_key": kwargs.get("policy_obs_key", "state"),
                    "policy_hidden_layer_sizes": list(kwargs.get("policy_hidden_layer_sizes", (32, 32, 32, 32)))}
        observation_size = network_config["observation_size"]
        if hasattr(observation_size, "to_dict"):
            observation_size = observation_size.to_dict()
        expected["policy_observation_size"] = _policy_observation_width(
            observation_size, expected["policy_obs_key"])
        if any(reference[key] != value for key, value in expected.items()):
            raise ValueError("Frozen leg noise checkpoint reference architecture differs")
        restored.leg_noise_reference_metadata = _leg_noise_metadata(reference)
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
