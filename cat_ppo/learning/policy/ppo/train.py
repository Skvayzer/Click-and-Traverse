# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import collections
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from cat_ppo.learning.policy.ppo.field_arguments import FieldArguments
from cat_ppo.learning.policy.ppo.reference_policy import (
    freeze_reference, normalize_reference_config, reference_regularization,
)


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: types.UInt64


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _set_optimizer_learning_rate(state, learning_rate):
    """Change an injected Adam rate without changing optimizer tree structure."""
    if hasattr(state, "hyperparams") and "learning_rate" in state.hyperparams:
        values = dict(state.hyperparams)
        values["learning_rate"] = jnp.asarray(learning_rate, dtype=values["learning_rate"].dtype)
        return state._replace(hyperparams=values)
    if isinstance(state, tuple):
        values = [_set_optimizer_learning_rate(value, learning_rate) for value in state]
        return type(state)(*values) if hasattr(state, "_fields") else tuple(values)
    return state


def _recover_training_state(training_state, params, optimizer, learning_rate):
    """Restore safe model weights and fresh Adam while preserving elapsed transitions."""
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Recovery learning rate must be finite and positive")
    if not isinstance(params, (list, tuple)) or len(params) != 3:
        raise ValueError("Recovery requires normalizer, actor and critic")
    params = tuple(params)  # Orbax's native loader returns a list.
    expected = (training_state.normalizer_params, training_state.params.policy, training_state.params.value)
    _assert_same_tree_shapes("recovery parameters", expected, params)
    for original, value in zip(jax.tree.leaves(expected), jax.tree.leaves(params)):
        if np.asarray(original).dtype != np.asarray(value).dtype:
            raise ValueError("Recovery parameter dtypes differ from learner")
    if any(not np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(params)):
        raise ValueError("Recovery parameters must be finite")
    params = jax.tree.map(jnp.asarray, params)
    model = training_state.params.replace(policy=params[1], value=params[2])
    state = _set_optimizer_learning_rate(optimizer.init(model), learning_rate)
    return training_state.replace(normalizer_params=params[0], params=model, optimizer_state=state)


def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def _runtime_tree_state(tree):
    """Portable array leaves; the fresh environment supplies static MJX metadata."""
    pairs, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {
        "paths": [jax.tree_util.keystr(path) for path, _ in pairs],
        "leaves": [np.asarray(value) for _, value in pairs],
    }


def _restore_runtime_tree(name, template, state):
    pairs, structure = jax.tree_util.tree_flatten_with_path(template)
    if state["paths"] != [jax.tree_util.keystr(path) for path, _ in pairs]:
        raise ValueError(f"Runtime {name} tree structure differs")
    values = state["leaves"]
    if len(values) != len(pairs):
        raise ValueError(f"Runtime {name} leaf count differs")
    for (path, original), value in zip(pairs, values):
        value = np.asarray(value)
        if np.shape(original) != value.shape or np.dtype(original.dtype) != value.dtype:
            raise ValueError(f"Runtime {name}{jax.tree_util.keystr(path)} shape/dtype differs")
    return jax.tree_util.tree_unflatten(structure, [jnp.asarray(x) for x in values])


def _generate_unroll_with_scene_ids(env, state, policy, key, unroll_length, extra_fields):
    """Attribute terminal transitions to the scene before its autoreset."""
    def step(carry, _):
        current, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        scene_id = current.info["pf_id"]  # CAT wrappers can mutate info in-place.
        next_state, transition = acting.actor_step(env, current, policy, current_key, extra_fields)
        transition = transition._replace(extras={
            **transition.extras,
            "state_extras": {**transition.extras["state_extras"], "pf_id": scene_id},
        })
        return (next_state, next_key), transition

    (state, _), data = jax.lax.scan(step, (state, key), (), length=unroll_length)
    return state, data


_WHOLEBODY_EPISODE_RATE_KEYS = {
    "wb_goal_reached": "training/goal_success_rate",
    "wb_fall": "training/fall_rate",
    "wb_obstacle": "training/obstacle_failure_rate",
    "wb_body_collision": "training/body_collision_rate",
    "wb_reset_replaced": "training/reset_pose_replacement_rate",
    **{f"wb_body_collision_{region}": f"training/body_collision_{region}_rate"
       for region in ("feet", "legs", "trunk", "head", "arms", "hands")},
    "wb_hand_violation": "training/hand_violation_rate",
    "wb_elbow_violation": "training/elbow_violation_rate",
    "wb_outside_bounds": "training/outside_bounds_rate",
    "wb_self_contact": "training/self_contact_rate",
    "wb_numerical": "training/numerical_failure_rate",
}


def _current_distribution_metrics(parametric_distribution, scales):
    """Reduce actual bounded scales on device; only scalar values reach the host."""
    config = getattr(parametric_distribution, "config", None)
    if not config or config.get("kind") not in (
            "wholebody_bounded_normal_tanh_v1", "wholebody_arm_conditional_correlated_tanh_v2"):
        return {}
    split = config["leg_action_count"]
    upper = scales[..., split:]
    innovation_scale = config.get("arm_innovation_scale", 1.)
    bound_scale = jnp.ones(upper.shape[-1], upper.dtype)
    if config.get("kind") == "wholebody_arm_conditional_correlated_tanh_v2":
        bound_scale = bound_scale.at[3:].set(innovation_scale)
    lower, higher = config["upper_std_min"] * bound_scale, config["upper_std_max"] * bound_scale
    invalid = (~jnp.isfinite(upper) | (upper < lower) | (upper > higher))
    metrics = {
        "training/leg_std_mean": jnp.mean(scales[..., :split]),
        "training/upper_std_mean": jnp.mean(upper),
        "training/upper_std_min": jnp.min(upper),
        "training/upper_std_max": jnp.max(upper),
        "training/upper_std_bounds_violation_rate": jnp.mean(invalid.astype(jnp.float32)),
        "training/upper_std_at_floor_fraction": jnp.mean((upper <= lower).astype(jnp.float32)),
        "training/upper_std_at_ceiling_fraction": jnp.mean((upper >= higher).astype(jnp.float32)),
    }
    if config.get("kind") == "wholebody_arm_conditional_correlated_tanh_v2":
        metrics.update({
            "training/arm_conditional_std_mean": jnp.mean(scales[..., 15:]),
            "training/arm_conditional_std_at_floor_fraction": jnp.mean(
                (scales[..., 15:] <= config["upper_std_min"] * innovation_scale).astype(jnp.float32)),
            "training/arm_conditional_std_at_ceiling_fraction": jnp.mean(
                (scales[..., 15:] >= config["upper_std_max"] * innovation_scale).astype(jnp.float32)),
            "training/arm_persistence": jnp.asarray(config["arm_persistence"]),
            "training/arm_innovation_scale": jnp.asarray(innovation_scale),
            "training/arm_stationary_std_proxy_mean": jnp.mean(scales[..., 15:]) / jnp.sqrt(
                1. - config["arm_persistence"] ** 2),
        })
    return metrics


class TrainingMetricsLogger:
    """Aggregates rollout episode metrics without flattening names under one prefix."""

    def __init__(self, buffer_size=100, steps_between_logging=1e5, progress_fn=None):
        self._metrics_buffer = collections.defaultdict(
            lambda: collections.deque(maxlen=buffer_size)
        )
        self._rollout_buffer = collections.defaultdict(
            lambda: collections.deque(maxlen=buffer_size)
        )
        self._buffer_size = buffer_size
        self._steps_between_logging = steps_between_logging
        self._num_steps = 0
        self._completed_episodes = 0
        self._last_log_steps = 0
        self._log_count = 0
        self._progress_fn = progress_fn
        self._latest_rollout_metrics = {}

    def state_dict(self):
        """Preserve the shared episode windows and logging axis across a resume."""
        return {
            "buffer_size": self._buffer_size,
            "steps_between_logging": self._steps_between_logging,
            "num_steps": self._num_steps,
            "completed_episodes": self._completed_episodes,
            "last_log_steps": self._last_log_steps,
            "log_count": self._log_count,
            "metrics_buffer": {k: list(v) for k, v in self._metrics_buffer.items()},
            "rollout_buffer": {k: list(v) for k, v in self._rollout_buffer.items()},
            "latest_rollout_metrics": dict(self._latest_rollout_metrics),
        }

    def load_state_dict(self, state):
        if (state["buffer_size"] != self._buffer_size or
                state["steps_between_logging"] != self._steps_between_logging):
            raise ValueError("Runtime metric logger configuration differs")
        for name in ("num_steps", "completed_episodes", "last_log_steps", "log_count"):
            setattr(self, f"_{name}", int(state[name]))
        for name in ("metrics_buffer", "rollout_buffer"):
            target = getattr(self, f"_{name}")
            target.clear()
            for key, values in state[name].items():
                target[key].extend(values)
        # Older snapshots predate current-rollout telemetry.
        self._latest_rollout_metrics = dict(state.get("latest_rollout_metrics", {}))

    def discard_windows(self):
        """A rollback starts fresh episode windows without rewinding the logging axis."""
        self._metrics_buffer.clear()
        self._rollout_buffer.clear()
        self._latest_rollout_metrics.clear()

    @staticmethod
    def _metric_key(name: str) -> str:
        if name.startswith("reward/"):
            return f"episode/{name}"
        if name in ("sum_reward", "length"):
            return f"episode/{name}"
        return f"episode_metrics/{name}"

    def update_rollout_metrics(self, metrics, dones, truncations, pf_ids=None, action_std=None,
                               distribution_metrics=None):
        dones = np.asarray(dones).astype(bool)
        truncations = np.asarray(truncations).astype(bool)
        pf_ids = None if pf_ids is None else np.asarray(pf_ids).astype(np.int32)
        self._num_steps += int(np.prod(dones.shape))
        if action_std is not None:
            self._rollout_buffer["training/action_std"].append(float(np.asarray(action_std)))

        done_count = int(np.sum(dones))
        # These values describe this callback's rollout, never the historical
        # 1000-callback buffer.  Replace the dictionary to prevent stale rates
        # when a rollout contains no completed episodes or a key is unavailable.
        self._latest_rollout_metrics = {"training/completed_episode_count": done_count}
        if distribution_metrics is not None:
            self._latest_rollout_metrics.update({
                key: float(np.asarray(value)) for key, value in distribution_metrics.items()
            })
        self._completed_episodes += done_count
        if done_count > 0:
            timeout_count = int(np.sum(dones & truncations))
            termination_count = done_count - timeout_count
            self._latest_rollout_metrics.update({
                "training/timeout_rate": timeout_count / done_count,
                "training/termination_rate": termination_count / done_count,
            })
            total_count = int(np.prod(dones.shape))
            self._rollout_buffer["done_rate"].append(done_count / total_count)
            self._rollout_buffer["termination_step_rate"].append(termination_count / total_count)
            self._rollout_buffer["timeout_step_rate"].append(timeout_count / total_count)
            self._rollout_buffer["timeout_rate"].append(timeout_count / done_count)
            self._rollout_buffer["termination_rate"].append(termination_count / done_count)
            self._rollout_buffer["episodes"].append(done_count)
            if pf_ids is not None:
                done_pf_ids = pf_ids[dones]
                done_truncations = truncations[dones]
                for pf_id in np.unique(done_pf_ids):
                    pf_mask = done_pf_ids == pf_id
                    pf_done_count = int(np.sum(pf_mask))
                    pf_timeout_count = int(np.sum(done_truncations[pf_mask]))
                    pf_termination_count = pf_done_count - pf_timeout_count
                    prefix = f"pfid/{int(pf_id)}"
                    self._rollout_buffer[f"{prefix}/episodes"].append(pf_done_count)
                    self._rollout_buffer[f"{prefix}/timeout_rate"].append(pf_timeout_count / pf_done_count)
                    self._rollout_buffer[f"{prefix}/termination_rate"].append(
                        pf_termination_count / pf_done_count
                    )

            done_mask = dones.astype(bool)
            for name, metric in metrics.items():
                metric = np.asarray(metric)
                done_metrics = metric[done_mask].flatten().tolist()
                self._metrics_buffer[self._metric_key(name)].extend(done_metrics)
                if name in _WHOLEBODY_EPISODE_RATE_KEYS:
                    self._latest_rollout_metrics[_WHOLEBODY_EPISODE_RATE_KEYS[name]] = float(
                        np.mean(done_metrics)
                    )
                elif name.startswith("wb_mean_"):
                    # The wrapper has already normalized each terminal metric
                    # by its own episode length; average completed episodes.
                    self._latest_rollout_metrics[f"training/{name[3:]}"] = float(
                        np.mean(done_metrics)
                    )
        else:
            self._rollout_buffer["done_rate"].append(0.0)
            self._rollout_buffer["termination_step_rate"].append(0.0)
            self._rollout_buffer["timeout_step_rate"].append(0.0)
            self._rollout_buffer["timeout_rate"].append(0.0)
            self._rollout_buffer["termination_rate"].append(0.0)
            self._rollout_buffer["episodes"].append(0.0)

        if self._num_steps - self._last_log_steps >= self._steps_between_logging:
            self.log_metrics()
            self._last_log_steps = self._num_steps

    def log_metrics(self, pad=35):
        self._log_count += 1
        log_string = f"\n{'Steps':>{pad}} Env: {self._num_steps} Log: {self._log_count}\n"
        mean_metrics = {"rollout/completed_episodes": self._completed_episodes,
                        "rollout/completed_episodes_in_buffer": len(self._metrics_buffer.get("episode/length", ()))}
        for metric_name in self._metrics_buffer:
            mean_metrics[metric_name] = np.mean(self._metrics_buffer[metric_name])
            log_string += f"{f'{metric_name}:':>{pad}} {mean_metrics[metric_name]:.4f}\n"
        for metric_name in self._rollout_buffer:
            key = metric_name if metric_name.startswith("training/") else f"rollout/{metric_name}"
            mean_metrics[key] = np.mean(self._rollout_buffer[metric_name])
            log_string += f"{f'{key}:':>{pad}} {mean_metrics[key]:.4f}\n"
        mean_metrics.update(self._latest_rollout_metrics)
        for key, value in self._latest_rollout_metrics.items():
            log_string += f"{f'{key}:':>{pad}} {value:.4f}\n"
        logging.info(log_string)
        if self._progress_fn is not None:
            self._progress_fn(int(self._num_steps), mean_metrics)


def _validate_madrona_args(
    madrona_backend: bool,
    num_envs: int,
    num_eval_envs: int,
    action_repeat: int,
    eval_env: Optional[envs.Env] = None,
):
    """Validates arguments for Madrona-MJX."""
    if madrona_backend:
        if eval_env:
            raise ValueError("Madrona-MJX doesn't support multiple env instances")
        if num_eval_envs != num_envs:
            raise ValueError("Madrona-MJX requires a fixed batch size")
        if action_repeat != 1:
            raise ValueError(
                "Implement action_repeat using PipelineEnv's _n_frames to avoid unnecessary rendering!"
            )


def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    local_device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
    """Wraps the environment for training/eval if wrap_env is True."""
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError("episode_length must be specified in ppo.train")
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        # all devices gets the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)
    if wrap_env_fn is not None:
        wrap_for_training = wrap_env_fn
    else:
        wrap_for_training = envs.training.wrap
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )  # pytype: disable=wrong-keyword-args
    return env


def _random_translate_pixels(
    obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
    """Apply random translations to B x T x ... pixel observations.

    The same shift is applied across the unroll_length (T) dimension.

    Args:
      obs: a dictionary of observations
      key: a PRNGKey

    Returns:
      A dictionary of observations with translated pixels
    """

    @jax.vmap
    def rt_all_views(
        ub_obs: Mapping[str, jax.Array], key: PRNGKey
    ) -> Mapping[str, jax.Array]:
        # Expects dictionary of unbatched observations.
        def rt_view(img: jax.Array, padding: int, key: PRNGKey) -> jax.Array:  # TxHxWxC
            # Randomly translates a set of pixel inputs.
            # Adapted from
            # https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
            crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
            zero = jnp.zeros((1,), dtype=jnp.int32)
            crop_from = jnp.concatenate([zero, crop_from, zero])
            padded_img = jnp.pad(
                img,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode="edge",
            )
            return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

        out = {}
        for k_view, v_view in ub_obs.items():
            if k_view.startswith("pixels/"):
                key, key_shift = jax.random.split(key)
                out[k_view] = rt_view(v_view, 4, key_shift)
        return {**ub_obs, **out}

    bdim = next(iter(obs.items()), None)[1].shape[0]
    keys = jax.random.split(key, bdim)
    obs = rt_all_views(obs, keys)
    return obs


def _remove_pixels(
    obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
    """Removes pixel observations from the observation dict."""
    if not isinstance(obs, Mapping):
        return obs
    return {k: v for k, v in obs.items() if not k.startswith("pixels/")}


def _cfg_get(config: Optional[Any], name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _stack_trees(trees):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trees)


def _mlp_network_factory_kwargs(network_factory_config: Any) -> dict[str, Any]:
    kwargs = dict(network_factory_config)
    network_kind = kwargs.pop("network_kind", "mlp")
    if network_kind != "mlp":
        raise ValueError(f"DAgger teacher network_kind={network_kind!r} is not supported")
    for key in (
        "transformer_embed_dim",
        "transformer_num_heads",
        "transformer_ff_dim",
        "transformer_num_layers",
    ):
        kwargs.pop(key, None)
    return kwargs


def _assert_same_tree_shapes(label: str, expected: Any, actual: Any):
    expected_leaves, expected_def = jax.tree_util.tree_flatten(expected)
    actual_leaves, actual_def = jax.tree_util.tree_flatten(actual)
    if expected_def != actual_def:
        raise ValueError(f"{label} pytree structure does not match")
    for idx, (expected_leaf, actual_leaf) in enumerate(zip(expected_leaves, actual_leaves)):
        if np.shape(expected_leaf) != np.shape(actual_leaf):
            raise ValueError(
                f"{label} leaf {idx} shape mismatch: expected {np.shape(expected_leaf)}, "
                f"got {np.shape(actual_leaf)}"
            )


def _assert_policy_shapes_for_dagger_teacher(
    label: str,
    expected: Any,
    actual: Any,
    allow_input_dim_mismatch: bool,
):
    if not allow_input_dim_mismatch:
        _assert_same_tree_shapes(label, expected, actual)
        return

    expected_items, expected_def = jax.tree_util.tree_flatten_with_path(expected)
    actual_items, actual_def = jax.tree_util.tree_flatten_with_path(actual)
    if expected_def != actual_def:
        raise ValueError(f"{label} pytree structure does not match")

    for idx, ((path, expected_leaf), (_, actual_leaf)) in enumerate(zip(expected_items, actual_items)):
        expected_shape = np.shape(expected_leaf)
        actual_shape = np.shape(actual_leaf)
        if expected_shape == actual_shape:
            continue
        path_keys = [getattr(entry, "key", str(entry)) for entry in path]
        is_first_kernel = path_keys[-2:] == ["hidden_0", "kernel"]
        same_output_dim = (
            len(expected_shape) == 2
            and len(actual_shape) == 2
            and expected_shape[1:] == actual_shape[1:]
        )
        if is_first_kernel and same_output_dim:
            continue
        raise ValueError(
            f"{label} leaf {idx} shape mismatch: expected {expected_shape}, "
            f"got {actual_shape}"
        )


def _teacher_observation(
    observation: Any,
    teacher_obs_key: Optional[str],
    teacher_privileged_obs_key: Optional[str],
) -> Any:
    if teacher_obs_key is None:
        return observation
    if not isinstance(observation, Mapping):
        raise ValueError("teacher_obs_key requires dict observations")
    if teacher_obs_key not in observation:
        raise ValueError(f"Missing DAgger teacher obs key: {teacher_obs_key}")

    teacher_obs = {"state": observation[teacher_obs_key]}
    if teacher_privileged_obs_key is not None:
        if teacher_privileged_obs_key not in observation:
            raise ValueError(
                f"Missing DAgger teacher privileged obs key: {teacher_privileged_obs_key}"
            )
        teacher_obs["privileged_state"] = observation[teacher_privileged_obs_key]
    elif "privileged_state" in observation:
        teacher_obs["privileged_state"] = observation["privileged_state"]
    return teacher_obs


def _teacher_observation_shape(
    observation_size: Any,
    teacher_obs_key: Optional[str],
    teacher_privileged_obs_key: Optional[str],
) -> Any:
    return _teacher_observation(
        observation_size, teacher_obs_key, teacher_privileged_obs_key
    )


def _uint64_lt(value: types.UInt64, other: int) -> jnp.ndarray:
    other_hi = jnp.uint32(other >> 32)
    other_lo = jnp.uint32(other & 0xFFFFFFFF)
    return (value.hi < other_hi) | ((value.hi == other_hi) & (value.lo < other_lo))


def compute_dagger_loss(
    params: ppo_losses.PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: ppo_networks.PPONetworks,
    teacher_ppo_network: ppo_networks.PPONetworks,
    teacher_normalizer_params: Any,
    teacher_policy_params: Any,
    num_teachers: int,
    kl_eps: float = 1e-5,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    actor_loss_scale: float = 1.0,
    value_loss_scale: float = 1.0,
    teacher_obs_key: Optional[str] = None,
    teacher_privileged_obs_key: Optional[str] = None,
) -> Tuple[jnp.ndarray, types.Metrics]:
    del rng
    policy_apply = ppo_network.policy_network.apply
    teacher_policy_apply = teacher_ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply
    parametric_action_distribution = ppo_network.parametric_action_distribution

    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    student_logits = policy_apply(normalizer_params, params.policy, data.observation)
    baseline = value_apply(normalizer_params, params.value, data.observation)
    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, terminal_obs)

    teacher_observation = _teacher_observation(
        data.observation, teacher_obs_key, teacher_privileged_obs_key
    )

    def teacher_apply(teacher_normalizer, teacher_policy):
        return teacher_policy_apply(teacher_normalizer, teacher_policy, teacher_observation)

    teacher_logits_all = jax.vmap(teacher_apply)(teacher_normalizer_params, teacher_policy_params)
    pf_id = data.extras["state_extras"]["pf_id"].astype(jnp.int32)
    pf_id = jnp.clip(pf_id, 0, num_teachers - 1)
    teacher_weight = jax.nn.one_hot(pf_id, num_teachers, dtype=student_logits.dtype)
    teacher_logits = jnp.einsum("tbn,ntba->tba", teacher_weight, teacher_logits_all)

    student_dist = parametric_action_distribution.create_dist(student_logits)
    teacher_dist = parametric_action_distribution.create_dist(teacher_logits)
    student_std = jnp.maximum(student_dist.scale, kl_eps)
    teacher_std = jnp.maximum(teacher_dist.scale, kl_eps)
    log_term = jnp.log(teacher_std / student_std + kl_eps)
    numerator = jnp.square(student_std) + jnp.square(student_dist.loc - teacher_dist.loc)
    denominator = 2.0 * jnp.square(teacher_std)
    dagger_kl = jnp.sum(log_term + numerator / denominator - 0.5, axis=-1)
    dagger_kl = jnp.mean(dagger_kl)

    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)
    vs, _ = ppo_losses.compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )
    v_loss = jnp.mean(jnp.square(vs - baseline)) * 0.5 * 0.5

    actor_loss = actor_loss_scale * dagger_kl
    value_loss = value_loss_scale * v_loss
    total_loss = actor_loss + value_loss
    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": actor_loss,
        "v_loss": value_loss,
        "entropy_loss": jnp.zeros_like(total_loss),
        "dagger_kl": dagger_kl,
        "student_std": jnp.mean(student_std),
        "teacher_std": jnp.mean(teacher_std),
        "loss_mode": jnp.ones_like(total_loss),
    }


def compute_dagger_then_ppo_loss(
    params: ppo_losses.PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    dagger_phase: jnp.ndarray,
    ppo_network: ppo_networks.PPONetworks,
    teacher_ppo_network: ppo_networks.PPONetworks,
    teacher_normalizer_params: Any,
    teacher_policy_params: Any,
    num_teachers: int,
    kl_eps: float = 1e-5,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    actor_loss_scale: float = 1.0,
    value_loss_scale: float = 1.0,
    teacher_obs_key: Optional[str] = None,
    teacher_privileged_obs_key: Optional[str] = None,
) -> Tuple[jnp.ndarray, types.Metrics]:
    def dagger_loss(_):
        return compute_dagger_loss(
            params,
            normalizer_params,
            data,
            rng,
            ppo_network=ppo_network,
            teacher_ppo_network=teacher_ppo_network,
            teacher_normalizer_params=teacher_normalizer_params,
            teacher_policy_params=teacher_policy_params,
            num_teachers=num_teachers,
            kl_eps=kl_eps,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            actor_loss_scale=actor_loss_scale,
            value_loss_scale=value_loss_scale,
            teacher_obs_key=teacher_obs_key,
            teacher_privileged_obs_key=teacher_privileged_obs_key,
        )

    def ppo_loss(_):
        total_loss, metrics = ppo_losses.compute_ppo_loss(
            params,
            normalizer_params,
            data,
            rng,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage,
        )
        metrics = {
            **metrics,
            "dagger_kl": jnp.zeros_like(total_loss),
            "student_std": jnp.zeros_like(total_loss),
            "teacher_std": jnp.zeros_like(total_loss),
            "loss_mode": jnp.zeros_like(total_loss),
        }
        return total_loss, metrics

    return jax.lax.cond(dagger_phase, dagger_loss, ppo_loss, operand=None)


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    # high-level control flow
    wrap_env: bool = True,
    madrona_backend: bool = False,
    augment_pixels: bool = False,
    # environment wrapper
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    randomize_initial_episode_steps: bool = True,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # ppo params
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    seed: int = 0,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    training_metrics_buffer_size: int = 10,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = False,
    dagger_config: Optional[Any] = None,
    scored_checkpoint_fn: Optional[Callable[..., None]] = None,
    num_training_epochs: Optional[int] = None,
    eval_episode_length: Optional[int] = None,
    should_stop_fn: Optional[Callable[[], bool]] = None,
    continuous: bool = False,
    training_steps_per_epoch: int = 1,
    runtime_checkpoint_fn: Optional[Callable[[int, Mapping[str, Any]], None]] = None,
    restore_runtime_state: Optional[Mapping[str, Any]] = None,
    runtime_metadata: Optional[Mapping[str, Any]] = None,
    reference_kl_config: Optional[Mapping[str, Any]] = None,
    recovery_fn: Optional[Callable[[int], Optional[Mapping[str, Any]]]] = None,
    sapg_config: Optional[Mapping[str, Any]] = None,
):
    """PPO training.

    Args:
      environment: the environment to train
      num_timesteps: the total number of environment steps to use during training
      max_devices_per_host: maximum number of chips to use per host process
      wrap_env: If True, wrap the environment for training. Otherwise use the
        environment as is.
      madrona_backend: whether to use Madrona backend for training
      augment_pixels: whether to add image augmentation to pixel inputs
      num_envs: the number of parallel environments to use for rollouts
        NOTE: `num_envs` must be divisible by the total number of chips since each
          chip gets `num_envs // total_number_of_chips` environments to roll out
        NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
          data generated by `num_envs` parallel envs gets used for gradient
          updates over `num_minibatches` of data, where each minibatch has a
          leading dimension of `batch_size`
      episode_length: the length of an environment episode
      action_repeat: the number of timesteps to repeat an action
      randomize_initial_episode_steps: whether to randomize the EpisodeWrapper
        steps counter after the initial training reset to avoid synchronized
        timeout waves across vectorized environments. This does not affect eval
        or later autoresets.
      wrap_env_fn: a custom function that wraps the environment for training. If
        not specified, the environment is wrapped with the default training
        wrapper.
      randomization_fn: a user-defined callback function that generates randomized
        environments
      learning_rate: learning rate for ppo loss
      entropy_cost: entropy reward for ppo loss, higher values increase entropy of
        the policy
      discounting: discounting rate
      unroll_length: the number of timesteps to unroll in each environment. The
        PPO loss is computed over `unroll_length` timesteps
      batch_size: the batch size for each minibatch SGD step
      num_minibatches: the number of times to run the SGD step, each with a
        different minibatch with leading dimension of `batch_size`
      num_updates_per_batch: the number of times to run the gradient update over
        all minibatches before doing a new environment rollout
      num_resets_per_eval: the number of environment resets to run between each
        eval. The environment resets occur on the host
      normalize_observations: whether to normalize observations
      reward_scaling: float scaling for reward
      clipping_epsilon: clipping epsilon for PPO loss
      gae_lambda: General advantage estimation lambda
      max_grad_norm: gradient clipping norm value. If None, no clipping is done
      normalize_advantage: whether to normalize advantage estimate
      network_factory: function that generates networks for policy and value
        functions
      seed: random seed
      num_evals: the number of evals to run during the entire training run.
        Increasing the number of evals increases total training time
      eval_env: an optional environment for eval only, defaults to `environment`
      num_eval_envs: the number of envs to use for evluation. Each env will run 1
        episode, and all envs run in parallel during eval.
      deterministic_eval: whether to run the eval with a deterministic policy
      log_training_metrics: whether to log training metrics and callback to
        progress_fn
      training_metrics_steps: the number of environment steps between logging
        training metrics
      training_metrics_buffer_size: log buf size
      progress_fn: a user-defined callback function for reporting/plotting metrics
      policy_params_fn: a user-defined callback function that can be used for
        saving custom policy checkpoints or creating policy rollouts and videos
      save_checkpoint_path: the path used to save checkpoints. If None, no
        checkpoints are saved.
      restore_checkpoint_path: the path used to restore previous model params
      restore_params: raw network parameters to restore the TrainingState from.
        These override `restore_checkpoint_path`. These paramaters can be obtained
        from the return values of ppo.train().
      restore_value_fn: whether to restore the value function from the checkpoint
        or use a random initialization
      continuous: keep the same PPO learner and environments running until
        should_stop_fn requests a cooperative stop. Requires num_evals=0 and
        num_resets_per_eval=0; num_timesteps is ignored in this mode.
      training_steps_per_epoch: completed PPO updates between host callbacks in
        continuous mode. A PPO update retains the native batch accumulation:
        batch_size * num_minibatches / num_envs rollout chunks.
      runtime_checkpoint_fn: receives a complete, host-resident snapshot after
        initialization (step zero) and each callback epoch, including Adam, RNG,
        environment and sampler state. Exact resumes retain their existing
        initial snapshot. This is independent of the best-model callback.
      restore_runtime_state: exact snapshot from runtime_checkpoint_fn. Cannot
        be combined with either params-only restore argument.
      runtime_metadata: caller's immutable run contract (e.g. scene hashes,
        observation contract and code version), checked on exact resume.
      reference_kl_config: optional coefficient, action_indices and boolean
        scene_mask for frozen initial actor retention on the same observations.
        Adds mean selected-transition KL(reference || current), summing selected
        action dimensions. Frozen actor/statistics are included in exact resumes.

    Returns:
      Tuple of (make_policy function, network params, metrics)
    """
    from cat_ppo.learning.policy.sapg.config import normalize_config
    sapg_config = normalize_config(sapg_config)
    if sapg_config is not None:
        if (num_evals or normalize_observations or augment_pixels or madrona_backend
                or bool(_cfg_get(dagger_config, "enable", False))
                or reference_kl_config is not None or recovery_fn is not None
                or save_checkpoint_path is not None or restore_checkpoint_path is not None):
            raise ValueError("SAPG supports vector-observation training only, without evaluation, "
                             "normalization, DAgger, retention, or generic checkpoint loading/saving; "
                             "use the whole-body launcher for original-model initialization and leader export")
        groups = sapg_config["num_policies"]
        if num_envs % groups or batch_size % groups:
            raise ValueError("SAPG num_envs and batch_size must be divisible by num_policies")
    if continuous:
        if num_evals != 0 or num_resets_per_eval != 0 or num_training_epochs is not None:
            raise ValueError("continuous training requires num_evals=0, num_resets_per_eval=0 and no num_training_epochs")
        if type(training_steps_per_epoch) is not int or training_steps_per_epoch < 1:
            raise ValueError("training_steps_per_epoch must be a positive integer")
        if should_stop_fn is None:
            raise ValueError("continuous training requires a cooperative should_stop_fn")
        if save_checkpoint_path is not None:
            raise ValueError("continuous training requires a bounded checkpoint callback, not save_checkpoint_path")
    if restore_runtime_state is not None and (restore_checkpoint_path is not None or restore_params is not None):
        raise ValueError("Exact runtime restore cannot be combined with params-only restore")
    reference_config = normalize_reference_config(reference_kl_config, environment.action_size)
    reference_state = None
    if reference_config is not None and all(value is None for value in (
            restore_params, restore_checkpoint_path, restore_runtime_state)):
        raise ValueError("Reference regularization requires a warm-start checkpoint/params or exact runtime")
    assert batch_size * num_minibatches % num_envs == 0
    _validate_madrona_args(
        madrona_backend, num_envs, num_eval_envs, action_repeat, eval_env
    )

    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count
    if sapg_config is not None and device_count != 1:
        raise ValueError("SAPG currently supports exactly one device on one host")
    if (continuous or runtime_checkpoint_fn is not None or restore_runtime_state is not None) and process_count != 1:
        raise ValueError("Continuous runtime snapshots currently require a single host")

    # The number of environment steps executed for every training step.
    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat
    )
    num_evals_after_init = max(num_evals - 1, 1)
    if num_training_epochs is not None:
        if num_evals != 0 or num_training_epochs < 1:
            raise ValueError("num_training_epochs requires num_evals=0 and a positive epoch count")
        num_evals_after_init = num_training_epochs
    # The number of training_step calls per training_epoch call.
    # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
    #                                 num_resets_per_eval))
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        )
    ).astype(int)
    if continuous:
        num_training_steps_per_epoch = training_steps_per_epoch

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value = jax.random.split(global_key)
    del global_key

    assert num_envs % device_count == 0

    env = _maybe_wrap_env(
        environment,
        wrap_env,
        num_envs,
        episode_length,
        action_repeat,
        local_device_count,
        key_env,
        wrap_env_fn,
        randomization_fn,
    )
    # Keep large immutable field banks out of the compiled HLO constants. These
    # shared operands never enter per-environment state, rollouts or checkpoints.
    field_arguments = FieldArguments(environment)
    field_values = field_arguments.values

    def reset_with_field_arguments(keys, fields):
        with field_arguments.bind(fields):
            return jax.vmap(env.reset)(keys)

    reset_fn = jax.jit(reset_with_field_arguments)
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs, field_values)
    has_scene_ids = "pf_id" in env_state.info
    if reference_config is not None:
        if not has_scene_ids:
            raise ValueError("Reference regularization requires pre-autoreset pf_id scene attribution")
        count = getattr(environment, "num_pf_scenes", None)
        if count is not None and len(reference_config["scene_mask"]) != count:
            raise ValueError("Reference scene mask must match the environment scene count")
    if randomize_initial_episode_steps and "steps" in env_state.info:
        if episode_length is None:
            raise ValueError("episode_length must be specified to randomize initial episode steps")
        initial_steps = jax.random.randint(
            jax.random.fold_in(key_env, 1),
            env_state.info["steps"].shape,
            minval=0,
            maxval=episode_length,
            dtype=jnp.int32,
        )
        env_state.info["steps"] = initial_steps.astype(env_state.info["steps"].dtype)
    # Discard the batch axes over devices and envs.
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    ppo_network = network_factory(
        obs_shape, env.action_size, preprocess_observations_fn=normalize
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    make_rollout_policy = make_policy
    if sapg_config is not None:
        from cat_ppo.learning.policy.sapg import losses as sapg_losses, networks as sapg_networks
        make_policy = sapg_networks.make_inference_fn(ppo_network)
        policy_ids = jnp.repeat(jnp.arange(sapg_config["num_policies"], dtype=jnp.int32),
                               num_envs // sapg_config["num_policies"])
        make_rollout_policy = sapg_networks.make_inference_fn(ppo_network, policy_ids=policy_ids)

    use_dagger = bool(_cfg_get(dagger_config, "enable", False))
    if use_dagger and reference_config is not None:
        raise ValueError("Frozen reference retention is additive to PPO; simultaneous DAgger is unsupported")
    teacher_checkpoint_paths = list(_cfg_get(dagger_config, "teacher_checkpoint_paths", []))
    dagger_timesteps = int(_cfg_get(dagger_config, "dagger_timesteps", 0))
    teacher_params = ()
    teacher_normalizer_params = None
    teacher_policy_params = None
    teacher_ppo_network = None
    if use_dagger:
        loss_name = _cfg_get(dagger_config, "loss", "kl")
        if loss_name != "kl":
            raise ValueError(f"Unsupported DAgger loss: {loss_name!r}; only 'kl' is implemented")
        if not teacher_checkpoint_paths:
            raise ValueError("dagger_config.enable=True requires teacher_checkpoint_paths")
        teacher_obs_key = _cfg_get(dagger_config, "teacher_obs_key", None)
        teacher_privileged_obs_key = _cfg_get(
            dagger_config, "teacher_privileged_obs_key", None
        )
        teacher_network_factory = _cfg_get(dagger_config, "teacher_network_factory", None)
        if teacher_network_factory is None:
            raise ValueError("dagger_config.teacher_network_factory must be set for DAgger")
        teacher_ppo_network = ppo_networks.make_ppo_networks(
            _teacher_observation_shape(
                obs_shape, teacher_obs_key, teacher_privileged_obs_key
            ),
            env.action_size,
            preprocess_observations_fn=normalize,
            **_mlp_network_factory_kwargs(teacher_network_factory),
        )
        teacher_params = tuple(checkpoint.load(path) for path in teacher_checkpoint_paths)
        teacher_normalizer_params = _stack_trees([params[0] for params in teacher_params])
        teacher_policy_params = _stack_trees([params[1] for params in teacher_params])

    adam = optax.inject_hyperparams(optax.adam) if recovery_fn is not None else optax.adam
    optimizer = adam(learning_rate=learning_rate)
    if max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            adam(learning_rate=learning_rate),
        )

    if use_dagger:
        loss_fn = functools.partial(
            compute_dagger_then_ppo_loss,
            ppo_network=ppo_network,
            teacher_ppo_network=teacher_ppo_network,
            teacher_normalizer_params=teacher_normalizer_params,
            teacher_policy_params=teacher_policy_params,
            num_teachers=len(teacher_checkpoint_paths),
            kl_eps=_cfg_get(dagger_config, "kl_eps", 1e-5),
            teacher_obs_key=_cfg_get(dagger_config, "teacher_obs_key", None),
            teacher_privileged_obs_key=_cfg_get(
                dagger_config, "teacher_privileged_obs_key", None
            ),
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage,
            actor_loss_scale=_cfg_get(dagger_config, "actor_loss_scale", 1.0),
            value_loss_scale=_cfg_get(dagger_config, "value_loss_scale", 1.0),
        )
    else:
        loss_fn = functools.partial(
            ppo_losses.compute_ppo_loss,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage,
        )

    if sapg_config is not None:
        loss_fn = functools.partial(
            sapg_losses.compute_sapg_loss, ppo_network=ppo_network,
            entropy_cost=entropy_cost, clipping_epsilon=clipping_epsilon,
            num_policies=sapg_config["num_policies"])

    if reference_config is not None:
        original_ppo_loss = loss_fn

        def loss_with_reference(params, normalizer_params, data, rng):
            loss, metrics = original_ppo_loss(params, normalizer_params, data, rng)
            penalty, reference_metrics = reference_regularization(
                params, normalizer_params, data, network=ppo_network,
                reference=reference_state, config=reference_config)
            total = loss + penalty
            return total, {**metrics, **reference_metrics, "total_loss": total}

        loss_fn = loss_with_reference

    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    metrics_aggregator = TrainingMetricsLogger(
        buffer_size=training_metrics_buffer_size,
        steps_between_logging=training_metrics_steps or env_step_per_training_step,
        progress_fn=progress_fn,
    )

    ckpt_config = checkpoint.network_config(
        observation_size=obs_shape,
        action_size=env.action_size,
        normalize_observations=normalize_observations,
        network_factory=network_factory,
    )
    if sapg_config is not None:
        # Best exports fold one embedding into the first-layer biases. Their
        # metadata must describe the resulting ordinary CAT/Brax network.
        for name in ("num_policies", "embedding_dim"):
            if name in ckpt_config.network_factory_kwargs:
                del ckpt_config.network_factory_kwargs[name]

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
        dagger_phase: jnp.ndarray,
    ):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        if use_dagger:
            (_, metrics), params, optimizer_state = gradient_update_fn(
                params,
                normalizer_params,
                data,
                key_loss,
                dagger_phase,
                optimizer_state=optimizer_state,
            )
        else:
            (_, metrics), params, optimizer_state = gradient_update_fn(
                params,
                normalizer_params,
                data,
                key_loss,
                optimizer_state=optimizer_state,
            )

        return (optimizer_state, params, key), metrics

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
        dagger_phase: jnp.ndarray,
    ):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        if augment_pixels:
            key, key_rt = jax.random.split(key)
            r_translate = functools.partial(_random_translate_pixels, key=key_rt)
            data = types.Transition(
                observation=r_translate(data.observation),
                action=data.action,
                reward=data.reward,
                discount=data.discount,
                next_observation=r_translate(data.next_observation),
                extras=data.extras,
            )

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                minibatch_step,
                normalizer_params=normalizer_params,
                dagger_phase=dagger_phase,
            ),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key), metrics

    def training_step(
        carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_rollout_policy(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )
        )

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            extra_fields = ("truncation", "episode_metrics", "episode_done")
            generate_unroll = _generate_unroll_with_scene_ids if has_scene_ids else acting.generate_unroll
            next_state, data = generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=extra_fields,
            )
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        assert data.discount.shape[1:] == (unroll_length,)

        optimization_data = data
        rollout_reward = jnp.mean(data.reward)
        if sapg_config is not None:
            key_sgd, key_prepare = jax.random.split(key_sgd)
            optimization_data = sapg_losses.prepare_rollout(
                training_state.params, training_state.normalizer_params, data, key_prepare,
                ppo_network, num_policies=sapg_config["num_policies"],
                discounting=discounting, reward_scaling=reward_scaling, gae_lambda=gae_lambda,
                normalize_advantage=normalize_advantage,
                preparation_chunk_size=sapg_config["prepare_chunk_size"])
            leader = data.extras["policy_extras"]["policy_id"] == 0
            rollout_reward = jnp.sum(jnp.where(leader, data.reward, 0.)) / jnp.sum(leader)

        if log_training_metrics:  # log unroll metrics
            if sapg_config is not None:
                # Preparation already computes this in bounded chunks; avoid
                # materializing a full-rollout actor activation tensor again.
                action_std = jnp.mean(optimization_data.extras["policy_extras"]["sapg_action_std"][:data.reward.shape[0]])
                distribution_metrics = {}
            else:
                rollout_logits = ppo_network.policy_network.apply(
                    training_state.normalizer_params,
                    training_state.params.policy,
                    data.observation,
                )
                parametric_distribution = ppo_network.parametric_action_distribution
                rollout_scales = parametric_distribution.create_dist(rollout_logits).scale
                action_std = jnp.mean(rollout_scales)
                distribution_metrics = _current_distribution_metrics(
                    parametric_distribution, rollout_scales,
                )
            # Stabilized runs use unified current-rollout rates plus fixed
            # validation. Avoid thousands of stale per-scene chart series.
            if has_scene_ids and not distribution_metrics:
                jax.debug.callback(
                    metrics_aggregator.update_rollout_metrics,
                    data.extras["state_extras"]["episode_metrics"],
                    data.extras["state_extras"]["episode_done"],
                    data.extras["state_extras"]["truncation"],
                    data.extras["state_extras"]["pf_id"],
                    action_std,
                    distribution_metrics,
                )
            else:
                jax.debug.callback(
                    metrics_aggregator.update_rollout_metrics,
                    data.extras["state_extras"]["episode_metrics"],
                    data.extras["state_extras"]["episode_done"],
                    data.extras["state_extras"]["truncation"],
                    None,
                    action_std,
                    distribution_metrics,
                )

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            _remove_pixels(data.observation),
            pmap_axis_name=_PMAP_AXIS_NAME,
        )
        dagger_phase = _uint64_lt(training_state.env_steps, dagger_timesteps)

        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step,
                data=optimization_data,
                normalizer_params=normalizer_params,
                dagger_phase=dagger_phase,
            ),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )
        metrics = {**metrics, "rollout_reward_mean": rollout_reward}
        return (new_training_state, state, new_key), metrics

    def training_epoch(
        training_state: TrainingState, state: envs.State, key: PRNGKey, fields
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        with field_arguments.bind(fields):
            (training_state, state, _), loss_metrics = jax.lax.scan(
                training_step,
                (training_state, state, key),
                (),
                length=num_training_steps_per_epoch,
            )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME,
                              in_axes=(0, 0, 0, None))

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key, field_values)
        training_state, env_state, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        # Read actual allocator statistics after completed GPU work. Keeping
        # this outside the compiled epoch avoids adding device callbacks and
        # distinguishes live tensors from the allocator's retained pool.
        for device in jax.local_devices()[:local_devices_to_use]:
            if device.platform == "gpu":
                memory = device.memory_stats() or {}
                for name in ("bytes_in_use", "peak_bytes_in_use", "bytes_reserved",
                             "peak_bytes_reserved", "bytes_limit", "pool_bytes"):
                    if name in memory:
                        metrics[f"training/gpu_{device.id}/{name}"] = memory[name]
        return (
            training_state,
            env_state,
            metrics,
        )  # pytype: disable=bad-return-type  # py311-upgrade

    # Initialize model params and training state.
    init_params = ppo_losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
    )
    if sapg_config is not None:
        table = init_params.policy.get("policy_embeddings")
        if table is None or table.shape != (sapg_config["num_policies"], sapg_config["embedding_dim"]):
            raise ValueError("SAPG requires make_sapg_networks with matching policy count and embedding dimension")
    if use_dagger:
        teacher_init_policy = teacher_ppo_network.policy_network.init(key_policy)
        for teacher_idx, loaded_params in enumerate(teacher_params):
            _assert_same_tree_shapes(
                f"DAgger teacher {teacher_idx} policy params",
                teacher_init_policy,
                loaded_params[1],
            )

    obs_shape = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs
    )
    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=optimizer.init(
            init_params
        ),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        normalizer_params=running_statistics.init_state(_remove_pixels(obs_shape)),
        env_steps=types.UInt64(hi=0, lo=0),
    )
    if use_dagger:
        teacher_obs_key = _cfg_get(dagger_config, "teacher_obs_key", None)
        if teacher_obs_key is None:
            for teacher_idx, loaded_params in enumerate(teacher_params):
                _assert_same_tree_shapes(
                    f"DAgger teacher {teacher_idx} normalizer params",
                    training_state.normalizer_params,
                    loaded_params[0],
                )

    if restore_checkpoint_path is not None:
        params = checkpoint.load(restore_checkpoint_path)
        value_params = params[2] if restore_value_fn else init_params.value
        training_state = training_state.replace(
            normalizer_params=params[0],
            params=training_state.params.replace(policy=params[1], value=value_params),
        )

    if restore_params is not None:
        logging.info("Restoring TrainingState from `restore_params`.")
        if sapg_config is not None:
            _assert_same_tree_shapes("SAPG restored actor", init_params.policy, restore_params[1])
            _assert_same_tree_shapes("SAPG restored normalizer", training_state.normalizer_params, restore_params[0])
            if restore_value_fn:
                _assert_same_tree_shapes("SAPG restored critic", init_params.value, restore_params[2])
        value_params = restore_params[2] if restore_value_fn else init_params.value
        training_state = training_state.replace(
            normalizer_params=restore_params[0],
            params=training_state.params.replace(
                policy=restore_params[1], value=value_params
            ),
        )

    if reference_config is not None:
        # On a new run these are the adapted warm-start parameters. On resume
        # this tree supplies only shape/type metadata; saved reference values
        # replace it below, before any optimizer trace or update is performed.
        reference_state = freeze_reference(training_state.params.policy,
                                           training_state.normalizer_params)

    if num_timesteps == 0 and not continuous and restore_runtime_state is None:
        return (
            make_policy,
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            ),
            {},
        )

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    runtime_contract = {
        "num_envs": num_envs, "local_devices": local_devices_to_use,
        "episode_length": episode_length, "action_repeat": action_repeat,
        "batch_size": batch_size, "num_minibatches": num_minibatches,
        "unroll_length": unroll_length, "num_updates_per_batch": num_updates_per_batch,
        "learning_rate": learning_rate, "entropy_cost": entropy_cost,
        "discounting": discounting, "reward_scaling": reward_scaling,
        "clipping_epsilon": clipping_epsilon, "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm, "normalize_advantage": normalize_advantage,
        "normalize_observations": normalize_observations,
        "training_steps_per_epoch": int(num_training_steps_per_epoch),
        "num_resets_per_eval": num_resets_per_eval, "num_evals": num_evals,
        "seed": seed, "use_dagger": use_dagger,
        "jax_version": jax.__version__, "flax_version": flax.__version__,
        "optax_version": optax.__version__,
        "device_platforms": [device.platform for device in jax.local_devices()[:local_devices_to_use]],
        "metadata": dict(runtime_metadata or {}),
        "distribution": getattr(ppo_network.parametric_action_distribution, "config", None),
    }
    if reference_config is not None:
        runtime_contract["reference_kl"] = reference_config
    if sapg_config is not None:
        runtime_contract["sapg"] = sapg_config
    if recovery_fn is not None:
        runtime_contract["recovery_optimizer"] = "mutable-adam-learning-rate-v1"
    restored_walltime = 0.0
    if restore_runtime_state is not None:
        if restore_runtime_state.get("schema") != "cat-ppo-runtime-v1":
            raise ValueError("Unrecognized exact runtime snapshot schema")
        if restore_runtime_state.get("contract") != runtime_contract:
            raise ValueError("Runtime training configuration differs; exact resume refused")
        if reference_config is not None:
            if "reference_policy" not in restore_runtime_state:
                raise ValueError("Runtime snapshot is missing its frozen reference policy")
            reference_state = _restore_runtime_tree("reference_policy", reference_state,
                                                    restore_runtime_state["reference_policy"])
        training_state = _restore_runtime_tree("training_state", training_state, restore_runtime_state["training_state"])
        env_state = _restore_runtime_tree("env_state", env_state, restore_runtime_state["env_state"])
        local_key = _restore_runtime_tree("local_key", local_key, restore_runtime_state["local_key"])
        key_envs = _restore_runtime_tree("key_envs", key_envs, restore_runtime_state["key_envs"])
        metrics_aggregator.load_state_dict(restore_runtime_state["metrics_logger"])
        restored_walltime = float(restore_runtime_state["training_walltime"])
        if int(_unpmap(training_state.env_steps)) != int(restore_runtime_state["step"]):
            raise ValueError("Runtime snapshot step disagrees with learner state")

    evaluator = None
    if num_evals > 0:
        evaluation_length = eval_episode_length or episode_length
        eval_env = _maybe_wrap_env(
            eval_env or environment,
            wrap_env,
            num_eval_envs,
            evaluation_length,
            action_repeat,
            local_device_count=1,  # eval on the host only
            key_env=eval_key,
            wrap_env_fn=wrap_env_fn,
            randomization_fn=randomization_fn,
        )
        evaluator = acting.Evaluator(
            eval_env,
            functools.partial(make_policy, deterministic=deterministic_eval),
            num_eval_envs=num_eval_envs,
            episode_length=evaluation_length,
            action_repeat=action_repeat,
            key=eval_key,
        )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap(
                (
                    training_state.normalizer_params,
                    training_state.params.policy,
                    training_state.params.value,
                )
            ),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    training_metrics = {}
    training_walltime = restored_walltime
    current_step = int(_unpmap(training_state.env_steps))
    stopped_by_request = False

    def publish_runtime():
        if runtime_checkpoint_fn is None:
            return
        snapshot = {
            "schema": "cat-ppo-runtime-v1", "contract": runtime_contract,
            "step": current_step,
            "training_state": _runtime_tree_state(jax.device_get(training_state)),
            "env_state": _runtime_tree_state(jax.device_get(env_state)),
            "local_key": _runtime_tree_state(jax.device_get(local_key)),
            "key_envs": _runtime_tree_state(jax.device_get(key_envs)),
            "metrics_logger": metrics_aggregator.state_dict(),
            "training_walltime": training_walltime,
        }
        if reference_config is not None:
            snapshot["reference_policy"] = _runtime_tree_state(jax.device_get(reference_state))
        runtime_checkpoint_fn(current_step, snapshot)

    # A first-update OOM must still have an exact starting state to resume.
    # This snapshot contains no learned progress and is not a best model.
    if restore_runtime_state is None:
        publish_runtime()
    it = 0
    while continuous or it < num_evals_after_init:
        if continuous and should_stop_fn():
            stopped_by_request = True
            break
        logging.info("starting iteration %s %s", it, time.time() - xt)
        it += 1

        navigation_before = None
        if "pf_navigation_outcome_counts" in env_state.info:
            from cat_ppo.furniture.hand_curriculum import navigation_count_snapshot
            # Materialize eight integers before the compiled epoch donates
            # env_state buffers; never retain a view into donated state.
            navigation_before = np.array(navigation_count_snapshot(env_state.info), copy=True)

        for _ in range(max(num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            # TODO: move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs, field_values) if num_resets_per_eval > 0 else env_state

        if process_id != 0:
            continue

        # Flush episode-metric callbacks before publishing one coherent boundary.
        jax.effects_barrier()

        # Process id == 0.
        params = _unpmap(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )
        )

        policy_params_fn(current_step, make_policy, params)

        if save_checkpoint_path is not None:
            checkpoint.save(save_checkpoint_path, current_step, params, ckpt_config)

        if num_evals > 0:
            metrics = evaluator.run_evaluation(
                params,
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)
        elif continuous:
            if "pf_hand_curriculum_stage" in env_state.info:
                from cat_ppo.furniture.hand_curriculum import curriculum_metrics
                training_metrics.update(curriculum_metrics(env_state.info))
            if navigation_before is not None:
                from cat_ppo.furniture.hand_curriculum import navigation_rollout_metrics
                training_metrics.update(navigation_rollout_metrics(
                    navigation_before, navigation_count_snapshot(env_state.info)))
            progress_fn(current_step, training_metrics)

        if scored_checkpoint_fn is not None:
            scored_checkpoint_fn(
                current_step, make_policy, params, ckpt_config,
                metrics if num_evals > 0 else training_metrics,
                "validation" if num_evals > 0 else "training_proxy",
            )

        recovery = recovery_fn(current_step) if recovery_fn is not None else None
        if recovery is not None:
            restored = _recover_training_state(_unpmap(training_state), recovery["params"],
                                               optimizer, recovery["learning_rate"])
            training_state = jax.device_put_replicated(restored, jax.local_devices()[:local_devices_to_use])
            local_key, recovery_reset_key = jax.random.split(local_key)
            key_envs = jax.random.split(recovery_reset_key, num_envs // process_count)
            key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
            # Discard trajectories/history generated by the regressed policy.
            # Curriculum restarts on easy so its counters describe the restored policy.
            env_state = reset_fn(key_envs, field_values)
            metrics_aggregator.discard_windows()
            logging.warning("Recovered learner at transition %s from %s; learning rate %s",
                            current_step, recovery.get("checkpoint"), recovery["learning_rate"])

        publish_runtime()

        # Cooperative stopping keeps the selected model from a completed PPO
        # epoch. It never interrupts a compiled update or checkpoint write.
        if should_stop_fn is not None and should_stop_fn():
            stopped_by_request = True
            logging.info("Manual stop requested at completed step %s", current_step)
            break

    total_steps = current_step
    if not continuous and not stopped_by_request and not total_steps >= num_timesteps:
        raise AssertionError(
            f"Total steps {total_steps} is less than `num_timesteps`= {num_timesteps}."
        )

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap(
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )
    )
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    metrics = {**metrics, "training/completed_steps": total_steps,
               "training/stopped_by_request": stopped_by_request}
    return (make_policy, params, metrics)
