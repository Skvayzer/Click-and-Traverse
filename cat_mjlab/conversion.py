"""Auditable Flax/Brax -> PyTorch model and Optax Adam conversion.

Array-only inputs avoid loading arbitrary Python checkpoint objects. Existing
Flax msgpack/Orbax loaders can be used in the original environment, then call
these functions and save ``learner.state_dict()`` with torch.save. The new
environment can load that file with torch.load(weights_only=True).

Transposition preserves all actor/critic weights, SAPG embeddings and optional
Adam first/second moments. Physics and random-generator states cannot be moved
between JAX MJX and mjlab: this is a documented simulator migration, never an
exact runtime resume, even when optimizer moments are retained.
"""
from __future__ import annotations

import re

import numpy as np
import torch

from cat_mjlab.learning import Learner
from cat_mjlab.checkpoint_arrays import expand_released_params, read_array_archive


def extract_native_leader(weights, source_config):
    """Collapse constant leader conditioning into both first-layer biases.

    Preserves actor (including scale head) and critic functions up to floating
    point rounding. Adam moments cannot be transformed by this weight mapping.
    """
    if source_config["algorithm"] != "sapg":
        raise ValueError("Leader extraction requires SAPG weights")
    table = weights["policy_embeddings"]
    if tuple(table.shape) != (source_config["num_policies"], source_config["embedding_dim"]):
        raise ValueError("SAPG embedding shape differs from source configuration")
    if any(not bool(torch.isfinite(value).all()) for value in weights.values()):
        raise ValueError("Nonfinite source model")
    result = {key: value.clone() for key, value in weights.items() if key != "policy_embeddings"}
    for trunk, width in (("actor", source_config["actor_obs"]), ("critic", source_config["critic_obs"])):
        name = f"{trunk}.layers.0"
        matrix = weights[name + ".weight"]
        if matrix.shape[1] != width + table.shape[1]:
            raise ValueError("SAPG input width differs from source configuration")
        result[name + ".weight"] = matrix[:, :width].clone()
        result[name + ".bias"] = weights[name + ".bias"] + matrix[:, width:] @ table[0]
    return result


def _array(value):
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError("Checkpoint parameters must be finite floating-point arrays")
    return value


def _network_arrays(params, prefix, model):
    layers = params.get("params", {})
    expected = {f"hidden_{i}" for i in range(len(getattr(model, prefix).layers))}
    if set(layers) != expected:
        raise ValueError(f"{prefix} layers differ from the configured MLP")
    result = {}
    for index in range(len(expected)):
        layer = layers[f"hidden_{index}"]
        if set(layer) != {"kernel", "bias"}:
            raise ValueError("Unexpected dense layer parameter names")
        result[f"{prefix}.layers.{index}.weight"] = _array(layer["kernel"]).T
        result[f"{prefix}.layers.{index}.bias"] = _array(layer["bias"])
    return result


def load_native_params(learner: Learner, params):
    """Load already expanded CAT (normalizer,actor,critic), optionally add SAPG.

    A leader-only best model has no follower state: loading it into SAPG creates
    fresh identical conditioned policies. Full SAPG continuation requires
    ``load_jax_runtime`` instead. Observation normalization must remain disabled.
    """
    if not isinstance(params, (tuple, list)) or len(params) != 3:
        raise ValueError("Expected (normalizer, actor, critic)")
    actor, critic = params[1:]
    if set(actor) - {"params", "policy_embeddings"} or set(critic) != {"params"}:
        raise ValueError("Unexpected actor/critic parameter collections")
    arrays = _network_arrays(actor, "actor", learner.model) | _network_arrays(critic, "critic", learner.model)
    table = actor.get("policy_embeddings")
    expected = learner.model.state_dict()
    added = False
    if table is not None:
        if learner.model.policy_embeddings is None:
            raise ValueError("Conditioned SAPG parameters cannot be silently collapsed to PPO")
        arrays["policy_embeddings"] = _array(table)
    elif learner.model.policy_embeddings is not None:
        # New conditioning inputs contribute zero, as in expand_ppo_params.
        width = learner.config.embedding_dim
        for trunk in ("actor", "critic"):
            name = f"{trunk}.layers.0.weight"
            arrays[name] = np.pad(arrays[name], ((0, 0), (0, width)))
        arrays["policy_embeddings"] = learner.model.policy_embeddings.detach().cpu().numpy()
        added = True
    if set(arrays) != set(expected):
        raise ValueError("Converted parameter keys differ")
    tensors = {}
    for name, value in arrays.items():
        if value.shape != tuple(expected[name].shape):
            raise ValueError(f"Checkpoint shape differs for {name}: {value.shape} != {tuple(expected[name].shape)}")
        tensors[name] = torch.as_tensor(np.array(value, copy=True), device=learner.device, dtype=expected[name].dtype)
    learner.model.load_state_dict(tensors, strict=True)
    return dict(kind="parameter_conversion", source_framework="flax", target_framework="pytorch",
                normalize_observations=False, actor_and_critic_preserved=True,
                sapg_embeddings="initialized_with_zero_input_weights" if added else "preserved" if table is not None else "absent",
                optimizer="fresh", exact_runtime_resume=False,
                simulator_state="new mjlab reset", rng="new PyTorch generator")


_DENSE_PATH = re.compile(r"^(policy|value)\['params'\]\['hidden_(\d+)'\]\['(kernel|bias)'\]$")


def _parameter_name(suffix):
    if suffix == "policy['policy_embeddings']":
        return "policy_embeddings", False
    match = _DENSE_PATH.fullmatch(suffix)
    if match is None:
        raise ValueError(f"Unrecognized native parameter path: {suffix}")
    trunk, index, kind = match.groups()
    return f"{'actor' if trunk == 'policy' else 'critic'}.layers.{index}.{'weight' if kind == 'kernel' else 'bias'}", kind == "kernel"


def _runtime_leaves(snapshot):
    if snapshot.get("schema") != "cat-ppo-runtime-v1":
        raise ValueError("Unrecognized JAX runtime schema")
    tree = snapshot["training_state"]
    if len(tree["paths"]) != len(tree["leaves"]) or len(set(tree["paths"])) != len(tree["paths"]):
        raise ValueError("Malformed runtime leaf paths")
    result = {}
    for path, value in zip(tree["paths"], tree["leaves"]):
        array = np.asarray(value)
        if not array.shape or array.shape[0] != 1:
            raise ValueError("Migration supports the single-device runtime's leading pmap axis")
        result[path] = array[0]
    return result


def _check_runtime_contract(learner, contract, *, restore_optimizer):
    if contract.get("normalize_observations") is not False:
        raise ValueError("Migration requires CAT normalize_observations=False")
    if contract.get("distribution") is not None or contract.get("reference_kl") is not None:
        raise ValueError("Bounded/correlated distributions and reference regularizers are not this training contract")
    sapg = contract.get("sapg")
    if learner.config.algorithm == "sapg":
        if sapg is None or any(sapg.get(key) != getattr(learner.config, key) for key in ("num_policies", "embedding_dim")):
            raise ValueError("SAPG conditioning configuration differs")
    elif sapg is not None:
        raise ValueError("SAPG runtime cannot be silently converted into PPO")
    if restore_optimizer:
        for key in ("learning_rate", "entropy_cost", "discounting", "reward_scaling", "clipping_epsilon",
                    "gae_lambda", "max_grad_norm", "normalize_advantage", "num_minibatches", "num_updates_per_batch"):
            if contract.get(key) != getattr(learner.config, key):
                raise ValueError(f"Learner setting {key} differs from source runtime")


def load_jax_runtime(learner: Learner, snapshot, *, restore_optimizer=True):
    """Convert all policies and optionally Adam moments, validating every leaf.

    Deliberately rejects actor-only best checkpoints and incompatible algorithm
    configurations. Environment, JAX RNG and old success windows are not copied.
    They describe the previous simulator and must start a new logged lineage.
    """
    _check_runtime_contract(learner, snapshot["contract"], restore_optimizer=restore_optimizer)
    leaves = _runtime_leaves(snapshot)
    expected = dict(learner.model.named_parameters())
    converted = {}
    for path, value in leaves.items():
        if path.startswith(".params."):
            name, transpose = _parameter_name(path.removeprefix(".params."))
            value = _array(value)
            if transpose:
                value = value.T
            if name not in expected or value.shape != tuple(expected[name].shape):
                raise ValueError(f"Runtime model shape differs for {name}")
            converted[name] = torch.as_tensor(value.copy(), device=learner.device, dtype=expected[name].dtype)
    if set(converted) != set(expected):
        raise ValueError("Full runtime is missing model parameters")
    moments = {"mu": {}, "nu": {}}
    counts = []
    if restore_optimizer:
        for path, value in leaves.items():
            if not path.startswith(".optimizer_state"):
                continue
            match = re.search(r"\.(mu|nu)\.(.+)$", path)
            if match:
                moment, suffix = match.groups()
                name, transpose = _parameter_name(suffix)
                value = _array(value)
                value = value.T if transpose else value
                if name not in expected or value.shape != tuple(expected[name].shape):
                    raise ValueError(f"Runtime Adam shape differs for {name}")
                if moment == "nu" and np.any(value < 0):
                    raise ValueError("Adam second moments must be nonnegative")
                moments[moment][name] = torch.as_tensor(value.copy(), device=learner.device, dtype=expected[name].dtype)
            elif path.endswith(".count"):
                if value.shape != () or not np.issubdtype(value.dtype, np.integer) or int(value) < 0:
                    raise ValueError("Invalid Adam update count")
                counts.append(int(value))
            else:
                raise ValueError(f"Unsupported optimizer leaf: {path}")
        if len(counts) != 1 or any(set(moment) != set(expected) for moment in moments.values()):
            raise ValueError("Incomplete or ambiguous Adam state")
    # Validate the complete payload before mutating the destination learner.
    learner.model.load_state_dict(converted, strict=True)
    if restore_optimizer:
        learner.optimizer.state.clear()
        for name, parameter in expected.items():
            learner.optimizer.state[parameter] = dict(step=torch.tensor(float(counts[0])),
                exp_avg=moments["mu"][name], exp_avg_sq=moments["nu"][name])
    learner.env_steps = int(snapshot["step"])
    updates_per_iteration = learner.config.num_minibatches * learner.config.num_updates_per_batch
    learner.updates = counts[0] // updates_per_iteration if counts else 0
    return dict(kind="simulator_migration", source_schema=snapshot["schema"], source_step=learner.env_steps,
                actor_and_critic_preserved=True, all_sapg_embeddings_preserved=learner.config.algorithm == "sapg",
                optimizer="adam_moments_and_count_preserved" if restore_optimizer else "fresh",
                adam_update_count=counts[0] if counts else None, exact_runtime_resume=False,
                simulator_state="new mjlab reset", rng="new PyTorch generator",
                success_windows="reset because simulator lineage changed")


def load_array_archive(learner, path, *, restore_optimizer=True):
    """Load an offline NPZ export without installing JAX, Brax, Flax or Orbax."""
    metadata, arrays = read_array_archive(path)
    from cat_ppo.furniture.control import mjlab_observation_contract
    if metadata.get('observation_contract', mjlab_observation_contract()) != mjlab_observation_contract():
        raise ValueError('Archive observation contract differs from native 222/310; no conversion is performed')
    arrays = arrays[:metadata.get("training_array_count", len(arrays))]
    paths = metadata["paths"]
    if len(paths) != len(arrays) or len(set(paths)) != len(paths):
        raise ValueError("Checkpoint array paths differ")
    if metadata["kind"] == "runtime":
        snapshot = dict(schema="cat-ppo-runtime-v1", step=metadata["step"], contract=metadata["contract"],
                        training_state=dict(paths=paths, leaves=arrays))
        report = load_jax_runtime(learner, snapshot, restore_optimizer=restore_optimizer)
    elif metadata["kind"] == "released_expanded":
        if metadata.get("normalize_observations") is not False:
            raise ValueError("Archive requires unsupported observation normalization")
        actor, critic = {"params": {}}, {"params": {}}
        for path, array in zip(paths, arrays):
            match = _DENSE_PATH.fullmatch(path)
            if match is None:
                raise ValueError(f"Unknown released archive parameter: {path}")
            trunk, index, key = match.groups()
            target = actor if trunk == "policy" else critic
            target["params"].setdefault(f"hidden_{index}", {})[key] = array
        report = load_native_params(learner, (None, actor, critic))
    else:
        raise ValueError("Unknown checkpoint archive kind")
    return dict(report, archive_metadata=metadata)
