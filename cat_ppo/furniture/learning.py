"""Native CAT parameter expansion with explicit observation and joint mappings.

Array transformation and parity checks use NumPy; JAX/Brax are imported only
when loading or initializing their native parameter trees.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import urllib.parse
import urllib.request

import numpy as np

DEFAULT_MANIFEST = Path(__file__).with_name("native_generalist_manifest.json")


def feature_names(contract, key):
    names = list(contract[{"state": "actor_features", "privileged_state": "critic_features"}[key]])
    if not names or not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{key} features must be nonempty, unique scalar names")
    return names


def index_mapping(old_names, new_names):
    if len(set(old_names)) != len(old_names) or len(set(new_names)) != len(new_names):
        raise ValueError("Duplicate feature/action names")
    lookup = {name: i for i, name in enumerate(new_names)}
    missing = [name for name in old_names if name not in lookup]
    if missing:
        raise ValueError(f"Target contract omits source names: {missing}")
    return np.asarray([lookup[name] for name in old_names], dtype=np.int64)


def _plain(tree):
    if isinstance(tree, dict) or hasattr(tree, "items"):
        return {key: _plain(value) for key, value in tree.items()}
    return copy.deepcopy(tree)


def dense_layers(params):
    layers = params["params"]
    names = sorted(layers, key=lambda name: int(name.removeprefix("hidden_"))
                   if re.fullmatch(r"hidden_\d+", name) else -1)
    if not names or any(name != f"hidden_{i}" for i, name in enumerate(names)):
        raise ValueError("Warm-start/export supports the released plain hidden_N MLP only")
    for name in names:
        layer = layers[name]
        if set(layer) != {"kernel", "bias"}:
            raise ValueError(f"Unexpected MLP parameters in {name}")
        kernel, bias = np.asarray(layer["kernel"]), np.asarray(layer["bias"])
        if kernel.ndim != 2 or bias.shape != (kernel.shape[1],):
            raise ValueError(f"Invalid Dense shapes in {name}")
        if not np.isfinite(kernel).all() or not np.isfinite(bias).all():
            raise ValueError(f"Nonfinite MLP parameters in {name}")
    return names


def mlp_hidden_sizes(params):
    names = dense_layers(params)
    return tuple(int(np.asarray(params["params"][name]["bias"]).size) for name in names[:-1])


def _expand_network(old, target, row_map, old_size, new_size, action_map=None,
                    old_actions=None, new_actions=None, new_action_std=.05):
    old_names, new_names = dense_layers(old), dense_layers(target)
    if old_names != new_names or len(old_names) < 2:
        raise ValueError("Warm-start requires identical MLP trunks with at least one hidden layer")
    result = _plain(target)
    src, dst = old["params"], result["params"]
    first, last = old_names[0], old_names[-1]
    if np.shape(src[first]["kernel"])[0] != old_size or np.shape(dst[first]["kernel"])[0] != new_size:
        raise ValueError("First-layer input shape disagrees with named observation contract")
    for name in old_names:
        for field in ("kernel", "bias"):
            a, b = np.asarray(src[name][field]), np.asarray(dst[name][field])
            if name == first and field == "kernel":
                if a.shape[1] != b.shape[1]:
                    raise ValueError("First hidden layer width changed")
                b = np.zeros_like(b)
                b[row_map] = a
                dst[name][field] = b
            elif name == last and action_map is not None:
                if a.shape[-1] != 2 * old_actions or b.shape[-1] != 2 * new_actions:
                    raise ValueError("Actor head must contain separate mean and raw-scale blocks")
                if field == "kernel" and a.shape[0] != b.shape[0]:
                    raise ValueError("Actor trunk width changed")
                b = np.zeros_like(b)
                if field == "bias":
                    if not math.isfinite(new_action_std) or new_action_std <= .001:
                        raise ValueError("New action standard deviation must exceed Brax min_std=.001")
                    b[new_actions:] = np.log(np.expm1(new_action_std - .001))
                b[..., action_map] = a[..., :old_actions]
                b[..., new_actions + action_map] = a[..., old_actions:]
                dst[name][field] = b
            else:
                if a.shape != b.shape:
                    raise ValueError(f"Cannot preserve {name}/{field}: {a.shape} != {b.shape}")
                dst[name][field] = a.copy()
    return result


def _stat_get(state, name):
    return state[name] if isinstance(state, dict) else getattr(state, name)


def adapt_native_params(source, target, source_contract, target_contract, *, new_action_std=.05):
    """Return an expanded native (normalizer, actor, critic) tuple and provenance.

    The native optimizer is intentionally absent. Added statistics have mean=0,
    std=1, and summed_variance=count, a documented unit-variance prior at the
    retained scalar count. Existing feature statistics remain byte-for-byte.
    """
    if len(source) != 3 or len(target) != 3:
        raise ValueError("Native CAT checkpoint must contain normalizer, actor and critic")
    old_actions, new_actions = source_contract["action_names"], target_contract["action_names"]
    action_map = index_mapping(old_actions, new_actions)
    maps = {key: index_mapping(feature_names(source_contract, key), feature_names(target_contract, key))
            for key in ("state", "privileged_state")}
    actor = _expand_network(source[1], target[1], maps["state"],
        len(feature_names(source_contract, "state")), len(feature_names(target_contract, "state")),
        action_map, len(old_actions), len(new_actions), new_action_std)
    critic = _expand_network(source[2], target[2], maps["privileged_state"],
        len(feature_names(source_contract, "privileged_state")),
        len(feature_names(target_contract, "privileged_state")))
    count = np.asarray(_stat_get(source[0], "count"))
    if count.shape != () or not np.isfinite(count) or count < 0:
        raise ValueError("Invalid normalizer count")
    values = {"count": count.copy()}
    for field in ("mean", "std", "summed_variance"):
        old_stats = _stat_get(source[0], field)
        values[field] = {}
        for key, mapping in maps.items():
            old_array = np.asarray(old_stats[key])
            if old_array.shape != (len(feature_names(source_contract, key)),) or not np.isfinite(old_array).all():
                raise ValueError(f"Invalid source normalizer {field}/{key}")
            if field == "std" and np.any(old_array <= 0):
                raise ValueError("Normalizer standard deviations must be positive")
            fill = 0 if field == "mean" else 1 if field == "std" else float(count)
            new_array = np.full(len(feature_names(target_contract, key)), fill, dtype=old_array.dtype)
            new_array[mapping] = old_array
            values[field][key] = new_array
    if isinstance(target[0], dict):
        normalizer = values
    elif hasattr(target[0], "replace"):
        normalizer = target[0].replace(**values)
    else:
        normalizer = dataclasses.replace(target[0], **values)
    report = {"kind": "native_parameter_finetuning", "optimizer": "fresh", "critic_restored": True,
        "source_action_count": len(old_actions), "target_action_count": len(new_actions),
        "action_mapping": action_map.tolist(), "input_mappings": {key: value.tolist() for key, value in maps.items()},
        "added_input_weights": "zero", "new_action_mean": 0.0, "new_action_std": new_action_std,
        "normalizer_added_features": "zero mean; unit variance prior at retained scalar count"}
    return (normalizer, actor, critic), report


def numpy_mlp(params, obs):
    # Float64 reference accumulation avoids backend-specific float32 BLAS
    # exception flags and supplies an independent check of JAX/ONNX arithmetic.
    x = np.asarray(obs, dtype=np.float64)
    names = dense_layers(params)
    for i, name in enumerate(names):
        layer = params["params"][name]
        x = np.einsum("...i,ij->...j", x, np.asarray(layer["kernel"], dtype=np.float64), optimize=False) + np.asarray(layer["bias"])
        if i != len(names) - 1:
            # Stable sigmoid, matching Brax's default swish activation.
            sigmoid = np.exp(-np.logaddexp(0, -x))
            x = x * sigmoid
    return x


def predict_numpy(params, observations, *, normalize_observations=False, critic=False):
    key = "privileged_state" if critic else "state"
    x = np.asarray(observations[key])
    if normalize_observations:
        x = (x - np.asarray(_stat_get(params[0], "mean")[key])) / np.asarray(_stat_get(params[0], "std")[key])
    output = numpy_mlp(params[2 if critic else 1], x)
    return output if critic else np.tanh(np.split(output, 2, axis=-1)[0])


def verify_warmstart_parity(source, expanded, source_contract, target_contract, *, normalize_observations=False,
                            seed=0, observations=None, atol=2e-6):
    """Check actor means/raw scales and critic on mapped observations, without simulation."""
    rng = np.random.default_rng(seed)
    old_obs = observations or {key: rng.normal(0, .3, (32, len(feature_names(source_contract, key)))).astype(np.float32)
                              for key in ("state", "privileged_state")}
    new_obs = {}
    for key in old_obs:
        mapping = index_mapping(feature_names(source_contract, key), feature_names(target_contract, key))
        new_obs[key] = rng.normal(0, .3, (*old_obs[key].shape[:-1], len(feature_names(target_contract, key)))).astype(np.float32)
        new_obs[key][..., mapping] = old_obs[key]
    action_map = index_mapping(source_contract["action_names"], target_contract["action_names"])
    errors = {}
    for key, slot in (("state", 1), ("privileged_state", 2)):
        inputs = []
        for params, obs in ((source, old_obs), (expanded, new_obs)):
            x = obs[key]
            if normalize_observations:
                x = (x - _stat_get(params[0], "mean")[key]) / _stat_get(params[0], "std")[key]
            inputs.append(numpy_mlp(params[slot], x))
        if slot == 1:
            columns = np.r_[action_map, len(target_contract["action_names"]) + action_map]
            actual = inputs[1][..., columns]
            name = "actor_mean_and_raw_scale_max_abs_error"
        else:
            actual = inputs[1]
            name = "critic_max_abs_error"
        errors[name] = float(np.max(np.abs(inputs[0] - actual)))
        if not np.allclose(inputs[0], actual, rtol=2e-6, atol=atol):
            raise ValueError(f"Warm-start parity failed: {name}={errors[name]}")
    return errors


def load_native(path):
    from brax.training.agents.ppo import checkpoint
    return checkpoint.load(Path(path).resolve())


def _safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe manifest path: {value!r}")
    return path


def fetch_native_checkpoint(destination, manifest_path=DEFAULT_MANIFEST):
    """Fetch only hash-pinned public HF files, without reading credential stores."""
    manifest = json.loads(Path(manifest_path).read_text())
    revision = manifest["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("HF revision must be an immutable 40-character commit")
    repo = manifest["repo_id"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid public HF repository")
    prefix = _safe_relative(manifest["checkpoint_path"])
    destination = Path(destination).absolute()
    if destination.is_symlink():
        raise ValueError("Refusing a symlink checkpoint destination")
    destination.mkdir(parents=True, exist_ok=True)
    entries = manifest["files"]
    if len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate checkpoint manifest paths")
    for entry in entries:
        relative = _safe_relative(entry["path"])
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) or entry["size"] < 0:
            raise ValueError("Every native file needs an exact size and SHA256")
        target = destination.joinpath(*relative.parts)
        parent = destination
        for part in relative.parts[:-1]:
            parent = parent / part
            if parent.is_symlink():
                raise ValueError(f"Refusing symlink parent: {parent}")
            parent.mkdir(exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file() or target.stat().st_size != entry["size"] or hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Existing checkpoint file does not match pinned source: {target}")
            continue
        path = urllib.parse.quote(str(prefix / relative), safe="/")
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
        fd, temporary = tempfile.mkstemp(prefix=".download-", dir=target.parent)
        try:
            digest = hashlib.sha256();size = 0
            with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > entry["size"]:
                        raise ValueError(f"Download larger than pinned file: {relative}")
                    digest.update(chunk);output.write(chunk)
                output.flush();os.fsync(output.fileno())
            if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Source hash/size mismatch: {relative}")
            # Hard-link publication refuses a raced-in existing destination.
            os.link(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return destination, manifest
