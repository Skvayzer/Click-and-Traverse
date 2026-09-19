"""NumPy-only checkpoint archive helpers usable inside the old JAX environment."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile

import numpy as np


SAMPLING_KEYS = ("pf_episode_ema", "pf_success_ema", "pf_sampling_logits",
                 "pf_hand_curriculum_stage", "pf_hand_curriculum_completed", "pf_hand_curriculum_goals")


def extract_shared_sampling(snapshot):
    """Extract simulator-independent shared state, rejecting divergent replicas."""
    tree = snapshot["env_state"]
    leaves = dict(zip(tree["paths"], tree["leaves"]))
    if len(leaves) != len(tree["paths"]) or len(tree["paths"]) != len(tree["leaves"]):
        raise ValueError("Malformed environment state paths")
    result = {}
    for key in SAMPLING_KEYS:
        path = f".info['{key}']"
        if path not in leaves:
            if key.startswith("pf_hand_curriculum"):
                continue
            raise ValueError(f"Missing shared sampler field: {key}")
        value = np.asarray(leaves[path])
        if value.ndim < 2 or value.shape[0] != 1 or value.shape[1] < 1:
            raise ValueError(f"Expected one device and replicated environments for {key}")
        first = value[0, 0]
        if not np.all(value == first):
            raise ValueError(f"Shared sampler replicas disagree for {key}")
        if key == "pf_sampling_logits":
            if np.isnan(first).any() or np.isposinf(first).any() or not np.isfinite(first).any():
                raise ValueError("Invalid sampler logits")
        elif not np.isfinite(first).all():
            raise ValueError(f"Nonfinite shared sampler state: {key}")
        result[key] = np.array(first, copy=True)
    curriculum = {key for key in result if key.startswith("pf_hand_curriculum")}
    if curriculum and len(curriculum) != 3:
        raise ValueError("Incomplete hand curriculum state")
    return result


def sampling_state_from_archive(metadata, arrays, bank_sha256):
    """Preserve shared state only for the exact source bank, never map scenes implicitly."""
    source_sha = metadata.get("contract", {}).get("metadata", {}).get("bank_sha256")
    if metadata.get("kind") != "runtime":
        return None, dict(status="fresh_model_curriculum", source_bank_sha256=source_sha)
    if not source_sha:
        raise ValueError("Runtime archive lacks source field-bank identity")
    if source_sha != bank_sha256:
        return None, dict(status="new_bank_new_curriculum", source_bank_sha256=source_sha,
                          target_bank_sha256=bank_sha256)
    sampling = metadata.get("sampling_state")
    if not sampling or sampling.get("bank_sha256") != source_sha:
        raise ValueError("Same-bank migration requires shared sampler/curriculum state; re-export the runtime")
    mapping = sampling.get("array_indices", {})
    if set(mapping) - set(SAMPLING_KEYS) or not set(SAMPLING_KEYS[:3]).issubset(mapping):
        raise ValueError("Invalid archived sampling field set")
    start = metadata.get("training_array_count", len(arrays))
    if (set(mapping.values()) != set(range(start, len(arrays)))
            or any(type(index) is not int for index in mapping.values())):
        raise ValueError("Invalid archived sampling array indices")
    state = {key: arrays[index] for key, index in mapping.items()}
    return state, dict(status="preserved", source_bank_sha256=source_sha,
                       target_bank_sha256=bank_sha256, fields=sorted(state),
                       curriculum_stage=int(state["pf_hand_curriculum_stage"]) if "pf_hand_curriculum_stage" in state else None)


def expand_released_params(params, target_contract=None, *, new_action_std=.05):
    """Use the existing named 162/250/12 -> 222/310/29 expansion unchanged."""
    from cat_ppo.furniture.control import legacy_observation_contract, wholebody_observation_contract
    from cat_ppo.furniture.learning import adapt_native_params, dense_layers

    source_contract = legacy_observation_contract()
    target_contract = target_contract or wholebody_observation_contract()
    target = [dict(), copy.deepcopy(params[1]), copy.deepcopy(params[2])]
    for slot, feature in ((1, "actor_features"), (2, "critic_features")):
        names = dense_layers(target[slot])
        first = target[slot]["params"][names[0]]
        first["kernel"] = np.zeros((len(target_contract[feature]), np.shape(first["kernel"])[1]), dtype=np.float32)
        if slot == 1:
            last = target[slot]["params"][names[-1]]
            size = 2 * len(target_contract["action_names"])
            last["kernel"] = np.zeros((np.shape(last["kernel"])[0], size), dtype=np.float32)
            last["bias"] = np.zeros(size, dtype=np.float32)
    return adapt_native_params(params, tuple(target), source_contract, target_contract, new_action_std=new_action_std)


def write_array_archive(path, metadata, arrays):
    """Write one atomic NPZ containing numeric leaves and JSON, never pickle."""
    path = Path(path)
    if path.is_symlink() or path.exists():
        raise ValueError("Archive destination must be new; preserved inputs are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, schema="cat-mjlab-array-archive-v1", array_count=len(arrays))
    encoded = np.frombuffer(json.dumps(metadata, sort_keys=True, allow_nan=False).encode(), dtype=np.uint8)
    payload = {f"array_{i:04d}": np.asarray(array) for i, array in enumerate(arrays)}
    if any(array.dtype.hasobject for array in payload.values()):
        raise ValueError("Object arrays are not supported")
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, metadata=encoded, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return metadata


def read_array_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode())
        if metadata.get("schema") != "cat-mjlab-array-archive-v1":
            raise ValueError("Unrecognized checkpoint archive schema")
        count = metadata["array_count"]
        if type(count) is not int or count <= 0:
            raise ValueError("Invalid checkpoint archive count")
        expected = {"metadata"} | {f"array_{i:04d}" for i in range(count)}
        if set(archive.files) != expected:
            raise ValueError("Checkpoint archive array set differs from metadata")
        arrays = [archive[f"array_{i:04d}"] for i in range(count)]
    return metadata, arrays
