"""NumPy-only checkpoint archive helpers usable inside the old JAX environment."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile

import numpy as np


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
