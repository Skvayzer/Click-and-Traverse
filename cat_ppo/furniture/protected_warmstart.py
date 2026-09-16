"""Verify a permanent selected-best archive before starting a new optimizer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verified_best_archive(directory, *, target_contract=None, network_config=None):
    directory = Path(directory).resolve()
    backup = json.loads((directory / "backup.json").read_text())
    selection = json.loads((directory / "checkpoint/selection.json").read_text())
    if backup.get("schema") != "cat-protected-best-backup-v1":
        raise ValueError("Not a protected CAT best-checkpoint archive")
    if (selection.get("selection_source") != "retention_validation"
            or not selection.get("metrics", {}).get("selection", {}).get("eligible")
            or selection.get("step") != backup.get("source_checkpoint_step")):
        raise ValueError("Warm start requires an eligible selected retention checkpoint")
    files = backup.get("checkpoint_files", {})
    if not files or files != selection.get("files"):
        raise ValueError("Backup and selected-checkpoint file inventories differ")
    native = directory / "checkpoint/native"
    actual = {str(p.relative_to(native)) for p in native.rglob("*") if p.is_file()}
    if actual != set(files):
        raise ValueError("Native checkpoint inventory differs from protected backup")
    for name, expected in files.items():
        path = native / name
        if (path.is_symlink() or not path.resolve().is_relative_to(native.resolve())
                or path.stat().st_size != expected["size"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]):
            raise ValueError(f"Protected checkpoint integrity failed: {name}")
    contract = json.loads((native / "observation_contract.json").read_text())
    config = json.loads((native / "ppo_network_config.json").read_text())
    if target_contract is not None:
        for key in ("schema", "actor_features", "critic_features", "action_names", "observed_joint_names"):
            if contract.get(key) != target_contract.get(key):
                raise ValueError(f"Warm start changes network feature layout: {key}")
    if config.get("normalize_observations") is not False:
        raise ValueError("Selected-best warm start requires unnormalized CAT observations")
    if network_config is not None:
        for key in ("policy_hidden_layer_sizes", "value_hidden_layer_sizes"):
            if list(config["network_factory_kwargs"][key]) != list(network_config[key]):
                raise ValueError(f"Warm start changes network architecture: {key}")
    provenance = {
        "kind": "protected_retention_best", "archive": str(directory),
        "source_run": backup["source_run"], "source_step": selection["step"],
        "checkpoint_files": files, "optimizer": "fresh Adam; weights preserved",
        "distribution_change": "v1 independent arms to v2 previous-action-conditioned correlated arms",
    }
    return native, contract, provenance
