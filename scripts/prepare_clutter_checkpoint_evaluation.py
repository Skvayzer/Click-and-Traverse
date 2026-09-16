#!/usr/bin/env python3
"""Freeze the selected model and a verified, immutable evaluation field subset.

Run with the training Python environment but CPU-only JAX. This does not copy
the continuously overwritten optimizer/resume checkpoint or alter the learner.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Frozen training source root")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True, help="Original complete manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, str(args.source.resolve()))
    from cat_ppo.furniture.checkpoint import BestCheckpointStore, _manifest
    from cat_ppo.furniture.generalist_fields import load_generalist_manifest, _json_hash
    from cat_ppo.furniture.retention_validation import FIXED_SCENE_IDS

    output = args.output.absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing evaluation directory: {output}")
    output.mkdir(parents=True)
    checkpoint = output / "checkpoint"
    store = BestCheckpointStore.open_existing(args.run)
    with store._lock():
        selected = store.selected(verify=True)
        if selected is None:
            raise ValueError("The run has no selected checkpoint")
        shutil.copytree(selected["generation"], checkpoint, symlinks=False)
        if _manifest(checkpoint / "native") != selected["files"]:
            raise ValueError("Frozen checkpoint differs from verified source")
    for name in ("run.json", "validation_baseline.json"):
        shutil.copyfile(args.run / name, output / name)

    # Capture complete JSONL records only: the learner may be appending a line.
    validation_history = []
    latest_step = 0
    for line in (args.run / "metrics.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest_step = max(latest_step, int(row.get("global_step", 0)))
        if "validation/clutter_goal_success_rate" in row:
            validation_history.append({key: value for key, value in row.items()
                if key == "global_step" or "clutter_goal_success_rate" in key
                or "cat_goal_success_rate" in key or "retention_eligible" in key
                or key == "selection/best_updated"})
    peak = max(validation_history, key=lambda row: row["validation/clutter_goal_success_rate"])
    write_json(output / "validation_history.json", validation_history)

    source_manifest = load_generalist_manifest(args.bank, verify_files=False)
    wanted = set(FIXED_SCENE_IDS)
    scenes = [scene for scene in source_manifest["scenes"]
              if scene["family"] == "original_cat" or scene["scene_id"] in wanted]
    identities = {scene["scene_id"] for scene in scenes}
    if missing := wanted - identities:
        raise ValueError(f"Fixed validation scenes are missing: {sorted(missing)}")
    bank_root = args.bank.resolve().parent
    subset_root = output / "bank"
    linked_bytes = 0
    linked_files = 0
    for scene in scenes:
        directory = (bank_root / scene["path"]).resolve()
        if not directory.is_relative_to(bank_root):
            raise ValueError("Source scene escapes field bank")
        for source in sorted(directory.rglob("*")):
            if source.is_symlink():
                raise ValueError(f"Refusing a symlink within immutable scene: {source}")
            if not source.is_file():
                continue
            destination = subset_root / scene["path"] / source.relative_to(directory)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            linked_bytes += source.stat().st_size
            linked_files += 1
    subset = dict(source_manifest)
    subset.pop("manifest_sha256")
    subset["scenes"] = scenes  # Preserve each source record and its ordering exactly.
    subset["scene_count"] = len(scenes)
    subset["fields_bytes"] = sum(field["size_bytes"] for scene in scenes
                                  for field in scene["fields"].values())
    subset["retained_family_counts"] = dict(Counter(scene["family"] for scene in scenes))
    subset["evaluation_subset"] = {
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "source_scene_count": source_manifest["scene_count"],
        "selection": "All 37 configured originals plus FIXED_SCENE_IDS",
        "fixed_scene_ids": list(FIXED_SCENE_IDS),
        "linked_files_are_immutable": True,
    }
    # Generation-request/coverage metadata describes the complete parent bank.
    subset["manifest_sha256"] = _json_hash(subset)
    subset_path = subset_root / "manifest.json"
    write_json(subset_path, subset)
    verified = load_generalist_manifest(subset_path, verify_files=True)
    if verified["scenes"] != scenes:
        raise ValueError("Evaluation subset changed source scene records")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_source": str(args.source.resolve()),
        "source_run": str(args.run.resolve()),
        "source_bank": str(args.bank.resolve()),
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "source_manifest_file_sha256": sha256(args.bank),
        "checkpoint": str(checkpoint / "native"),
        "checkpoint_step": selected["step"],
        "checkpoint_selection_score": selected["score"],
        "checkpoint_selection_rules": selected["rules"],
        "checkpoint_source_generation": selected["generation"],
        "checkpoint_selection_sha256": sha256(checkpoint / "selection.json"),
        "latest_observed_training_step": latest_step,
        "observed_peak_deterministic_clutter_validation": peak,
        "subset_manifest": str(subset_path),
        "subset_manifest_sha256": subset["manifest_sha256"],
        "subset_scene_count": len(scenes),
        "subset_family_counts": subset["retained_family_counts"],
        "subset_fields_bytes": subset["fields_bytes"],
        "hardlinked_files": linked_files,
        "hardlinked_bytes": linked_bytes,
        "notes": ["Only the selected best model was frozen; the rolling resume was not copied.",
                  "Subset scenes and files are unchanged; original identities/order are retained.",
                  "Do not edit hardlinked immutable fields or scene metadata.",
                  "This is fixed training-bank evaluation, not held-out generalization."],
    }
    write_json(output / "preparation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
