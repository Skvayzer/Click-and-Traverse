#!/usr/bin/env python3
"""Append certified hand-protection passages without rewriting the CAT bank.

Every existing scene record, path and file is retained through hardlinks. Only
new passages are generated. A resumable plan pins all source hashes, and the
production manifest is published atomically after complete file verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import multiprocessing
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Offline FK/certification also uses JAX geometry kernels. Keep them off the
# training GPU, including in spawned field-generation workers.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from cat_ppo.furniture.expanded_fields import check_free_space_connectivity
from cat_ppo.furniture.generalist_fields import (
    EXPANDED_SCHEMA, _field_records, _json_hash, load_generalist_manifest,
    make_clutter_fields, sha256,
)
from scripts.rebuild_clutter_navigation_bank import (
    _contained_directory, _hardlink_tree, _require_corrected_room_source, _write_json,
)

BUILDER_VERSION = "append-certified-hand-protection-bank-v1"
KINDS = {"hand_table_aisle": "furniture", "hand_shelf_passage": "generic_clutter"}
DIFFICULTIES = ("easy", "medium", "hard")
FIELD_DX = .04
CURRICULUM = dict(schema="hand-protection-success-curriculum-v1",
                  fraction_within_room_family=.5, minimum_completed_episodes=64,
                  clean_goal_success_threshold=.6,
                  difficulties=list(DIFFICULTIES), retain_unlocked_levels=True)


def generate_scene(seed, *, kind, difficulty):
    # Kept lazy so small builder tests can mock geometry without loading MuJoCo.
    from cat_ppo.furniture.hand_passages import generate_hand_passage_scene
    return generate_hand_passage_scene(seed, split="train", kind=kind,
                                      difficulty=difficulty, certify=True)


def _code_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("scripts/build_hand_protection_bank.py", "scripts/rebuild_clutter_navigation_bank.py",
             "cat_ppo/furniture/hand_passages.py",
             "cat_ppo/furniture/generalist_fields.py", "cat_ppo/furniture/room_geometry.py",
             "cat_ppo/furniture/scenes.py", "cat_ppo/furniture/control.py",
             "cat_ppo/furniture/grippers.py", "cat_ppo/furniture/body_collision_geometry.py",
             "cat_ppo/envs/g1/env_cat_wholebody.py", "cat_ppo/envs/g1/constants.py",
             "docs/assets/collision-proxy-proposal-20260916/proposal.json")
    return {name: sha256(root / name) for name in names}


def generation_jobs(variants_per_level=4):
    if type(variants_per_level) is not int or not 1 <= variants_per_level <= 100:
        raise ValueError("variants_per_level must be an integer in [1, 100]")
    jobs = []
    for kind_index, (kind, family) in enumerate(KINDS.items()):
        for level, difficulty in enumerate(DIFFICULTIES):
            for variant in range(variants_per_level):
                seed = 6001 + 1000 * kind_index + level * variants_per_level + variant
                jobs.append(dict(kind=kind, family=family, difficulty=difficulty,
                    level=level, seed=seed, split="train",
                    path=f"hand_protection/{kind}/{seed:06d}"))
    return jobs


def _validate_new_record(record, directory):
    _require_corrected_room_source(record["source"])
    if record["fields"] != _field_records(directory):
        raise ValueError(f"Hand-protection field fingerprint differs: {directory}")
    for filename, expected in (("scene.json", record["scene_sha256"]),
                               ("source.json", record["source"]["metadata_sha256"]),
                               ("obs.npy", record["source"]["occupancy_sha256"])):
        if sha256(directory / filename) != expected:
            raise ValueError(f"Hand-protection {filename} fingerprint differs: {directory}")
    scene = json.loads((directory / "scene.json").read_text())
    if scene["geometry_hash"] != record["source"]["geometry_sha256"]:
        raise ValueError("Canonical hand-protection geometry hash differs")
    if scene.get("hand_protection") != record["source"]["hand_protection"]:
        raise ValueError("Canonical hand-protection certificate and source metadata disagree")
    if json.loads((directory / "source.json").read_text()) != {
            key: value for key, value in record["source"].items() if key != "metadata_sha256"}:
        raise ValueError("Hand-protection record and source metadata disagree")


def _build_scene(payload):
    job, output, plan = payload
    directory = _contained_directory(Path(output), job["path"])
    cache_path = directory / "hand_protection_record.json"
    key = _json_hash(dict(plan_sha256=plan["plan_sha256"], job=job))
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("cache_key") != key:
            raise ValueError("Cached hand scene belongs to a different generation plan")
        _validate_new_record(cached["record"], directory)
        return cached["record"]
    scene = generate_scene(job["seed"], kind=job["kind"], difficulty=job["difficulty"])
    record = make_clutter_fields(scene, directory, dx=FIELD_DX)
    if record["family"] != job["family"]:
        raise ValueError("Hand-protection geometry was assigned to the wrong room family")
    _require_corrected_room_source(record["source"])
    occupancy = np.load(directory / "obs.npy", allow_pickle=False)
    connectivity = check_free_space_connectivity(
        occupancy, np.asarray(record["origin"]), FIELD_DX, record["start"], record["goal"])
    hand = copy.deepcopy(scene["hand_protection"])
    if any(hand.get(key) != job[key] for key in ("kind", "difficulty", "level")):
        raise ValueError("Hand-protection scene metadata differs from its generation job")
    if not isinstance(hand.get("certificate"), dict) or not hand["certificate"]:
        raise ValueError("Hand-protection scenes require a nonempty static certificate")
    source = dict(record["source"], kind="certified-hand-protection-room",
        hand_protection=hand, source_hashes=plan["source_hashes"],
        generation_parameters={key: job[key] for key in ("seed", "split", "kind", "difficulty")},
        geometry_sha256=scene["geometry_hash"], occupancy_sha256=sha256(directory / "obs.npy"),
        connectivity=connectivity, append_plan_sha256=plan["plan_sha256"])
    _write_json(directory / "source.json", source)
    record.update(path=job["path"], task_kind="room", reset_mode="room",
                  crossed_mode="goal_radius", episode_length=4000,
                  sampling_group=job["family"], sampling_weight=1.,
                  source=dict(source, metadata_sha256=sha256(directory / "source.json")))
    _validate_new_record(record, directory)
    _write_json(cache_path, dict(cache_key=key, record=record))
    return record


def build_bank(base_manifest, output, *, variants_per_level=4, workers=1, progress=None):
    """Append six groups of passages, preserving every old scene byte-for-byte."""
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("Build hand fields with JAX_PLATFORMS=cpu to avoid allocating the training GPU")
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("workers must be an integer in [1, 32]")
    jobs = generation_jobs(variants_per_level)
    base_manifest, output = Path(base_manifest).resolve(), Path(output).resolve()
    source_root = base_manifest.parent
    if output == source_root or output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("Use separate, non-nested source and destination bank directories")
    original = load_generalist_manifest(base_manifest)
    if original["schema"] != EXPANDED_SCHEMA:
        raise ValueError("Hand-protection append requires the expanded CAT bank")
    if original.get("hand_protection_curriculum") or any(
            record.get("source", {}).get("hand_protection") for record in original["scenes"]):
        raise ValueError("Base bank already contains hand-protection tasks")
    if not {"furniture", "generic_clutter"}.issubset({s["family"] for s in original["scenes"]}):
        raise ValueError("Base bank must retain ordinary furniture and generic-clutter rooms")
    for job in jobs:
        candidate = Path(job["path"])
        for old in original["scenes"]:
            prior = Path(old["path"])
            if candidate == prior or candidate.is_relative_to(prior) or prior.is_relative_to(candidate):
                raise ValueError("New hand scene path overlaps an existing scene")
    base_file_hash = sha256(base_manifest)
    plan = dict(builder=BUILDER_VERSION, base_manifest=str(base_manifest),
        base_manifest_file_sha256=base_file_hash,
        base_manifest_sha256=original["manifest_sha256"],
        source_scene_count=original["scene_count"], append_scene_count=len(jobs),
        variants_per_level=variants_per_level, field_dx=FIELD_DX,
        source_hashes=_code_hashes(), jobs=jobs, curriculum=CURRICULUM,
        preserved_storage="hardlinks only; every old scene path and record unchanged")
    plan["plan_sha256"] = _json_hash(plan)
    plan_path = output / "hand_protection_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise FileExistsError("Destination has a different append plan; choose a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Destination is not empty and has no matching append plan")
        output.mkdir(parents=True, exist_ok=True)
        if source_root.stat().st_dev != output.stat().st_dev:
            raise ValueError("Source and destination must share a filesystem for hardlinks")
        _write_json(plan_path, plan)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        complete = load_generalist_manifest(manifest_path)
        if complete.get("hand_protection_curriculum", {}).get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("Complete bank has different hand-protection provenance")
        if complete["scenes"][:len(original["scenes"])] != original["scenes"]:
            raise ValueError("Complete bank changed an original scene record")
        for scene in complete["scenes"][len(original["scenes"]):]:
            _validate_new_record(scene, _contained_directory(output, scene["path"]))
        return manifest_path

    files, linked_bytes = 0, 0
    for record in original["scenes"]:
        count, size = _hardlink_tree(_contained_directory(source_root, record["path"]),
                                    _contained_directory(output, record["path"]))
        files += count
        linked_bytes += size
    # Pin the complete parent manifest and its historical generation reports.
    # They remain explicitly historical rather than describing appended scenes.
    archive = output / "parent_bank"
    archive.mkdir(exist_ok=True)
    sources = [(base_manifest, "manifest.json")] + [
        (path, path.name) for path in sorted(source_root.glob("*.json")) if path != base_manifest]
    for source, filename in sources:
        target = archive / filename
        if target.exists():
            if not os.path.samefile(source, target):
                raise FileExistsError(f"Historical provenance differs: {target}")
        else:
            os.link(source, target)
    if (source_root / "parent_generation").is_dir():
        _hardlink_tree(source_root / "parent_generation", archive / "parent_generation")

    payloads = [(job, str(output), plan) for job in jobs]
    generated = {}
    if workers == 1:
        for index, payload in enumerate(payloads, 1):
            record = _build_scene(payload)
            generated[record["path"]] = record
            if progress:
                progress(index, len(jobs), record["scene_id"])
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(_build_scene, payload) for payload in payloads]
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                generated[record["path"]] = record
                if progress:
                    progress(index, len(jobs), record["scene_id"])
    if base_file_hash != sha256(base_manifest) or plan["source_hashes"] != _code_hashes():
        raise ValueError("Base manifest or builder code changed during append")
    records = copy.deepcopy(original["scenes"]) + [generated[job["path"]] for job in jobs]
    if len({record["scene_id"] for record in records}) != len(records):
        raise ValueError("Appended hand-protection scene ID is not unique")
    counts = dict(Counter(record["family"] for record in records))
    report = dict(builder=BUILDER_VERSION, plan_sha256=plan["plan_sha256"],
        source_scene_count=len(original["scenes"]), appended_scene_count=len(jobs),
        scene_count=len(records), preserved_records_identical=True,
        preserved_scene_order_identical=True, all_original_scene_files_hardlinked=True,
        hardlinked_file_count=files, hardlinked_logical_bytes=linked_bytes,
        retained_family_counts=counts, appended_scene_ids=[record["scene_id"] for record in records[len(original["scenes"]):]])
    _write_json(output / "hand_protection_report.json", report)
    manifest = copy.deepcopy(original)
    manifest.pop("manifest_sha256")
    historical = {key: manifest.pop(key) for key in (
        "navigation_migration", "preparation_plan_sha256", "generation_coverage_sha256",
        "requested_family_counts") if key in manifest}
    manifest.update(scenes=records, scene_count=len(records), retained_family_counts=counts,
        fields_bytes=sum(field["size_bytes"] for record in records for field in record["fields"].values()),
        hand_protection_curriculum=dict(CURRICULUM, builder=BUILDER_VERSION,
            plan_sha256=plan["plan_sha256"], base_manifest_sha256=original["manifest_sha256"],
            report_sha256=sha256(output / "hand_protection_report.json"),
            parent_metadata=historical, parent_manifest="parent_bank/manifest.json"))
    from cat_ppo.furniture.hand_curriculum import curriculum_levels
    curriculum_levels(manifest, enabled=True)
    manifest["manifest_sha256"] = _json_hash(manifest)
    pending = output / ".manifest.validating.json"
    _write_json(pending, manifest)
    load_generalist_manifest(pending)
    pending.replace(manifest_path)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variants-per-level", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = build_bank(args.base_manifest, args.output, variants_per_level=args.variants_per_level,
                        workers=args.workers, progress=lambda i, n, identity:
                        print(f"[{i}/{n}] {identity}", flush=True))
    print(json.dumps(dict(manifest=str(result),
                         manifest_sha256=load_generalist_manifest(result, verify_files=False)["manifest_sha256"])))


if __name__ == "__main__":
    main()
