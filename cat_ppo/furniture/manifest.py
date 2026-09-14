"""Generate split-specific benchmark specifications without running evaluations."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from cat_ppo.furniture.scenes import FAMILIES, generate_scene, write_scene_bundle, load_scene, validate_scene


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_manifest(output, *, split, layouts_per_family=20, seed=20260914,
                   training_seeds=(0, 1, 2)):
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation, or test")
    if layouts_per_family < 1 or not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("positive layout count and distinct training seeds required")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace benchmark manifest: {output}")
    cases = []
    for family_index, family in enumerate(FAMILIES):
        for layout in range(layouts_per_family):
            scene_seed = seed + family_index * 100000 + layout
            scene = generate_scene(seed=scene_seed, split=split, family=family, difficulty="dense")
            variants = scene["start_goals"]
            if len(variants) != 3:
                raise ValueError("benchmark contract requires three start/goal variants")
            for goal_index, _ in enumerate(variants):
                cases.append({"case_id": f"{family}-{layout:03d}-g{goal_index}",
                              "scene_id": scene["scene_id"], "scene": scene,
                              "scene_sha256": digest(scene), "goal_index": goal_index,
                              "episode_seed": scene_seed * 3 + goal_index})
    manifest = {"schema": "cat-furniture-benchmark-v1", "split": split,
                "base_seed": seed, "families": list(FAMILIES),
                "layouts_per_family": layouts_per_family,
                "layouts": len(FAMILIES)*layouts_per_family, "cases": cases,
                "training_seeds": list(training_seeds),
                "episodes_per_controller": len(cases)*len(training_seeds),
                "status": "specified; no evaluations run",
                "geometry_validation": "scene generator checks are not dynamic traversal evidence"}
    manifest["manifest_sha256"] = digest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    expected = manifest.pop("manifest_sha256")
    if digest(manifest) != expected:
        raise ValueError("benchmark manifest hash mismatch")
    manifest["manifest_sha256"] = expected
    if manifest["schema"] != "cat-furniture-benchmark-v1":
        raise ValueError("unsupported benchmark manifest schema")
    identities = set()
    for case in manifest["cases"]:
        if digest(case["scene"]) != case["scene_sha256"]:
            raise ValueError(f"scene specification hash mismatch: {case['case_id']}")
        validate_scene(case["scene"])
        if case["scene_id"] != case["scene"]["scene_id"] or case["scene"]["split"] != manifest["split"]:
            raise ValueError("benchmark case scene identity/split differs from its specification")
        goal_index = case["goal_index"]
        if type(goal_index) is not int or not 0 <= goal_index < len(case["scene"]["start_goals"]):
            raise ValueError("benchmark case goal_index is outside start_goals")
        identity = (case["scene_id"], case["case_id"], case["episode_seed"])
        if identity in identities:
            raise ValueError("duplicate benchmark episode identity")
        identities.add(identity)
    return manifest


def _case_specification(scene, *, goal_index=None):
    """Canonical requested data, excluding derived arrays and reset distance.

    The writer recomputes reset center clearance for the selected start. Its
    radius, vertical band and validation labels remain part of the comparison.
    """
    specification = copy.deepcopy(scene)
    for key in ("grid", "fields", "field_provenance"):
        specification.pop(key, None)
    if goal_index is not None:
        cases = specification.get("start_goals", [scene])
        if type(goal_index) is not int or not 0 <= goal_index < len(cases):
            raise ValueError("goal_index is outside start_goals")
        for key in ("start", "goal", "route", "time_budget", "bottlenecks", "route_length_m"):
            if key in cases[goal_index]:
                specification[key] = copy.deepcopy(cases[goal_index][key])
        specification["goal_index"] = goal_index
    if "reset_clearance" in specification:
        specification["reset_clearance"].pop("center_clearance_m", None)
    return specification


def materialize_case(case, cache, *, voxel_size=0.10):
    """Materialize one matched physical scene and goal-specific PF bundle."""
    if digest(case["scene"]) != case["scene_sha256"]:
        raise ValueError("scene hash mismatch")
    expected = _case_specification(case["scene"], goal_index=case["goal_index"])
    key = digest({"scene": case["scene_sha256"], "goal": case["goal_index"], "dx": voxel_size})[:24]
    directory = Path(cache).resolve() / key
    if not (directory / "scene.json").exists():
        write_scene_bundle(case["scene"], directory, voxel_size=voxel_size, goal_index=case["goal_index"])
    loaded = load_scene(directory)
    if (_case_specification(loaded) != expected
            or loaded["grid"]["voxel_size"] != voxel_size):
        raise ValueError("cached bundle differs from the full requested scene/case specification or resolution")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--layouts-per-family", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    result = build_manifest(args.output, split=args.split, layouts_per_family=args.layouts_per_family,
                            seed=args.seed, training_seeds=args.training_seeds)
    print(json.dumps({k: result[k] for k in ("split", "layouts", "episodes_per_controller", "manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
