import copy
import json

import pytest

from cat_ppo.furniture import manifest
from cat_ppo.furniture.scenes import generate_scene, write_scene_bundle


def test_manifest_has_distinct_cases_and_detects_modified_scene(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "FAMILIES", ("table",))
    path = tmp_path / "benchmark.json"
    generated = manifest.build_manifest(path, split="test", layouts_per_family=1)
    assert generated["episodes_per_controller"] == 9
    assert len({x["case_id"] for x in generated["cases"]}) == 3
    assert len({x["scene"]["geometry_hash"] for x in generated["cases"]}) == 1
    assert all(x["scene"]["counts"]["tables"] == 9 for x in generated["cases"])
    assert manifest.load_manifest(path) == generated
    generated["cases"][0]["scene"]["boxes"][0]["center"][0] += 1
    path.write_text(json.dumps(generated))
    with pytest.raises(ValueError, match="hash mismatch"):
        manifest.load_manifest(path)


def test_manifest_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "FAMILIES", ("table",))
    path = tmp_path / "exists.json"
    path.write_text("important previous results")
    with pytest.raises(FileExistsError):
        manifest.build_manifest(path, split="validation", layouts_per_family=1)
    assert path.read_text() == "important previous results"


def _case(scene, goal_index=0):
    return {"scene": scene, "scene_sha256": manifest.digest(scene), "goal_index": goal_index}


@pytest.mark.parametrize("mutation", ["time_budget", "route", "identity", "split", "reset_radius"])
def test_cache_rejects_valid_bundle_from_different_requested_specification(tmp_path, mutation):
    scene = generate_scene(difficulty="open_floor")
    case = _case(scene)
    key = manifest.digest({"scene": case["scene_sha256"], "goal": 0, "dx": .2})[:24]
    changed = copy.deepcopy(scene)
    if mutation == "time_budget":
        changed["time_budget"] += 100
        changed["start_goals"][0]["time_budget"] += 100
    elif mutation == "route":
        changed["route"].insert(1, [2, 1.60])
        changed["start_goals"][0]["route"].insert(1, [2, 1.60])
    elif mutation == "identity":
        changed["scene_id"] += "-different"
    elif mutation == "split":
        changed["split"] = "test"
    else:
        changed["reset_clearance"]["root_radius_m"] += .05
    write_scene_bundle(changed, tmp_path / key, voxel_size=.2)
    with pytest.raises(ValueError, match="full requested"):
        manifest.materialize_case(case, tmp_path, voxel_size=.2)


def test_cache_accepts_exact_reverse_case_and_reuses_verified_fields(tmp_path):
    scene = generate_scene(difficulty="open_floor")
    case = _case(scene, goal_index=2)
    first = manifest.materialize_case(case, tmp_path, voxel_size=.2)
    before = (first / "gf.npy").stat().st_mtime_ns
    second = manifest.materialize_case(case, tmp_path, voxel_size=.2)
    assert first == second and (second / "gf.npy").stat().st_mtime_ns == before
    loaded = json.loads((second / "scene.json").read_text())
    assert loaded["start"] == scene["start_goals"][2]["start"]
    assert loaded["goal"] == scene["start_goals"][2]["goal"]


@pytest.mark.parametrize("mismatch", ["split", "scene_id", "duplicate"])
def test_rehashed_manifest_still_requires_consistent_case_identity(tmp_path, monkeypatch, mismatch):
    monkeypatch.setattr(manifest, "FAMILIES", ("table",))
    path = tmp_path / "manifest.json"
    generated = manifest.build_manifest(path, split="validation", layouts_per_family=1)
    if mismatch == "split":
        generated["split"] = "test"
    elif mismatch == "scene_id":
        generated["cases"][0]["scene_id"] = "incorrect-label"
    else:
        generated["cases"].append(copy.deepcopy(generated["cases"][0]))
    generated.pop("manifest_sha256")
    generated["manifest_sha256"] = manifest.digest(generated)
    path.write_text(json.dumps(generated))
    with pytest.raises(ValueError, match="identity"):
        manifest.load_manifest(path)
