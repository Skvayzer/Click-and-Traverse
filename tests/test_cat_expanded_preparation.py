"""Coverage, resumability and source-integrity checks without network fixtures."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cat_ppo.furniture import expanded_fields as expanded
from cat_ppo.furniture.generalist_fields import (
    DATASET_REPO, DATASET_REVISION, FIELD_NAMES, _field_records, sha256,
)
from cat_ppo.furniture.legacy_scenes import TYPICAL_SCENES, _upstream


def _procedural_job():
    return dict(kind="procedural_cat", scene_id="procedural-fixture", path="procedural/fixture",
                parameters=dict(difficulty=.7, seed=2001, n_rect_F=2, n_rect_L=3, n_rect_R=3, n_rect_C=1))


def test_default_covers_every_original_count_combination_difficulty_and_seed():
    plan = expanded.generation_plan()
    assert plan["requested_total_count"] == 2338
    assert plan["family_counts"] == dict(published_cat=27, procedural_cat=2226, furniture=24, generic_clutter=24)
    procedural = [job["parameters"] for job in plan["jobs"] if job["kind"] == "procedural_cat"]
    for difficulty in expanded.DEFAULT_DIFFICULTIES:
        matching = [p for p in procedural if p["difficulty"] == difficulty]
        assert len(matching) == 318
        assert {(bool(p["n_rect_F"]), bool(p["n_rect_L"]), bool(p["n_rect_C"])) for p in matching} == {
            (True, False, False), (False, True, False), (False, False, True),
            (True, True, False), (True, False, True), (False, True, True), (True, True, True)}
    assert {p["seed"] for p in procedural}.isdisjoint(range(101, 111))
    assert {j["name"] for j in plan["jobs"] if j["kind"] == "published_cat"} == set(TYPICAL_SCENES)
    assert len({job["path"] for job in plan["jobs"]}) == 2301


@pytest.mark.parametrize("kwargs", [dict(seeds=[101]), dict(difficulties=[float("nan")]),
                                    dict(difficulties=[.45]), dict(lateral_counts=[10]),
                                    dict(seeds=[True]), dict(ground_counts=[-1]),
                                    dict(typical_scenes=["not-released"])])
def test_bad_or_held_out_generation_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        expanded.generation_plan(**kwargs)


def test_connectivity_rejects_impenetrable_wall_not_just_free_endpoints():
    occupancy = np.zeros((9, 9, 9), dtype=bool)
    occupancy[4, :, :] = True
    with pytest.raises(expanded.SceneAdmissionError, match="disconnected"):
        expanded.check_free_space_connectivity(occupancy, np.zeros(3), 1., [2., 4., 4.], [6., 4., 4.])
    occupancy[4, 4, 4] = False
    report = expanded.check_free_space_connectivity(occupancy, np.zeros(3), 1., [2., 4., 4.], [6., 4., 4.])
    assert report["start_goal_connected"]
    assert report["root_radius_validated"] is False


def test_real_original_pipeline_is_native_and_cache_skips_regeneration(tmp_path, monkeypatch):
    job = _procedural_job()
    original = expanded.generate_procedural_fields
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(expanded, "generate_procedural_fields", counted)
    first = expanded._build_job((job, str(tmp_path), {"fixture": "fixed"}, False))
    second = expanded._build_job((job, str(tmp_path), {"fixture": "fixed"}, False))
    assert first == second
    assert len(calls) == 1
    generator = _upstream("random_obstacle")
    expected, *_ = generator.generate_and_save(generator.Cfg(**job["parameters"]), save=False)
    np.testing.assert_array_equal(np.load(tmp_path / job["path"] / "obs.npy"), expected)
    assert first["origin"] == [-.5, -1., 0.]
    assert first["source"]["generation_sample_origin"] == pytest.approx([-.48, -.98, .02])
    assert first["task_kind"] == "cat" and first["episode_length"] == 1000
    assert first["reset_xy_scale"] == [1., 1.]
    assert first["source"]["connectivity"]["start_goal_connected"]
    assert all(first["fields"][field]["dtype"] == "float32" for field in FIELD_NAMES)
    source = tmp_path / job["path"] / "source.json"
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="Cached source"):
        expanded._build_job((job, str(tmp_path), {"fixture": "fixed"}, False))


def test_geometry_rejection_is_cached_but_io_errors_propagate(tmp_path, monkeypatch):
    job = _procedural_job()
    def reject(*args):
        raise expanded.SceneAdmissionError("disconnected fixture")
    monkeypatch.setattr(expanded, "generate_procedural_fields", reject)
    record = expanded._build_job((job, str(tmp_path), {}, False))
    assert record["rejected"] and record["reason"] == "disconnected fixture"
    def unexpected(*args):
        raise OSError("disk unavailable")
    monkeypatch.setattr(expanded, "generate_procedural_fields", unexpected)
    assert expanded._build_job((job, str(tmp_path), {}, False)) == record
    with pytest.raises(OSError, match="disk unavailable"):
        expanded._build_job((dict(job, path="uncached"), str(tmp_path), {}, False))


def _fake_scene(identity, family, fingerprint):
    return dict(scene_id=identity, family=family, path=identity,
                task_kind="cat", fields={name: {"sha256": fingerprint + name} for name in FIELD_NAMES},
                origin=[-.5, -1., 0.], shape=[75, 50, 38], dx=.04,
                start=[0., 0., .8], goal=[2., 0., .75], source={"geometry_sha256": fingerprint})


def test_dedup_omits_duplicate_slots_retains_parameter_aliases_and_mandatory_anchors():
    anchor = _fake_scene("anchor", "original_cat", "same")
    job = _procedural_job()
    duplicate = _fake_scene("duplicate", "procedural_cat", "same")
    unique = _fake_scene("unique", "procedural_cat", "different")
    published = _fake_scene("published", "published_cat", "same")
    jobs = [job, dict(job, scene_id="unique"), dict(kind="published_cat")]
    kept, coverage = expanded.deduplicate_generated([anchor], [duplicate, unique, published], jobs)
    assert [s["scene_id"] for s in kept] == ["anchor", "unique", "published"]
    assert coverage[0]["duplicate_of"] == "anchor"
    assert coverage[0]["generation"]["parameters"] == job["parameters"]
    assert coverage[2]["included_in_bank"]


def test_rejection_of_entire_requested_obstacle_family_fails_coverage():
    job = _procedural_job()
    rejected = dict(rejected=True, scene_id="rejected", family="procedural_cat", path="rejected", reason="blocked")
    with pytest.raises(ValueError, match="coverage lost"):
        expanded.deduplicate_generated([], [rejected], [job])


def test_published_arrays_verified_against_pinned_cached_lfs_metadata(tmp_path):
    job = dict(kind="published_cat", name="hurdle0", scene_id="published-hurdle0", path="published/hurdle0")
    shape = (75, 50, 38)
    values = {"obs": np.zeros(shape, np.uint8), "sdf": np.ones(shape, np.float32),
              "bf": np.ones((*shape, 3), np.float32), "gf": np.ones((*shape, 3), np.float32)}
    files = {}
    relative = "assets_v1/TypiObs/hurdle0"
    for name, array in values.items():
        file = tmp_path / f"{name}.npy"
        np.save(file, array, allow_pickle=False)
        files[file.name] = dict(path=f"{relative}/{file.name}", lfs={"oid": sha256(file)})
    expanded._write_json(tmp_path / "release_metadata.json",
        dict(repo=DATASET_REPO, revision=DATASET_REVISION, scene_path=relative, files=files))
    record = expanded._fetch_published_fields(job, tmp_path, {}, False)
    assert record["source"]["arrays_unchanged"]
    assert record["source"]["revision"] == DATASET_REVISION
    assert record["family"] == "published_cat" and record["sampling_group"] == "original_cat"
    with (tmp_path / "gf.npy").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="Missing/corrupt pinned"):
        expanded._fetch_published_fields(job, tmp_path, {}, False)
