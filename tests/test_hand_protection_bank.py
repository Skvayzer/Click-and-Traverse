"""Append invariants, interruption recovery and actual on-disk hash validation."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from cat_ppo.furniture import generalist_fields as fields
from cat_ppo.furniture.generalist_config import released_config
from cat_ppo.furniture.hand_curriculum import curriculum_levels
from scripts import build_hand_protection_bank as builder


def arrays(directory, value=0.):
    directory.mkdir(parents=True, exist_ok=True)
    for name in fields.FIELD_NAMES:
        shape = (3, 3, 3) if name == "sdf" else (3, 3, 3, 3)
        np.save(directory / f"{name}.npy", np.full(shape, value, np.float32))
    np.save(directory / "obs.npy", np.zeros((3, 3, 3), np.uint8))


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "old"
    paths = released_config()["env_config"]["pf_config"]["paths"]
    specs = [("original_cat", Path(path).name, path) for path in paths]
    specs += [(kind, kind, None) for kind in ("published_cat", "procedural_cat", "furniture", "generic_clutter")]
    records = []
    for index, (family, identity, original) in enumerate(specs):
        directory = root / family / identity
        arrays(directory, index)
        room = family in ("furniture", "generic_clutter")
        metadata = dict(kind="fixture")
        if original:
            metadata["arrays_unchanged"] = True
            if original.endswith("D8G2L3O2S13"):
                metadata.update(kind="reconstructed-missing-original", arrays_unchanged=False)
        builder._write_json(directory / "source.json", metadata)
        record = dict(scene_id=identity, family=family, path=str(directory.relative_to(root)),
            fields=fields._field_records(directory), source=dict(metadata, metadata_sha256=fields.sha256(directory / "source.json")),
            shape=[3, 3, 3], origin=[0., 0., 0.], dx=1., start=[0., 0., .8], goal=[2., 0., .75],
            reset_xy_scale=[.08, .08] if room else [1., 1.], reset_yaw=0., sampling_weight=.7 if room else 1.,
            task_kind="room" if room else "cat", reset_mode="room" if room else "cat",
            crossed_mode="goal_radius" if room else "x_plane", episode_length=4000 if room else 1000,
            sampling_group="original_cat" if family == "published_cat" else family)
        if original:
            record["original_config_path"] = original
        if room:
            builder._write_json(directory / "scene.json", dict(scene_id=identity, geometry_hash="fixture"))
            record["scene_sha256"] = fields.sha256(directory / "scene.json")
        records.append(record)
    manifest = dict(schema=fields.EXPANDED_SCHEMA, scene_count=len(records), scenes=records,
        original_count=37, byte_verified_original_count=36, reconstructed_original_count=1,
        released_config_sha256=fields.RELEASED_CONFIG_SHA256, dataset_revision=fields.DATASET_REVISION,
        sampling_group_masses=fields.DEFAULT_SAMPLING_GROUP_MASSES)
    manifest["manifest_sha256"] = fields._json_hash(manifest)
    builder._write_json(root / "manifest.json", manifest)
    fields.load_generalist_manifest(root / "manifest.json")
    return root / "manifest.json"


@pytest.fixture
def generated(monkeypatch):
    calls = []
    def scene(seed, *, kind, difficulty):
        calls.append(seed)
        return dict(scene_id=f"{kind}-{difficulty}-{seed}", geometry_hash=f"geometry-{seed}",
            counts=dict(generic_objects=int(kind == "hand_shelf_passage")),
            hand_protection=dict(kind=kind, difficulty=difficulty, level=builder.DIFFICULTIES.index(difficulty),
                certificate=dict(method="small-fixture-certificate", dynamic_feasibility_validated=False)))
    def generate_fields(scene, directory, *, dx):
        assert dx == .04
        arrays(directory, len(calls))
        builder._write_json(directory / "scene.json", scene)
        return dict(scene_id=scene["scene_id"], family="generic_clutter" if scene["counts"]["generic_objects"] else "furniture",
            shape=[3, 3, 3], origin=[0., 0., .76], dx=dx,
            start=[0., 0., .8], goal=[.08, .08, .8], reset_xy_scale=[.08, .08], reset_yaw=0.,
            fields=fields._field_records(directory), scene_sha256=fields.sha256(directory / "scene.json"),
            source=dict(occupancy="conservative-voxel-cell-OBB-intersection-v1",
                        room_navigation="ordered-certified-route-v1", route_used_for_guidance=True))
    monkeypatch.setattr(builder, "generate_scene", scene)
    monkeypatch.setattr(builder, "make_clutter_fields", generate_fields)
    return calls, scene, generate_fields


def test_append_preserves_all_records_paths_bytes_and_inodes(source, tmp_path, generated):
    calls, _, _ = generated
    before = fields.load_generalist_manifest(source)
    hashes = {str(p.relative_to(source.parent)): fields.sha256(p) for p in source.parent.rglob("*") if p.is_file()}
    output = tmp_path / "new"
    result = builder.build_bank(source, output, variants_per_level=1)
    after = fields.load_generalist_manifest(result)
    assert after["scene_count"] == before["scene_count"] + 6
    assert after["scenes"][:before["scene_count"]] == before["scenes"]
    assert after["sampling_group_masses"] == before["sampling_group_masses"]
    for record in before["scenes"]:
        for file in (source.parent / record["path"]).rglob("*"):
            if file.is_file():
                assert os.path.samefile(file, output / file.relative_to(source.parent))
    assert hashes == {str(p.relative_to(source.parent)): fields.sha256(p) for p in source.parent.rglob("*") if p.is_file()}
    assert os.path.samefile(source, output / "parent_bank/manifest.json")
    levels = curriculum_levels(after, enabled=True)
    np.testing.assert_array_equal(levels[:before["scene_count"]], -1)
    np.testing.assert_array_equal(levels[-6:], [0, 1, 2, 0, 1, 2])
    assert calls == [6001, 6002, 6003, 7001, 7002, 7003]
    for record in after["scenes"][-6:]:
        metadata = json.loads((output / record["path"] / "source.json").read_text())
        assert metadata["hand_protection"]["certificate"]["method"] == "small-fixture-certificate"
        assert metadata["connectivity"]["start_goal_connected"]
        assert metadata["source_hashes"]
        assert record["dx"] == .04
    assert builder.build_bank(source, output, variants_per_level=1) == result
    assert len(calls) == 6


def test_interrupted_build_does_not_publish_and_resumes_verified_scenes(source, tmp_path, generated, monkeypatch):
    calls, generate, _ = generated
    output = tmp_path / "new"
    def fail(seed, **kwargs):
        if seed == 6002:
            raise RuntimeError("intentional interruption")
        return generate(seed, **kwargs)
    monkeypatch.setattr(builder, "generate_scene", fail)
    with pytest.raises(RuntimeError, match="interruption"):
        builder.build_bank(source, output, variants_per_level=1)
    assert not (output / "manifest.json").exists()
    assert calls == [6001]
    monkeypatch.setattr(builder, "generate_scene", generate)
    builder.build_bank(source, output, variants_per_level=1)
    assert calls == [6001, 6002, 6003, 7001, 7002, 7003]


@pytest.mark.parametrize("filename", ["sdf.npy", "scene.json", "source.json", "obs.npy"])
def test_completed_bank_detects_appended_file_tampering(source, tmp_path, generated, filename):
    output = tmp_path / "new"
    result = builder.build_bank(source, output, variants_per_level=1)
    record = fields.load_generalist_manifest(result)["scenes"][-1]
    file = output / record["path"] / filename
    if filename.endswith(".npy"):
        data = np.load(file)
        data.flat[0] += 1
        np.save(file, data)
    else:
        file.write_text(file.read_text() + " ")
    with pytest.raises(ValueError, match="fingerprint|hashes"):
        builder.build_bank(source, output, variants_per_level=1)


def test_in_place_or_changed_plan_or_unowned_output_refused(source, tmp_path, generated):
    for output in (source.parent, source.parent / "nested", tmp_path):
        with pytest.raises(ValueError, match="non-nested"):
            builder.build_bank(source, output, variants_per_level=1)
    output = tmp_path / "new"
    builder.build_bank(source, output, variants_per_level=1)
    with pytest.raises(FileExistsError, match="different append plan"):
        builder.build_bank(source, output, variants_per_level=2)
    other = tmp_path / "other"
    other.mkdir()
    (other / "keep.txt").write_text("untouched")
    with pytest.raises(FileExistsError, match="not empty"):
        builder.build_bank(source, other, variants_per_level=1)
    assert (other / "keep.txt").read_text() == "untouched"


def test_uncertified_or_legacy_field_generation_never_publishes(source, tmp_path, generated, monkeypatch):
    _, generate, generate_fields = generated
    def uncertified(seed, **kwargs):
        scene = generate(seed, **kwargs)
        del scene["hand_protection"]["certificate"]
        return scene
    monkeypatch.setattr(builder, "generate_scene", uncertified)
    with pytest.raises(ValueError, match="static certificate"):
        builder.build_bank(source, tmp_path / "uncertified", variants_per_level=1)
    assert not (tmp_path / "uncertified/manifest.json").exists()
    monkeypatch.setattr(builder, "generate_scene", generate)
    def old_fields(*args, **kwargs):
        record = generate_fields(*args, **kwargs)
        record["source"]["route_used_for_guidance"] = False
        return record
    monkeypatch.setattr(builder, "make_clutter_fields", old_fields)
    with pytest.raises(ValueError, match="has not integrated"):
        builder.build_bank(source, tmp_path / "old_fields", variants_per_level=1)
    assert not (tmp_path / "old_fields/manifest.json").exists()


def test_default_plan_has_24_scenes_in_two_existing_families():
    jobs = builder.generation_jobs()
    assert len(jobs) == 24
    assert [job["seed"] for job in jobs] == list(range(6001, 6013)) + list(range(7001, 7013))
    assert {(job["family"], job["level"]) for job in jobs} == {
        (family, level) for family in ("furniture", "generic_clutter") for level in range(3)}
    for invalid in (0, -1, True, 101, 1.5):
        with pytest.raises(ValueError):
            builder.generation_jobs(invalid)
