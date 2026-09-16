"""Bank migration preserves source bytes and publishes only complete replacements."""

import copy
import errno
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from cat_ppo.furniture import generalist_fields as fields
from cat_ppo.furniture.generalist_config import released_config


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rebuild_clutter_navigation_bank.py"
SPEC = importlib.util.spec_from_file_location("rebuild_clutter_navigation_bank", SCRIPT)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


def _write_arrays(directory, value):
    directory.mkdir(parents=True, exist_ok=True)
    for name in fields.FIELD_NAMES:
        shape = (3, 3, 3) if name == "sdf" else (3, 3, 3, 3)
        np.save(directory / f"{name}.npy", np.full(shape, value, dtype=np.float32))
    np.save(directory / "obs.npy", np.zeros((3, 3, 3), dtype=np.uint8))


@pytest.fixture
def source_bank(tmp_path):
    root = tmp_path / "source"
    configured = released_config()["env_config"]["pf_config"]["paths"]
    specifications = [("original_cat", Path(p).name, p) for p in configured]
    specifications += [(family, family, None) for family in
                       ("published_cat", "procedural_cat", "furniture", "generic_clutter")]
    records = []
    for index, (family, identity, original_path) in enumerate(specifications):
        directory = root / family / identity
        _write_arrays(directory, index)
        is_room = family in migration.ROOM_FAMILIES
        source = dict(kind="test-fixture")
        if original_path:
            source["arrays_unchanged"] = True
            if original_path.endswith("D8G2L3O2S13"):
                source.update(kind="reconstructed-missing-original", arrays_unchanged=False)
        migration._write_json(directory / "source.json", source)
        record = dict(scene_id=identity, family=family, path=str(directory.relative_to(root)),
            shape=[3, 3, 3], origin=[0., 0., 0.], dx=1., start=[0., 0., .8], goal=[2., 0., .75],
            reset_xy_scale=[.08, .08] if is_room else [1., 1.], reset_yaw=0.,
            sampling_weight=.75 if is_room else 1., fields=fields._field_records(directory),
            source=dict(source, metadata_sha256=fields.sha256(directory / "source.json")),
            task_kind="room" if is_room else "cat", reset_mode="room" if is_room else "cat",
            crossed_mode="goal_radius" if is_room else "x_plane", episode_length=4000 if is_room else 1000,
            sampling_group="original_cat" if family == "published_cat" else family)
        if original_path:
            record["original_config_path"] = original_path
        if is_room:
            scene = dict(scene_id=identity, family=family, route=[[0., 0.], [2., 0.]], boxes=[],
                         cat_field_generation={"route_used_for_guidance": False})
            migration._write_json(directory / "scene.json", scene)
            record["scene_sha256"] = fields.sha256(directory / "scene.json")
        records.append(record)
    manifest = dict(schema=fields.EXPANDED_SCHEMA, scene_count=len(records), scenes=records,
        original_count=37, byte_verified_original_count=36, reconstructed_original_count=1,
        released_config_sha256=fields.RELEASED_CONFIG_SHA256, dataset_revision=fields.DATASET_REVISION,
        sampling_group_masses=fields.DEFAULT_SAMPLING_GROUP_MASSES)
    manifest["manifest_sha256"] = fields._json_hash(manifest)
    migration._write_json(root / "manifest.json", manifest)
    fields.load_generalist_manifest(root / "manifest.json")
    return root / "manifest.json"


@pytest.fixture
def generator(monkeypatch, source_bank):
    originals = {s["scene_id"]: s for s in fields.load_generalist_manifest(source_bank)["scenes"]}
    calls = []

    def generate(scene, directory, *, dx):
        calls.append(scene["scene_id"])
        _write_arrays(directory, 123.)
        updated_scene = copy.deepcopy(scene)
        updated_scene["cat_field_generation"] = dict(migration.REQUIRED_ROOM_METADATA)
        migration._write_json(directory / "scene.json", updated_scene)
        old = originals[scene["scene_id"]]
        result = {key: copy.deepcopy(old[key]) for key in
                  ("scene_id", "family", "shape", "origin", "dx", "start", "goal", "reset_xy_scale", "reset_yaw")}
        result.update(fields=fields._field_records(directory), source=dict(migration.REQUIRED_ROOM_METADATA),
                      scene_sha256=fields.sha256(directory / "scene.json"))
        return result

    monkeypatch.setattr(migration, "make_clutter_fields", generate)
    return calls, generate


def test_rebuild_preserves_source_bytes_and_hardlinks_every_unchanged_field(source_bank, tmp_path, generator):
    calls, _ = generator
    source_hashes = {str(p.relative_to(source_bank.parent)): fields.sha256(p)
                     for p in source_bank.parent.rglob("*") if p.is_file()}
    output = tmp_path / "corrected"
    path = migration.rebuild_bank(source_bank, output)
    before, after = fields.load_generalist_manifest(source_bank), fields.load_generalist_manifest(path)
    assert calls == ["furniture", "generic_clutter"]
    assert before["manifest_sha256"] != after["manifest_sha256"]
    for old, new in zip(before["scenes"], after["scenes"], strict=True):
        if old["family"] in migration.CAT_FAMILIES:
            assert old == new
            for filename in ("sdf.npy", "bf.npy", "gf.npy", "source.json", "obs.npy"):
                assert os.path.samefile(source_bank.parent / old["path"] / filename,
                                        output / new["path"] / filename)
        else:
            assert old["sampling_weight"] == new["sampling_weight"] == .75
            assert old["fields"] != new["fields"]
            assert not os.path.samefile(source_bank.parent / old["path"] / "sdf.npy",
                                       output / new["path"] / "sdf.npy")
            assert new["source"]["navigation_migration"]["canonical_geometry_unchanged"]
    assert source_hashes == {str(p.relative_to(source_bank.parent)): fields.sha256(p)
                             for p in source_bank.parent.rglob("*") if p.is_file()}
    report = json.loads((output / "navigation_migration_report.json").read_text())
    assert report["unchanged_cat_scene_count"] == 39
    assert report["rebuilt_room_count"] == 2
    assert report["hardlinked_file_count"] == 39 * 5
    # A completed matching output validates and returns without regenerating.
    assert migration.rebuild_bank(source_bank, output) == path
    assert calls == ["furniture", "generic_clutter"]


def test_failed_room_never_publishes_and_retry_reuses_verified_completed_room(source_bank, tmp_path, generator, monkeypatch):
    calls, generate = generator
    output = tmp_path / "corrected"

    def fail_second(scene, directory, *, dx):
        if scene["scene_id"] == "generic_clutter":
            raise RuntimeError("simulated field generation interruption")
        return generate(scene, directory, dx=dx)

    monkeypatch.setattr(migration, "make_clutter_fields", fail_second)
    with pytest.raises(RuntimeError, match="interruption"):
        migration.rebuild_bank(source_bank, output)
    assert not (output / "manifest.json").exists()
    assert calls == ["furniture"]
    monkeypatch.setattr(migration, "make_clutter_fields", generate)
    migration.rebuild_bank(source_bank, output)
    assert calls == ["furniture", "generic_clutter"]


def test_old_builder_cannot_publish_a_claimed_navigation_fix(source_bank, tmp_path, generator, monkeypatch):
    _, generate = generator

    def old_builder(scene, directory, *, dx):
        record = generate(scene, directory, dx=dx)
        record["source"]["route_used_for_guidance"] = False
        return record

    monkeypatch.setattr(migration, "make_clutter_fields", old_builder)
    output = tmp_path / "incorrect"
    with pytest.raises(ValueError, match="has not integrated"):
        migration.rebuild_bank(source_bank, output)
    assert not (output / "manifest.json").exists()


def test_in_place_nested_or_unowned_destination_is_refused(source_bank, tmp_path, generator):
    for output in (source_bank.parent, source_bank.parent / "nested", tmp_path):
        with pytest.raises(ValueError, match="non-nested"):
            migration.rebuild_bank(source_bank, output)
    output = tmp_path / "unrelated"
    output.mkdir()
    (output / "important.txt").write_text("retain this")
    with pytest.raises(FileExistsError, match="not empty"):
        migration.rebuild_bank(source_bank, output)
    assert (output / "important.txt").read_text() == "retain this"


def test_hardlink_failure_has_no_silent_copy_fallback(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "field.npy").write_bytes(b"source bytes")

    def fail(*args, **kwargs):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(migration.os, "link", fail)
    with pytest.raises(OSError, match="copy fallback is intentionally disabled"):
        migration._hardlink_tree(source, output)
    assert not (output / "field.npy").exists()
    assert (source / "field.npy").read_bytes() == b"source bytes"
