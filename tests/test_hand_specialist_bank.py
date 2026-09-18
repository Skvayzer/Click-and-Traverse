"""Lossless specialist extraction, identity proofs and curriculum distribution."""
from copy import deepcopy
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest

from cat_ppo.furniture import generalist_fields as fields
from cat_ppo.furniture.body_collision_bank import ARRAY_NAMES, load_body_collision_bank
from cat_ppo.furniture.generalist_config import released_config
from cat_ppo.furniture.hand_curriculum import curriculum_levels, hand_scene_logits
from cat_ppo.furniture.hand_specialist import (
    build_hand_specialist_bank, compact_collision_arrays, validate_specialist_manifest,
)


def _write(path, data, hashed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = deepcopy(data)
    if hashed:
        data.pop("manifest_sha256", None)
        data["manifest_sha256"] = fields._json_hash(data)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data


def _arrays(count):
    # Three boxes and three cells per scene. Include an empty middle cell and
    # repeated box IDs, so compaction must preserve CSR holes and multiplicity.
    boxes = count * 3 + 1
    return dict(
        centers=np.arange(boxes * 3, dtype=np.float32).reshape(boxes, 3),
        half_sizes=np.full((boxes, 3), .1, np.float32),
        rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (boxes, 3, 3)).copy(),
        scene_box_offsets=np.arange(count, dtype=np.int32) * 3 + 1,
        scene_box_counts=np.full(count, 3, np.int32),
        scene_grid_origins=np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        scene_grid_shapes=np.tile(np.array([3, 1, 1], np.int32), (count, 1)),
        scene_grid_offsets=np.arange(count, dtype=np.int32) * 3,
        cell_starts=np.concatenate([np.array([1, 3, 3], np.int32) + i * 4 for i in range(count)]),
        cell_counts=np.tile(np.array([2, 0, 2], np.int32), count),
        candidate_ids=np.concatenate([np.zeros(1, np.int32), *[
            np.array([1, 2, 2, 3], np.int32) + i * 3 for i in range(count)]]),
        cell_size=np.array(.25, np.float32))


def _fixture_banks(root):
    field_root, collision_root, reset_root = (root / name for name in ("fields", "collision", "resets"))
    scenes = []
    original_paths = released_config()["env_config"]["pf_config"]["paths"]
    descriptions = [("original_cat", index, None, original) for index, original in enumerate(original_paths)]
    for kind, family in (("hand_table_aisle", "furniture"), ("hand_shelf_passage", "generic_clutter")):
        for level in range(3):
            for repeat in range(4):
                descriptions.append((family, len(descriptions), (kind, level), None))
    for family, index, hand, original in descriptions:
        directory = field_root / f"scenes/{index}"
        directory.mkdir(parents=True)
        for name in fields.FIELD_NAMES:
            shape = (3, 3, 3) if name == "sdf" else (3, 3, 3, 3)
            np.save(directory / (name + ".npy"), np.full(shape, index / 100., np.float32))
        source = {"arrays_unchanged": True} if hand is None else {
            "hand_protection": {"kind": hand[0], "level": hand[1],
                                "difficulty": ("easy", "medium", "hard")[hand[1]]}}
        _write(directory / "source.json", source)
        source["metadata_sha256"] = fields.sha256(directory / "source.json")
        record = dict(scene_id=f"scene{index}", family=family, path=f"scenes/{index}",
                      shape=[3, 3, 3], origin=[0., 0., 0.], dx=.1,
                      start=[0., 0., .8], goal=[2., 0., .75], reset_xy_scale=[1., 1.], reset_yaw=0.,
                      sampling_weight=1., fields=fields._field_records(directory), source=source,
                      sampling_group=family, task_kind="cat" if hand is None else "room",
                      reset_mode="cat" if hand is None else "room",
                      crossed_mode="x_plane" if hand is None else "goal_radius",
                      episode_length=1000 if hand is None else 4000)
        if original is not None:
            record["original_config_path"] = original
        else:
            _write(directory / "scene.json", {"id": index})
            record["scene_sha256"] = fields.sha256(directory / "scene.json")
        scenes.append(record)
    field_path = field_root / "manifest.json"
    source = _write(field_path, dict(schema=fields.EXPANDED_SCHEMA, scenes=scenes, scene_count=len(scenes),
                                    released_config_sha256=fields.RELEASED_CONFIG_SHA256,
                                    dataset_revision=fields.DATASET_REVISION,
                                    original_count=37, byte_verified_original_count=37,
                                    reconstructed_original_count=0, hand_protection_curriculum=True,
                                    sampling_group_masses=fields.DEFAULT_SAMPLING_GROUP_MASSES), hashed=True)
    arrays = _arrays(len(scenes))
    collision = dict(schema="cat-body-collision-bank-v1", scene_count=len(scenes),
                     field_manifest_sha256=fields.sha256(field_path),
                     field_manifest_content_sha256=source["manifest_sha256"],
                     proxy_sha256="a" * 64, arrays={}, scenes=[], max_candidates=2, static_candidate_count=2)
    collision_root.mkdir()
    for name in ARRAY_NAMES:
        path = collision_root / (name + ".npy")
        np.save(path, arrays[name], allow_pickle=False)
        collision["arrays"][name] = dict(file=path.name, sha256=fields.sha256(path),
                                         shape=list(arrays[name].shape), dtype=str(arrays[name].dtype),
                                         bytes=arrays[name].nbytes)
    for index, scene in enumerate(scenes):
        cache = collision_root / f"geometry/{index}.npz"
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, centers=arrays["centers"][index * 3 + 1:(index + 1) * 3 + 1])
        collision["scenes"].append(dict(index=index, scene_id=scene["scene_id"],
                                        geometry_file=str(cache.relative_to(collision_root)),
                                        geometry_sha256=fields.sha256(cache)))
    collision_path = collision_root / "manifest.json"
    _write(collision_path, collision, hashed=True)
    reset_root.mkdir()
    pool = np.arange(len(scenes) * 2 * 36, dtype=np.float32).reshape(len(scenes), 2, 36)
    np.save(reset_root / "qpos.npy", pool, allow_pickle=False)
    reset_path = reset_root / "manifest.json"
    _write(reset_path, dict(schema="cat-body-collision-clear-reset-pool-v1", status="complete",
                           field_manifest_sha256=fields.sha256(field_path),
                           collision_bank_sha256=fields.sha256(collision_path), proxy_sha256="a" * 64,
                           scene_count=len(scenes), file="qpos.npy", sha256=fields.sha256(reset_root / "qpos.npy"),
                           shape=list(pool.shape), dtype=str(pool.dtype),
                           scenes=[dict(index=i, scene_id=scene["scene_id"], complete=True, selected_poses=2)
                                   for i, scene in enumerate(scenes)]))
    return field_path, collision_path, reset_path, arrays, pool


@pytest.mark.parametrize("selected", [[1], [3, 0, 2], [0, 1, 2, 3]])
def test_compaction_preserves_box_geometry_and_every_cell_candidate(selected):
    source = _arrays(4)
    result = compact_collision_arrays(source, selected)
    for new_index, old_index in enumerate(selected):
        old_box, new_box = source["scene_box_offsets"][old_index], result["scene_box_offsets"][new_index]
        for name in ("centers", "half_sizes", "rotations"):
            np.testing.assert_array_equal(result[name][new_box:new_box + 3], source[name][old_box:old_box + 3])
            np.testing.assert_array_equal(result[name][0], source[name][0])
        for cell in range(3):
            old_cell, new_cell = old_index * 3 + cell, new_index * 3 + cell
            count = source["cell_counts"][old_cell]
            old_start, new_start = source["cell_starts"][old_cell], result["cell_starts"][new_cell]
            assert result["cell_counts"][new_cell] == count
            np.testing.assert_array_equal(result["candidate_ids"][new_start:new_start + count] - new_box,
                                          source["candidate_ids"][old_start:old_start + count] - old_box)


def test_compaction_rejects_cross_scene_candidates_and_noncontiguous_csr():
    source = _arrays(2)
    source["candidate_ids"][1] = source["scene_box_offsets"][1]
    with pytest.raises(ValueError, match="another scene"):
        compact_collision_arrays(source, [0])
    source = _arrays(2)
    source["cell_starts"][2] += 1
    with pytest.raises(ValueError, match="contiguous"):
        compact_collision_arrays(source, [0])


def test_complete_builder_preserves_selected_fields_resets_and_provenance(tmp_path):
    field_path, collision_path, reset_path, source_arrays, source_pool = _fixture_banks(tmp_path / "source")
    output = tmp_path / "specialist"
    result = build_hand_specialist_bank(field_path, collision_path, reset_path, output)
    assert result["scene_count"] == 24
    assert result["kind_counts"] == {"hand_table_aisle": 12, "hand_shelf_passage": 12}
    actual = fields.load_generalist_manifest(result["field_manifest"])
    assert actual["original_count"] == 0
    assert actual["specialist"]["source_scene_indices"] == list(range(37, 61))
    source = fields.load_generalist_manifest(field_path)
    assert actual["scenes"] == source["scenes"][37:]
    for scene in actual["scenes"]:
        for name in fields.FIELD_NAMES:
            assert (output / "fields" / scene["path"] / (name + ".npy")).read_bytes() == (
                field_path.parent / scene["path"] / (name + ".npy")).read_bytes()
    compact, metadata = load_body_collision_bank(result["collision_bank"], expected_field_manifest=result["field_manifest"])
    for name, value in compact_collision_arrays(source_arrays, list(range(37, 61))).items():
        np.testing.assert_array_equal(compact[name], value)
    np.testing.assert_array_equal(np.load(output / "resets/qpos.npy"), source_pool[37:])
    assert metadata["specialist"]["geometry_regenerated"] is False
    with pytest.raises(FileExistsError):
        build_hand_specialist_bank(field_path, collision_path, reset_path, output)


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_specialist_curriculum_keeps_exact_half_mass_per_kind(tmp_path, stage):
    field_path, collision_path, reset_path, *_ = _fixture_banks(tmp_path / "source")
    result = build_hand_specialist_bank(field_path, collision_path, reset_path, tmp_path / "specialist")
    manifest = fields.load_generalist_manifest(result["field_manifest"])
    levels = curriculum_levels(manifest, enabled=True)
    groups, masses = fields.sampling_groups(manifest)
    probabilities = np.asarray(jax.nn.softmax(hand_scene_logits(
        jp.linspace(.001, 1., len(levels)), jp.asarray(groups), jp.asarray(masses), jp.asarray(levels), jp.int32(stage))))
    np.testing.assert_allclose([probabilities[groups == 2].sum(), probabilities[groups == 3].sum()], [.5, .5], atol=1e-7)
    assert np.all(probabilities[levels > stage] == 0.)
    assert np.all(probabilities[levels <= stage] > 0.)


def test_forged_specialist_cannot_drop_change_or_reorder_source_scenes(tmp_path):
    field_path, collision_path, reset_path, *_ = _fixture_banks(tmp_path / "source")
    result = build_hand_specialist_bank(field_path, collision_path, reset_path, tmp_path / "specialist")
    path = Path(result["field_manifest"])
    manifest = json.loads(path.read_text())
    manifest["scenes"][0]["goal"][0] += .1
    _write(path, manifest, hashed=True)
    with pytest.raises(ValueError, match="immutable source selection"):
        fields.load_generalist_manifest(path, verify_files=False)
    manifest["specialist"]["source_scene_indices"].reverse()
    with pytest.raises(ValueError, match="mapping"):
        validate_specialist_manifest(manifest)


def test_builder_rejects_mismatched_reset_certification_before_output(tmp_path):
    field_path, collision_path, reset_path, *_ = _fixture_banks(tmp_path / "source")
    reset = json.loads(reset_path.read_text())
    reset["collision_bank_sha256"] = "f" * 64
    _write(reset_path, reset)
    output = tmp_path / "specialist"
    with pytest.raises(ValueError, match="certification"):
        build_hand_specialist_bank(field_path, collision_path, reset_path, output)
    assert not output.exists()
