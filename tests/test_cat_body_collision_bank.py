import json

import numpy as np
import pytest

from cat_ppo.furniture.body_collision_bank import (
    _extract_scene, boxes_from_occupancy, build_scene_index, canonical_room_boxes,
    load_body_collision_bank, lookup_collision_candidates, proxy_bounding_radius,
)
from cat_ppo.furniture.generalist_fields import sha256


def _query_arrays(index, *, cell_size=.25):
    jnp = pytest.importorskip("jax.numpy")
    return {key: jnp.asarray(value) for key, value in dict(
        scene_grid_origins=index["origin"][None], scene_grid_shapes=index["shape"][None],
        scene_grid_offsets=np.array([0], np.int32), cell_size=np.float32(cell_size),
        cell_starts=index["starts"] + 1, cell_counts=index["counts"],
        candidate_ids=np.r_[np.int32(0), index["ids"] + 1]).items()}


def test_voxel_merge_preserves_union_and_runtime_half_cell_convention():
    occupancy = np.zeros((6, 5, 4), np.uint8)
    occupancy[1:4, 2:4, :2] = 1
    occupancy[3:5, 1:3, 1:3] = 1
    centers, half, rotation = boxes_from_occupancy(occupancy, [-.5, -1, 0], .04)
    grid = np.stack(np.meshgrid(*[o + np.arange(n) * .04 for o, n in zip(
        [-.5, -1, 0], occupancy.shape)], indexing="ij"), axis=-1)
    rebuilt = np.any(np.all(np.abs(grid[..., None, :] - centers) <= half + 1e-12, axis=-1), axis=-1)
    np.testing.assert_array_equal(rebuilt, occupancy)
    assert np.min(centers[:, 2] - half[:, 2]) == pytest.approx(-.02)
    np.testing.assert_allclose(rotation, np.broadcast_to(np.eye(3), rotation.shape))


def test_canonical_room_yaw_and_counts_are_retained_without_floor():
    scene = {"boxes": [{"center": [2, 3, .8], "half_size": [1, .03, .08],
                         "yaw": np.pi / 3, "category": "table"}]}
    centers, half, rotation = canonical_room_boxes(scene)
    np.testing.assert_array_equal(centers, [[2, 3, .8]])
    np.testing.assert_array_equal(half, [[1, .03, .08]])
    np.testing.assert_allclose(rotation[0] @ [1, 0, 0], [.5, np.sqrt(3) / 2, 0], atol=1e-12)
    scene["boxes"][0]["category"] = "floor"
    with pytest.raises(ValueError, match="floor"):
        canonical_room_boxes(scene)


def test_index_never_truncates_even_when_more_than_64_boxes_overlap():
    index = build_scene_index(np.zeros((101, 3)), np.ones((101, 3)) * .1,
                              np.tile(np.eye(3), (101, 1, 1)), query_radius=.22)
    assert index["max_candidates"] == 101
    assert np.max(index["counts"]) == 101
    ids, mask = lookup_collision_candidates(_query_arrays(index), 0, np.zeros((1, 3)), max_candidates=101)
    assert np.asarray(mask).sum() == 101
    np.testing.assert_array_equal(np.asarray(ids)[0], np.arange(1, 102))


def test_jitted_lookup_covers_random_rotated_box_sphere_intersections_and_boundaries():
    jax = pytest.importorskip("jax")
    rng = np.random.default_rng(11)
    centers = rng.uniform(-2, 2, (35, 3))
    half = rng.uniform(.01, .65, (35, 3))
    yaws = rng.uniform(-np.pi, np.pi, 35)
    rotations = np.array([[[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]] for y in yaws])
    radius = .21254
    index = build_scene_index(centers, half, rotations, query_radius=radius)
    points = np.concatenate([rng.uniform(-3, 3, (2500, 3)), centers,
                             np.round(rng.uniform(-2, 2, (100, 3)) / .25) * .25])
    arrays = _query_arrays(index)
    ids, valid = jax.jit(lambda p: lookup_collision_candidates(
        arrays, 0, p, max_candidates=index["max_candidates"]))(points.astype(np.float32))
    ids, valid = np.asarray(ids), np.asarray(valid)
    local = np.einsum("bji,pbj->pbi", rotations, points[:, None, :] - centers[None])
    intersects = np.linalg.norm(np.maximum(np.abs(local) - half, 0), axis=-1) <= radius
    for row, required in enumerate(intersects):
        assert set(np.flatnonzero(required) + 1).issubset(set(ids[row][valid[row]]))
    far_ids, far_valid = lookup_collision_candidates(arrays, 0, np.array([[1e4, 1e4, 1e4]]),
                                                     max_candidates=index["max_candidates"])
    assert not np.asarray(far_valid).any()
    assert not np.asarray(far_ids).any()


def test_empty_scene_has_one_empty_grid_cell():
    index = build_scene_index(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3, 3)), query_radius=.22)
    assert index["max_candidates"] == 0
    ids, valid = lookup_collision_candidates(_query_arrays(index), 0, np.zeros((1, 3)), max_candidates=1)
    assert np.asarray(ids).tolist() == [[0]] and not np.asarray(valid).any()


def test_proxy_capsule_radius_is_about_actual_segment_midpoint():
    proposal = {"shapes": [{"kind": "capsule", "center": [0, 0, .1],
                             "endpoints": [[0, 0, -.1], [0, 0, .3]], "radius": .05}]}
    assert proxy_bounding_radius(proposal) == pytest.approx(.25)
    proposal["shapes"][0]["center"] = [0, 0, 0]
    with pytest.raises(ValueError, match="midpoint"):
        proxy_bounding_radius(proposal)


def test_extraction_rejects_hash_valid_occupancy_disagreeing_with_training_sdf(tmp_path):
    root = tmp_path / "fields"; directory = root / "scene"; directory.mkdir(parents=True)
    np.save(directory / "obs.npy", np.zeros((3, 3, 3), np.uint8))
    np.save(directory / "sdf.npy", -np.ones((3, 3, 3), np.float32))
    source = {"kind": "generated", "occupancy_sha256": sha256(directory / "obs.npy")}
    (directory / "source.json").write_text(json.dumps(source))
    record = dict(path="scene", scene_id="test", family="procedural_cat", shape=[3, 3, 3],
                  origin=[0, 0, 0], dx=.04, source={"metadata_sha256": sha256(directory / "source.json")},
                  fields={"sdf": {"sha256": sha256(directory / "sdf.npy")}})
    with pytest.raises(ValueError, match="inconsistent"):
        _extract_scene((0, record, str(root), str(tmp_path / "out"), False))


def test_loader_rejects_corrupted_manifest_before_reading_arrays(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "cat-body-collision-bank-v1", "manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="hash/schema"):
        load_body_collision_bank(tmp_path)
