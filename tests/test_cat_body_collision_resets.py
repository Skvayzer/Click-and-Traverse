"""Fallback sampling preserves CAT reset law and never publishes failed scenes."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cat_ppo.envs.g1.constants import DEFAULT_QPOS


PATH = Path(__file__).resolve().parents[1] / "scripts/build_body_collision_resets.py"
SPEC = importlib.util.spec_from_file_location("build_body_collision_resets", PATH)
resets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resets)


def _scenes(count=2):
    return [{"scene_id": f"scene-{i}", "reset_mode": "cat", "family": "procedural_cat"}
            for i in range(count)]


def test_native_reset_bounds_and_joint_zero_values_are_preserved():
    draws = np.random.default_rng(13).random((100, 32), dtype=np.float32)
    lower, upper = np.full(29, -2.), np.full(29, 2.)
    lower[3], upper[3] = .2, .35
    poses = resets.sample_native_reset_poses(draws, DEFAULT_QPOS, lower, upper, _scenes()[0])
    assert poses.dtype == np.float32 and poses.shape == (100, 36)
    assert np.all(np.abs(poses[:, :2]) <= 1.)
    np.testing.assert_array_equal(poses[:, 2], np.full(100, np.float32(.8)))
    np.testing.assert_allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1., atol=1e-7)
    yaw = 2. * np.arctan2(poses[:, 6], poses[:, 3])
    assert np.all(np.abs(yaw) <= np.pi / 2)
    assert np.all(poses[:, 10] >= .2) and np.all(poses[:, 10] <= .35)
    np.testing.assert_array_equal(poses[:, 7:][:, DEFAULT_QPOS[7:] == 0.], 0.)


def test_room_transform_only_changes_xy_and_heading_from_same_native_draws():
    draws = np.full((3, 32), .5, dtype=np.float32)
    draws[:, :2] = [[0., .5], [.5, .5], [.75, .25]]
    scene = {"scene_id": "room", "reset_mode": "room", "family": "furniture",
             "start": [4., 7., .8], "reset_xy_scale": [.08, .08], "reset_yaw": np.pi / 2}
    poses = resets.sample_native_reset_poses(draws, DEFAULT_QPOS, np.full(29, -3.), np.full(29, 3.), scene)
    np.testing.assert_allclose(poses[:, :2], [[3.92, 7.], [4., 7.], [4.04, 6.96]], atol=4e-7)
    np.testing.assert_allclose(poses[:, 3:7], np.tile([2 ** -.5, 0., 0., 2 ** -.5], (3, 1)), atol=1e-7)
    np.testing.assert_array_equal(poses[:, 7:], np.broadcast_to(DEFAULT_QPOS[7:], (3, 29)))


def test_rejection_pool_filters_every_scene_and_padding_never_counts():
    calls = []

    def collision(scene_ids, poses):
        calls.append((scene_ids.copy(), poses.copy()))
        assert len(scene_ids) == 7
        return np.stack([poses[:, 0] < 0., poses[:, 1] < -.75], axis=-1)

    kwargs = dict(poses_per_scene=5, seed=123, batch_size=7,
                  candidates_per_scene=3, max_attempts_per_scene=100)
    pool, records, failed = resets.collect_reset_pool(
        _scenes(3), DEFAULT_QPOS, np.full(29, -3.), np.full(29, 3.), collision, **kwargs)
    assert not failed and pool.shape == (3, 5, 36)
    assert np.isfinite(pool).all() and np.all(pool[:, :, 0] >= 0.) and np.all(pool[:, :, 1] >= -.75)
    assert all(item["selected_poses"] == 5 and item["complete"] for item in records)
    assert all(item["attempts"] % 3 == 0 for item in records)
    assert sum(item["attempts"] for item in records) < len(calls) * 7
    again, again_records, again_failed = resets.collect_reset_pool(
        _scenes(3), DEFAULT_QPOS, np.full(29, -3.), np.full(29, 3.), collision, **kwargs)
    np.testing.assert_array_equal(pool, again)
    assert records == again_records and failed == again_failed


def test_failed_scene_is_explicit_never_dropped_or_replaced_by_other_scene():
    def collision(scene_ids, poses):
        return scene_ids == 1

    pool, records, failed = resets.collect_reset_pool(
        _scenes(), DEFAULT_QPOS, np.full(29, -3.), np.full(29, 3.), collision,
        poses_per_scene=3, batch_size=8, candidates_per_scene=4, max_attempts_per_scene=9)
    assert failed == [1]
    assert pool.shape == (2, 3, 36)
    assert np.isfinite(pool[0]).all() and np.isnan(pool[1]).all()
    assert records[1]["attempts"] == 9 and records[1]["selected_poses"] == 0
    assert records[1]["rejection_fraction"] == 1.
    assert records[1]["collision_counts_by_shape"] == [9]


def test_invalid_checker_shape_and_invalid_draws_are_rejected():
    with pytest.raises(ValueError, match="batch shape"):
        resets.collect_reset_pool(_scenes(), DEFAULT_QPOS, np.full(29, -3.), np.full(29, 3.),
                                  lambda scene, qpos: np.zeros(3), poses_per_scene=2, batch_size=4)
    with pytest.raises(ValueError, match="U\\[0,1"):
        resets.sample_native_reset_poses(np.ones((1, 32)), DEFAULT_QPOS,
                                        np.full(29, -3.), np.full(29, 3.), _scenes()[0])
