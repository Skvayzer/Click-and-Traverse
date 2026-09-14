import copy
import json

import numpy as np
import pytest

from cat_ppo.furniture.legacy_scenes import (generate_legacy_scene, merge_occupied_voxels,
    TYPICAL_SCENES, _upstream, GUIDANCE_METHOD)
from cat_ppo.furniture.scenes import write_scene_bundle, load_scene


def test_cuboid_merge_is_exact_nonoverlapping_for_irregular_union():
    rng = np.random.default_rng(4)
    occupancy = rng.random((9, 7, 6)) > .6
    occupancy[1:7, 2:5, 1:5] = True
    restored = np.zeros_like(occupancy, dtype=np.int32)
    for begin, end in merge_occupied_voxels(occupancy):
        restored[tuple(slice(a, b) for a, b in zip(begin, end))] += 1
    np.testing.assert_array_equal(restored, occupancy.astype(np.int32))


@pytest.mark.parametrize('scene_type', ['forward', 'hurdle1', 'crouch1', 'side0', 'side-hurdle-crouch1'])
def test_typical_boxes_reconstruct_original_voxels_and_translation(scene_type):
    scene = generate_legacy_scene(scene_type=scene_type)
    original = scene['legacy_cat']
    grid = _upstream('grid_config').load_grid_config()
    axes = [origin + (np.arange(n) + .5) * grid['voxel'] for origin, n in zip(grid['origin_w'], grid['shape'])]
    source = _upstream('typical_obstacle').build_obstacles(scene_type, np.meshgrid(*axes, indexing='ij'))
    restored = np.zeros_like(source, dtype=np.int32)
    for box in scene['boxes']:
        if box['category'] != 'legacy_obstacle':
            continue
        begin, end = box['source_voxel_begin'], box['source_voxel_end']
        restored[tuple(slice(a, b) for a, b in zip(begin, end))] += 1
        physical_low = np.asarray(box['center']) - box['half_size']
        expected = np.asarray(original['source_edge_origin']) + np.asarray(begin) * original['source_voxel_size_m'] + original['translation']
        np.testing.assert_allclose(physical_low, expected, atol=1e-12)
    np.testing.assert_array_equal(restored, source.astype(np.int32))
    assert not scene['feasibility']['full_body_validated']
    assert scene['generator']['split_holdout']['typical_template_geometry_shared_across_splits']


def test_random_generator_is_deterministic_and_split_seeds_are_disjoint():
    train = generate_legacy_scene(kind='random', seed=3)
    again = generate_legacy_scene(kind='random', seed=3)
    validation = generate_legacy_scene(kind='random', seed=3, split='validation')
    assert train == again
    assert train['geometry_hash'] != validation['geometry_hash']
    assert train['legacy_cat']['source_seed'] != validation['legacy_cat']['source_seed']
    assert train['legacy_cat']['occupied_voxels'] > 0


def test_original_three_dimensional_guidance_is_used_and_verified(tmp_path):
    scene = generate_legacy_scene(scene_type='crouch1')
    bundle = write_scene_bundle(scene, tmp_path / 'crouch', voxel_size=.1)
    loaded = load_scene(bundle)
    gf = np.load(bundle / 'gf.npy')
    sdf = np.load(bundle / 'sdf.npy')
    assert loaded['field_provenance']['guidance'] == GUIDANCE_METHOD
    assert not loaded['field_provenance']['original_fields_bit_identical']
    assert np.isfinite(gf).all()
    assert np.max(np.abs(gf[..., 2][sdf > 0])) > .2
    grid = loaded['grid']
    point = np.asarray([1.4, 1.2, 1.15])  # before overhead bar, above goal's nominal height
    index = np.round((point - grid['sample_origin']) / grid['voxel_size']).astype(int)
    assert gf[tuple(index)][2] < 0  # retained crouching/downward guidance
    raw = json.loads((bundle / 'scene.json').read_text())
    raw['legacy_cat']['goal_height_m'] += .1
    (bundle / 'scene.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='guidance'):
        load_scene(bundle)


def test_all_original_typical_dispatch_entries_are_supported():
    for scene_type in TYPICAL_SCENES:
        scene = generate_legacy_scene(scene_type=scene_type)
        assert scene['legacy_cat']['scene_type'] == scene_type


def test_source_resolution_bundle_preserves_translated_occupancy_sample_centers(tmp_path):
    scene = generate_legacy_scene(scene_type='hurdle1')
    legacy = scene['legacy_cat']
    bundle = write_scene_bundle(scene, tmp_path / 'original-resolution', voxel_size=.04)
    loaded = load_scene(bundle)
    sdf = np.load(bundle / 'sdf.npy')
    source_grid = _upstream('grid_config').load_grid_config()
    axes = [origin + (np.arange(n) + .5) * source_grid['voxel']
            for origin, n in zip(source_grid['origin_w'], source_grid['shape'])]
    occupancy = _upstream('typical_obstacle').build_obstacles('hurdle1', np.meshgrid(*axes, indexing='ij'))
    source_first = np.asarray(legacy['source_edge_origin']) + legacy['translation'] + .5 * .04
    index_float = (source_first - loaded['grid']['sample_origin']) / .04
    np.testing.assert_allclose(index_float, np.round(index_float), atol=1e-10)
    begin = np.round(index_float).astype(int)
    subset = sdf[tuple(slice(b, b+n) for b, n in zip(begin, occupancy.shape))]
    np.testing.assert_array_equal(subset < 0, occupancy)
