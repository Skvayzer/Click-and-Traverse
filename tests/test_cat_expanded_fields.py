"""Expanded banks retain released anchors, task semantics, and source integrity."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest

from cat_ppo.furniture import generalist_fields as fields
from cat_ppo.furniture.generalist_config import released_config


def _write_manifest(path, manifest):
    manifest = copy.deepcopy(manifest)
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = fields._json_hash(manifest)
    path.write_text(json.dumps(manifest) + "\n")
    return path


def _scene(root, identity, family, index, original_path=None):
    directory = root / family / identity
    directory.mkdir(parents=True)
    shape = [3, 4, 5] if index % 2 else [4, 3, 3]
    for name in fields.FIELD_NAMES:
        array_shape = shape if name == "sdf" else shape + [3]
        np.save(directory / f"{name}.npy", np.full(array_shape, index, dtype=np.float32))
    source = dict(kind="synthetic-test-fixture", fixture_index=index)
    if original_path:
        source["arrays_unchanged"] = True
        if original_path.endswith("D8G2L3O2S13"):
            source.update(arrays_unchanged=False, kind="reconstructed-missing-original")
    (directory / "source.json").write_text(json.dumps(source))
    source["metadata_sha256"] = fields.sha256(directory / "source.json")
    is_cat = family in ("original_cat", "published_cat", "procedural_cat")
    scene = dict(scene_id=identity, family=family, path=str(directory.relative_to(root)),
                 shape=shape, origin=[-.5, -1., 0.], dx=.04,
                 start=[0., 0., .8] if is_cat else [3., 2., .8],
                 goal=[2., 0., .75] if is_cat else [6., 5., .75],
                 reset_xy_scale=[1., 1.] if is_cat else [.08, .08], reset_yaw=0.,
                 fields=fields._field_records(directory), source=source, sampling_weight=1.,
                 task_kind="cat" if is_cat else "room", episode_length=1000 if is_cat else 4000,
                 reset_mode="cat" if is_cat else "room", crossed_mode="x_plane" if is_cat else "goal_radius",
                 sampling_group="original_cat" if family == "published_cat" else family)
    if original_path:
        scene["original_config_path"] = original_path
    if not is_cat:
        (directory / "scene.json").write_text(json.dumps({"synthetic_room": identity}))
        scene["scene_sha256"] = fields.sha256(directory / "scene.json")
    return scene


@pytest.fixture
def expanded_bank(tmp_path):
    configured = released_config()["env_config"]["pf_config"]["paths"]
    scenes = [_scene(tmp_path, Path(path).name, "original_cat", index, path)
              for index, path in enumerate(configured)]
    for family in ("published_cat", "procedural_cat", "furniture", "generic_clutter"):
        scenes.append(_scene(tmp_path, f"extra_{family}", family, len(scenes)))
    manifest = dict(schema=fields.EXPANDED_SCHEMA, scene_count=len(scenes), scenes=scenes,
                    original_count=37, byte_verified_original_count=36, reconstructed_original_count=1,
                    released_config_sha256=fields.RELEASED_CONFIG_SHA256,
                    dataset_revision=fields.DATASET_REVISION,
                    sampling_group_masses=fields.DEFAULT_SAMPLING_GROUP_MASSES)
    return _write_manifest(tmp_path / "manifest.json", manifest)


class _Base:
    def __init__(self, task_type, config, config_overrides):
        self._config = config

    def reset(self, rng):
        return SimpleNamespace(info={})


class _BankEnv(fields.RaggedSceneMixin, _Base):
    pass


def _environment(path, weights=None):
    return _BankEnv(config=SimpleNamespace(
        episode_length=1000, clutter_episode_length=4000,
        pf_config=SimpleNamespace(bank_manifest=str(path), sampling_weights=weights)))


def test_expanded_bank_accepts_extra_cat_fields_and_retains_all_anchors(expanded_bank):
    manifest = fields.load_generalist_manifest(expanded_bank)
    assert manifest["scene_count"] == 41
    assert manifest["original_count"] == 37
    assert manifest["byte_verified_original_count"] == 36
    assert manifest["reconstructed_original_count"] == 1
    config = fields.bank_config(expanded_bank)
    assert len(config["paths"]) == 41
    env = _environment(expanded_bank)
    np.testing.assert_array_equal(env._pf_scene_original, [True] * 39 + [False] * 2)
    np.testing.assert_array_equal(env._pf_reset_is_cat, env._pf_scene_original)
    np.testing.assert_array_equal(env._pf_crossed_is_x_plane, env._pf_scene_original)
    np.testing.assert_array_equal(env._pf_scene_episode_lengths, [1000] * 39 + [4000] * 2)
    assert env.sdf.shape[0] == sum(np.prod(s["shape"]) for s in manifest["scenes"])
    # Additional CAT fields retain CAT's reset distribution, even though their
    # source family is not original_cat. Room placement remains separate.
    qpos = jp.array([.8, -.8, .8, 1., 0., 0., 0., .2])
    for scene_id in (0, 37, 38):
        env._field_pf_id = jp.array(scene_id)
        np.testing.assert_array_equal(env.adjust_reset_pose(qpos), qpos)
        actual = env.sample_field(env.sdf, jp.array([[0., 0., .8]]))
        np.testing.assert_allclose(actual, scene_id, atol=1e-6)
    env._field_pf_id = jp.array(39)
    np.testing.assert_allclose(env.adjust_reset_pose(qpos)[:2], [3.064, 1.936], atol=1e-6)


def test_initial_sampling_keeps_fixed_family_mass_with_unequal_scene_counts(expanded_bank):
    env = _environment(expanded_bank)
    ids = np.asarray(env._pf_sampling_group_ids)
    probabilities = np.asarray(jax.nn.softmax(env._pf_sampling_logits))
    masses = np.bincount(ids, weights=probabilities, minlength=4)
    np.testing.assert_allclose(masses, [.20, .40, .25, .15], atol=1e-7)
    np.testing.assert_array_equal(ids, [0] * 38 + [1, 2, 3])
    # Explicit scene weights affect only their own group's relative shares.
    weights = np.ones(41)
    weights[0] = 3.
    env = _environment(expanded_bank, weights.tolist())
    probabilities = np.asarray(jax.nn.softmax(env._pf_sampling_logits))
    np.testing.assert_allclose(probabilities[0] / probabilities[1], 3., atol=1e-6)
    np.testing.assert_allclose(np.bincount(ids, weights=probabilities), masses, atol=1e-7)
    state = env.reset_with_pf_id(jax.random.key(0), jp.array(38))
    assert set(state.info) == {"pf_id", "pf_success_ema", "pf_episode_ema", "pf_sampling_logits",
                               "pf_sampling_alpha", "pf_sampling_ema_decay",
                               "pf_sampling_group_ids", "pf_sampling_group_masses"}
    np.testing.assert_array_equal(state.info["pf_sampling_group_ids"], ids)
    np.testing.assert_allclose(state.info["pf_sampling_group_masses"], [.20, .40, .25, .15])


def test_absent_groups_renormalize_without_losing_cat_metadata(expanded_bank):
    manifest = fields.load_generalist_manifest(expanded_bank)
    manifest["scenes"] = manifest["scenes"][:39]
    manifest["scene_count"] = 39
    _write_manifest(expanded_bank, manifest)
    env = _environment(expanded_bank)
    np.testing.assert_allclose(env._pf_sampling_group_masses, [1 / 3, 2 / 3, 0., 0.])
    masses = np.bincount(np.asarray(env._pf_sampling_group_ids),
                        weights=np.asarray(jax.nn.softmax(env._pf_sampling_logits)), minlength=4)
    np.testing.assert_allclose(masses, [1 / 3, 2 / 3, 0., 0.], atol=1e-7)


def test_v1_keeps_its_original_sampling_and_info_contract(expanded_bank):
    manifest = fields.load_generalist_manifest(expanded_bank)
    manifest["schema"] = fields.SCHEMA
    manifest.pop("sampling_group_masses")
    manifest["scenes"] = manifest["scenes"][:37] + manifest["scenes"][39:]
    manifest["scene_count"] = 39
    for scene in manifest["scenes"]:
        for key in ("task_kind", "episode_length", "reset_mode", "crossed_mode", "sampling_group"):
            scene.pop(key)
    _write_manifest(expanded_bank, manifest)
    env = _environment(expanded_bank)
    expected = jp.log(jp.ones(39) / 39 + 1e-8)
    np.testing.assert_array_equal(env._pf_sampling_logits, expected)
    np.testing.assert_array_equal(env._pf_scene_original, [True] * 37 + [False] * 2)
    state = env.reset_with_pf_id(jax.random.key(0), jp.array(0))
    assert set(state.info) == {"pf_id", "pf_success_ema", "pf_episode_ema", "pf_sampling_logits",
                               "pf_sampling_alpha", "pf_sampling_ema_decay"}
    assert not hasattr(env, "_pf_sampling_group_ids")


@pytest.mark.parametrize("key,value,message", [
    ("task_kind", "room", "task_kind/sampling_group"),
    ("sampling_group", "furniture", "task_kind/sampling_group"),
    ("reset_mode", "room", "reset_mode/crossed_mode"),
    ("crossed_mode", "goal_radius", "reset_mode/crossed_mode"),
    ("episode_length", 4000, "episode_length"),
    ("reset_xy_scale", [.08, .08], "untransformed reset"),
    ("sampling_weight", -1., "sampling_weight"),
    ("family", "unknown", "scene family"),
    ("source", {}, "source metadata SHA256"),
])
def test_expanded_cat_task_metadata_is_validated(expanded_bank, key, value, message):
    manifest = fields.load_generalist_manifest(expanded_bank)
    manifest["scenes"][38][key] = value
    _write_manifest(expanded_bank, manifest)
    with pytest.raises(ValueError, match=message):
        fields.load_generalist_manifest(expanded_bank, verify_files=False)


@pytest.mark.parametrize("change,message", [
    ("reorder", "scenes/order"), ("remove", "37 distinct"),
    ("provenance", "provenance counts"), ("count", "scene count"),
    ("duplicate", "IDs must be nonempty and unique"),
    ("escape", "path escapes"), ("weightless_group", "positive scene weight"),
    ("missing_group_mass", "all four"),
])
def test_expansion_cannot_remove_anchor_or_integrity_requirements(expanded_bank, change, message):
    manifest = fields.load_generalist_manifest(expanded_bank)
    if change == "reorder":
        manifest["scenes"][0], manifest["scenes"][1] = manifest["scenes"][1], manifest["scenes"][0]
    elif change == "remove":
        manifest["scenes"].pop(0)
        manifest["scene_count"] -= 1
    elif change == "provenance":
        manifest["reconstructed_original_count"] = 0
    elif change == "count":
        manifest["scene_count"] += 1
    elif change == "duplicate":
        manifest["scenes"][-1]["scene_id"] = manifest["scenes"][0]["scene_id"]
    elif change == "escape":
        manifest["scenes"][-1]["path"] = "../outside_bank"
    elif change == "weightless_group":
        manifest["scenes"][38]["sampling_weight"] = 0.
    elif change == "missing_group_mass":
        manifest["sampling_group_masses"].pop("furniture")
    _write_manifest(expanded_bank, manifest)
    with pytest.raises(ValueError, match=message):
        fields.load_generalist_manifest(expanded_bank, verify_files=False)


@pytest.mark.parametrize("family,file,message", [
    ("published_cat", "source.json", "source fingerprint"),
    ("procedural_cat", "source.json", "source fingerprint"),
    ("furniture", "source.json", "source fingerprint"),
    ("generic_clutter", "scene.json", "geometry/source fingerprint"),
    ("procedural_cat", "sdf.npy", "file hashes/shapes"),
])
def test_expanded_scene_source_geometry_and_field_corruption_is_detected(expanded_bank, family, file, message):
    manifest = fields.load_generalist_manifest(expanded_bank)
    scene = next(s for s in manifest["scenes"] if s["family"] == family)
    path = expanded_bank.parent / scene["path"] / file
    if file.endswith(".npy"):
        np.save(path, np.load(path) + 1.)
    else:
        path.write_text('{"tampered": true}')
    with pytest.raises(ValueError, match=message):
        fields.load_generalist_manifest(expanded_bank)


def test_runtime_override_cannot_zero_a_whole_sampling_group(expanded_bank):
    weights = [1.] * 41
    weights[38] = 0.
    with pytest.raises(ValueError, match="positive scene weight"):
        _environment(expanded_bank, weights)


@pytest.mark.parametrize("key_factory", [jax.random.key, jax.random.PRNGKey])
def test_inverse_cdf_sampling_is_deterministic_and_has_requested_distribution(key_factory):
    keys = jax.random.split(key_factory(20260916), 20_000)
    logits = jp.log(jp.array([.10, .30, 0., .60]))
    sampled = jax.jit(fields.sample_scene_ids)(keys, logits)
    assert sampled.dtype == jp.int32
    np.testing.assert_array_equal(sampled, fields.sample_scene_ids(keys, logits))
    np.testing.assert_array_equal(sampled[:16], jax.vmap(lambda key: fields.sample_scene_ids(key, logits))(keys[:16]))
    frequencies = np.bincount(np.asarray(sampled), minlength=4) / len(keys)
    np.testing.assert_allclose(frequencies, [.10, .30, 0., .60], atol=.012)
    assert frequencies[2] == 0.


def test_large_inverse_cdf_sampler_uses_valid_scene_indices():
    # Match the order of magnitude of the intended bank and environment batch.
    keys = jax.random.split(jax.random.key(7), 8192)
    logits = jp.linspace(-10., 0., 2338)
    sampled = jax.jit(fields.sample_scene_ids)(keys, logits)
    assert sampled.shape == (8192,)
    assert np.isfinite(sampled).all()
    assert int(sampled.min()) >= 0 and int(sampled.max()) < len(logits)
    singleton = fields.sample_scene_ids(keys, jp.array([0.]))
    np.testing.assert_array_equal(singleton, np.zeros(8192, dtype=np.int32))
