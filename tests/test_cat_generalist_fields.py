"""Behavioral compatibility and provenance checks for mixed-size CAT fields."""
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest

from cat_ppo.envs.g1.env_cat import G1CatEnv
from cat_ppo.furniture import generalist_fields as fields


@pytest.mark.parametrize("shape,channels", [((7, 9, 5), 1), ((4, 5, 8), 3)])
def test_ragged_sampler_is_exact_cat_at_interior_and_clipped_boundaries(shape, channels):
    rng = np.random.default_rng(7)
    array = rng.normal(size=(*shape, channels)).astype(np.float32)
    origin = np.asarray([-.5, -1., 0.], dtype=np.float32)
    dx = .04
    positions = rng.uniform(-2, 3, size=(500, 3)).astype(np.float32)
    positions[:100] = origin + rng.uniform(0, np.asarray(shape) - 1, size=(100, 3)) * dx
    positions[100:108] = origin + np.asarray([[0, 0, 0], [0, 0, 3], [0, 3, 0], [3, 0, 0],
                                             [3, 3, 3], [-1, -1, -1], [50, 50, 50], [1, 2, 3]]) * dx
    original = object.__new__(G1CatEnv)
    original.pf_origin = jp.array(origin)
    original.dx = dx
    original.Nx, original.Ny, original.Nz = shape
    prefix = np.full((19, channels), -999, dtype=np.float32)
    flattened = jp.array(np.concatenate([prefix, array.reshape(-1, channels), prefix]))
    reference = jax.jit(lambda pos: original.sample_field(jp.array(array), pos))(jp.array(positions))
    actual = jax.jit(lambda pos: fields.sample_ragged_field(flattened, pos,
        origin=jp.array(origin), dx=jp.array(dx), shape=jp.array(shape), offset=jp.array(19)))(jp.array(positions))
    np.testing.assert_array_equal(actual, reference)


def test_original_download_preserves_source_bytes_and_rejects_corruption(tmp_path, monkeypatch):
    source = tmp_path / "fixture"
    source.mkdir()
    metadata = {}
    for name in fields.FIELD_NAMES:
        shape = (3, 4, 5) if name == "sdf" else (3, 4, 5, 3)
        path = source / f"{name}.npy"
        np.save(path, np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
        metadata[path.name] = {"path": f"assets_v0/RandObs/test/{path.name}", "lfs": {"oid": fields.sha256(path)}}
    monkeypatch.setattr(fields, "_source_metadata", lambda name: metadata)
    def fake_download(relative, destination, expected_sha256):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source / Path(relative).name).read_bytes())
    monkeypatch.setattr(fields, "_download_file", fake_download)
    output = tmp_path / "bank"
    record = fields.fetch_original_scene("data/assets/RandObs/test", output,
        {"origin": [-.5, -1., 0.], "dx": .04})
    for name in fields.FIELD_NAMES:
        assert (output / record["path"] / f"{name}.npy").read_bytes() == (source / f"{name}.npy").read_bytes()
    assert record["source"]["arrays_unchanged"]
    (output / record["path"] / "gf.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Missing/corrupt"):
        fields.fetch_original_scene("data/assets/RandObs/test", output,
            {"origin": [-.5, -1., 0.], "dx": .04}, download=False)


def test_clutter_uses_all_three_original_cat_field_functions(tmp_path, monkeypatch):
    from cat_ppo.furniture.scenes import generate_scene
    module = fields._upstream("pf_modular")
    calls = []
    for name in ("make_sdf", "grad3", "make_guidance_field_progressive"):
        original = getattr(module, name)
        def wrapped(*args, _name=name, _fn=original, **kwargs):
            calls.append(_name)
            return _fn(*args, **kwargs)
        monkeypatch.setattr(module, name, wrapped)
    scene = generate_scene(seed=20260915, difficulty="dense")
    record = fields.make_clutter_fields(scene, tmp_path, dx=.12)
    assert calls == ["make_sdf", "grad3", "make_guidance_field_progressive"]
    assert scene["counts"]["tables"] == 9 and scene["counts"]["chairs"] == 36
    assert record["source"]["free_voxel_start_goal_connected"]
    assert record["source"]["root_route_geometry_validated"]
    assert record["source"]["route_used_for_guidance"] is False
    assert record["source"]["dynamic_feasibility_validated"] is False
    gf = np.load(tmp_path / "gf.npy")
    # The old route-lookahead supplied zero vertical guidance everywhere. CAT's
    # three-dimensional field includes vertical navigation around furniture.
    assert np.count_nonzero(np.abs(gf[..., 2]) > .01) > 100
    assert np.isfinite(gf).all()


def test_released_configuration_has_exact_original_scene_list():
    path = Path(__file__).resolve().parents[1] / "configs/cat_generalist_released.json"
    assert fields.sha256(path) == fields.RELEASED_CONFIG_SHA256
    config = json.loads(path.read_text())
    scenes = config["env_config"]["pf_config"]["paths"]
    assert len(scenes) == len(set(scenes)) == 37
    assert scenes[0].endswith("D8G0L1O0S3")
    assert scenes[-1].endswith("D8G2L3O2S81")


def test_missing_original_reconstruction_is_explicit_and_labelled(tmp_path, monkeypatch):
    import urllib.error
    def missing(name):
        raise urllib.error.HTTPError("fixture", 404, "not found", {}, None)
    monkeypatch.setattr(fields, "_source_metadata", missing)
    scene = "data/assets/RandObs/D8G2L3O2S13"
    pf = {"origin": [-.5, -1., 0.], "dx": .04}
    with pytest.raises(urllib.error.HTTPError):
        fields.fetch_original_scene(scene, tmp_path, pf)
    record = fields.fetch_original_scene(scene, tmp_path, pf, reconstruct_missing=True)
    assert record["source"]["arrays_unchanged"] is False
    assert record["source"]["released_original_byte_identity_verified"] is False
    assert record["source"]["kind"] == "reconstructed-missing-original"
    assert record["shape"] == [75, 50, 38]
    with pytest.raises(ValueError, match="reconstructed source"):
        fields.fetch_original_scene(scene, tmp_path, pf, download=False)


def test_new_scene_reset_extension_keeps_original_pose_exact():
    mixin = object.__new__(fields.RaggedSceneMixin)
    mixin._pf_scene_starts = jp.array([[0, 0, .8], [.6, .55, .8]])
    mixin._pf_reset_xy_scale = jp.array([[1., 1.], [.08, .08]])
    mixin._pf_scene_yaws = jp.array([0., 0.])
    mixin._pf_scene_original = jp.array([True, False])
    qpos = jp.array([.8, -.8, .8, 1., 0., 0., 0., .2])
    mixin._field_pf_id = jp.array(0)
    np.testing.assert_array_equal(mixin.adjust_reset_pose(qpos), qpos)
    mixin._field_pf_id = jp.array(1)
    moved = mixin.adjust_reset_pose(qpos)
    np.testing.assert_allclose(moved[:2], [.664, .486], atol=1e-7)
    np.testing.assert_array_equal(moved[2:], qpos[2:])
