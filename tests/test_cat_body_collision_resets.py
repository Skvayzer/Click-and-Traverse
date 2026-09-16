"""Fallback sampling preserves CAT reset law and never publishes failed scenes."""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil

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


def test_suffix_uses_global_checker_ids_and_exact_full_bank_random_streams():
    calls = []

    def collision(ids, poses):
        calls.append(ids.copy())
        # Deliberately scene-dependent: passing local suffix IDs would change
        # both rejection decisions and the poses chosen for the new scenes.
        return poses[:, 0] < ids * .05 - .5

    kwargs = dict(poses_per_scene=5, seed=97, batch_size=8,
                  candidates_per_scene=4, max_attempts_per_scene=64)
    full, records, failed = resets.collect_reset_pool(
        _scenes(5), DEFAULT_QPOS, np.full(29, -3.), np.full(29, 3.), collision, **kwargs)
    calls.clear()
    suffix, suffix_records, suffix_failed = resets.collect_reset_pool(
        _scenes(5)[3:], DEFAULT_QPOS, np.full(29, -3.), np.full(29, 3.), collision,
        scene_index_offset=3, **kwargs)
    assert not failed and not suffix_failed
    assert all(np.all((ids >= 3) & (ids <= 4)) for ids in calls)
    np.testing.assert_array_equal(suffix, full[3:])
    assert suffix_records == records[3:]
    _, rejected, failed = resets.collect_reset_pool(
        _scenes(5)[3:], DEFAULT_QPOS, np.full(29, -3.), np.full(29, 3.),
        lambda ids, qpos: ids == 4, scene_index_offset=3, **kwargs)
    assert failed == [4] and rejected[1]["index"] == 4


def _publish_json(path, metadata, *, content_hash=False):
    from cat_ppo.furniture.generalist_fields import _json_hash
    value = dict(metadata)
    if content_hash:
        value.pop("manifest_sha256", None)
        value["manifest_sha256"] = _json_hash(value)
    path.write_text(json.dumps(value))
    return value


@pytest.fixture
def append_fixture(tmp_path):
    """Small real manifests exercising the normal field/collision validators."""
    from cat_ppo.furniture import body_collision_bank as bank
    from cat_ppo.furniture import generalist_fields as fields
    from cat_ppo.furniture.generalist_config import released_config

    old_root, new_root = tmp_path / "old_fields", tmp_path / "new_fields"
    old_root.mkdir()
    configured = released_config()["env_config"]["pf_config"]["paths"]

    def scene(root, index, original=None):
        folder = root / str(index)
        folder.mkdir()
        for name in fields.FIELD_NAMES:
            np.save(folder / f"{name}.npy", np.ones((3, 3, 3) + (() if name == "sdf" else (3,)), np.float32))
        source = {"kind": "fixture"}
        if original:
            source["arrays_unchanged"] = not original.endswith("D8G2L3O2S13")
            if not source["arrays_unchanged"]:
                source["kind"] = "reconstructed-missing-original"
        _publish_json(folder / "source.json", source)
        source["metadata_sha256"] = resets.sha256(folder / "source.json")
        record = dict(scene_id=f"scene-{index}", family="original_cat" if original else "furniture",
                      path=str(index), shape=[3, 3, 3], origin=[0, 0, 0], dx=.04,
                      start=[0, 0, .8], goal=[2, 0, .75], reset_xy_scale=[1., 1.], reset_yaw=0.,
                      sampling_weight=1., fields=fields._field_records(folder), source=source,
                      task_kind="cat" if original else "room", episode_length=1000 if original else 4000,
                      reset_mode="cat" if original else "room", crossed_mode="x_plane" if original else "goal_radius",
                      sampling_group="original_cat" if original else "furniture")
        if original:
            record["original_config_path"] = original
        else:
            _publish_json(folder / "scene.json", {"boxes": []})
            record["scene_sha256"] = resets.sha256(folder / "scene.json")
        return record

    old_scenes = [scene(old_root, index, original) for index, original in enumerate(configured)]
    old_meta = dict(schema=fields.EXPANDED_SCHEMA, scene_count=37, scenes=old_scenes,
                    original_count=37, byte_verified_original_count=36, reconstructed_original_count=1,
                    released_config_sha256=fields.RELEASED_CONFIG_SHA256, dataset_revision=fields.DATASET_REVISION,
                    sampling_group_masses=fields.DEFAULT_SAMPLING_GROUP_MASSES)
    old_field = old_root / "manifest.json"
    _publish_json(old_field, old_meta, content_hash=True)
    shutil.copytree(old_root, new_root)
    new_meta = dict(old_meta, scene_count=38, scenes=old_scenes + [scene(new_root, 37)])
    new_field = new_root / "manifest.json"
    _publish_json(new_field, new_meta, content_hash=True)
    proposal = tmp_path / "proposal.json"
    _publish_json(proposal, {"shape": "fixture-approved"})

    def collision_bank(root, field_path, field_meta, base_root=None):
        root.mkdir()
        count = field_meta["scene_count"]
        arrays = dict(centers=np.zeros((1, 3), np.float32), half_sizes=np.zeros((1, 3), np.float32),
                      rotations=np.eye(3, dtype=np.float32)[None], scene_box_offsets=np.ones(count, np.int32),
                      scene_box_counts=np.zeros(count, np.int32), scene_grid_origins=np.zeros((count, 3), np.float32),
                      scene_grid_shapes=np.ones((count, 3), np.int32), scene_grid_offsets=np.arange(count, dtype=np.int32),
                      cell_starts=np.zeros(count, np.int32), cell_counts=np.zeros(count, np.int32),
                      candidate_ids=np.zeros(1, np.int32), cell_size=np.array(.25, np.float32))
        scene_records = []
        for index, field in enumerate(field_meta["scenes"]):
            path = root / f"geometry-{index}.npz"
            if base_root is not None and (base_root / path.name).exists():
                shutil.copyfile(base_root / path.name, path)
            else:
                np.savez(path, centers=np.empty((0, 3)), half_sizes=np.empty((0, 3)), rotations=np.empty((0, 3, 3)))
            scene_records.append(dict(index=index, scene_id=field["scene_id"], family=field["family"],
                                      boxes=0, geometry_file=path.name, geometry_sha256=resets.sha256(path),
                                      provenance={"kind": "fixture"}))
        meta = dict(schema=bank.SCHEMA, field_manifest_sha256=resets.sha256(field_path),
                    proxy_sha256=resets.sha256(proposal), scene_count=count, scenes=scene_records,
                    max_candidates=0, static_candidate_count=1, arrays={})
        for name, array in arrays.items():
            path = root / f"{name}.npy"
            np.save(path, array)
            meta["arrays"][name] = dict(file=path.name, sha256=resets.sha256(path), shape=list(array.shape), dtype=str(array.dtype))
        manifest = root / "manifest.json"
        _publish_json(manifest, meta, content_hash=True)
        return manifest

    old_collision = collision_bank(tmp_path / "old_collision", old_field, old_meta)
    new_collision = collision_bank(tmp_path / "new_collision", new_field, new_meta, old_collision.parent)
    reset_root = tmp_path / "old_resets"
    reset_root.mkdir()
    pool = np.broadcast_to(DEFAULT_QPOS, (37, 3, 36)).copy().astype(np.float32)
    pool[:, :, 0] = np.arange(37)[:, None]  # Preservation is observable per scene.
    np.save(reset_root / "qpos.npy", pool)
    records = [dict(index=index, scene_id=field["scene_id"], family=field["family"], attempts=4,
                    clear_candidates=4, selected_poses=3, rejection_fraction=0., complete=True)
               for index, field in enumerate(old_scenes)]
    reset_meta = dict(schema=resets.SCHEMA, status="complete", field_manifest=str(old_field),
                      collision_bank=str(old_collision), proposal=str(proposal), file="qpos.npy",
                      sha256=resets.sha256(reset_root / "qpos.npy"), field_manifest_sha256=resets.sha256(old_field),
                      collision_bank_sha256=resets.sha256(old_collision), proxy_sha256=resets.sha256(proposal),
                      robot_xml_sha256="xml-fixture", scene_count=37, poses_per_scene=3, validated_pose_count=111,
                      shape=[37, 3, 36], dtype="float32", shape_names=["fixture-shape"],
                      soft_joint_pos_limit_factor=.95, extra_clearance_margin_m=0., dropped_scenes=0,
                      invented_reset_postures=False, scenes=records, seed=23)
    base_path = reset_root / "manifest.json"
    _publish_json(base_path, reset_meta)
    arrays, metadata = bank.load_body_collision_bank(new_collision)
    kwargs = dict(field_path=new_field, fields=fields.load_generalist_manifest(new_field),
                  collision_path=new_collision, collision_arrays=arrays, collision_meta=metadata,
                  proposal_path=proposal, robot_xml_sha256="xml-fixture", poses_per_scene=3,
                  nq=36, shape_names=["fixture-shape"])
    return base_path, kwargs, pool, records


def test_verified_append_base_preserves_rows_order_statistics_and_proofs(append_fixture):
    path, kwargs, pool, records = append_fixture
    preserved, stats, proof = resets.load_append_base(path, **kwargs)
    np.testing.assert_array_equal(preserved, pool)
    assert stats == records
    assert proof["preserved_scene_count"] == 37 and proof["preserved_pose_count"] == 111
    assert proof["collision_array_prefixes_identical"] and proof["field_scene_prefix_identical"]
    assert not np.shares_memory(preserved, pool)


@pytest.mark.parametrize("change,message", [
    ("proposal", "proposal"), ("robot", "robot XML"), ("incomplete", "complete validated"),
    ("stats", "statistics"), ("pool", "checksum"), ("field_file", "file hashes"),
    ("geometry_file", "geometry cache"), ("prefix", "array prefix"), ("scene_record", "field-scene prefix"),
])
def test_append_fails_closed_on_source_or_prefix_changes(append_fixture, change, message):
    from cat_ppo.furniture import body_collision_bank as bank
    from cat_ppo.furniture import generalist_fields as fields
    path, kwargs, _, _ = append_fixture
    base = json.loads(path.read_text())
    if change == "proposal":
        alternate = path.parent / "other-proposal.json"
        alternate.write_text("{}")
        kwargs["proposal_path"] = alternate
    elif change == "robot":
        kwargs["robot_xml_sha256"] = "different"
    elif change == "incomplete":
        base["status"] = "failed"
        _publish_json(path, base)
    elif change == "stats":
        base["scenes"][2]["complete"] = False
        _publish_json(path, base)
    elif change == "pool":
        (path.parent / base["file"]).write_bytes(b"bad")
    elif change == "field_file":
        np.save(Path(base["field_manifest"]).parent / "0/sdf.npy", np.zeros((3, 3, 3), np.float32))
    elif change == "geometry_file":
        (Path(base["collision_bank"]).parent / "geometry-0.npz").write_bytes(b"bad")
    elif change == "prefix":
        collision = kwargs["collision_path"]
        meta = json.loads(collision.read_text())
        file = collision.parent / meta["arrays"]["scene_grid_origins"]["file"]
        array = np.load(file)
        array[0, 0] = 1.
        np.save(file, array)
        meta["arrays"]["scene_grid_origins"]["sha256"] = resets.sha256(file)
        _publish_json(collision, meta, content_hash=True)
        kwargs["collision_arrays"], kwargs["collision_meta"] = bank.load_body_collision_bank(collision)
    elif change == "scene_record":
        field = kwargs["field_path"]
        meta = json.loads(field.read_text())
        meta["scenes"][0]["sampling_weight"] = 2.
        _publish_json(field, meta, content_hash=True)
        kwargs["fields"] = fields.load_generalist_manifest(field)
        collision = kwargs["collision_path"]
        meta = json.loads(collision.read_text())
        meta["field_manifest_sha256"] = resets.sha256(field)
        _publish_json(collision, meta, content_hash=True)
        kwargs["collision_arrays"], kwargs["collision_meta"] = bank.load_body_collision_bank(collision)
    with pytest.raises(ValueError, match=message):
        resets.load_append_base(path, **kwargs)


@pytest.fixture
def relocated_robot(tmp_path):
    old_asset = tmp_path / "old-source/mesh.stl"
    new_asset = tmp_path / "new-source/renamed.stl"
    old_asset.parent.mkdir()
    new_asset.parent.mkdir()
    old_asset.write_bytes(b"same robot triangle mesh bytes")
    new_asset.write_bytes(old_asset.read_bytes())
    template = ('<mujoco><compiler angle="radian"/><asset><mesh name="hand" file="{file}" scale="1 1 1"/>'
                '</asset><worldbody><body name="wrist" pos="0 0 1"><geom mesh="hand" mass="1"/>'
                '</body></worldbody></mujoco>')
    old_xml = template.format(file=old_asset)
    new_xml = template.format(file=new_asset)
    old_path = tmp_path / "old-assembled.xml"
    old_path.write_text(old_xml)
    digest = lambda xml: hashlib.sha256(xml.encode()).hexdigest()
    return old_path, old_asset, new_asset, old_xml, new_xml, digest


def test_robot_xml_relocation_is_bound_to_old_raw_hash_and_mesh_contents(relocated_robot):
    path, _, _, old, new, digest = relocated_robot
    proof = resets.verify_robot_xml_identity(digest(old), digest(new), new_xml=new, base_robot_xml=path)
    assert proof["verification"] == "content-verified-mesh-path-relocation"
    assert proof["old_raw_sha256"] != proof["new_raw_sha256"]
    assert proof["old_normalized_sha256"] == proof["new_normalized_sha256"]
    assert proof["mesh_file_count"] == 1 and proof["all_other_robot_attributes_unchanged"]
    # Exact matching XML retains the existing path without requiring proof I/O.
    fast = resets.verify_robot_xml_identity(digest(old), digest(old))
    assert fast["verification"] == "raw-XML-identity"


@pytest.mark.parametrize("change,message", [("mesh", "mesh content"), ("body", "non-path"),
    ("mass", "non-path"), ("scale", "non-path"), ("old_raw", "raw SHA256"),
    ("new_raw", "raw SHA256"), ("missing_proof", "supply --base-robot-xml")])
def test_robot_xml_relocation_rejects_geometry_parameters_and_unbound_xml(relocated_robot, change, message):
    path, _, new_asset, old, new, digest = relocated_robot
    expected_old, expected_new = digest(old), digest(new)
    if change == "mesh":
        new_asset.write_bytes(b"different vertices")
    elif change == "body":
        new = new.replace('pos="0 0 1"', 'pos="0 0 2"')
        expected_new = digest(new)
    elif change == "mass":
        new = new.replace('mass="1"', 'mass="2"')
        expected_new = digest(new)
    elif change == "scale":
        new = new.replace('scale="1 1 1"', 'scale="2 1 1"')
        expected_new = digest(new)
    elif change == "old_raw":
        path.write_text(old + "\n")
    elif change == "new_raw":
        expected_new = "0" * 64
    elif change == "missing_proof":
        path = None
    with pytest.raises(ValueError, match=message):
        resets.verify_robot_xml_identity(expected_old, expected_new, new_xml=new, base_robot_xml=path)


def test_append_base_accepts_proven_asset_relocation_and_records_proof(append_fixture, relocated_robot):
    path, kwargs, pool, records = append_fixture
    xml_path, _, _, old, new, digest = relocated_robot
    base = json.loads(path.read_text())
    base["robot_xml_sha256"] = digest(old)
    _publish_json(path, base)
    kwargs.update(robot_xml_sha256=digest(new), robot_xml=new, base_robot_xml=xml_path)
    preserved, stats, proof = resets.load_append_base(path, **kwargs)
    np.testing.assert_array_equal(preserved, pool)
    assert stats == records
    assert proof["robot_xml_identity"]["verification"] == "content-verified-mesh-path-relocation"
