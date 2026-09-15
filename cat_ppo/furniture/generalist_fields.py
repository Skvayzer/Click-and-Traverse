"""Released CAT fields, additive FMM clutter, and a lossless ragged scene bank.

Available released arrays are downloaded byte-for-byte at an immutable dataset
revision. One configured scene is absent from that release and requires an
explicit, prominently recorded reconstruction option. New rooms use the original occupancy -> SDF -> gradient -> progressive
3-D fast-marching pipeline. Ragged storage avoids expanding every small original
scene to room size; interpolation deliberately retains CAT's corner ordering.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import urllib.error
import urllib.request

import numpy as np

from cat_ppo.furniture.legacy_scenes import SOURCE_ROOT, UPSTREAM_COMMIT, _upstream

DATASET_REPO = "Axian12138/Click-and-Traverse"
DATASET_REVISION = "db8c202fa724bc4d3dba2cb9fead1267f15fd811"
RELEASED_CONFIG_SHA256 = "0e38cac262a3c1c95bacdf859193340056fef53b8d2f8414293789cee5d45d2d"
SCHEMA = "cat-generalist-field-bank-v1"
FIELD_NAMES = ("sdf", "bf", "gf")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _source_metadata(scene_name):
    url = (f"https://huggingface.co/api/datasets/{DATASET_REPO}/tree/"
           f"{DATASET_REVISION}/assets_v0/RandObs/{scene_name}")
    with urllib.request.urlopen(url, timeout=60) as response:
        records = json.load(response)
    return {Path(record["path"]).name: record for record in records}


def _download_file(relative, destination, expected_sha256):
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != expected_sha256:
            raise ValueError(f"Existing original field differs from pinned dataset: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{relative}"
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, stream)
            stream.close()
            if sha256(temporary) != expected_sha256:
                raise ValueError(f"Downloaded original field hash mismatch: {relative}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def _field_records(directory):
    records = {}
    for name in FIELD_NAMES:
        file = Path(directory) / f"{name}.npy"
        array = np.load(file, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Field must be finite float32: {file}")
        records[name] = dict(file=file.name, sha256=sha256(file), shape=list(array.shape),
                             dtype=str(array.dtype), size_bytes=file.stat().st_size)
    shape = records["sdf"]["shape"]
    if len(shape) != 3 or min(shape) < 3:
        raise ValueError("SDF must have three spatial dimensions of at least three samples")
    if any(records[name]["shape"] != shape + [3] for name in ("bf", "gf")):
        raise ValueError("Boundary and guidance fields must match SDF spatial dimensions")
    return records


def reconstruct_missing_original(scene_path, output, pf_config):
    """Explicitly reconstruct the one configured scene absent from the release.

    Upstream's current generator does not reproduce even known released fields
    exactly. This is consequently a same-parameter reconstruction, never a claim
    that the unavailable original bytes have been recovered.
    """
    scene_name = Path(scene_path).name
    if scene_name != "D8G2L3O2S13":
        raise ValueError("Reconstruction is limited to the verified missing release scene")
    directory = Path(output) / "original" / scene_name
    directory.mkdir(parents=True, exist_ok=True)
    generator, module = _upstream("random_obstacle"), _upstream("pf_modular")
    config = generator.Cfg(difficulty=.8, seed=13, n_rect_L=3, n_rect_R=3, n_rect_F=2, n_rect_C=2)
    occupancy, *axes = generator.generate_and_save(config, save=False)
    sdf = module.make_sdf(occupancy, config.voxel)
    bf = module.grad3(sdf, config.voxel)
    _, gf = module.make_guidance_field_progressive(module.PFConfig(), np.meshgrid(*axes, indexing="ij"),
                                                  occupancy, config.goal_w, bf, sdf)
    for name, array in (("sdf", sdf), ("bf", bf), ("gf", gf)):
        np.save(directory / f"{name}.npy", array.astype(np.float32), allow_pickle=False)
    np.save(directory / "obs.npy", np.asarray(occupancy, dtype=np.uint8), allow_pickle=False)
    source = dict(kind="reconstructed-missing-original", repo=DATASET_REPO, revision=DATASET_REVISION,
                  scene_path=scene_path, upstream_commit=UPSTREAM_COMMIT,
                  generation_parameters=dict(difficulty=.8, seed=13, n_rect_L=3, n_rect_R=3, n_rect_F=2, n_rect_C=2),
                  source_hashes={name: sha256(SOURCE_ROOT / name) for name in
                                 ("random_obstacle.py", "grid_config.py", "pf_grid_config.yaml", "pf_modular.py")},
                  reason="D8G2L3O2S13 is referenced by the model config but absent from the public dataset",
                  arrays_unchanged=False, released_original_byte_identity_verified=False,
                  warning="Same-parameter task reconstruction; current upstream generator differs from released assets")
    (directory / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    records = _field_records(directory)
    return dict(scene_id=scene_name, family="original_cat", path=str(directory.relative_to(output)),
                shape=records["sdf"]["shape"], origin=list(pf_config["origin"]), dx=float(pf_config["dx"]),
                start=[0., 0., .8], goal=[2., 0., .75], reset_xy_scale=[1., 1.], reset_yaw=0.,
                original_config_path=scene_path, sampling_weight=1., fields=records,
                source=dict(source, metadata_sha256=sha256(directory / "source.json")),
                sample_coordinates="original runtime origin, including its released half-cell convention")


def fetch_original_scene(scene_path, output, pf_config, *, download=True, reconstruct_missing=False):
    """Verify source LFS hashes and retain all original NPY bytes unchanged."""
    scene_name = Path(scene_path).name
    directory = Path(output) / "original" / scene_name
    source_file = directory / "source.json"
    if source_file.exists():
        source = json.loads(source_file.read_text())
        if source.get("revision") != DATASET_REVISION or source.get("scene_path") != scene_path:
            raise ValueError("Cached original source pin differs")
        if source.get("kind") == "reconstructed-missing-original":
            if not reconstruct_missing:
                raise ValueError("Cached bank includes a reconstructed source; pass --reconstruct-missing-original explicitly")
            return reconstruct_missing_original(scene_path, output, pf_config)
        metadata = source["files"]
    elif download:
        try:
            metadata = _source_metadata(scene_name)
        except urllib.error.HTTPError as error:
            if error.code != 404 or not reconstruct_missing or scene_name != "D8G2L3O2S13":
                raise
            return reconstruct_missing_original(scene_path, output, pf_config)
    else:
        raise FileNotFoundError(f"Missing verified original asset metadata: {source_file}; use --download")
    for name in FIELD_NAMES:
        record = metadata[f"{name}.npy"]
        expected = record.get("lfs", {}).get("oid")
        if not expected or len(expected) != 64:
            raise ValueError("Original source metadata lacks an immutable LFS SHA256")
        file = directory / f"{name}.npy"
        if download:
            _download_file(record["path"], file, expected)
        if not file.exists() or sha256(file) != expected:
            raise ValueError(f"Missing/corrupt original field: {file}")
    source = dict(repo=DATASET_REPO, revision=DATASET_REVISION, scene_path=scene_path, files=metadata)
    source_file.write_text(json.dumps(source, indent=2) + "\n")
    fields = _field_records(directory)
    return dict(scene_id=scene_name, family="original_cat", path=str(directory.relative_to(output)),
                shape=fields["sdf"]["shape"], origin=list(pf_config["origin"]), dx=float(pf_config["dx"]),
                start=[0., 0., .8], goal=[2., 0., .75], reset_xy_scale=[1., 1.], reset_yaw=0.,
                original_config_path=scene_path, sampling_weight=1., fields=fields,
                source=dict(repo=DATASET_REPO, revision=DATASET_REVISION,
                            arrays_unchanged=True, metadata_sha256=sha256(source_file)),
                sample_coordinates="original runtime origin, including its released half-cell convention")


def rasterize_boxes(boxes, shape, sample_origin, dx):
    """Voxel-center occupancy of the canonical oriented boxes, without a floor."""
    shape = np.asarray(shape, dtype=int)
    origin = np.asarray(sample_origin, dtype=float)
    occupancy = np.zeros(tuple(shape), dtype=bool)
    for box in boxes:
        center, half = np.asarray(box["center"]), np.asarray(box["half_size"])
        c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
        extent = np.asarray([abs(c) * half[0] + abs(s) * half[1],
                             abs(s) * half[0] + abs(c) * half[1], half[2]])
        begin = np.maximum(0, np.ceil((center - extent - origin) / dx - 1e-8).astype(int))
        end = np.minimum(shape, np.floor((center + extent - origin) / dx + 1e-8).astype(int) + 1)
        if np.any(end <= begin):
            continue
        axes = [origin[i] + np.arange(begin[i], end[i]) * dx - center[i] for i in range(3)]
        x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)
        inside = ((np.abs(c * x + s * y) <= half[0] + 1e-8)
                  & (np.abs(-s * x + c * y) <= half[1] + 1e-8)
                  & (np.abs(z) <= half[2] + 1e-8))
        occupancy[tuple(slice(a, b) for a, b in zip(begin, end))] |= inside
    return occupancy


def make_clutter_fields(scene, directory, *, dx=.04):
    """Use CAT's unmodified three field functions on dense furniture occupancy."""
    from scipy import ndimage
    from cat_ppo.furniture.scenes import validate_scene

    validate_scene(scene)
    if not math.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    module = _upstream("pf_modular")
    shape = np.ceil((np.asarray(scene["room_dimensions"]) + [.2, .2, 0.]) / dx).astype(int)
    origin = np.asarray([-.1, -.1, 0.]) + .5 * dx
    axes = [origin[i] + np.arange(shape[i]) * dx for i in range(3)]
    occupancy = rasterize_boxes(scene["boxes"], shape, origin, dx)
    if not occupancy.any() or occupancy.all():
        raise ValueError("Clutter must contain both obstacles and free space")
    start = np.asarray([*scene["start"][:2], .8])
    goal = np.asarray([*scene["goal"], .75])
    start_index = np.rint((start - origin) / dx).astype(int)
    goal_index = np.rint((goal - origin) / dx).astype(int)
    labels, _ = ndimage.label(~occupancy)
    start_label, goal_label = labels[tuple(start_index)], labels[tuple(goal_index)]
    if start_label == 0 or start_label != goal_label:
        raise ValueError("Clutter start and goal are disconnected in the 3-D free-space occupancy")
    del labels
    config = module.PFConfig()
    config.voxel = dx
    sdf = module.make_sdf(occupancy, dx)
    bf = module.grad3(sdf, dx)
    _, gf = module.make_guidance_field_progressive(
        config, np.meshgrid(*axes, indexing="ij"), occupancy, goal, bf, sdf)
    for name, array in (("sdf", sdf), ("bf", bf), ("gf", gf)):
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite CAT FMM {name}")
        np.save(directory / f"{name}.npy", array.astype(np.float32), allow_pickle=False)
    np.save(directory / "obs.npy", occupancy.astype(np.uint8), allow_pickle=False)
    specification = copy.deepcopy(scene)
    specification["cat_field_generation"] = dict(
        pipeline="original-CAT-occupancy-SDF-gradient-progressive-3D-FMM",
        upstream_commit=UPSTREAM_COMMIT, pf_modular_sha256=sha256(SOURCE_ROOT / "pf_modular.py"),
        grid_config_sha256=sha256(SOURCE_ROOT / "grid_config.py"),
        dx=dx, shape=shape.tolist(), sample_origin=origin.tolist(),
        goal=goal.tolist(), route_used_for_guidance=False, physical_obstacles_in_training=False,
        floor_in_obstacle_sdf=False, free_voxel_start_goal_connected=True,
        root_route_geometry_validated=bool(scene["feasibility"]["root_route_validated"]),
        dynamic_feasibility_validated=False, full_body_reset_clearance_validated=False)
    (directory / "scene.json").write_text(json.dumps(specification, indent=2) + "\n")
    fields = _field_records(directory)
    return dict(scene_id=scene["scene_id"], family="generic_clutter" if scene["counts"].get("generic_objects") else "furniture",
                shape=shape.tolist(), origin=origin.tolist(), dx=dx,
                start=start.tolist(), goal=goal.tolist(), reset_xy_scale=[.08, .08],
                reset_yaw=scene["start"][2], sampling_weight=1., fields=fields,
                source=specification["cat_field_generation"],
                scene_sha256=sha256(directory / "scene.json"))


def prepare_generalist_fields(released_config, output, *, download=False, clutter=True,
                              seed=20260915, workers=4, reconstruct_missing=False):
    """Prepare all configured scene slots plus one dense room of each new family."""
    released_config, output = Path(released_config), Path(output).resolve()
    if sha256(released_config) != RELEASED_CONFIG_SHA256:
        raise ValueError("Released generalist config is not the pinned original file")
    config = json.loads(released_config.read_text())
    pf = config["env_config"]["pf_config"]
    if len(pf["paths"]) != 37 or len(set(pf["paths"])) != 37:
        raise ValueError("Pinned generalist configuration must contain 37 distinct scenes")
    existing_path = output / "manifest.json"
    if existing_path.exists():
        existing = load_generalist_manifest(existing_path)
        if (existing.get("seed") != seed or existing["scene_count"] != (39 if clutter else 37)
                or existing["released_config_sha256"] != RELEASED_CONFIG_SHA256
                or existing["dataset_revision"] != DATASET_REVISION):
            raise FileExistsError("Existing bank describes a different configuration; use a new output directory")
        if existing["reconstructed_original_count"] and not reconstruct_missing:
            raise ValueError("Existing bank contains a reconstructed scene; pass --reconstruct-missing-original explicitly")
        return existing_path
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        scenes = list(executor.map(lambda path: fetch_original_scene(path, output, pf, download=download,
                                      reconstruct_missing=reconstruct_missing), pf["paths"]))
    if clutter:
        from cat_ppo.furniture.scenes import generate_scene
        from cat_ppo.furniture.clutter import generate_clutter_scene
        for name, generator in (("furniture", generate_scene), ("generic_clutter", generate_clutter_scene)):
            directory = output / "clutter" / name
            record = make_clutter_fields(generator(seed=seed, difficulty="dense"), directory, dx=pf["dx"])
            record["path"] = str(directory.relative_to(output))
            scenes.append(record)
    manifest = dict(schema=SCHEMA, released_config_sha256=RELEASED_CONFIG_SHA256,
                    original_count=37, scene_count=len(scenes), scenes=scenes,
                    byte_verified_original_count=sum(s["source"].get("arrays_unchanged", False) for s in scenes),
                    reconstructed_original_count=sum(s["source"].get("kind") == "reconstructed-missing-original" for s in scenes),
                    upstream_commit=UPSTREAM_COMMIT, dataset_repo=DATASET_REPO,
                    dataset_revision=DATASET_REVISION, seed=seed,
                    storage="ragged-concatenated-xyz; original clipping and interpolation retained",
                    fields_bytes=sum(f["size_bytes"] for scene in scenes for f in scene["fields"].values()))
    manifest["manifest_sha256"] = _json_hash(manifest)
    destination = output / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    load_generalist_manifest(destination)
    return destination


def load_generalist_manifest(path, *, verify_files=True):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    expected = manifest.pop("manifest_sha256")
    if _json_hash(manifest) != expected or manifest.get("schema") != SCHEMA:
        raise ValueError("Field-bank manifest hash/schema mismatch")
    manifest["manifest_sha256"] = expected
    if len(manifest["scenes"]) != manifest["scene_count"]:
        raise ValueError("Manifest scene count differs")
    if manifest.get("released_config_sha256") != RELEASED_CONFIG_SHA256 or manifest.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("Field-bank release/configuration source pin differs")
    originals = [scene for scene in manifest["scenes"] if scene["family"] == "original_cat"]
    if (len(originals) != manifest["original_count"] or len(originals) != 37
            or len({scene["original_config_path"] for scene in originals}) != 37):
        raise ValueError("Field-bank must retain 37 distinct configured original scene slots")
    from cat_ppo.furniture.generalist_config import released_config
    configured_paths = released_config()["env_config"]["pf_config"]["paths"]
    if [scene["original_config_path"] for scene in originals] != configured_paths:
        raise ValueError("Field-bank original scenes/order differ from the released configuration")
    unchanged = sum(scene["source"].get("arrays_unchanged", False) for scene in originals)
    reconstructed = sum(scene["source"].get("kind") == "reconstructed-missing-original" for scene in originals)
    if (unchanged != manifest["byte_verified_original_count"]
            or reconstructed != manifest["reconstructed_original_count"] or unchanged + reconstructed != 37):
        raise ValueError("Field-bank original-source provenance counts differ")
    for scene in manifest["scenes"]:
        directory = (path.parent / scene["path"]).resolve()
        if not directory.is_relative_to(path.parent):
            raise ValueError("Field-bank path escapes its directory")
        if len(scene["origin"]) != 3 or len(scene["shape"]) != 3 or min(scene["shape"]) < 3:
            raise ValueError("Invalid field-bank spatial metadata")
        if not np.isfinite(scene["origin"]).all() or not math.isfinite(scene["dx"]) or scene["dx"] <= 0:
            raise ValueError("Invalid field-bank coordinates")
        if verify_files:
            records = _field_records(directory)
            if records != scene["fields"] or records["sdf"]["shape"] != scene["shape"]:
                raise ValueError(f"Field-bank file hashes/shapes differ: {scene['scene_id']}")
            if scene["family"] == "original_cat":
                if sha256(directory / "source.json") != scene["source"]["metadata_sha256"]:
                    raise ValueError("Original scene source fingerprint differs")
            elif sha256(directory / "scene.json") != scene["scene_sha256"]:
                raise ValueError("Clutter scene geometry/source fingerprint differs")
    return manifest


def bank_config(path):
    """PF config values consumed by G1CatWholeBodyEnv and the native wrapper."""
    path = Path(path).resolve()
    manifest = load_generalist_manifest(path)
    scenes = manifest["scenes"]
    paths = [str(path.parent / scene["path"]) for scene in scenes]
    return dict(bank_manifest=str(path), path=paths[0], paths=paths,
                origin=scenes[0]["origin"], dx=scenes[0]["dx"],
                sampling_weights=[scene["sampling_weight"] for scene in scenes],
                sampling_alpha=1., sampling_ema_decay=.95)


def sample_ragged_field(field, pos, *, origin, dx, shape, offset):
    """CAT's exact interpolation, changing only corner storage addressing."""
    import jax.numpy as jp

    idx = (pos - origin) / dx
    xyz = jp.clip(idx, 0, shape - 2)
    base = jp.floor(xyz).astype(jp.int32)
    fractional = xyz - base
    offsets = jp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=jp.int32)
    corners = base[:, None, :] + offsets[None, :, :]
    addresses = offset + (corners[..., 0] * shape[1] + corners[..., 1]) * shape[2] + corners[..., 2]
    values = field[addresses, :]
    wx = jp.stack([1. - fractional[:, 0], fractional[:, 0]], axis=1)
    wy = jp.stack([1. - fractional[:, 1], fractional[:, 1]], axis=1)
    wz = jp.stack([1. - fractional[:, 2], fractional[:, 2]], axis=1)
    weights = (wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]).reshape(-1, 8)
    return jp.einsum("ne,nec->nc", weights, values)


class RaggedSceneMixin:
    """Original DAgger scene/reset API backed by immutable mixed-size fields."""

    def __init__(self, task_type="flat_terrain", config=None, config_overrides=None, **kwargs):
        import jax.numpy as jp

        # G1CatEnv needs one original field for initialization only. No reset is
        # performed by its constructor, so routing can be installed afterwards.
        super().__init__(task_type=task_type, config=config, config_overrides=config_overrides, **kwargs)
        manifest_path = Path(config.pf_config.bank_manifest).resolve()
        self.field_bank_manifest = load_generalist_manifest(manifest_path)
        scenes = self.field_bank_manifest["scenes"]
        self.num_pf_scenes = len(scenes)
        arrays = {name: [] for name in FIELD_NAMES}
        offsets, position = [], 0
        for scene in scenes:
            offsets.append(position)
            position += int(np.prod(scene["shape"]))
            for name in FIELD_NAMES:
                channels = 1 if name == "sdf" else 3
                array = np.load(manifest_path.parent / scene["path"] / f"{name}.npy", allow_pickle=False)
                arrays[name].append(array.reshape(-1, channels))
        for name in FIELD_NAMES:
            setattr(self, name, jp.array(np.concatenate(arrays[name], axis=0)))
        self._pf_offsets = jp.array(offsets, dtype=jp.int32)
        self._pf_shapes = jp.array([s["shape"] for s in scenes], dtype=jp.int32)
        self._pf_origins = jp.array([s["origin"] for s in scenes], dtype=jp.float32)
        self._pf_dxs = jp.array([s["dx"] for s in scenes], dtype=jp.float32)
        self._pf_scene_starts = jp.array([s["start"] for s in scenes], dtype=jp.float32)
        self._pf_reset_xy_scale = jp.array([s["reset_xy_scale"] for s in scenes], dtype=jp.float32)
        self._pf_scene_yaws = jp.array([s["reset_yaw"] for s in scenes], dtype=jp.float32)
        self._pf_scene_goals = jp.array([s["goal"] for s in scenes], dtype=jp.float32)
        self._pf_scene_original = jp.array([s["family"] == "original_cat" for s in scenes])
        self._field_pf_id = jp.array(0, dtype=jp.int32)
        weights = np.asarray(getattr(config.pf_config, "sampling_weights", []) or
                             [s["sampling_weight"] for s in scenes], dtype=np.float32)
        if weights.shape != (self.num_pf_scenes,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("Field sampling weights must be finite, nonnegative and match scene count")
        self._pf_sampling_logits = jp.log(jp.array(weights / weights.sum()) + 1e-8)
        self._pf_sampling_alpha = float(getattr(config.pf_config, "sampling_alpha", 1.))
        self._pf_sampling_ema_decay = float(getattr(config.pf_config, "sampling_ema_decay", .95))

    def _add_pf_info(self, state):
        import jax.numpy as jp
        state.info.update(pf_id=self._field_pf_id, pf_success_ema=jp.zeros(self.num_pf_scenes),
                          pf_episode_ema=jp.zeros(self.num_pf_scenes), pf_sampling_logits=self._pf_sampling_logits,
                          pf_sampling_alpha=jp.array(self._pf_sampling_alpha, dtype=jp.float32),
                          pf_sampling_ema_decay=jp.array(self._pf_sampling_ema_decay, dtype=jp.float32))
        return state

    def reset(self, rng):
        import jax
        import jax.numpy as jp
        rng, pf_key = jax.random.split(rng)
        self._field_pf_id = jax.random.categorical(pf_key, self._pf_sampling_logits).astype(jp.int32)
        return self._add_pf_info(super().reset(rng))

    def reset_with_pf_id(self, rng, pf_id):
        import jax.numpy as jp
        self._field_pf_id = pf_id.astype(jp.int32)
        return self._add_pf_info(super().reset(rng))

    def step(self, state, action):
        import jax.numpy as jp
        self._field_pf_id = state.info["pf_id"].astype(jp.int32)
        return super().step(state, action)

    def sample_field(self, field, pos):
        scene = self._field_pf_id
        return sample_ragged_field(field, pos, origin=self._pf_origins[scene],
                                   dx=self._pf_dxs[scene], shape=self._pf_shapes[scene],
                                   offset=self._pf_offsets[scene])

    def adjust_reset_pose(self, qpos):
        """Keep released resets exact; place new rooms around their admitted start."""
        import jax.numpy as jp
        from mujoco.mjx._src import math as mj_math
        scene = self._field_pf_id
        position = self._pf_scene_starts[scene, :2] + qpos[:2] * self._pf_reset_xy_scale[scene]
        quat = mj_math.axis_angle_to_quat(jp.array([0., 0., 1.]), self._pf_scene_yaws[scene][None])
        adjusted = qpos.at[:2].set(position).at[3:7].set(mj_math.quat_mul(quat, qpos[3:7]))
        return jp.where(self._pf_scene_original[scene], qpos, adjusted)
