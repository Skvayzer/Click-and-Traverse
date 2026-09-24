#!/usr/bin/env python3
"""Build immutable obstacle-clear fallback resets from CAT's native distribution.

This only generates a small offline qpos cache. Native reset candidates that
are already clear remain unchanged at runtime. No scene is dropped, and failed
rejection sampling never silently substitutes another upper-body posture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
SCHEMA = "cat-body-collision-clear-reset-pool-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, record):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".partial")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _quat_multiply(first, second):
    first, second = np.broadcast_arrays(first, second)
    scalar = first[..., :1] * second[..., :1] - np.sum(first[..., 1:] * second[..., 1:], axis=-1, keepdims=True)
    vector = (first[..., :1] * second[..., 1:] + second[..., :1] * first[..., 1:]
              + np.cross(first[..., 1:], second[..., 1:]))
    return np.concatenate([scalar, vector], axis=-1)


def sample_native_reset_poses(uniforms, nominal_qpos, soft_lower, soft_upper, scene):
    """Map independent U[0,1) draws to the existing native/room reset law.

    Rows contain two XY draws, one yaw draw, and one draw per actuated joint.
    This preserves the native distribution, not its exact JAX random-key stream.
    Float32 candidate rows make validation and runtime cache values identical.
    """
    nominal = np.asarray(nominal_qpos, dtype=np.float32)
    draws = np.asarray(uniforms, dtype=np.float32)
    if (nominal.ndim != 1 or len(nominal) < 8 or draws.ndim != 2
            or draws.shape[1] != len(nominal) - 4 or not np.isfinite(draws).all()
            or np.any(draws < 0.) or np.any(draws >= 1.)):
        raise ValueError("Expected one finite native qpos and U[0,1) rows with XY/yaw/joint draws")
    qpos = np.broadcast_to(nominal, (len(draws), len(nominal))).copy()
    qpos[:, :2] += 2. * draws[:, :2] - 1.
    qpos[:, 2] = .8
    yaw = (draws[:, 2] - .5) * np.pi
    yaw_quat = np.zeros((len(draws), 4), dtype=np.float32)
    yaw_quat[:, 0], yaw_quat[:, 3] = np.cos(.5 * yaw), np.sin(.5 * yaw)
    qpos[:, 3:7] = _quat_multiply(qpos[:, 3:7], yaw_quat)
    qpos[:, 7:] = np.clip(qpos[:, 7:] * (.5 + draws[:, 3:]), soft_lower, soft_upper)
    reset_mode = scene.get("reset_mode", scene.get("task_kind",
                          "room" if scene.get("family") in ("furniture", "generic_clutter") else "cat"))
    if reset_mode not in ("cat", "room"):
        raise ValueError(f"Unknown scene reset mode: {reset_mode}")
    if reset_mode == "room":
        qpos[:, :2] = np.asarray(scene["start"][:2], np.float32) + qpos[:, :2] * np.asarray(scene["reset_xy_scale"], np.float32)
        room_yaw = float(scene["reset_yaw"])
        room_quat = np.array([np.cos(.5 * room_yaw), 0., 0., np.sin(.5 * room_yaw)], np.float32)
        qpos[:, 3:7] = _quat_multiply(room_quat, qpos[:, 3:7])
    return qpos.astype(np.float32, copy=False)


def collect_reset_pool(scenes, nominal_qpos, soft_lower, soft_upper, check_collisions, *,
                       poses_per_scene=32, seed=20260916, batch_size=1024,
                       candidates_per_scene=64, max_attempts_per_scene=4096, progress=None,
                       scene_index_offset=0):
    """Deterministic rejection sampling with fixed-size checker batches.

    ``check_collisions(scene_ids, qpos)`` returns either [batch] collision flags
    or [batch, shape] flags. Final partial batches repeat their last valid row;
    padding never contributes to counts or accepted poses. Returns the pool,
    per-scene provenance, and failed scene indices. A failed pool is not valid
    for publication. ``scene_index_offset`` maps a suffix to its actual full-bank
    checker IDs and PRNG streams. Returned record/failed indices are global.
    No runtime or scene state is modified.
    """
    if (not scenes or min(poses_per_scene, batch_size, candidates_per_scene) < 1
            or max_attempts_per_scene < poses_per_scene or seed < 0
            or type(scene_index_offset) is not int or scene_index_offset < 0):
        raise ValueError("Invalid reset pool sampling sizes/seed")
    count, nq = len(scenes), len(nominal_qpos)
    pool = np.full((count, poses_per_scene, nq), np.nan, dtype=np.float32)
    selected = np.zeros(count, dtype=np.int64)
    attempts = np.zeros(count, dtype=np.int64)
    clear_count = np.zeros(count, dtype=np.int64)
    generators = [np.random.default_rng(np.random.SeedSequence([seed, index + scene_index_offset]))
                  for index in range(count)]
    collision_by_shape = None
    batches = 0
    started = time.monotonic()
    while True:
        active = np.flatnonzero((selected < poses_per_scene) & (attempts < max_attempts_per_scene))
        if not len(active):
            break
        for offset in range(0, len(active), max(1, batch_size // candidates_per_scene)):
            members = active[offset:offset + max(1, batch_size // candidates_per_scene)]
            ids, positions = [], []
            available = batch_size
            for index in members:
                number = min(candidates_per_scene, max_attempts_per_scene - attempts[index], available)
                if number <= 0:
                    continue
                draws = generators[index].random((number, nq - 4), dtype=np.float32)
                positions.append(sample_native_reset_poses(draws, nominal_qpos, soft_lower, soft_upper, scenes[index]))
                ids.append(np.full(number, index, dtype=np.int32))
                available -= number
            if not ids:
                continue
            ids, positions = np.concatenate(ids), np.concatenate(positions)
            used = len(ids)
            padded_ids = np.pad(ids, (0, batch_size - used), mode="edge")
            padded_qpos = np.concatenate([positions, np.repeat(positions[-1:], batch_size - used, axis=0)])
            flags = np.asarray(check_collisions(padded_ids + scene_index_offset, padded_qpos), dtype=bool)
            if flags.ndim == 1:
                flags = flags[:, None]
            if flags.ndim != 2 or flags.shape[0] != batch_size:
                raise ValueError("Collision checker returned an invalid batch shape")
            flags = flags[:used]
            invalid = np.any(flags, axis=-1)
            if collision_by_shape is None:
                collision_by_shape = np.zeros((count, flags.shape[1]), dtype=np.int64)
            elif flags.shape[1] != collision_by_shape.shape[1]:
                raise ValueError("Collision checker changed its number of primitive flags")
            for index in np.unique(ids):
                rows = ids == index
                candidates = positions[rows & ~invalid]
                attempts[index] += int(rows.sum())
                clear_count[index] += len(candidates)
                collision_by_shape[index] += flags[rows].sum(axis=0)
                take = min(len(candidates), poses_per_scene - selected[index])
                first = int(selected[index])
                pool[index, first:first + take] = candidates[:take]
                selected[index] += take
            batches += 1
            if progress is not None:
                progress(dict(status="sampling", batches=batches, scene_count=count,
                              completed_scenes=int(np.sum(selected == poses_per_scene)),
                              attempted_poses=int(attempts.sum()), selected_poses=int(selected.sum()),
                              rejected_poses=int((attempts - clear_count).sum()),
                              elapsed_seconds=time.monotonic() - started))
    failed = (np.flatnonzero(selected != poses_per_scene) + scene_index_offset).tolist()
    records = []
    for index, scene in enumerate(scenes):
        record = dict(index=index + scene_index_offset, scene_id=scene["scene_id"], family=scene.get("family"),
                      attempts=int(attempts[index]), clear_candidates=int(clear_count[index]),
                      selected_poses=int(selected[index]),
                      rejection_fraction=float(1. - clear_count[index] / attempts[index]),
                      complete=bool(selected[index] == poses_per_scene))
        if index + scene_index_offset in failed:
            record["collision_counts_by_shape"] = collision_by_shape[index].tolist()
        records.append(record)
    return pool, records, failed


def _manifest_reference(owner, value):
    path = Path(value)
    path = (owner.parent / path).resolve() if not path.is_absolute() else path.resolve()
    return path / "manifest.json" if path.is_dir() else path


def _verified_geometry_cache(manifest_path, scene):
    path = (manifest_path.parent / scene["geometry_file"]).resolve()
    if not path.is_relative_to(manifest_path.parent) or sha256(path) != scene["geometry_sha256"]:
        raise ValueError("Base append collision geometry cache path/hash differs")


def _load_reset_array(path):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if loaded.files not in (["qpos"], ["arr_0"]):
                raise ValueError("Reset NPZ must contain exactly one qpos array")
            return loaded[loaded.files[0]]
        finally:
            loaded.close()
    return loaded


def _normalized_robot_xml(xml):
    """Canonicalize only mesh file paths to verified content hashes.

    Every element, mesh name, transform, physical parameter, compiler setting,
    and non-file attribute remains part of the comparison. Assembled training
    XML uses absolute mesh paths, so relative files fail rather than being
    resolved against an unrelated directory containing this proof artifact.
    """
    root = ET.fromstring(xml)
    mesh_count = 0
    for mesh in root.findall("./asset/mesh"):
        if "file" not in mesh.attrib:
            continue
        path = Path(mesh.get("file"))
        if not path.is_absolute() or not path.is_file():
            raise ValueError("Robot relocation proof requires existing absolute mesh file paths")
        mesh.set("file", "sha256:" + sha256(path))
        mesh_count += 1
    if not mesh_count:
        raise ValueError("Robot relocation proof requires file-backed robot meshes")
    canonical = ET.canonicalize(ET.tostring(root, encoding="unicode"), strip_text=True)
    return hashlib.sha256(canonical.encode()).hexdigest(), mesh_count


def _without_contact(xml_text):
    root = ET.fromstring(xml_text)
    for contact in root.findall("contact"):
        root.remove(contact)
    return ET.canonicalize(ET.tostring(root, encoding="unicode"), strip_text=True)


def verify_robot_xml_identity(expected_old_sha256, new_sha256, *, new_xml=None, base_robot_xml=None,
                              allow_contact_pair_change=False):
    """Accept identical XML, a hash-bound proof of mesh-path relocation, or -- when explicitly
    allowed -- an XML that differs ONLY in its <contact> pairs (self-contact pairs do not change
    the robot geometry the reset poses were certified against)."""
    if expected_old_sha256 == new_sha256:
        return dict(verification="raw-XML-identity", old_raw_sha256=expected_old_sha256,
                    new_raw_sha256=new_sha256, old_normalized_sha256=None, new_normalized_sha256=None)
    if base_robot_xml == "UNVERIFIED":
        # Operator override: the base bank's compiled XML could not be reproduced (its provenance
        # records only a hash). Recorded loudly in the output manifest; use only when the robot
        # geometry is known to be unchanged (e.g. self-contact pairs added, no geoms touched).
        return dict(verification="UNVERIFIED-base-robot-xml-accepted-by-operator", old_raw_sha256=expected_old_sha256,
                    new_raw_sha256=new_sha256, old_normalized_sha256=None, new_normalized_sha256=None)
    if base_robot_xml is None or new_xml is None:
        raise ValueError("Base reset compiled robot XML differs; supply --base-robot-xml for an asset-relocation proof")
    old_path = Path(base_robot_xml).resolve()
    old_bytes = old_path.read_bytes()
    if hashlib.sha256(old_bytes).hexdigest() != expected_old_sha256:
        raise ValueError("--base-robot-xml does not hash to the base reset bank's recorded robot XML")
    if allow_contact_pair_change and _without_contact(old_bytes.decode()) == _without_contact(new_xml):
        return dict(verification="contact-pairs-only-change", old_raw_sha256=expected_old_sha256,
                    new_raw_sha256=new_sha256, old_normalized_sha256=None, new_normalized_sha256=None)
    old_raw = hashlib.sha256(old_bytes).hexdigest()
    if old_raw != expected_old_sha256:
        raise ValueError("Base robot XML proof raw SHA256 differs from the reset manifest")
    actual_new_sha256 = hashlib.sha256(new_xml.encode()).hexdigest()
    if actual_new_sha256 != new_sha256:
        raise ValueError("New robot XML proof raw SHA256 differs from the assembled model")
    old_normalized, old_count = _normalized_robot_xml(old_bytes.decode())
    new_normalized, new_count = _normalized_robot_xml(new_xml)
    if old_normalized != new_normalized or old_count != new_count:
        raise ValueError("Robot XML relocation changes mesh content or non-path robot attributes")
    return dict(verification="content-verified-mesh-path-relocation", base_robot_xml=str(old_path),
                old_raw_sha256=old_raw, new_raw_sha256=actual_new_sha256,
                old_normalized_sha256=old_normalized, new_normalized_sha256=new_normalized,
                mesh_file_count=new_count, normalized_attributes=["asset/mesh@file"],
                all_other_robot_attributes_unchanged=True)


def load_append_base(base_path, *, field_path, fields, collision_path, collision_arrays,
                     collision_meta, proposal_path, robot_xml_sha256, poses_per_scene,
                     nq, shape_names, robot_xml=None, base_robot_xml=None, allow_contact_pair_change=False):
    """Verify an immutable base and prove that every old scene is unchanged.

    Both field banks and collision arrays are verified with their normal
    loaders. Full field records, geometry fingerprints, and all old indexed
    geometry/CSR array rows must be an exact prefix of the new bank. A complete
    reset pool is then copied without changing row order or scene statistics.
    The build still rechecks every retained pose under the final new checker.
    """
    from cat_ppo.furniture.body_collision_bank import load_body_collision_bank
    from cat_ppo.furniture.generalist_fields import load_generalist_manifest

    base_path = Path(base_path).resolve()
    if base_path.is_dir():
        base_path = base_path / "manifest.json"
    base = json.loads(base_path.read_text())
    if base.get("schema") != SCHEMA or base.get("status") != "complete":
        raise ValueError("Base reset manifest must be a complete validated reset pool")
    base_field = _manifest_reference(base_path, base["field_manifest"])
    base_collision = _manifest_reference(base_path, base["collision_bank"])
    base_proposal = _manifest_reference(base_path, base["proposal"])
    proxy_hash = sha256(proposal_path)
    for key, path in (("field_manifest_sha256", base_field),
                      ("collision_bank_sha256", base_collision), ("proxy_sha256", base_proposal)):
        if base.get(key) != sha256(path):
            raise ValueError(f"Base reset {key} differs from its source")
    if base["proxy_sha256"] != proxy_hash:
        raise ValueError("Base reset proposal differs from the approved collision proposal")
    robot_proof = verify_robot_xml_identity(base.get("robot_xml_sha256"), robot_xml_sha256,
                                            new_xml=robot_xml, base_robot_xml=base_robot_xml,
                                            allow_contact_pair_change=allow_contact_pair_change)
    old_fields = load_generalist_manifest(base_field, verify_files=True)
    # Validate new bytes too; a matching record alone cannot prove its file.
    verified_fields = load_generalist_manifest(field_path, verify_files=True)
    if verified_fields != fields:
        raise ValueError("New field manifest changed while preparing append")
    old_arrays, old_collision = load_body_collision_bank(
        base_collision, expected_field_manifest=base_field, expected_proxy_sha256=proxy_hash)
    verified_arrays, verified_collision = load_body_collision_bank(
        collision_path, expected_field_manifest=field_path, expected_proxy_sha256=proxy_hash)
    if (verified_collision != collision_meta or set(verified_arrays) != set(collision_arrays)
            or any(not np.array_equal(value, collision_arrays[key]) for key, value in verified_arrays.items())):
        raise ValueError("New collision bank changed while preparing append")
    count = old_fields["scene_count"]
    if (not 0 < count < fields["scene_count"] or fields["scenes"][:count] != old_fields["scenes"]
            or old_collision["scene_count"] != count or base.get("scene_count") != count):
        raise ValueError("Append requires an exact old field-scene prefix and at least one new scene")
    for key in ("builder", "query_radius_m", "cell_size_m", "outward_epsilon_m", "shape_count",
                "floor_included", "geometry_mapping", "geometry_float32", "index"):
        if old_collision.get(key) != collision_meta.get(key):
            raise ValueError(f"Append collision geometry/index contract differs: {key}")
    for index, old_scene in enumerate(old_collision["scenes"]):
        new_scene = collision_meta["scenes"][index]
        if (old_scene.get("index") != index or old_scene["scene_id"] != old_fields["scenes"][index]["scene_id"]
                or {key: value for key, value in old_scene.items() if key != "geometry_file"}
                != {key: value for key, value in new_scene.items() if key != "geometry_file"}):
            raise ValueError("Append collision scene geometry rows/fingerprints differ")
        _verified_geometry_cache(base_collision, old_scene)
        _verified_geometry_cache(collision_path, new_scene)
    if set(old_arrays) != set(collision_arrays):
        raise ValueError("Append collision array set differs")
    for key, old in old_arrays.items():
        new = collision_arrays[key]
        if (old.dtype != new.dtype or old.ndim != new.ndim
                or (old.ndim and (new.shape[1:] != old.shape[1:] or new.shape[0] < old.shape[0]))
                or not np.array_equal(old, new[:old.shape[0]] if old.ndim else new)):
            raise ValueError(f"Append collision array prefix differs: {key}")
    pool_path = (base_path.parent / base["file"]).resolve()
    if not pool_path.is_relative_to(base_path.parent) or sha256(pool_path) != base["sha256"]:
        raise ValueError("Base reset pose cache path/checksum differs")
    pool = _load_reset_array(pool_path)
    expected_shape = (count, poses_per_scene, nq)
    if (pool.shape != expected_shape or pool.dtype != np.float32 or not np.isfinite(pool).all()
            or base.get("shape") != list(expected_shape) or base.get("dtype") != "float32"
            or base.get("poses_per_scene") != poses_per_scene
            or base.get("validated_pose_count") != count * poses_per_scene
            or base.get("shape_names") != list(shape_names)
            or base.get("soft_joint_pos_limit_factor") != .95
            or base.get("extra_clearance_margin_m") != 0.
            or base.get("dropped_scenes") != 0 or base.get("invented_reset_postures") is not False):
        raise ValueError("Base reset pool shape, dtype, completeness or reset-law contract differs")
    records = base.get("scenes", [])
    if len(records) != count:
        raise ValueError("Base reset scene statistics count differs")
    for index, record in enumerate(records):
        attempts, clear = record.get("attempts", 0), record.get("clear_candidates", 0)
        if (record.get("index") != index or record.get("scene_id") != old_fields["scenes"][index]["scene_id"]
                or record.get("family") != old_fields["scenes"][index].get("family")
                or record.get("complete") is not True or record.get("selected_poses") != poses_per_scene
                or not attempts >= clear >= poses_per_scene
                or not np.isclose(record.get("rejection_fraction", np.nan), 1. - clear / attempts,
                                  rtol=0., atol=1e-12)):
            raise ValueError("Base reset per-scene statistics are incomplete or inconsistent")
    proof = dict(manifest=str(base_path), manifest_sha256=sha256(base_path), pool_sha256=base["sha256"],
                 field_manifest_sha256=base["field_manifest_sha256"],
                 collision_bank_sha256=base["collision_bank_sha256"],
                 preserved_scene_count=count, preserved_pose_count=count * poses_per_scene,
                 source_seed=base.get("seed"), field_scene_prefix_identical=True,
                 collision_geometry_fingerprints_identical=True, collision_array_prefixes_identical=True,
                 qpos_rows_and_order_preserved=True, per_scene_statistics_preserved=True,
                 robot_xml_identity=robot_proof)
    return pool.copy(), list(records), proof


def _make_checker(model, compiled, arrays, *, max_candidates):
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    from cat_ppo.envs.g1.body_collision import body_collisions

    mjx_model = mjx.put_model(model)
    template = mjx.make_data(mjx_model)
    bank = {key: jp.asarray(value) for key, value in arrays.items()}

    def query(scene_id, qpos):
        data = mjx.kinematics(mjx_model, template.replace(qpos=qpos))
        return body_collisions(compiled, bank, scene_id, data.xpos, data.xmat,
                               max_candidates=max_candidates)

    return jax.jit(jax.vmap(query))


def _make_native_cpu_checker(model, proposal, arrays, *, max_candidates):
    """CPU MuJoCo FK plus the exact native Torch collision kernels; no simulation steps."""
    import mujoco
    import torch
    from cat_mjlab.collision import compile_proposal, body_collisions
    torch.set_num_threads(2)
    compiled=compile_proposal(proposal,model,'cpu')
    bank={key:torch.as_tensor(np.asarray(value).copy()) for key,value in arrays.items()}
    data=mujoco.MjData(model)
    def query(scene_ids,qpos):
        positions=[];rotations=[]
        for pose in qpos:
            data.qpos[:]=pose
            mujoco.mj_kinematics(model,data)
            positions.append(data.xpos.copy());rotations.append(data.xmat.copy().reshape(-1,3,3))
        with torch.no_grad():
            return body_collisions(compiled,bank,torch.as_tensor(scene_ids,dtype=torch.long),
                torch.as_tensor(np.asarray(positions),dtype=torch.float32),
                torch.as_tensor(np.asarray(rotations),dtype=torch.float32),max_candidates=max_candidates).numpy()
    return query


def build(args):
    import jax
    import mujoco
    if getattr(args,'cpu_native',False):
        from cat_mjlab import constants
        from cat_mjlab.model import assemble_training_xml
    else:
        from cat_ppo.envs.g1 import constants
        from cat_ppo.envs.g1.env_cat_wholebody import assemble_training_xml
    from cat_ppo.furniture.body_collision_bank import load_body_collision_bank
    from cat_ppo.furniture.body_collision_geometry import compile_proposal
    from cat_ppo.furniture.generalist_fields import load_generalist_manifest
    from scripts.collision_proxy_preview_geometry import validate_proposal

    started = time.monotonic()
    field_path, collision_path, proposal_path, output = map(lambda path: Path(path).resolve(), (
        args.field_manifest, args.collision_bank, args.proposal, args.output))
    if collision_path.is_dir():
        collision_path = collision_path / "manifest.json"
    if output.exists():
        raise FileExistsError(f"Use a new immutable reset output directory: {output}")
    fields = load_generalist_manifest(field_path, verify_files=False)
    arrays, collision_meta = load_body_collision_bank(
        collision_path, expected_field_manifest=field_path, expected_proxy_sha256=sha256(proposal_path))
    scenes = fields["scenes"]
    if (len(scenes) != collision_meta["scene_count"]
            or any(first["scene_id"] != second["scene_id"]
                   for first, second in zip(scenes, collision_meta["scenes"]))):
        raise ValueError("Field and collision bank scene order differs")
    proposal = json.loads(proposal_path.read_text())
    xml = assemble_training_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    if model.nq != 36 or model.nv != 35 or model.njnt != 30:
        raise ValueError("Expected the current 29-joint G1 with fixed Dex3 hands")
    nominal_data = mujoco.MjData(model)
    nominal_data.qpos[:] = constants.DEFAULT_QPOS
    mujoco.mj_forward(model, nominal_data)
    mesh_coverage = validate_proposal(model, nominal_data, proposal)
    if not mesh_coverage["all_covered"] or mesh_coverage["missing_geom_ids"]:
        raise ValueError("Approved primitives do not cover the actual compiled robot meshes")
    compiled = compile_proposal(proposal, model)
    lower, upper = model.jnt_range[1:].T
    center, span = .5 * (lower + upper), upper - lower
    soft_lower = (center - .5 * span * .95).astype(np.float32)
    soft_upper = (center + .5 * span * .95).astype(np.float32)
    if getattr(args,'cpu_native',False):
        checker=_make_native_cpu_checker(model,proposal,arrays,max_candidates=int(collision_meta['static_candidate_count']))
    else:
        checker = _make_checker(model, compiled, arrays,
                                max_candidates=int(collision_meta["static_candidate_count"]))
    base_pool, base_records, base_proof = None, [], None
    if getattr(args, "base_reset_manifest", None) is not None:
        base_pool, base_records, base_proof = load_append_base(
            args.base_reset_manifest, field_path=field_path, fields=fields,
            collision_path=collision_path, collision_arrays=arrays, collision_meta=collision_meta,
            proposal_path=proposal_path, robot_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
            poses_per_scene=args.poses_per_scene, nq=model.nq, shape_names=compiled["shape_names"],
            robot_xml=xml, base_robot_xml=getattr(args, "base_robot_xml", None),
            allow_contact_pair_change=getattr(args, "allow_contact_pair_change", False))
    preserved_count = len(base_records)
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(
        schema=SCHEMA, collision_bank_sha256=sha256(collision_path),
        proxy_sha256=sha256(proposal_path), field_manifest_sha256=sha256(field_path),
        collision_bank=str(collision_path), field_manifest=str(field_path),
        proposal=str(proposal_path), robot_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        scene_count=len(scenes), poses_per_scene=args.poses_per_scene, seed=args.seed,
        batch_size=args.batch_size, candidates_per_scene=args.candidates_per_scene,
        max_attempts_per_scene=args.max_attempts_per_scene,
        sampling="Native CAT XY/yaw/nominal-joint law followed by unchanged room reset transforms",
        random_stream="NumPy PCG64: SeedSequence([seed, scene_index]), 32 float32 draws per candidate",
        soft_joint_pos_limit_factor=.95, extra_clearance_margin_m=0.,
        collision_rule="All 35 approved primitives clear under the same indexed exact-volume checker as training",
        dropped_scenes=0, invented_reset_postures=False,
        body_margin_m=proposal["body_margin_m"],
        mesh_coverage=mesh_coverage,
        shape_names=list(compiled["shape_names"]),
        backend='native-cpu' if getattr(args,'cpu_native',False) else jax.default_backend(),
        devices=['CPU'] if getattr(args,'cpu_native',False) else [str(device) for device in jax.devices()],
        source_sha256={name: sha256(ROOT / name) for name in (
            "scripts/build_body_collision_resets.py", "cat_ppo/envs/g1/body_collision.py",
            "scripts/collision_proxy_preview_geometry.py",
            "cat_ppo/furniture/body_collision_geometry.py", "cat_ppo/furniture/body_collision_bank.py",
            "cat_ppo/envs/g1/constants.py", "cat_ppo/furniture/grippers.py")})
    if base_proof is not None:
        provenance["append_base"] = base_proof
        provenance["generated_scene_count"] = len(scenes) - preserved_count
        provenance["generated_scene_index_offset"] = preserved_count
    _write_json(output / "preparation.json", provenance)

    def progress(record):
        if record["batches"] == 1 or record["batches"] % 10 == 0:
            if preserved_count:
                record = dict(record, scene_count=len(scenes), preserved_scenes=preserved_count,
                              completed_scenes=record["completed_scenes"] + preserved_count,
                              selected_poses=record["selected_poses"] + preserved_count * args.poses_per_scene)
            _write_json(output / "progress.json", record)
            print(json.dumps(record), flush=True)

    pool, records, failed = collect_reset_pool(
        scenes[preserved_count:], constants.DEFAULT_QPOS, soft_lower, soft_upper, checker,
        poses_per_scene=args.poses_per_scene, seed=args.seed, batch_size=args.batch_size,
        candidates_per_scene=args.candidates_per_scene,
        max_attempts_per_scene=args.max_attempts_per_scene, progress=progress,
        scene_index_offset=preserved_count)
    if base_pool is not None:
        pool = np.concatenate([base_pool, pool], axis=0)
        records = base_records + records
        if not np.array_equal(pool[:preserved_count], base_pool):
            raise AssertionError("Appending changed preserved reset poses")
    if failed:
        failure = dict(provenance, status="failed", failed_scene_indices=failed,
                       scenes=records, elapsed_seconds=time.monotonic() - started)
        _write_json(output / "failure.json", failure)
        first = records[failed[0]]
        raise RuntimeError(f"Could not fill clear native reset pool for {len(failed)} scenes; "
                           f"first: {first['scene_id']} ({first['selected_poses']}/{args.poses_per_scene} "
                           f"after {first['attempts']} attempts). See {output / 'failure.json'}")
    if not np.isfinite(pool).all():
        raise ValueError("Completed reset pool contains nonfinite poses")
    # Recheck every selected float32 row before publishing, independently of
    # collection indexing and padding. Runtime consumes precisely these rows.
    flat = pool.reshape(-1, model.nq)
    scene_ids = np.repeat(np.arange(len(scenes), dtype=np.int32), args.poses_per_scene)
    for start in range(0, len(flat), args.batch_size):
        qpos, ids = flat[start:start + args.batch_size], scene_ids[start:start + args.batch_size]
        used = len(ids)
        flags = np.asarray(checker(np.pad(ids, (0, args.batch_size - used), mode="edge"),
                           np.concatenate([qpos, np.repeat(qpos[-1:], args.batch_size - used, axis=0)])))
        if np.any(flags[:used]):
            raise ValueError("Final reset pool verification found a colliding pose; no manifest published")
    temporary = output / ".qpos.partial"
    with temporary.open("wb") as stream:
        np.save(stream, pool, allow_pickle=False)
    pool_path = output / "qpos.npy"
    temporary.replace(pool_path)
    result = dict(provenance, status="complete", file=pool_path.name, sha256=sha256(pool_path),
                  shape=list(pool.shape), dtype=str(pool.dtype), scenes=records,
                  validated_pose_count=len(flat), elapsed_seconds=time.monotonic() - started)
    _write_json(output / "manifest.json", result)
    _write_json(output / "progress.json", dict(status="complete", completed_scenes=len(scenes),
                                              selected_poses=len(flat), elapsed_seconds=result["elapsed_seconds"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-native", action="store_true", help="CPU-only MuJoCo FK and native Torch collision certification")
    parser.add_argument("--field-manifest", type=Path, required=True)
    parser.add_argument("--collision-bank", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, default=ROOT / "docs/assets/collision-proxy-proposal-20260916/proposal.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-reset-manifest", type=Path,
                        help="Preserve a verified reset pool prefix; generate only appended scenes")
    parser.add_argument("--base-robot-xml", type=lambda v: v if v == "UNVERIFIED" else Path(v),
                        help="Exact old assembled XML for proving unchanged mesh assets after source relocation")
    parser.add_argument("--allow-contact-pair-change", action="store_true",
                   help="Accept a base robot XML that differs from the current one ONLY in <contact> pairs")
    parser.add_argument("--poses-per-scene", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--candidates-per-scene", type=int, default=64)
    parser.add_argument("--max-attempts-per-scene", type=int, default=4096)
    args = parser.parse_args()
    result = build(args)
    print(json.dumps({key: value for key, value in result.items() if key != "scenes"}, indent=2))


if __name__ == "__main__":
    main()
