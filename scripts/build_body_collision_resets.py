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
                       candidates_per_scene=64, max_attempts_per_scene=4096, progress=None):
    """Deterministic rejection sampling with fixed-size checker batches.

    ``check_collisions(scene_ids, qpos)`` returns either [batch] collision flags
    or [batch, shape] flags. Final partial batches repeat their last valid row;
    padding never contributes to counts or accepted poses. Returns the pool,
    per-scene provenance, and failed scene indices. A failed pool is not valid
    for publication. No runtime or scene state is modified.
    """
    if (not scenes or min(poses_per_scene, batch_size, candidates_per_scene) < 1
            or max_attempts_per_scene < poses_per_scene or seed < 0):
        raise ValueError("Invalid reset pool sampling sizes/seed")
    count, nq = len(scenes), len(nominal_qpos)
    pool = np.full((count, poses_per_scene, nq), np.nan, dtype=np.float32)
    selected = np.zeros(count, dtype=np.int64)
    attempts = np.zeros(count, dtype=np.int64)
    clear_count = np.zeros(count, dtype=np.int64)
    generators = [np.random.default_rng(np.random.SeedSequence([seed, index])) for index in range(count)]
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
            flags = np.asarray(check_collisions(padded_ids, padded_qpos), dtype=bool)
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
    failed = np.flatnonzero(selected != poses_per_scene).tolist()
    records = []
    for index, scene in enumerate(scenes):
        record = dict(index=index, scene_id=scene["scene_id"], family=scene.get("family"),
                      attempts=int(attempts[index]), clear_candidates=int(clear_count[index]),
                      selected_poses=int(selected[index]),
                      rejection_fraction=float(1. - clear_count[index] / attempts[index]),
                      complete=bool(selected[index] == poses_per_scene))
        if index in failed:
            record["collision_counts_by_shape"] = collision_by_shape[index].tolist()
        records.append(record)
    return pool, records, failed


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


def build(args):
    import jax
    import mujoco
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
    checker = _make_checker(model, compiled, arrays,
                            max_candidates=int(collision_meta["static_candidate_count"]))
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
        backend=jax.default_backend(), devices=[str(device) for device in jax.devices()],
        source_sha256={name: sha256(ROOT / name) for name in (
            "scripts/build_body_collision_resets.py", "cat_ppo/envs/g1/body_collision.py",
            "scripts/collision_proxy_preview_geometry.py",
            "cat_ppo/furniture/body_collision_geometry.py", "cat_ppo/furniture/body_collision_bank.py",
            "cat_ppo/envs/g1/constants.py", "cat_ppo/furniture/grippers.py")})
    _write_json(output / "preparation.json", provenance)

    def progress(record):
        if record["batches"] == 1 or record["batches"] % 10 == 0:
            _write_json(output / "progress.json", record)
            print(json.dumps(record), flush=True)

    pool, records, failed = collect_reset_pool(
        scenes, constants.DEFAULT_QPOS, soft_lower, soft_upper, checker,
        poses_per_scene=args.poses_per_scene, seed=args.seed, batch_size=args.batch_size,
        candidates_per_scene=args.candidates_per_scene,
        max_attempts_per_scene=args.max_attempts_per_scene, progress=progress)
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
    parser.add_argument("--field-manifest", type=Path, required=True)
    parser.add_argument("--collision-bank", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, default=ROOT / "docs/assets/collision-proxy-proposal-20260916/proposal.json")
    parser.add_argument("--output", type=Path, required=True)
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
