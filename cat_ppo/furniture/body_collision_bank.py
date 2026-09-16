"""Shared obstacle OBBs and a conservative, untruncated 3-D collision index.

CAT occupancy is interpreted at the *runtime/render* sample coordinates:
voxel i is centered at record.origin + i * dx. This intentionally retains the
released half-cell difference from the generator's nominal coordinates.
Furniture retains its canonical scene.json boxes. No floor is added.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cat_ppo.furniture.generalist_fields import (
    DATASET_REPO, DATASET_REVISION, _download_file, _json_hash,
    load_generalist_manifest, sha256,
)
from cat_ppo.furniture.legacy_scenes import merge_occupied_voxels

SCHEMA = "cat-body-collision-bank-v1"
BUILDER = "canonical-OBB-or-lossless-runtime-centered-voxels-v1"
OUTWARD_EPSILON_M = 1e-4
ARRAY_NAMES = ("centers", "half_sizes", "rotations", "scene_box_offsets", "scene_box_counts",
               "scene_grid_origins", "scene_grid_shapes", "scene_grid_offsets",
               "cell_starts", "cell_counts", "candidate_ids", "cell_size")


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def proxy_bounding_radius(proposal):
    """Exact enclosing radius about each proxy's own geometric center."""
    radii = []
    for shape in proposal["shapes"]:
        if shape["kind"] == "box":
            radius = np.linalg.norm(shape["half_size"])
        elif shape["kind"] == "sphere":
            radius = shape["radius"]
        elif shape["kind"] == "capsule":
            first, last = np.asarray(shape["endpoints"], dtype=float)
            if not np.allclose((first + last) / 2, shape["center"], atol=1e-10, rtol=0):
                raise ValueError("Capsule center must be the segment midpoint")
            radius = np.linalg.norm(last - first) / 2 + shape["radius"]
        else:
            raise ValueError("Unknown proxy kind")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Invalid proxy enclosing radius")
        radii.append(float(radius))
    if not radii:
        raise ValueError("An empty proxy proposal cannot define index coverage")
    return max(radii)


def boxes_from_occupancy(occupancy, sample_origin, dx):
    """Losslessly merge occupied cells; do not shift to generator coordinates."""
    occupancy = np.asarray(occupancy)
    if occupancy.ndim != 3 or not np.isin(occupancy, (0, 1)).all():
        raise ValueError("Expected binary xyz occupancy")
    if not math.isfinite(dx) or dx <= 0:
        raise ValueError("Invalid voxel size")
    edge_origin = np.asarray(sample_origin, dtype=np.float64) - dx / 2
    cuboids = merge_occupied_voxels(occupancy)
    coverage = np.zeros(occupancy.shape, dtype=np.uint8)
    centers, halves = [], []
    for begin, end in cuboids:
        coverage[tuple(slice(a, b) for a, b in zip(begin, end))] += 1
        low = edge_origin + np.asarray(begin) * dx
        high = edge_origin + np.asarray(end) * dx
        centers.append((low + high) / 2)
        halves.append((high - low) / 2)
    if not np.array_equal(coverage, occupancy):
        raise AssertionError("Voxel cuboids overlap or change the occupied union")
    return (np.asarray(centers, dtype=np.float64).reshape(-1, 3),
            np.asarray(halves, dtype=np.float64).reshape(-1, 3),
            np.broadcast_to(np.eye(3), (len(centers), 3, 3)).copy())


def canonical_room_boxes(scene):
    centers, halves, rotations = [], [], []
    for box in scene["boxes"]:
        center, half = np.asarray(box["center"], dtype=float), np.asarray(box["half_size"], dtype=float)
        if (center.shape != (3,) or half.shape != (3,) or not np.isfinite(center).all()
                or not np.isfinite(half).all() or np.any(half <= 0)
                or not math.isfinite(box["yaw"])):
            raise ValueError("Invalid canonical room box")
        if box.get("category") == "floor":
            raise ValueError("The canonical obstacle bank must not contain a floor")
        c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
        centers.append(center); halves.append(half)
        rotations.append(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    return (np.asarray(centers, dtype=np.float64).reshape(-1, 3),
            np.asarray(halves, dtype=np.float64).reshape(-1, 3),
            np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3))


def build_scene_index(centers, half_sizes, rotations, *, query_radius, cell_size=.25,
                      outward_epsilon=OUTWARD_EPSILON_M):
    """CSR cell lists, including every expanded obstacle AABB/cell overlap.

    Any proxy contained in a sphere of query_radius about its queried center
    can intersect only obstacles listed in that center's cell. Inclusive upper
    bounds and outward epsilon cover exact cell boundaries and float32 rounding.
    IDs here are local zero-based IDs; bank assembly adds the sentinel offset.
    """
    centers, half_sizes, rotations = (np.asarray(x, dtype=np.float64)
                                    for x in (centers, half_sizes, rotations))
    if (centers.shape != half_sizes.shape or centers.ndim != 2 or centers.shape[1] != 3
            or rotations.shape != (len(centers), 3, 3)
            or not all(np.isfinite(x).all() for x in (centers, half_sizes, rotations))
            or np.any(half_sizes <= 0)):
        raise ValueError("Invalid obstacle geometry")
    if (not all(math.isfinite(x) and x > 0 for x in (query_radius, cell_size, outward_epsilon))):
        raise ValueError("Invalid spatial-index dimensions")
    if len(centers):
        extent = np.einsum("nij,nj->ni", np.abs(rotations), half_sizes)
        low = centers - extent - query_radius - outward_epsilon
        high = centers + extent + query_radius + outward_epsilon
        origin = np.floor(low.min(axis=0) / cell_size) * cell_size
        shape = np.floor((high.max(axis=0) - origin) / cell_size).astype(np.int64) + 1
    else:
        low = high = np.empty((0, 3)); origin = np.zeros(3); shape = np.ones(3, np.int64)
    size = int(np.prod(shape))
    if size > np.iinfo(np.int32).max:
        raise ValueError("Scene grid exceeds int32 indexing")
    rows = [[] for _ in range(size)]
    for box_id, (lower, upper) in enumerate(zip(low, high)):
        begin = np.floor((lower - origin) / cell_size).astype(np.int64)
        end = np.floor((upper - origin) / cell_size).astype(np.int64) + 1
        for x in range(begin[0], end[0]):
            for y in range(begin[1], end[1]):
                offset = (x * shape[1] + y) * shape[2]
                for z in range(begin[2], end[2]):
                    rows[offset + z].append(box_id)
    counts = np.fromiter((len(row) for row in rows), dtype=np.int32, count=size)
    starts = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)[:-1])).astype(np.int64)
    ids = np.fromiter((i for row in rows for i in row), dtype=np.int32,
                      count=int(counts.sum(dtype=np.int64)))
    return dict(origin=origin, shape=shape.astype(np.int32), starts=starts,
                counts=counts, ids=ids, max_candidates=int(counts.max(initial=0)))


def _extract_scene(payload):
    index, record, bank_root, output, download = payload
    directory = Path(bank_root) / record["path"]
    output = Path(output)
    if not directory.resolve().is_relative_to(Path(bank_root).resolve()):
        raise ValueError("Source scene escapes its bank")
    source_path = directory / "source.json"
    if sha256(source_path) != record["source"]["metadata_sha256"]:
        raise ValueError("Scene provenance fingerprint differs")
    source = json.loads(source_path.read_text())
    if record["family"] in ("furniture", "generic_clutter"):
        path = directory / "scene.json"
        if sha256(path) != record["scene_sha256"]:
            raise ValueError("Canonical room geometry fingerprint differs")
        geometry = canonical_room_boxes(json.loads(path.read_text()))
        provenance = dict(kind="canonical-room-OBBs", source_sha256=sha256(path),
                          exact_canonical_box_count=len(geometry[0]))
    else:
        path = directory / "obs.npy"
        expected = source.get("occupancy_sha256")
        if source.get("arrays_unchanged", record["source"].get("arrays_unchanged", False)):
            if source.get("repo") != DATASET_REPO or source.get("revision") != DATASET_REVISION:
                raise ValueError("Released occupancy source pin differs")
            released = source["files"]["obs.npy"]
            expected = released.get("lfs", {}).get("oid")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("Released occupancy lacks immutable SHA256")
            relative = released["path"]
            if (not relative.startswith(("assets_v0/RandObs/", "assets_v1/TypiObs/"))
                    or Path(relative).name != "obs.npy" or ".." in Path(relative).parts):
                raise ValueError("Unexpected released occupancy path")
            if not path.exists():
                path = output / "downloaded_occupancy" / str(index) / "obs.npy"
                if download:
                    _download_file(relative, path, expected)
        if not path.is_file() or (expected and sha256(path) != expected):
            raise ValueError(f"Missing/corrupt source occupancy {path}; use --download")
        occupancy = np.load(path, allow_pickle=False)
        sdf_path = directory / "sdf.npy"
        if sha256(sdf_path) != record["fields"]["sdf"]["sha256"]:
            raise ValueError("Source SDF fingerprint differs")
        sdf = np.load(sdf_path, allow_pickle=False)
        if (list(occupancy.shape) != record["shape"] or not np.isfinite(sdf).all()
                or not np.array_equal(occupancy.astype(bool), sdf < 0)):
            raise ValueError("Occupancy is inconsistent with the exact training SDF")
        geometry = boxes_from_occupancy(occupancy, record["origin"], record["dx"])
        provenance = dict(kind="lossless-occupied-voxel-cuboids", source_sha256=sha256(path),
                          sdf_sha256=sha256(sdf_path), occupied_voxels=int(occupancy.sum()),
                          voxel_size_m=record["dx"], sample_origin=record["origin"],
                          edge_origin=(np.asarray(record["origin"]) - record["dx"] / 2).tolist(),
                          sdf_sign_identity=True, lossless_occupied_union_verified=True,
                          dataset_revision=source.get("revision"),
                          original_source_kind=source.get("kind", "released-original"))
    cache = output / "geometry" / f"{index:05d}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Small geometry caches only; the multi-GB field bank is never copied.
    np.savez(cache, centers=geometry[0], half_sizes=geometry[1], rotations=geometry[2])
    return dict(index=index, scene_id=record["scene_id"], family=record["family"],
                boxes=len(geometry[0]), geometry_file=str(cache.relative_to(output)),
                geometry_sha256=sha256(cache), provenance=provenance)


def build_body_collision_bank(field_manifest, proposal_path, output, *, cell_size=.25,
                              workers=4, download=False, progress=None, inventory_only=False):
    """Build provenance-verified geometry, measure exact K, then publish last."""
    field_manifest, proposal_path, output = map(lambda p: Path(p).resolve(),
                                                (field_manifest, proposal_path, output))
    if output == field_manifest.parent or output.is_relative_to(field_manifest.parent):
        raise ValueError("Write the collision bank outside its source field bank")
    if (output / "manifest.json").exists():
        raise FileExistsError("A published collision bank is immutable; use a fresh output path")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    manifest = load_generalist_manifest(field_manifest, verify_files=False)
    proposal = json.loads(proposal_path.read_text())
    radius = proxy_bounding_radius(proposal)
    key = dict(builder=BUILDER, field_manifest_sha256=sha256(field_manifest),
               field_manifest_content_sha256=manifest["manifest_sha256"],
               proxy_sha256=sha256(proposal_path), query_radius_m=radius, cell_size_m=cell_size,
               outward_epsilon_m=OUTWARD_EPSILON_M)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "build-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != key:
        raise ValueError("Partial collision bank belongs to a different build")
    _write_json(plan_path, key)
    inventory_path = output / "geometry-inventory.json"
    if inventory_path.exists():
        scenes = json.loads(inventory_path.read_text())["scenes"]
        if len(scenes) != manifest["scene_count"] or any(
                sha256(output / s["geometry_file"]) != s["geometry_sha256"] for s in scenes):
            raise ValueError("Cached geometry inventory differs")
    else:
        scenes = [None] * manifest["scene_count"]
        payloads = [(i, record, str(field_manifest.parent), str(output), download)
                    for i, record in enumerate(manifest["scenes"])]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_extract_scene, payload) for payload in payloads]
            completed = 0
            for future in as_completed(futures):
                result = future.result(); scenes[result["index"]] = result; completed += 1
                if progress and (completed % 100 == 0 or completed == len(scenes)):
                    progress(dict(phase="geometry", completed=completed, total=len(scenes)))
        _write_json(inventory_path, dict(build=key, scenes=scenes))
    all_geometry = [[], [], []]
    indexes = []
    box_count, cells, entries, max_k = 1, 0, 1, 0  # Sentinel box/entry at index zero.
    offsets, counts = [], []
    for i, scene in enumerate(scenes):
        with np.load(output / scene["geometry_file"], allow_pickle=False) as data:
            geometry = [data[name] for name in ("centers", "half_sizes", "rotations")]
        index = build_scene_index(*geometry, query_radius=radius, cell_size=cell_size)
        offsets.append(box_count); counts.append(len(geometry[0])); box_count += len(geometry[0])
        for chunks, array in zip(all_geometry, geometry): chunks.append(array)
        index["cell_offset"] = cells; index["entry_offset"] = entries
        cells += len(index["counts"]); entries += len(index["ids"])
        max_k = max(max_k, index["max_candidates"]); indexes.append(index)
        scene.update(grid_shape=index["shape"].tolist(), grid_cells=len(index["counts"]),
                     candidate_entries=len(index["ids"]), max_candidates=index["max_candidates"])
        if progress and (i % 100 == 0 or i + 1 == len(scenes)):
            progress(dict(phase="index", completed=i + 1, total=len(scenes), max_k=max_k,
                          boxes=box_count - 1, cells=cells, candidate_entries=entries - 1))
    if max(cells, entries, box_count) >= np.iinfo(np.int32).max:
        raise ValueError("Bank exceeds int32 addressing; no truncation is permitted")
    summary = dict(schema=SCHEMA, **key, scene_count=len(scenes), shape_count=len(proposal["shapes"]),
                   obstacle_count=box_count - 1, grid_cells=cells, candidate_entries=entries - 1,
                   max_candidates=max_k, static_candidate_count=max(1, max_k),
                   floor_included=False, geometry_mapping="CAT voxel centers at runtime/render origin+i*dx; canonical room OBBs",
                   index="ragged 3-D scene grids and CSR obstacle IDs; inclusive expanded AABB coverage",
                   geometry_float32="centers/rotations rounded; half sizes outward-inflated by 2e-6 m plus center rounding error",
                   scenes=scenes)
    # CSR avoids padding every cell to the global worst-case candidate count.
    estimated = box_count * 15 * 4 + len(scenes) * (3 + 3 + 1 + 1 + 1) * 4 + cells * 8 + entries * 4 + 4
    summary["estimated_shared_array_bytes"] = estimated
    _write_json(output / "inventory.json", summary)
    if inventory_only:
        return summary
    arrays = {}
    center = np.concatenate([np.zeros((1, 3)), *all_geometry[0]])
    half = np.concatenate([np.zeros((1, 3)), *all_geometry[1]])
    rotation = np.concatenate([np.eye(3)[None], *all_geometry[2]])
    arrays["centers"] = center.astype(np.float32)
    center_error = np.einsum("nji,nj->ni", np.abs(rotation), np.abs(center - arrays["centers"]))
    arrays["half_sizes"] = np.nextafter((half + center_error + 2e-6).astype(np.float32), np.float32(np.inf))
    arrays["half_sizes"][0] = 0
    arrays["rotations"] = rotation.astype(np.float32)
    arrays["scene_box_offsets"] = np.asarray(offsets, np.int32)
    arrays["scene_box_counts"] = np.asarray(counts, np.int32)
    arrays["scene_grid_origins"] = np.asarray([x["origin"] for x in indexes], np.float32)
    arrays["scene_grid_shapes"] = np.asarray([x["shape"] for x in indexes], np.int32)
    arrays["scene_grid_offsets"] = np.asarray([x["cell_offset"] for x in indexes], np.int32)
    arrays["cell_starts"] = np.concatenate([x["starts"] + x["entry_offset"] for x in indexes]).astype(np.int32)
    arrays["cell_counts"] = np.concatenate([x["counts"] for x in indexes])
    arrays["candidate_ids"] = np.concatenate([np.zeros(1, np.int32), *[
        x["ids"] + offset for x, offset in zip(indexes, offsets)]])
    arrays["cell_size"] = np.asarray(cell_size, np.float32)
    summary["arrays"] = {}
    for name in ARRAY_NAMES:
        path = output / (name + ".npy")
        np.save(path, arrays[name], allow_pickle=False)
        summary["arrays"][name] = dict(file=path.name, sha256=sha256(path),
                                      shape=list(arrays[name].shape), dtype=str(arrays[name].dtype),
                                      bytes=arrays[name].nbytes)
    summary["shared_array_bytes"] = sum(x["bytes"] for x in summary["arrays"].values())
    summary["manifest_sha256"] = _json_hash(summary)
    _write_json(output / "manifest.json", summary)
    return summary


def load_body_collision_bank(path, *, expected_field_manifest=None, expected_proxy_sha256=None):
    """Return (NumPy array dict, metadata dict), verifying every stored hash."""
    path = Path(path).resolve()
    if path.is_dir(): path = path / "manifest.json"
    metadata = json.loads(path.read_text())
    expected = metadata.pop("manifest_sha256")
    if metadata.get("schema") != SCHEMA or _json_hash(metadata) != expected:
        raise ValueError("Collision-bank manifest hash/schema differs")
    metadata["manifest_sha256"] = expected
    if expected_field_manifest is not None and sha256(expected_field_manifest) != metadata["field_manifest_sha256"]:
        raise ValueError("Collision bank was built for a different field manifest")
    if expected_proxy_sha256 is not None and expected_proxy_sha256 != metadata["proxy_sha256"]:
        raise ValueError("Collision bank was built for different robot shapes")
    arrays = {}
    for name in ARRAY_NAMES:
        record = metadata["arrays"][name]
        file = (path.parent / record["file"]).resolve()
        if not file.is_relative_to(path.parent) or sha256(file) != record["sha256"]:
            raise ValueError("Collision-bank array path/hash differs")
        array = np.load(file, allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError("Collision-bank array shape/dtype differs")
        arrays[name] = array
    if int(arrays["cell_counts"].max(initial=0)) != metadata["max_candidates"]:
        raise ValueError("Static collision candidate capacity is inconsistent")
    return arrays, metadata


def lookup_collision_candidates(arrays, scene_id, shape_centers, *, max_candidates):
    """JAX lookup for one scene and any leading center dimensions; returns IDs/mask."""
    import jax.numpy as jp

    shape_centers = jp.asarray(shape_centers)
    shape = arrays["scene_grid_shapes"][scene_id]
    index = jp.floor((shape_centers - arrays["scene_grid_origins"][scene_id]) / arrays["cell_size"]).astype(jp.int32)
    inside = jp.all((index >= 0) & (index < shape), axis=-1)
    index = jp.clip(index, 0, shape - 1)
    cell = arrays["scene_grid_offsets"][scene_id] + (index[..., 0] * shape[1] + index[..., 1]) * shape[2] + index[..., 2]
    count, start = arrays["cell_counts"][cell], arrays["cell_starts"][cell]
    offset = jp.arange(max_candidates)
    address = jp.minimum(start[..., None] + offset, arrays["candidate_ids"].shape[0] - 1)
    valid = inside[..., None] & (offset < count[..., None])
    ids = jp.where(valid, arrays["candidate_ids"][address], 0)
    return ids, valid
