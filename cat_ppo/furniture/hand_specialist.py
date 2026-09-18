"""Lossless, provenance-checked extraction of existing hand-protection scenes.

The source banks remain immutable. Fields and geometry caches are copied, never
regenerated; only collision CSR addresses and scene indices are compacted. The
reset poses retain their original values and within-scene ordering.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import numpy as np

from cat_ppo.furniture.generalist_fields import (
    EXPANDED_SCHEMA, SAMPLING_GROUPS, _json_hash, sha256,
)

SCHEMA = "cat-hand-protection-specialist-v1"
KINDS = {"hand_table_aisle": "furniture", "hand_shelf_passage": "generic_clutter"}
DIFFICULTIES = ("easy", "medium", "hard")


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Specialist provenance path escapes its bank")
    return path


def _hand_indices(manifest):
    return [index for index, scene in enumerate(manifest["scenes"])
            if scene.get("source", {}).get("hand_protection") is not None]


def validate_specialist_manifest(manifest, path=None):
    """Validate the explicit specialist opt-in; verify its source when path is set.

    Structural checks are also usable by the curriculum. The field-bank loader
    must provide its path to prove exact correspondence to an immutable full
    bank, whose own normal loader still verifies all 37 original CAT anchors.
    """
    marker = manifest.get("specialist", {})
    if (manifest.get("schema") != EXPANDED_SCHEMA or marker.get("schema") != SCHEMA
            or marker.get("scope") != "hand_protection"
            or marker.get("selection") != "all-source-hand-protection-scenes"):
        raise ValueError("Invalid hand specialist schema/scope/selection")
    scenes = manifest["scenes"]
    indices, identities = marker.get("source_scene_indices"), marker.get("source_scene_ids")
    if (not isinstance(indices, list) or len(indices) != len(scenes) or not indices
            or any(type(index) is not int or index < 0 for index in indices)
            or indices != sorted(set(indices))
            or identities != [scene["scene_id"] for scene in scenes]
            or manifest.get("scene_count") != len(scenes)):
        raise ValueError("Invalid specialist source scene mapping")
    if any(manifest.get(key) != 0 for key in (
            "original_count", "byte_verified_original_count", "reconstructed_original_count")):
        raise ValueError("Specialist bank must declare zero selected original scenes")
    masses = dict(zip(SAMPLING_GROUPS, (0., 0., .5, .5)))
    if (manifest.get("sampling_group_masses") != masses
            or marker.get("kind_masses") != {name: .5 for name in KINDS}
            or not manifest.get("hand_protection_curriculum")):
        raise ValueError("Specialist requires equal hand-kind masses and the existing curriculum")
    levels = {name: set() for name in KINDS}
    for scene in scenes:
        task = scene.get("source", {}).get("hand_protection", {})
        kind, level = task.get("kind"), task.get("level")
        if (kind not in KINDS or type(level) is not int or level not in range(3)
                or task.get("difficulty") != DIFFICULTIES[level]
                or scene.get("sampling_group") != KINDS[kind]
                or scene.get("family") != KINDS[kind]):
            raise ValueError("Specialist contains a non-hand or malformed hand scene")
        levels[kind].add(level)
    if any(value != {0, 1, 2} for value in levels.values()):
        raise ValueError("Specialist must retain all three levels of both hand kinds")
    for key in ("source_field_sha256", "source_field_content_sha256"):
        fingerprint = marker.get(key, "")
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)):
            raise ValueError("Invalid specialist source fingerprint")
    if path is None:
        return
    source_path = _inside(Path(path).resolve().parent, marker["source_field_manifest"])
    if sha256(source_path) != marker["source_field_sha256"]:
        raise ValueError("Specialist source manifest fingerprint differs")
    source_raw = json.loads(source_path.read_text())
    if "specialist" in source_raw:
        raise ValueError("Specialist source must be a complete generalist bank")
    from cat_ppo.furniture.generalist_fields import load_generalist_manifest
    source = load_generalist_manifest(source_path, verify_files=False)
    if (source["manifest_sha256"] != marker["source_field_content_sha256"]
            or indices != _hand_indices(source)
            or scenes != [source["scenes"][index] for index in indices]):
        raise ValueError("Specialist scenes differ from the complete immutable source selection")


def compact_collision_arrays(arrays, selected):
    """Select complete scene grids, preserving every box and candidate exactly."""
    selected = np.asarray(selected)
    scene_count = len(arrays["scene_box_counts"])
    if (selected.ndim != 1 or not len(selected) or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0) or np.any(selected >= scene_count)
            or len(np.unique(selected)) != len(selected)):
        raise ValueError("Invalid collision scene selection")
    result = {name: [np.array(arrays[name][:1], copy=True)]
              for name in ("centers", "half_sizes", "rotations")}
    result.update(candidate_ids=[np.array(arrays["candidate_ids"][:1], copy=True)],
                  cell_starts=[], cell_counts=[], scene_box_offsets=[], scene_grid_offsets=[])
    box_offset, grid_offset, entry_offset = 1, 0, 1
    for index in selected:
        old_box = int(arrays["scene_box_offsets"][index])
        boxes = int(arrays["scene_box_counts"][index])
        old_grid = int(arrays["scene_grid_offsets"][index])
        cells = int(np.prod(arrays["scene_grid_shapes"][index], dtype=np.int64))
        starts = np.asarray(arrays["cell_starts"][old_grid:old_grid + cells])
        counts = np.asarray(arrays["cell_counts"][old_grid:old_grid + cells])
        if not cells or len(starts) != cells or len(counts) != cells or np.any(counts < 0):
            raise ValueError("Invalid source collision scene grid")
        first, entries = int(starts[0]), int(counts.sum(dtype=np.int64))
        expected = first + np.concatenate(([0], np.cumsum(counts, dtype=np.int64)[:-1]))
        if not np.array_equal(starts, expected):
            raise ValueError("Source collision CSR is not contiguous within a scene")
        ids = np.asarray(arrays["candidate_ids"][first:first + entries])
        if len(ids) != entries or np.any(ids < old_box) or np.any(ids >= old_box + boxes):
            raise ValueError("Source collision candidates reference another scene")
        for name in ("centers", "half_sizes", "rotations"):
            chunk = arrays[name][old_box:old_box + boxes]
            if len(chunk) != boxes:
                raise ValueError("Source collision box range is invalid")
            result[name].append(np.array(chunk, copy=True))
        result["scene_box_offsets"].append(box_offset)
        result["scene_grid_offsets"].append(grid_offset)
        result["cell_starts"].append(starts - first + entry_offset)
        result["cell_counts"].append(counts.copy())
        result["candidate_ids"].append(ids - old_box + box_offset)
        box_offset += boxes
        grid_offset += cells
        entry_offset += entries
    for name in ("centers", "half_sizes", "rotations", "cell_starts", "cell_counts", "candidate_ids"):
        result[name] = np.concatenate(result[name]).astype(arrays[name].dtype, copy=False)
    for name in ("scene_box_offsets", "scene_grid_offsets"):
        result[name] = np.asarray(result[name], dtype=arrays[name].dtype)
    for name in ("scene_box_counts", "scene_grid_origins", "scene_grid_shapes"):
        result[name] = np.array(arrays[name][selected], copy=True)
    result["cell_size"] = np.array(arrays["cell_size"], copy=True)
    return result


def _copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Independent files preserve immutability even if a future source is edited.
    shutil.copy2(source, destination)


def build_hand_specialist_bank(field_manifest, collision_bank, reset_manifest, output):
    """Publish a compact hand-only field/collision/reset bundle; never regenerate."""
    from cat_ppo.furniture.generalist_fields import load_generalist_manifest
    from cat_ppo.furniture.body_collision_bank import ARRAY_NAMES, load_body_collision_bank

    field_manifest, collision_bank, reset_manifest, output = (
        Path(value).resolve() for value in (field_manifest, collision_bank, reset_manifest, output))
    if output.exists():
        raise FileExistsError("Specialist output must be a fresh immutable directory")
    if any(output.is_relative_to(path.parent) for path in (field_manifest, collision_bank, reset_manifest)):
        raise ValueError("Specialist output must be outside each source bank")
    source = load_generalist_manifest(field_manifest, verify_files=False)
    if "specialist" in source:
        raise ValueError("Select a complete source generalist bank")
    selected = _hand_indices(source)
    scenes = [copy.deepcopy(source["scenes"][index]) for index in selected]
    marker = dict(schema=SCHEMA, scope="hand_protection", selection="all-source-hand-protection-scenes",
                  source_field_manifest="_provenance/source-field-manifest.json",
                  source_field_sha256=sha256(field_manifest),
                  source_field_content_sha256=source["manifest_sha256"],
                  source_scene_indices=selected, source_scene_ids=[scene["scene_id"] for scene in scenes],
                  kind_masses={name: .5 for name in KINDS})
    fields = copy.deepcopy(source)
    fields.pop("manifest_sha256")
    fields.update(scenes=scenes, scene_count=len(scenes), original_count=0,
                  byte_verified_original_count=0, reconstructed_original_count=0,
                  sampling_group_masses=dict(zip(SAMPLING_GROUPS, (0., 0., .5, .5))),
                  specialist=marker,
                  fields_bytes=sum(record["size_bytes"] for scene in scenes for record in scene["fields"].values()))
    # Recompute summary counts when inherited from an expanded source builder.
    for key in ("family_counts", "retained_family_counts", "sampling_group_counts"):
        if key in fields:
            attr = "sampling_group" if key == "sampling_group_counts" else "family"
            fields[key] = {value: sum(scene[attr] == value for scene in scenes)
                           for value in sorted({scene[attr] for scene in scenes})}
    generation_keys = ("requested_generated_count", "duplicate_generated_count",
                       "rejected_generated_count", "procedural_generation")
    marker["source_generation"] = {key: fields.pop(key) for key in generation_keys if key in fields}
    validate_specialist_manifest(fields)
    arrays, collision = load_body_collision_bank(collision_bank, expected_field_manifest=field_manifest)
    if collision["scene_count"] != source["scene_count"]:
        raise ValueError("Source collision scene count differs")
    reset = json.loads(reset_manifest.read_text())
    if (reset.get("schema") != "cat-body-collision-clear-reset-pool-v1" or reset.get("status") != "complete"
            or reset.get("field_manifest_sha256") != sha256(field_manifest)
            or reset.get("collision_bank_sha256") != sha256(collision_bank)
            or reset.get("proxy_sha256") != collision["proxy_sha256"]
            or reset.get("scene_count") != source["scene_count"]):
        raise ValueError("Source reset certification belongs to different or incomplete banks")
    pool_path = _inside(reset_manifest.parent, reset["file"])
    if sha256(pool_path) != reset["sha256"]:
        raise ValueError("Source reset array fingerprint differs")
    pool = np.load(pool_path, allow_pickle=False)
    if (pool.ndim != 3 or pool.shape[0] != source["scene_count"] or pool.shape[1] < 1
            or list(pool.shape) != reset["shape"] or str(pool.dtype) != reset["dtype"]
            or not np.isfinite(pool).all()):
        raise ValueError("Source reset array shape/dtype/content differs")
    for index in selected:
        identity = source["scenes"][index]["scene_id"]
        coll_record, reset_record = collision["scenes"][index], reset["scenes"][index]
        if (coll_record.get("index") != index or coll_record["scene_id"] != identity
                or reset_record.get("index") != index or reset_record["scene_id"] != identity
                or reset_record.get("complete") is not True
                or reset_record.get("selected_poses") != pool.shape[1]):
            raise ValueError("Source collision/reset scene identities or certification differ")
    compact = compact_collision_arrays(arrays, selected)
    field_root, collision_root, reset_root = (output / name for name in ("fields", "collision", "resets"))
    output.mkdir(parents=True)
    _copy(field_manifest, field_root / marker["source_field_manifest"])
    for scene in scenes:
        source_dir = _inside(field_manifest.parent, scene["path"])
        destination = _inside(field_root, scene["path"])
        # Copy complete small scene assets to retain renderer/navigation metadata.
        shutil.copytree(source_dir, destination)
    fields["manifest_sha256"] = _json_hash(fields)
    new_field = field_root / "manifest.json"
    _write_json(new_field, fields)
    load_generalist_manifest(new_field, verify_files=True)
    provenance = dict(schema=SCHEMA, source_field_sha256=sha256(field_manifest),
                      source_collision_sha256=sha256(collision_bank), source_reset_sha256=sha256(reset_manifest),
                      source_scene_indices=selected, source_scene_ids=marker["source_scene_ids"],
                      geometry_regenerated=False, fields_changed=False, reset_poses_changed=False)
    new_collision = copy.deepcopy(collision)
    new_collision.pop("manifest_sha256")
    new_collision.update(field_manifest_sha256=sha256(new_field),
                         field_manifest_content_sha256=fields["manifest_sha256"], scene_count=len(selected),
                         obstacle_count=len(compact["centers"]) - 1, grid_cells=len(compact["cell_counts"]),
                         candidate_entries=len(compact["candidate_ids"]) - 1,
                         max_candidates=int(compact["cell_counts"].max(initial=0)),
                         specialist=provenance, arrays={}, scenes=[])
    new_collision["static_candidate_count"] = max(1, new_collision["max_candidates"])
    for new_index, index in enumerate(selected):
        record = copy.deepcopy(collision["scenes"][index])
        record.update(index=new_index, source_index=index)
        cache = _inside(collision_bank.parent, record["geometry_file"])
        if sha256(cache) != record["geometry_sha256"]:
            raise ValueError("Source collision geometry cache fingerprint differs")
        _copy(cache, _inside(collision_root, record["geometry_file"]))
        new_collision["scenes"].append(record)
    for name in ARRAY_NAMES:
        path = collision_root / (name + ".npy")
        np.save(path, compact[name], allow_pickle=False)
        new_collision["arrays"][name] = dict(file=path.name, sha256=sha256(path),
                                             shape=list(compact[name].shape), dtype=str(compact[name].dtype),
                                             bytes=compact[name].nbytes)
    total_bytes = sum(record["bytes"] for record in new_collision["arrays"].values())
    new_collision.update(shared_array_bytes=total_bytes, estimated_shared_array_bytes=total_bytes)
    new_collision["manifest_sha256"] = _json_hash(new_collision)
    new_collision_path = collision_root / "manifest.json"
    _write_json(new_collision_path, new_collision)
    load_body_collision_bank(new_collision_path, expected_field_manifest=new_field,
                             expected_proxy_sha256=collision["proxy_sha256"])
    reset_root.mkdir(parents=True)
    new_pool = np.array(pool[selected], copy=True)
    new_pool_path = reset_root / "qpos.npy"
    np.save(new_pool_path, new_pool, allow_pickle=False)
    new_reset = copy.deepcopy(reset)
    new_reset["source_reset_generation"] = {key: new_reset.pop(key) for key in (
        "append_base", "generated_scene_count", "generated_scene_index_offset") if key in new_reset}
    records = []
    for new_index, index in enumerate(selected):
        record = copy.deepcopy(reset["scenes"][index])
        record.update(index=new_index, source_index=index)
        records.append(record)
    new_reset.update(field_manifest_sha256=sha256(new_field), field_manifest=str(new_field),
                     collision_bank_sha256=sha256(new_collision_path), collision_bank=str(new_collision_path),
                     scene_count=len(selected), file=new_pool_path.name, sha256=sha256(new_pool_path),
                     shape=list(new_pool.shape), dtype=str(new_pool.dtype), scenes=records,
                     validated_pose_count=int(np.prod(new_pool.shape[:2])), specialist=provenance)
    _write_json(reset_root / "manifest.json", new_reset)
    for name, source_path in (("collision", collision_bank), ("resets", reset_manifest)):
        _copy(source_path, output / "_provenance" / ("source-" + name + "-manifest.json"))
    result = dict(schema=SCHEMA, status="complete", **{key: value for key, value in provenance.items() if key != "schema"},
                  scene_count=len(scenes), kind_counts={name: sum(
                      scene["source"]["hand_protection"]["kind"] == name for scene in scenes) for name in KINDS},
                  field_manifest=str(new_field), collision_bank=str(new_collision_path),
                  reset_manifest=str(reset_root / "manifest.json"),
                  field_manifest_sha256=sha256(new_field), collision_bank_sha256=sha256(new_collision_path),
                  reset_manifest_sha256=sha256(reset_root / "manifest.json"),
                  fields_bytes=fields["fields_bytes"], collision_array_bytes=total_bytes)
    _write_json(output / "manifest.json", result)
    return result
