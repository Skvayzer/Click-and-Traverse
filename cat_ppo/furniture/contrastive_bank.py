"""Explicit standalone contrastive specialist banks and role-balanced sampling."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

ROLES = ("open", "forward_protected", "narrow", "transition")
MARKER = "hand-contrast-specialist-v1"
TABLE_MARKER = "hand-contrast-specialist-v2"
GROUP_ROLES = {"cabinet": set(ROLES), "table_edges": {"open", "forward_protected"}}


def validate_contrastive_manifest(manifest, *, path=None, verify_files=False):
    marker = manifest.get("contrastive_specialist", {})
    if (marker.get("schema") not in (MARKER, TABLE_MARKER) or manifest.get("schema") != "cat-generalist-field-bank-v2"
            or "specialist" in manifest or manifest.get("hand_protection_curriculum")
            or any(manifest.get(key) != 0 for key in
                   ("original_count", "byte_verified_original_count", "reconstructed_original_count"))):
        raise ValueError("Invalid explicit contrastive specialist bank")
    masses=marker.get('role_reset_masses',{})
    if set(masses)!=set(ROLES) or any(not np.isfinite(v) or v<=0 for v in masses.values()) or not np.isclose(sum(masses.values()),1.):
        raise ValueError('Role reset masses must be positive and sum to one')
    from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
    from cat_ppo.furniture.room_navigation import scene_navigation_radius
    groups, seen_seeds, families = defaultdict(dict), {}, {}
    for record in manifest["scenes"]:
        contrast = record.get("source", {}).get("hand_contrast", {})
        if contrast.get('certificate_semantics')=='width-curriculum-scaffold-v1':
            raise ValueError('Width scaffolds require their explicit curriculum manifest, not a specialist label')
        group, role = contrast.get("group_id"), contrast.get("role")
        if (record.get("family") != "generic_clutter" or role not in ROLES
                or not isinstance(group, str) or role in groups[group]
                or record.get("dx") != .04):
            raise ValueError("Invalid or repeated contrastive group/role")
        family = contrast.get("geometry_family", "cabinet")
        if (family not in GROUP_ROLES or (family != "cabinet" and marker["schema"] != TABLE_MARKER)
                or (group in families and families[group] != family)):
            raise ValueError("Invalid or inconsistent contrastive geometry family")
        families[group] = family
        groups[group][role] = record
        pack_hand_contrast([{"hand_contrast": contrast}])
        cert = contrast.get("certificate", {})
        if (cert.get("schema") != "hand-contrast-certificate-v1"
                or cert.get("route_transition_validated") is not True
                or cert.get("sampled_kinematic_stances_validated") is not True
                or cert.get("primitive_count") != 35 or cert.get("voxel_size_m") != .04
                or cert.get("exterior_lanes_sealed") is not True):
            raise ValueError("Contrastive scenes require geometry and stance certificates")
        if family == "table_edges":
            clearance = cert.get("raised_hand_above_table_min_m", float("nan"))
            if (not isinstance(clearance, (int, float)) or not np.isfinite(clearance) or clearance <= 0
                    or cert.get("tabletop_geometry_validated") is not True):
                raise ValueError("Table scenes require raised hand clearance above the tabletop")
            if role == "forward_protected" and any(zone["region_valid"] != [True, False]
                    for zone in contrast["zones"]):
                raise ValueError("Table hand hazards require raised-only target regions")
        if verify_files:
            directory = (Path(path).parent / record["path"]).resolve()
            if not directory.is_relative_to(Path(path).parent.resolve()):
                raise ValueError("Contrastive scene path escapes bank")
            scene = json.loads((directory / "scene.json").read_text())
            if scene.get("hand_contrast") != contrast or scene.get("scene_id") != record["scene_id"]:
                raise ValueError("Contrastive source metadata differs from canonical scene")
            scene_navigation_radius(scene)
            seed_split = (family, scene["seed"], scene["split"])
            if group in seen_seeds and seen_seeds[group] != seed_split:
                raise ValueError("Matched group variants must share seed and split")
            seen_seeds[group] = seed_split
    if not groups or any(set(group) != GROUP_ROLES[families[name]] for name, group in groups.items()):
        raise ValueError("Each contrastive group must contain every role required by its geometry family")
    if marker["schema"] == TABLE_MARKER:
        actual = {family: sum(value == family for value in families.values()) for family in GROUP_ROLES}
        if marker.get("geometry_group_counts") != actual or not all(actual.values()):
            raise ValueError("Mixed contrastive bank must retain cabinet groups and add table pairs")
        from cat_ppo.furniture.generalist_fields import _json_hash
        preserved = marker.get("preserved_cabinet_scene_count")
        if (preserved != actual["cabinet"] * 4
                or any(s["source"]["hand_contrast"].get("geometry_family", "cabinet") != "cabinet"
                       for s in manifest["scenes"][:preserved])
                or _json_hash(manifest["scenes"][:preserved]) != marker.get("preserved_scene_records_sha256")):
            raise ValueError("Preserved cabinet scene records differ from the pinned source bank")
    if marker.get("group_count") != len(groups):
        raise ValueError("Contrastive group count differs")
    if verify_files and len({(family, seed) for family, seed, split in seen_seeds.values()}) != len(groups):
        raise ValueError("Seeds may not leak across contrastive groups or splits")


def contrastive_roles(manifest):
    if 'width_curriculum' in manifest:
        from .width_curriculum import validate_width_manifest
        validate_width_manifest(manifest)
        return np.asarray([ROLES.index(s['source']['hand_contrast']['role']) if s.get('source',{}).get('hand_contrast') else -1 for s in manifest['scenes']],np.int32)
    if "contrastive_specialist" not in manifest:
        return None
    validate_contrastive_manifest(manifest)
    return np.asarray([ROLES.index(s["source"]["hand_contrast"]["role"])
                       for s in manifest["scenes"]], np.int32)


def role_balanced_logits(weights, roles, masses=None):
    """Configured role masses; the legacy default remains 25% each."""
    import jax.numpy as jp
    masses = jp.asarray([.25]*4 if masses is None else masses, dtype=weights.dtype)
    totals = jp.zeros(len(masses), weights.dtype).at[roles].add(weights)
    probabilities = masses[roles] * weights / jp.maximum(totals[roles], 1e-20)
    return jp.where(probabilities > 0, jp.log(jp.maximum(probabilities, 1e-30)), -jp.inf)


def build_contrastive_bank(output, *, seed=20260919, groups=16, split="train", progress=print):
    from cat_ppo.furniture.contrastive_passages import generate_contrastive_group
    from cat_ppo.furniture.generalist_fields import (
        DATASET_REVISION, EXPANDED_SCHEMA, RELEASED_CONFIG_SHA256, _json_hash,
        load_generalist_manifest, make_clutter_fields, sha256,
    )
    if type(groups) is not int or groups < 1:
        raise ValueError("Expected at least one matched group")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "manifest.json"
    if destination.exists():
        existing = load_generalist_manifest(destination)
        if (existing["seed"], existing["contrastive_specialist"]["group_count"], existing["split"]) != (seed, groups, split):
            raise ValueError("Existing immutable bank uses different arguments")
        return destination
    records = []
    for group_seed in range(seed, seed + groups):
        cached = sorted((output / "scenes").glob(f"contrast-{split}-{group_seed:06d}-*/scene-input.json"))
        if len(cached) == 4:
            scenes = [json.loads(p.read_text()) for p in cached]
            scenes.sort(key=lambda scene: ROLES.index(scene["hand_contrast"]["role"]))
        else:
            scenes = generate_contrastive_group(group_seed, split=split)
        for scene in scenes:
            relative = Path("scenes") / scene["scene_id"]
            directory = output / relative
            directory.mkdir(parents=True, exist_ok=True)
            record_path = directory / "field-record.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                from cat_ppo.furniture.generalist_fields import _field_records
                if (record["fields"] != _field_records(directory)
                        or record["scene_sha256"] != sha256(directory / "scene.json")
                        or record["source"]["metadata_sha256"] != sha256(directory / "source.json")):
                    raise ValueError("Cached field bytes changed")
                if json.loads((directory / "scene.json").read_text())["hand_contrast"] != scene["hand_contrast"]:
                    raise ValueError("Cached contrastive geometry metadata changed")
            else:
                record = make_clutter_fields(scene, directory, dx=.04)
                source = dict(record["source"], kind="contrastive-hand-passage", hand_contrast=scene["hand_contrast"])
                (directory / "source.json").write_text(json.dumps(source, indent=2) + "\n")
                source["metadata_sha256"] = sha256(directory / "source.json")
                record.update(path=str(relative), source=source, task_kind="room", reset_mode="room",
                    crossed_mode="goal_radius", episode_length=4000, sampling_group="generic_clutter")
                record_path.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            progress(f"Prepared {len(records)}/{groups * 4}: {scene['scene_id']}")
    manifest = dict(schema=EXPANDED_SCHEMA, dataset_revision=DATASET_REVISION,
        released_config_sha256=RELEASED_CONFIG_SHA256, seed=seed, split=split,
        original_count=0, byte_verified_original_count=0, reconstructed_original_count=0,
        scene_count=len(records), scenes=records,
        sampling_group_masses=dict(original_cat=0., procedural_cat=0., furniture=0., generic_clutter=1.),
        contrastive_specialist=dict(schema=MARKER, group_count=groups,
            role_reset_masses=dict.fromkeys(ROLES, .25),
            purpose="Matched forward/hand posture specialist; not a full generalist bank",
            split_unit="matched seed group", training_only=True),
        fields_bytes=sum(f["size_bytes"] for s in records for f in s["fields"].values()))
    validate_contrastive_manifest(manifest, path=destination, verify_files=True)
    manifest["manifest_sha256"] = _json_hash(manifest)
    temporary = output / "manifest.pending.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    load_generalist_manifest(temporary)
    temporary.replace(destination)
    return destination
