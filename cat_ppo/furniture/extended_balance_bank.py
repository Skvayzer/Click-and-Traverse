"""Bank composition that can grow, without letting composition change silently.

Every shipped bank so far pins its contents by an exact identity. ``v2`` is
literally ``v1.scenes + 24 pinned objectives``; appending even one scene fails
validation. That is the right property — a checkpoint's contract records the
bank hash, and a bank whose contents could drift would make those contracts
meaningless — but it also means no bank could ever be extended.

This schema keeps the property and drops the ceiling. An extended bank is:

    parent.scenes  ++  addition[0].scenes  ++  addition[1].scenes  ++  ...

The parent is pinned by hash and revalidated in full, so it must still be a
legal bank in its own right. Each addition is a separate manifest pinned by its
own hash, appended in declared order. The parent's records must appear as an
exact immutable prefix, so extending can only ever add, never rewrite or
reorder what a previous checkpoint trained on.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "cat-extended-balance-v1"


def validate(manifest, *, path=None, verify_files=False):
    from .generalist_fields import load_generalist_manifest, sha256

    marker = manifest["flat_balance"]
    if marker.get("schema") != SCHEMA:
        raise ValueError("Expected an extended balance marker")
    if any(key in manifest for key in ("width_curriculum", "hand_protection_curriculum",
                                       "hand_posture_upgrade", "contrastive_specialist",
                                       "specialist", "external_retention_bank")):
        raise ValueError("Extended balance cannot inherit another sampler or curriculum")

    pin = marker["parent"]
    parent_path = Path(pin["manifest"])
    if sha256(parent_path) != pin["sha256"]:
        raise ValueError("Extended balance parent pin differs")
    parent = load_generalist_manifest(parent_path, verify_files=False)

    scenes = manifest["scenes"]
    count = len(parent["scenes"])
    if scenes[:count] != parent["scenes"]:
        raise ValueError("Extended balance must preserve the parent bank as an exact prefix")

    offset = count
    additions = marker.get("additions")
    if not isinstance(additions, list) or not additions:
        raise ValueError("Extended balance must declare at least one pinned addition")
    for addition in additions:
        addition_path = Path(addition["manifest"])
        if sha256(addition_path) != addition["sha256"]:
            raise ValueError(f"Addition pin differs: {addition.get('kind')}")
        # Read as plain JSON, not through load_generalist_manifest: an addition is a
        # fragment, not a standalone bank, so it has no 37 original CAT anchors. Its
        # integrity comes from the sha256 pin above, and every record it contributes is
        # still validated by the ordinary per-scene checks in the caller.
        added = json.loads(addition_path.read_text())["scenes"]
        if len(added) != addition["count"]:
            raise ValueError(f"Addition count differs: {addition.get('kind')}")
        if scenes[offset:offset + len(added)] != added:
            raise ValueError(f"Addition records differ from their pinned manifest: {addition.get('kind')}")
        offset += len(added)
    if offset != len(scenes):
        raise ValueError("Extended balance carries scenes outside the parent and its pinned additions")

    parent_marker = parent["flat_balance"]
    if marker.get("retention_storage_root") != parent_marker.get("retention_storage_root"):
        raise ValueError("Retention storage root differs from the parent bank")
    if marker.get("settings") != parent_marker.get("settings") or marker.get("masses") != parent_marker.get("masses"):
        raise ValueError("Extended balance must inherit the parent's balance settings and masses")
    if len({scene["scene_id"] for scene in scenes}) != len(scenes):
        raise ValueError("Duplicate scene identity")
    return marker
