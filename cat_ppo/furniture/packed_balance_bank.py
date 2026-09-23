"""A bank that is a faithful re-encoding of another bank, pinned to its source.

Packing changes how fields are stored, never which scenes a bank contains. So a
packed bank must prove exactly that: identical scene records, in identical
order, from a source manifest pinned by hash. The only permitted difference is
the retention storage root, which has to point at the packed bank itself
because every field file now lives inside it.

Decoding happens in SceneBank, so nothing downstream can tell the difference.
"""
from __future__ import annotations

from pathlib import Path

SCHEMA = "cat-packed-balance-v1"


def validate(manifest, *, path=None, verify_files=False):
    from .generalist_fields import load_generalist_manifest, sha256

    marker = manifest["flat_balance"]
    if marker.get("schema") != SCHEMA:
        raise ValueError("Expected a packed balance marker")
    if any(key in manifest for key in ("width_curriculum", "hand_protection_curriculum",
                                       "hand_posture_upgrade", "contrastive_specialist",
                                       "specialist", "external_retention_bank")):
        raise ValueError("Packed balance cannot inherit another sampler or curriculum")

    # Provenance is recorded, not re-derived. Re-validating the source at load time
    # would drag in the source's whole ancestor chain -- its parents, their retention
    # pins -- and a packed bank exists precisely so those raw inputs can be retired.
    # The claim "these are the source's scenes" is therefore pinned as a hash computed
    # when the bank was packed, and checked here against this bank's own records.
    from .generalist_fields import _json_hash
    pin = marker["source"]
    strip = lambda records: [{k: v for k, v in r.items() if k != "fields"} for r in records]
    if _json_hash(strip(manifest["scenes"])) != pin["scene_records_sha256"]:
        raise ValueError("Packed balance scene records differ from the source they were packed from")
    source_path = Path(pin["manifest"])
    if source_path.exists() and sha256(source_path) != pin["sha256"]:
        raise ValueError("Packed balance source pin differs")
    if path is not None and marker.get("retention_storage_root") != str(Path(path).resolve().parent):
        raise ValueError("Packed balance must resolve its fields inside itself")
    # sdf stays float32: the collision-bank builder verifies occupancy against it with
    # np.array_equal(occupancy, sdf < 0), and float16 rounding can flip a sign there.
    # Only the direction fields are packed, which costs 16% of the saving and keeps
    # every exact geometric check working.
    if marker.get("encoding") not in ("cat-packed-fields-v1", "cat-packed-fields-v1-float32-sdf"):
        raise ValueError("Unrecognized packed field encoding")
    return marker
