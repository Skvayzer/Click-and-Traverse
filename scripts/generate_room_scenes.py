#!/usr/bin/env python3
"""Generate clutter and furniture rooms, packed on the way out.

Sixty-one clutter rooms currently carry 40% of all training experience, so each
one is seen about forty times as often as a procedural CAT scene. That is a
direct invitation to memorise floor plans instead of learning navigation.

Rooms are large: a single clutter scene is ~98 MB of raw float32 fields, so two
hundred of them would need more disk than exists here. Each scene is therefore
packed as soon as its fields are built and the raw copy is dropped, which keeps
peak usage at one scene rather than the whole batch.

Output is an addition fragment plus an extended bank, exactly like the narrow
generator: the parent stays an immutable prefix and every addition is pinned.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cat_mjlab.packing.field_packing import (  # noqa: E402
    SCHEMA as PACK_ENCODING, pack_direction, pack_occupancy, pack_scalar,
)

EXTENDED_SCHEMA = "cat-extended-balance-v1"
# "open_floor" is a walled empty box: 11% occupancy, nothing for hands or navigation
# to deal with. A third of the first batch was wasted on those, so it is dropped.
DIFFICULTIES = ("pilot", "dense")


def pack_directory(source, target):
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "sdf.npy", pack_scalar(np.load(source / "sdf.npy")))
    for name in ("bf", "gf"):
        codes, valid = pack_direction(np.load(source / f"{name}.npy"))
        np.save(target / f"{name}.npy", codes)
        np.save(target / f"{name}_valid.npy", valid)
    if (source / "obs.npy").exists():
        np.save(target / "obs.npy", pack_occupancy(np.load(source / "obs.npy")))
    for extra in source.iterdir():
        if extra.suffix != ".npy":
            shutil.copy2(extra, target / extra.name)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clutter", type=int, default=200)
    p.add_argument("--furniture", type=int, default=80)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--split", default="train")
    p.add_argument("--dx", type=float, default=.04)
    p.add_argument("--base-manifest", type=Path,
                   default=ROOT / "data/furniture/narrow_widths_v1_20260923/manifest.json")
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output bank already exists; refusing to overwrite")

    from cat_ppo.furniture.random_rooms import generate_random_room
    from cat_ppo.furniture.generalist_fields import (
        _field_records, _json_hash, load_generalist_manifest, make_clutter_fields, sha256,
    )

    args.output.mkdir(parents=True)
    scratch = Path(tempfile.mkdtemp(prefix="room-build-", dir=args.output))
    records, failures = [], 0
    plan = ([("generic_clutter", i) for i in range(args.clutter)]
            + [("furniture", i) for i in range(args.furniture)])
    for index, (kind, offset) in enumerate(plan):
        seed = args.seed + index
        difficulty = DIFFICULTIES[offset % len(DIFFICULTIES)]
        try:
            # generate_clutter_scene hard-codes its dense layout -- "nine tables and 36
            # chairs" on a fixed 3x3 grid with only centimetre jitter -- so 12 of its rooms
            # differed pairwise by a median of 0.06, with pairs as close as 0.007.
            # generate_random_room reaches 0.28 on the same measure and varies room size
            # too, so both families are drawn from it.
            scene = generate_random_room(seed, split=args.split, kind=kind, difficulty=difficulty)
        except Exception as error:                       # infeasible layouts are expected
            failures += 1
            print(f"skip {kind} seed={seed} ({type(error).__name__}: {error})", flush=True)
            continue
        relative = Path("scenes") / scene["scene_id"]
        staging = scratch / scene["scene_id"]
        record = make_clutter_fields(scene, staging, dx=args.dx)
        target = args.output / relative
        pack_directory(staging, target)
        shutil.rmtree(staging)
        # scene_directory() routes a record to the retention root unless it carries
        # hand_contrast or flat_balance. New rooms live in THIS bank, so they must carry
        # the marker or they would be looked for inside the parent's packed directory.
        source = dict(record["source"], kind=f"procedural-{kind}", difficulty=difficulty,
                      flat_balance=True)
        (target / "source.json").write_text(json.dumps(source, indent=2) + "\n")
        source["metadata_sha256"] = sha256(target / "source.json")
        record.update(path=str(relative), source=source, task_kind="room", reset_mode="room",
                      crossed_mode="goal_radius", episode_length=4000,
                      # generate_clutter_scene works in "mixed" mode and sometimes returns a
                      # furniture-family room, so the group must follow the scene, not the plan.
                      sampling_group=record["family"],
                      fields=_field_records(target))
        (target / "field-record.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        if len(records) % 20 == 0:
            free = shutil.disk_usage(args.output).free / 1e9
            print(f"{len(records)}/{len(plan)} built, {failures} skipped, {free:.1f} GB free", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)

    fragment = args.output / "addition.json"
    fragment.write_text(json.dumps(dict(kind="procedural_rooms", scene_count=len(records),
                                        scenes=records), indent=1) + "\n")
    base_path = args.base_manifest.resolve()
    base = json.loads(base_path.read_text())
    for record in base["scenes"]:
        meta = record.get("source", {})
        if not (meta.get("hand_contrast") or meta.get("flat_balance")):
            continue
        target = args.output / record["path"]
        if target.exists():
            continue
        origin = (base_path.parent / record["path"]).resolve()
        target.mkdir(parents=True, exist_ok=True)
        for item in origin.iterdir():
            try:
                import os
                os.link(item, target / item.name)
            except OSError:
                shutil.copy2(item, target / item.name)

    combined = list(base["scenes"]) + records
    marker = dict(base["flat_balance"])
    # Only this addition: the parent's own additions are already inside the parent's
    # scenes, and the validator checks the parent as a prefix before the additions.
    additions = [dict(kind="procedural_rooms", manifest=str(fragment.resolve()),
                      sha256=sha256(fragment), count=len(records))]
    marker.update(schema=EXTENDED_SCHEMA,
                  parent=dict(manifest=str(base_path), sha256=sha256(base_path)),
                  additions=additions)
    manifest = dict(base, scene_count=len(combined), scenes=combined, flat_balance=marker,
                    fields_bytes=base.get("fields_bytes", 0)
                    + sum(f["size_bytes"] for s in records for f in s["fields"].values()))
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _json_hash(manifest)
    destination = args.output / "manifest.json"
    temporary = args.output / "manifest.pending.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    load_generalist_manifest(temporary)
    temporary.replace(destination)
    print(json.dumps(dict(built=len(records), skipped=failures, encoding=PACK_ENCODING,
                          scenes_total=len(combined)), indent=1))


if __name__ == "__main__":
    main()
