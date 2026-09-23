#!/usr/bin/env python3
"""Generate narrow-passage contrastive groups with continuously sampled widths.

The shipped bank holds twelve narrow scenes at exactly three widths
(0.58, 0.64, 0.70 m), and the narrow success rate is measured on those twelve.
That cannot distinguish a policy that understood passages from one that
memorised a dozen layouts.

Here the narrow width is drawn uniformly from the generator's full supported
range, minus a held-out band that is never generated for training. Evaluating
on the held-out band afterwards is what makes the number mean something.

generate_contrastive_group already accepts a continuous ``narrow_width_range``;
this script simply stops feeding it three fixed rungs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cat_ppo.furniture.contrastive_passages import generate_contrastive_group  # noqa: E402
from cat_ppo.furniture.contrastive_bank import ROLES  # noqa: E402

EXTENDED_SCHEMA = "cat-extended-balance-v1"
SUPPORTED = (.40, .74)          # enforced inside generate_contrastive_group


def sample_width(rng, holdout):
    """Uniform over the supported range, rejecting the held-out band."""
    low, high = holdout
    while True:
        width = rng.uniform(*SUPPORTED)
        if not (low <= width <= high):
            return width


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--groups", type=int, default=50, help="Each group yields four matched roles")
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--split", default="train")
    p.add_argument("--holdout", type=float, nargs=2, default=(.60, .66),
                   help="Width band reserved for evaluation; never generated here")
    p.add_argument("--dx", type=float, default=.04)
    p.add_argument("--base-manifest", type=Path,
                   default=ROOT / "data/furniture/cat_flat_hand_balance_v2_20260921/manifest.json",
                   help="Bank to extend; its scenes are referenced in place, never copied")
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output bank already exists; refusing to overwrite")

    from cat_ppo.furniture.generalist_fields import (
        DATASET_REVISION, EXPANDED_SCHEMA, RELEASED_CONFIG_SHA256, _json_hash,
        load_generalist_manifest, make_clutter_fields, sha256,
    )
    # A generalist manifest, exactly like the shipped v2 bank: these scenes carry
    # certificate_semantics='width-curriculum-scaffold-v1', which the contrastive
    # specialist validator rejects by design.

    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True)
    records, widths = [], []
    for index in range(args.groups):
        width = sample_width(rng, args.holdout)
        group_seed = args.seed + index
        scenes = generate_contrastive_group(group_seed, split=args.split,
                                            narrow_width_range=(width, width), curriculum_rung=0)
        scenes.sort(key=lambda scene: ROLES.index(scene["hand_contrast"]["role"]))
        for scene in scenes:
            relative = Path("scenes") / scene["scene_id"]
            directory = args.output / relative
            directory.mkdir(parents=True, exist_ok=True)
            record = make_clutter_fields(scene, directory, dx=args.dx)
            source = dict(record["source"], kind="contrastive-hand-passage",
                          hand_contrast=scene["hand_contrast"], sampled_narrow_width_m=width,
                          holdout_band_m=list(args.holdout))
            (directory / "source.json").write_text(json.dumps(source, indent=2) + "\n")
            source["metadata_sha256"] = sha256(directory / "source.json")
            record.update(path=str(relative), source=source, task_kind="room", reset_mode="room",
                          crossed_mode="goal_radius", episode_length=4000, sampling_group="generic_clutter")
            (directory / "field-record.json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
        widths.append(width)
        print(f"group {index + 1}/{args.groups}  width={width:.4f} m  scenes={len(records)}", flush=True)

    # Write the new scenes as a pinned addition fragment, then an extended bank that
    # is exactly parent ++ addition. The parent keeps resolving its fields through its
    # own retention root, so nothing is copied.
    fragment = args.output / "addition.json"
    fragment.write_text(json.dumps(dict(kind="narrow_passages", scene_count=len(records),
        scenes=records, narrow_width_sampling=dict(supported_range_m=list(SUPPORTED),
        holdout_band_m=list(args.holdout), sampled_widths_m=widths)), indent=1) + "\n")

    base_path = args.base_manifest.resolve()
    base = json.loads(base_path.read_text())
    # scene_directory() resolves records carrying `hand_contrast` against the manifest's
    # OWN directory, not the retention root, so the parent's contrastive scenes must be
    # reachable from here too. Symlink them: no bytes are copied and the parent is
    # untouched.
    for record in base["scenes"]:
        source_meta = record.get("source", {})
        # scene_directory() sends records to the retention root ONLY when they carry
        # neither marker; both kinds resolve against this bank's own directory.
        if not (source_meta.get("hand_contrast") or source_meta.get("flat_balance")):
            continue
        target = args.output / record["path"]
        if target.exists():
            continue
        # Hard-link the field files: scene_directory() refuses paths that resolve
        # outside the bank root, so symlinks are rejected by design. Hard links keep
        # the bytes shared on disk while every path stays inside this bank.
        source = (base_path.parent / record["path"]).resolve()
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            destination = target / item.name
            try:
                os.link(item, destination)
            except OSError:
                shutil.copy2(item, destination)
    combined = list(base["scenes"]) + records
    marker = dict(base["flat_balance"])
    marker.update(schema=EXTENDED_SCHEMA,
                  parent=dict(manifest=str(base_path), sha256=sha256(base_path)),
                  additions=[dict(kind="narrow_passages", manifest=str(fragment.resolve()),
                                  sha256=sha256(fragment), count=len(records))])
    manifest = dict(base, scene_count=len(combined), scenes=combined, flat_balance=marker,
        fields_bytes=base.get("fields_bytes", 0)
                     + sum(f["size_bytes"] for s in records for f in s["fields"].values()))
    manifest.pop("manifest_sha256", None)
    destination = args.output / "manifest.json"
    manifest["manifest_sha256"] = _json_hash(manifest)
    temporary = args.output / "manifest.pending.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    load_generalist_manifest(temporary)
    temporary.replace(destination)
    print(json.dumps(dict(groups=args.groups, scenes=len(records),
                          width_min=min(widths), width_max=max(widths),
                          holdout=list(args.holdout)), indent=1))


if __name__ == "__main__":
    main()
