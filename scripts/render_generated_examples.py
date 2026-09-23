#!/usr/bin/env python3
"""Contact sheets of the newly generated scenes, for eyeball validation.

Shows what was actually built, not what was intended:
  rooms   top-down occupancy projection plus a hand-height slice, so both the
          floor plan and what the hands would meet are visible
  narrow  the same, with the sampled corridor width printed
  reactive parametric, so there is no occupancy grid: the approach ray, the
          object's start and stop points and the certified surface gap are drawn
          relative to the targeted hand

Occupancy is bit-packed in the new banks and is unpacked here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cat_mjlab.packing.field_packing import unpack_occupancy  # noqa: E402
from cat_ppo.furniture.generalist_fields import load_generalist_manifest, scene_directory  # noqa: E402

HAND_HEIGHT_M = .80          # roughly where a standing robot's hands sweep


def load_occupancy(directory, shape):
    array = np.load(directory / "obs.npy", allow_pickle=False)
    if array.shape != tuple(shape):
        array = unpack_occupancy(array, tuple(shape))
    return np.asarray(array)


def draw_room(axis, record, directory, title):
    shape = tuple(record["shape"])
    occupancy = load_occupancy(directory, shape)
    dx = record["dx"]
    level = min(shape[2] - 1, max(0, int(round(HAND_HEIGHT_M / dx))))
    footprint = occupancy.any(axis=2).T            # anything, any height
    at_hands = occupancy[:, :, level].T            # what the hands would meet
    canvas = np.zeros(footprint.shape + (3,), dtype=float)
    canvas[...] = 1.
    canvas[footprint] = (.78, .82, .86)            # obstacle somewhere in the column
    canvas[at_hands.astype(bool)] = (.86, .32, .24)  # obstacle at hand height
    axis.imshow(canvas, origin="lower", interpolation="nearest")
    axis.set_title(title, fontsize=7)
    axis.set_xticks([]); axis.set_yticks([])
    occupied = float(footprint.mean())
    axis.set_xlabel(f"{shape[0]}x{shape[1]}  занято {occupied * 100:.0f}%", fontsize=6)


def sheet_rooms(manifest_path, picks, output, title):
    manifest_path = Path(manifest_path).resolve()
    manifest = load_generalist_manifest(manifest_path, verify_files=False)
    columns = 4
    rows = (len(picks) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.0 * rows))
    for axis, record in zip(np.ravel(axes), picks):
        directory = scene_directory(manifest, manifest_path, record)
        label = record["scene_id"]
        extra = record.get("source", {}).get("difficulty") or ""
        width = record.get("source", {}).get("sampled_narrow_width_m")
        if width is not None:
            extra = f"ширина {width:.3f} м"
        draw_room(axis, record, directory, f"{label[:34]}\n{record['family']} {extra}")
    for axis in np.ravel(axes)[len(picks):]:
        axis.axis("off")
    figure.suptitle(title + "   (серое — препятствие, красное — на высоте рук ~0.80 м)", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, .96))
    figure.savefig(output, dpi=110)
    plt.close(figure)
    print(f"wrote {output}")


def sheet_reactive(manifest_path, count, output, seed=0):
    manifest = json.loads(Path(manifest_path).read_text())
    rng = np.random.default_rng(seed)
    picks = [manifest["scenes"][i] for i in rng.choice(len(manifest["scenes"]), count, replace=False)]
    columns = 4
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.0 * rows))
    for axis, scene in zip(np.ravel(axes), picks):
        start = np.array(scene["start"][0]); end = np.array(scene["end"][0])
        ray = np.array(scene.get("approach_ray", (end - start) / max(np.linalg.norm(end - start), 1e-9)))
        hand = end + ray * 0.                      # the object stops short of the hand
        gap = scene["clearance"][0]
        centre = end + ray * (gap + .10344617)     # hand centre lies further along the ray
        axis.add_patch(Circle((centre[0], centre[1]), .10344617, fill=False, color="#1f77b4", lw=1.4))
        axis.plot([start[0], end[0]], [start[1], end[1]], color="#d62728", lw=1.2)
        axis.plot(start[0], start[1], "o", color="#d62728", ms=4)
        axis.plot(end[0], end[1], "s", color="#2ca02c", ms=5)
        axis.set_aspect("equal")
        span = max(.45, np.linalg.norm(start[:2] - centre[:2]) * 1.3)
        axis.set_xlim(centre[0] - span, centre[0] + span)
        axis.set_ylim(centre[1] - span, centre[1] + span)
        axis.set_title(f"{scene['bucket']}  {scene['shape']}  {scene['target']}\n"
                       f"зазор {gap * 1000:.0f} мм   {scene['speed'][0] * 100:.0f} см/с", fontsize=7)
        axis.tick_params(labelsize=5)
    for axis in np.ravel(axes)[count:]:
        axis.axis("off")
    figure.suptitle("reactive: синий — сфера кисти (103.45 мм), красный — путь объекта, "
                    "зелёный квадрат — точка остановки", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(output, dpi=110)
    plt.close(figure)
    print(f"wrote {output}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/generated_examples_20260923")
    p.add_argument("--per-sheet", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rooms_path = ROOT / "data/furniture/procedural_rooms_v2_20260923/manifest.json"
    rooms = load_generalist_manifest(rooms_path.resolve(), verify_files=False)
    new = [s for s in rooms["scenes"] if s.get("source", {}).get("kind", "").startswith("procedural-")]
    for family in ("generic_clutter", "furniture"):
        pool = [s for s in new if s["family"] == family]
        picks = [pool[i] for i in rng.choice(len(pool), min(args.per_sheet, len(pool)), replace=False)]
        sheet_rooms(rooms_path, picks, args.output / f"rooms-{family}.png",
                    f"Новые комнаты: {family}  ({len(pool)} сгенерировано)")

    narrow_path = ROOT / "data/furniture/narrow_widths_v1_20260923/manifest.json"
    fragment = json.loads((narrow_path.parent / "addition.json").read_text())
    pool = [s for s in fragment["scenes"] if "-narrow-" in s["scene_id"]]
    picks = [pool[i] for i in rng.choice(len(pool), min(args.per_sheet, len(pool)), replace=False)]
    sheet_rooms(narrow_path, picks, args.output / "narrow-passages.png",
                f"Узкие проходы: непрерывные ширины ({len(pool)} сгенерировано)")

    sheet_reactive(ROOT / "data/furniture/reactive_standing_v2_20260923/manifest.json",
                   args.per_sheet, args.output / "reactive-approaches.png", seed=args.seed)


if __name__ == "__main__":
    main()
