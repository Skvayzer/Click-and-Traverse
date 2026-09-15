"""Render four actual expanded-bank furniture scenes as one readable contact sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cat_ppo.furniture.generalist_fields import load_generalist_manifest, sha256
from cat_ppo.furniture.random_rooms import render_room_overview


DEFAULT_BANK = ROOT / "data/furniture/cat_diversity_v2_20260916/manifest.json"
DEFAULT_OUTPUT = ROOT / "docs/assets/cat-diversity-20260916/random-furniture-rooms.png"
DEFAULT_SEEDS = (4001, 4008, 4016, 4024)


def _font(size, bold=False):
    from PIL import ImageFont

    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def visualize_bank(bank=DEFAULT_BANK, output=DEFAULT_OUTPUT, seeds=DEFAULT_SEEDS):
    from PIL import Image, ImageDraw

    bank, output = Path(bank).resolve(), Path(output).resolve()
    if bank.is_dir():
        bank /= "manifest.json"
    manifest = load_generalist_manifest(bank, verify_files=False)
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError("The contact sheet requires four distinct furniture seeds")
    records = {}
    for record in manifest["scenes"]:
        if record["family"] != "furniture":
            continue
        seed = record["source"].get("generation_parameters", {}).get("seed")
        if seed in seeds:
            if seed in records:
                raise ValueError(f"Ambiguous furniture seed {seed} in completed bank")
            records[seed] = record
    missing = set(seeds) - records.keys()
    if missing:
        raise ValueError(f"Completed bank has no furniture scenes for seeds {sorted(missing)}")

    margin, gap, card_width, card_height = 60, 42, 1100, 1170
    header, footer = 205, 105
    width = 2 * margin + 2 * card_width + gap
    height = header + 2 * card_height + gap + footer
    image = Image.new("RGB", (width, height), "#f0f4f7")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 34), "Random furniture rooms", font=_font(55, True), fill="#172a3a")
    draw.text((margin, 105), "Actual training geometry · independent positions and orientations", font=_font(29), fill="#455b6b")
    legend_y = 163
    draw.rounded_rectangle((margin, legend_y, margin + 35, legend_y + 21), radius=3, fill="#cfad79")
    draw.text((margin + 47, legend_y - 5), "Tables", font=_font(26), fill="#344e60")
    draw.rounded_rectangle((margin + 215, legend_y, margin + 243, legend_y + 21), radius=3, fill="#7e9bad")
    draw.text((margin + 255, legend_y - 5), "Chairs", font=_font(26), fill="#344e60")
    draw.line((margin + 429, legend_y + 10, margin + 473, legend_y + 10), fill="#178962", width=5)
    draw.text((margin + 490, legend_y - 5), "A → B route", font=_font(26), fill="#344e60")

    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = []
    for index, seed in enumerate(seeds):
        record = records[seed]
        scene_path = bank.parent / record["path"] / "scene.json"
        if sha256(scene_path) != record["scene_sha256"]:
            raise ValueError(f"Scene geometry differs from completed bank: {scene_path}")
        scene = json.loads(scene_path.read_text())
        native_path = output.parent / f"furniture-seed-{seed}.png"
        render_room_overview(scene, native_path, size=1100)
        with Image.open(native_path) as native:
            # The helper includes its own title and certification footer. Keep
            # its exact geometry and A/B labels; the sheet supplies one footer.
            geometry = native.convert("RGB").crop((25, 70, 1075, 1136))
        x = margin + (index % 2) * (card_width + gap)
        y = header + (index // 2) * (card_height + gap)
        draw.rounded_rectangle((x, y, x + card_width, y + card_height), radius=18,
                               fill="#f8fafc", outline="#cdd8df", width=2)
        draw.text((x + 30, y + 20), f"Room {seed}", font=_font(35, True), fill="#172a3a")
        counts = scene["counts"]
        dims = scene["room_dimensions"]
        subtitle = f"{counts['tables']} tables  ·  {counts['chairs']} chairs  ·  {dims[0]:.1f} × {dims[1]:.1f} m"
        draw.text((x + 30, y + 64), subtitle, font=_font(27), fill="#455b6b")
        image.paste(geometry, (x + 25, y + 100))
        rendered.append(dict(seed=seed, scene_id=scene["scene_id"], scene_sha256=record["scene_sha256"],
                             geometry_hash=scene["geometry_hash"], source=str(scene_path),
                             individual_image=str(native_path)))

    draw.text((margin, height - 73),
              "Routes checked for a 23 cm root radius; whole-body dynamics still require evaluation.",
              font=_font(28), fill="#455b6b")
    image.save(output)
    provenance = dict(manifest=str(bank), manifest_sha256=manifest["manifest_sha256"],
                      output=str(output), scenes=rendered, rendering="Exact canonical oriented boxes, top view")
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs=4, default=DEFAULT_SEEDS)
    args = parser.parse_args(argv)
    print(visualize_bank(args.bank, args.output, args.seeds))


if __name__ == "__main__":
    main()
