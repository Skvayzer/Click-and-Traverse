#!/usr/bin/env python3
"""Render actual new training geometry with explicitly illustrative static poses."""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cat_ppo.envs.g1.env_cat_wholebody import assemble_training_xml
from cat_ppo.furniture.hand_passages import (
    arm_pose, generate_hand_passage_scene, pose_separations,
)


def font(size, bold=False):
    candidates = [f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
                  f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def render(scene, mode, width=980, height=630):
    xml = ET.fromstring(assemble_training_xml())
    world = xml.find("worldbody")
    floor = world.find("geom[@name='floor']")
    floor.attrib.pop("material", None)
    floor.set("rgba", ".94 .95 .96 1")
    ET.SubElement(world, "light", name="hand_preview_key", directional="true", pos="0 0 6",
                  dir="-.4 -.6 -1", diffuse=".4 .4 .4")
    colors = {"tabletop": ".69 .46 .27 1", "table_leg": ".33 .30 .25 1",
              "shelf_edge": ".72 .48 .24 1", "shelf_support": ".34 .40 .44 1",
              "wall": ".68 .72 .77 1"}
    for i, box in enumerate(scene["boxes"]):
        center, half = list(box["center"]), list(box["half_size"])
        # Only the visualization lowers perimeter walls; certificates consume
        # the unchanged full-height scene JSON saved beside the image.
        if box["category"] == "wall":
            center[2], half[2] = .035, .035
        ET.SubElement(world, "geom", name=f"hand_preview_{i}", type="box",
            pos=" ".join(map(str, center)), size=" ".join(map(str, half)),
            quat=f"{math.cos(box['yaw'] / 2)} 0 0 {math.sin(box['yaw'] / 2)}",
            rgba=colors.get(box["category"], ".55 .60 .65 1"), contype="0", conaffinity="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.global_.fovy = 38
    model.vis.headlight.ambient[:] = .35
    data = mujoco.MjData(model)
    qpos = arm_pose(mode)
    route = scene["route"]
    focus = .5 * (np.asarray(route[0]) + np.asarray(route[-1])) if len(route) == 2 else np.asarray(route[len(route) // 2])
    yaw = scene["generator"]["passage_yaw_rad"]
    qpos[:2], qpos[3:7] = focus, [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    option.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [focus[0] + .05, focus[1], .62]
    camera.distance, camera.azimuth, camera.elevation = 3.6, 145 + math.degrees(yaw), -24
    measurement = pose_separations(scene, [[*focus, yaw]], mode=mode)
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        for center, radius in zip(measurement["hand_centers"][0], measurement["hand_radii"]):
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            rgba = [.82, .16, .10, .38] if mode == "nominal" else [.02, .60, .32, .32]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                               center, np.eye(3).ravel(), np.array(rgba))
            renderer.scene.ngeom += 1
        picture = Image.fromarray(renderer.render().copy())
    return picture, float(measurement["hands"].min())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/assets/hand-protection-passages-20260917/preview.png")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scenes = [generate_hand_passage_scene(6001, kind="hand_table_aisle", difficulty="easy"),
              generate_hand_passage_scene(7009, kind="hand_shelf_passage", difficulty="hard")]
    canvas = Image.new("RGB", (2040, 1660), "white")
    draw = ImageDraw.Draw(canvas)
    navy = "#203864"
    draw.text((35, 24), "New hand-protection training scenes", font=font(46, True), fill=navy)
    draw.text((37, 85), "Actual generated geometry | static G1 / Dex3 poses | not a learned rollout",
              font=font(27), fill="#465971")
    for row, (scene, label) in enumerate(zip(scenes, ("Table-edge aisle / easy", "Offset shelf passage / hard"))):
        top = 140 + row * 715
        meta = scene["hand_protection"]
        draw.text((37, top), f"{label}  |  gap {meta['passage_width_m'] * 100:.1f} cm, top {meta['furniture_top_height_m'] * 100:.1f} cm",
                  font=font(30, True), fill=navy)
        for column, mode in enumerate(("nominal", "raised")):
            picture, clearance = render(scene, mode)
            left = 25 + column * 1010
            canvas.paste(picture, (left, top + 70))
            text = ("Nominal forward arms: hand intersection" if mode == "nominal" else "Illustrative raised arms: clear")
            draw.text((left + 12, top + 38), text, font=font(25, True), fill="#ab3529" if mode == "nominal" else "#176a48")
            draw.text((left + 12, top + 657), f"Hand-sphere surface clearance at this pose: {clearance * 100:+.1f} cm",
                      font=font(24), fill=navy)
        (args.output.parent / f"{scene['scene_id']}.json").write_text(json.dumps(scene, indent=2) + "\n")
    draw.text((37, 1587), "Raised route checked with all 35 body shapes and CAT's 4 cm field. Walls shown low for visibility.",
              font=font(24), fill="#465971")
    draw.text((37, 1620), "Sideways alternatives remain possible and are audited; these scenes encourage protective arm movement.",
              font=font(24), fill="#465971")
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
