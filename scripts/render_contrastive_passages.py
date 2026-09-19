#!/usr/bin/env python3
"""Render canonical contrastive scenes and explicitly illustrative G1/Dex3 poses."""
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
from cat_ppo.furniture.contrastive_passages import ROLES, _poses, generate_contrastive_group
from cat_ppo.furniture.hand_passages import arm_pose, pose_separations


LABELS = {
    "open": ("Open: forward, relaxed arms", "Nominal forward posture has comfortable clearance."),
    "forward_protected": ("Protected: forward, raise or tuck", "Nominal hands intersect; protected arm poses clear."),
    "narrow": ("Narrow: sideways is appropriate", "All three checked forward poses intersect the cabinets."),
    "transition": ("Transition: adapt posture and heading", "Protected / narrow / protected, with open turn bays."),
}
COLORS = {"open": ".32 .44 .48 1", "forward_protected": ".75 .55 .34 1", "narrow": ".45 .53 .67 1"}


def font(size, bold=False):
    for candidate in (f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
                      f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def render(scene, mode="nominal", *, sideways=False, overview=False, width=1100, height=690):
    xml = ET.fromstring(assemble_training_xml())
    world = xml.find("worldbody")
    floor = world.find("geom[@name='floor']")
    floor.attrib.pop("material", None)
    floor.set("rgba", ".93 .94 .95 1")
    ET.SubElement(world, "light", name="contrast_key", directional="true", pos="0 0 6",
                  dir="-.4 -.6 -1", diffuse=".4 .4 .4")
    for index, box in enumerate(scene["boxes"]):
        center, half = list(box["center"]), list(box["half_size"])
        color = ".68 .72 .77 1"
        if box["category"] == "wall":
            center[2], half[2] = .025, .025  # Display only; saved geometry is unchanged.
        elif box["category"] == "baffle":
            color = ".28 .34 .42 1"
        else:
            module_id = int(box["name"].split("_")[1])
            color = COLORS[scene["hand_contrast"]["modules"][module_id]["role"]]
        ET.SubElement(world, "geom", name=f"contrast_box_{index}", type="box",
                      pos=" ".join(map(str, center)), size=" ".join(map(str, half)),
                      quat=f"{math.cos(box['yaw']/2)} 0 0 {math.sin(box['yaw']/2)}",
                      rgba=color, contype="0", conaffinity="0")
    start, goal = scene["route"][0], scene["route"][-1]
    ET.SubElement(world, "geom", name="route", type="capsule", size=".015",
                  fromto=f"{start[0]} {start[1]} .015 {goal[0]} {goal[1]} .015",
                  rgba=".1 .45 .7 .75", contype="0", conaffinity="0")
    for name, xy, color in (("A", start, ".12 .45 .7 1"), ("B", goal, ".1 .6 .35 1")):
        ET.SubElement(world, "geom", name=f"route_{name}", type="cylinder", size=".10 .008",
                      pos=f"{xy[0]} {xy[1]} .01", rgba=color, contype="0", conaffinity="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.global_.fovy = 38
    model.vis.headlight.ambient[:] = .4
    data = mujoco.MjData(model)
    pose = _poses(scene, [0.], math.pi/2 if sideways else 0.)[0]
    qpos = arm_pose(mode, .95 if mode != "nominal" else 1.)
    qpos[:2], qpos[3:7] = pose[:2], [math.cos(pose[2]/2), 0, 0, math.sin(pose[2]/2)]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    option.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    yaw = scene["generator"]["passage_yaw_rad"]
    camera.lookat[:] = [*pose[:2], .62 if overview else .79]
    camera.distance = 9.5 if overview else 2.15
    camera.azimuth = math.degrees(yaw) + (90 if overview else 8)
    camera.elevation = -61 if overview else -18
    measurement = pose_separations(scene, [pose], mode=mode, fraction=.95 if mode != "nominal" else 1.)
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        for center, radius, clearance in zip(measurement["hand_centers"][0], measurement["hand_radii"], measurement["hands"][0]):
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            rgba = [.88, .08, .04, .5] if clearance < 0 else [.02, .66, .35, .32]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                               center, np.eye(3).ravel(), np.array(rgba))
            renderer.scene.ngeom += 1
        image = Image.fromarray(renderer.render().copy())
    return image, {"hand_min_separation_m": float(measurement["hands"].min()),
                   "body_min_separation_m": float(measurement["body"].min())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--scene-cache", type=Path, default=ROOT / "data/furniture/contrastive_hand_v2/scenes")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/contrastive-passages-20260919")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scenes = []
    for role in ROLES:
        found = sorted(args.scene_cache.glob(f"contrast-train-{args.seed:06d}-{role}-*/scene-input.json"))
        if found:
            scenes.append(json.loads(found[0].read_text()))
    if len(scenes) != 4:
        scenes = generate_contrastive_group(args.seed)
    audit = []
    sheet = Image.new("RGB", (2240, 1400), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((30, 20), "Matched contrastive training scenes", font=font(43, True), fill="#203864")
    draw.text((32, 76), "Same layout and route; change corridor width to change the appropriate behavior.", font=font(26), fill="#465971")
    draw.text((32, 113), "Actual MuJoCo geometry / static illustrative poses / not learned rollouts", font=font(26), fill="#465971")
    for index, scene in enumerate(scenes):
        role = scene["hand_contrast"]["role"]
        x, y = 20 + (index % 2) * 1120, 172 + (index // 2) * 592
        mode = "raised" if role == "forward_protected" else "nominal"
        sideways = role in ("narrow", "transition")
        picture, record = render(scene, mode, sideways=sideways, overview=True)
        picture.save(args.output / f"{role}-overview.png")
        closeup, _ = render(scene, mode, sideways=sideways)
        closeup.save(args.output / f"{role}-closeup.png")
        sheet.paste(picture.crop((0, 110, 1100, 620)), (x, y + 64))
        draw.text((x + 12, y), LABELS[role][0], font=font(29, True), fill="#203864")
        widths = " / ".join(f"{m['width_m']*100:.1f}" for m in scene["hand_contrast"]["modules"])
        draw.text((x + 12, y + 36), f"Three gaps: {widths} cm", font=font(24), fill="#465971")
        draw.text((x + 12, y + 552), LABELS[role][1], font=font(22), fill="#465971")
        (args.output / f"{role}-scene.json").write_text(json.dumps(scene, indent=2) + "\n")
        audit.append(dict(role=role, mode=mode, sideways=sideways, **record))
    draw.text((32, 1358), "Cabinets remain full height; perimeter walls are displayed low. Blue floor line is the commanded route.", font=font(23), fill="#465971")
    sheet.save(args.output / "contrastive-scenes.png")

    comparison = Image.new("RGB", (3360, 940), "white")
    draw = ImageDraw.Draw(comparison)
    draw.text((28, 18), "Same protected corridor: the arm posture changes clearance", font=font(44, True), fill="#203864")
    draw.text((30, 78), "Illustrative kinematic poses, not a trained policy. Red hand sphere intersects; green is clear.", font=font(28), fill="#465971")
    protected = scenes[1]
    for index, mode in enumerate(("nominal", "raised", "tucked")):
        picture, record = render(protected, mode)
        x = 20 + index * 1120
        comparison.paste(picture, (x, 185))
        draw.text((x + 10, 139), mode.capitalize() + " arms", font=font(32, True), fill="#203864")
        draw.text((x + 10, 866), f"Hand clearance {record['hand_min_separation_m']*100:+.1f} cm; body {record['body_min_separation_m']*100:+.1f} cm", font=font(25), fill="#465971")
        picture.save(args.output / f"protected-{mode}.png")
        audit.append(dict(role="forward_protected", mode=mode, sideways=False, **record))
    draw.text((30, 907), "35 approved collision shapes and the 4 cm obstacle field certify sampled routes and transitions; this is not a dynamic guarantee.", font=font(24), fill="#465971")
    comparison.save(args.output / "protected-postures.png")
    (args.output / "render-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(args.output / "contrastive-scenes.png")
    print(args.output / "protected-postures.png")


if __name__ == "__main__":
    main()
