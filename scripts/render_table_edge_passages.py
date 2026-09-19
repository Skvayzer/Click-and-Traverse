#!/usr/bin/env python3
"""Render certified table-edge geometry with illustrative G1/Dex3 arm poses.

These are MuJoCo renders of the canonical scene boxes, not policy rollouts.
Only the perimeter walls are lowered for visibility; table slabs, aprons,
legs, and exterior lane blockers retain their training dimensions.
"""
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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cat_ppo.envs.g1.env_cat_wholebody import assemble_training_xml
from cat_ppo.furniture.contrastive_passages import _poses
from cat_ppo.furniture.hand_passages import arm_pose, pose_separations
from render_contrastive_passages import font


PALETTE = {
    "table_top": ".72 .49 .29 1",
    "tabletop": ".72 .49 .29 1",
    "table_apron": ".52 .33 .19 1",
    "table_leg": ".27 .30 .32 1",
    "baffle": ".39 .45 .52 1",
    "wall": ".68 .72 .77 1",
}


def table_top_height(scene):
    tops = [b["center"][2] + b["half_size"][2] for b in scene["boxes"]
            if b["category"] in ("table_top", "tabletop")]
    if not tops:
        raise ValueError("Expected canonical table-top boxes")
    return max(tops)


def render(scene, mode="raised", *, overview=False, width=1100, height=740):
    xml = ET.fromstring(assemble_training_xml())
    world = xml.find("worldbody")
    floor = world.find("geom[@name='floor']")
    floor.attrib.pop("material", None)
    floor.set("rgba", ".94 .95 .96 1")
    ET.SubElement(world, "light", name="table_edge_key", directional="true", pos="0 0 6",
                  dir="-.4 -.6 -1", diffuse=".4 .4 .4")
    for index, box in enumerate(scene["boxes"]):
        center, half = list(box["center"]), list(box["half_size"])
        if box["category"] == "wall":
            center[2], half[2] = .025, .025
        ET.SubElement(world, "geom", name=f"table_edge_box_{index}", type="box",
                      pos=" ".join(map(str, center)), size=" ".join(map(str, half)),
                      quat=f"{math.cos(box['yaw']/2)} 0 0 {math.sin(box['yaw']/2)}",
                      rgba=PALETTE.get(box["category"], ".55 .60 .65 1"),
                      contype="0", conaffinity="0")
    start, goal = scene["route"][0], scene["route"][-1]
    ET.SubElement(world, "geom", name="table_edge_route", type="capsule", size=".015",
                  fromto=f"{start[0]} {start[1]} .015 {goal[0]} {goal[1]} .015",
                  rgba=".1 .45 .7 .75", contype="0", conaffinity="0")
    for name, xy, color in (("A", start, ".12 .45 .7 1"), ("B", goal, ".1 .6 .35 1")):
        ET.SubElement(world, "geom", name=f"table_route_{name}", type="cylinder", size=".10 .008",
                      pos=f"{xy[0]} {xy[1]} .01", rgba=color, contype="0", conaffinity="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.global_.fovy = 38
    model.vis.headlight.ambient[:] = .4
    data = mujoco.MjData(model)
    # Both nominal and raised hands lie longitudinally within the same table
    # module at this root position; use exactly the same root pose and camera.
    pose = _poses(scene, [-.20])[0]
    fraction = .95 if mode != "nominal" else 1.
    qpos = arm_pose(mode, fraction)
    qpos[:2], qpos[3:7] = pose[:2], [math.cos(pose[2]/2), 0, 0, math.sin(pose[2]/2)]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    option.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    yaw = math.degrees(scene["generator"]["passage_yaw_rad"])
    camera.lookat[:] = [*pose[:2], .62 if overview else .64]
    camera.distance = 9.3 if overview else 2.65
    camera.azimuth = yaw + (90 if overview else 160)
    camera.elevation = -58 if overview else -25
    measurement = pose_separations(scene, [pose], mode=mode, fraction=fraction)
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        for center, radius, clearance in zip(measurement["hand_centers"][0],
                                              measurement["hand_radii"], measurement["hands"][0]):
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            color = [.88, .08, .04, .5] if clearance < 0 else [.02, .66, .35, .36]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                               center, np.eye(3).ravel(), np.array(color))
            renderer.scene.ngeom += 1
        image = Image.fromarray(renderer.render().copy())
    hand_bottom = measurement["hand_centers"][0, :, 2] - measurement["hand_radii"]
    table_top = table_top_height(scene)
    return image, {
        "scene_id": scene["scene_id"], "mode": mode, "arm_fraction": fraction,
        "root_xyyaw": pose.tolist(), "tabletop_height_m": table_top,
        "hand_bottom_height_m": hand_bottom.tolist(),
        "hand_bottom_above_table_height_m": (hand_bottom - table_top).tolist(),
        "hand_min_surface_separation_m": float(measurement["hands"].min()),
        "body_min_surface_separation_m": float(measurement["body"].min()),
        "static_illustration_not_policy_rollout": True,
    }


def main():
    from cat_ppo.furniture.table_edge_passages import generate_table_edge_pair
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260919, 20260920, 20260921])
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/table-edge-passages-20260919")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scenes = []
    for seed in args.seeds[:3]:
        scene_path = args.output / f"scene-{seed}.json"
        # Always regenerate from the installed source so the image cannot
        # silently use a cached scene from an older geometry implementation.
        scene = next(s for s in generate_table_edge_pair(seed)
                     if s["hand_contrast"]["role"] == "forward_protected")
        scene_path.write_text(json.dumps(scene, indent=2) + "\n")
        scenes.append(scene)
    navy, muted = "#203864", "#465971"
    audit = []
    sheet = Image.new("RGB", (3360, 1060), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((30, 22), "Table-height passages: protect the hands while walking forward", font=font(45, True), fill=navy)
    draw.text((32, 85), "Actual generated table slabs, aprons and legs. G1 / Dex3 with illustrative raised-hand poses.", font=font(28), fill=muted)
    for index, scene in enumerate(scenes):
        image, record = render(scene)
        audit.append(record)
        x = 20 + index * 1120
        width = scene["hand_contrast"]["modules"][1]["width_m"]
        draw.text((x+10, 144), f"Example {index+1}: {record['tabletop_height_m']*100:.1f} cm tables", font=font(31, True), fill=navy)
        draw.text((x+10, 186), f"Gap between table edges: {width*100:.1f} cm", font=font(27), fill=muted)
        sheet.paste(image, (x, 225))
        vertical = min(record["hand_bottom_above_table_height_m"])*100
        draw.text((x+10, 971), f"Hand sphere bottoms {vertical:.1f} cm above table height", font=font(26), fill="#176a48")
        image.save(args.output / f"example-{index+1}-raised.png")
        overview, _ = render(scene, overview=True)
        overview.save(args.output / f"example-{index+1}-overview.png")
    draw.text((32, 1022), "Static prescribed poses, not newly learned behavior. Perimeter walls displayed low; obstacle dimensions are unchanged.", font=font(25), fill=muted)
    sheet.save(args.output / "table-edge-scenes.png")

    comparison = Image.new("RGB", (2240, 1090), "white")
    draw = ImageDraw.Draw(comparison)
    draw.text((30, 22), "Same passage and root pose: raising the hands changes clearance", font=font(41, True), fill=navy)
    draw.text((32, 82), "Red: hand envelope intersects a table. Green: clear hand envelope above tabletop height.", font=font(27), fill=muted)
    for index, mode in enumerate(("nominal", "raised")):
        image, record = render(scenes[0], mode)
        audit.append(record)
        x = 20 + index*1120
        draw.text((x+10, 140), "Nominal hanging hands" if mode == "nominal" else "Raised hands", font=font(33, True), fill=navy)
        comparison.paste(image, (x, 200))
        color = "#ab3529" if mode == "nominal" else "#176a48"
        draw.text((x+10, 947), f"Hand-envelope separation: {record['hand_min_surface_separation_m']*100:+.1f} cm", font=font(29), fill=color)
        draw.text((x+10, 989), f"Lowest point of hand sphere: {min(record['hand_bottom_height_m'])*100:.1f} cm above floor", font=font(25), fill=muted)
        image.save(args.output / f"comparison-{mode}.png")
    draw.text((32, 1044), "Illustrative static poses in MuJoCo. Certificates check the approved body shapes and conservative 4 cm obstacle field.", font=font(24), fill=muted)
    comparison.save(args.output / "table-edge-hand-comparison.png")
    (args.output / "render-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(args.output / "table-edge-scenes.png")
    print(args.output / "table-edge-hand-comparison.png")


if __name__ == "__main__":
    main()
