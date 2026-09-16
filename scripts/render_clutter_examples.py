"""Static 3D and plan views of exact scene JSONs from an active CAT bank.

Only the displayed wall heights, lights and colors change. No simulation or
policy runs; the route is the generator's root-clearance certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

PALETTE = {
    "tabletop": "#c49b68", "table_leg": "#645449",
    "chair_seat": "#548994", "chair_back": "#447985",
    "chair_leg": "#394d56", "chair_armrest": "#3d6670",
    "wall": "#87929b", "crate": "#c09a71", "low_block": "#b36f5f",
    "partition": "#768cab", "shelf_edge": "#819b7a", "support": "#536650",
}


def color(category):
    from PIL import ImageColor
    return [value / 255 for value in ImageColor.getrgb(PALETTE.get(category, "#83939a"))] + [1.]


def font(size, bold=False):
    from PIL import ImageFont
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def render_3d(scene, width=1400, height=1100):
    import mujoco
    import numpy as np
    from PIL import Image
    from cat_ppo.envs.g1.env_cat_wholebody import assemble_training_xml
    from cat_ppo.envs.g1.constants import DEFAULT_QPOS

    xml = ET.fromstring(assemble_training_xml())
    world = xml.find("worldbody")
    floor = world.find("geom[@name='floor']")
    floor.attrib.pop("material", None)
    floor.set("rgba", ".95 .96 .97 1")
    quality = xml.find("./visual/quality")
    quality.set("shadowsize", "4096")
    # Two directed lights reveal table legs and chair backs in the overview.
    for name, direction, diffuse in (("room_key", "-.4 -.3 -1", ".45 .45 .45"),
                                      ("room_fill", ".5 .6 -1", ".1 .1 .1")):
        ET.SubElement(world, "light", name=name, directional="true", pos="0 0 8",
                      dir=direction, diffuse=diffuse, castshadow="true" if name == "room_key" else "false")
    for index, box in enumerate(scene["boxes"]):
        center, size = list(box["center"]), list(box["half_size"])
        if box["category"] == "wall":
            center[2], size[2] = .045, .045  # Explicit cutaway display.
        yaw = box["yaw"]
        ET.SubElement(world, "geom", name=f"display_object_{index}", type="box",
                      pos=" ".join(map(str, center)), size=" ".join(map(str, size)),
                      quat=f"{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}",
                      rgba=" ".join(map(str, color(box["category"]))),
                      contype="0", conaffinity="0", group="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.global_.fovy = 40
    model.vis.headlight.ambient[:] = [.25] * 3
    model.vis.headlight.diffuse[:] = [.3] * 3
    model.vis.headlight.specular[:] = [.08] * 3
    data = mujoco.MjData(model)
    data.qpos[:] = DEFAULT_QPOS
    data.qpos[:2] = scene["start"][:2]
    yaw = scene["start"][2]
    data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    options.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    room_x, room_y = scene["room_dimensions"][:2]
    model.stat.center[:] = [room_x / 2, room_y / 2, .5]
    model.stat.extent = max(room_x, room_y)
    camera.lookat[:] = [room_x / 2, room_y / 2, -.6]
    camera.distance = max(room_x, room_y) * 1.79
    camera.azimuth, camera.elevation = -66, -48
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        for a, b in zip(scene["route"], scene["route"][1:]):
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                               np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([.06, .56, .39, 1.]))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .033,
                                np.array([*a, .025]), np.array([*b, .025]))
            renderer.scene.ngeom += 1
        for point, rgba in ((scene["start"][:2], [.08, .35, .76, 1.]),
                            (scene["goal"], [.87, .37, .08, 1.])):
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                               np.array([.18, .025, .025]), np.array([*point, .027]),
                               np.eye(3).ravel(), np.array(rgba))
            renderer.scene.ngeom += 1
        return Image.fromarray(renderer.render().copy())


def render_plan(scene, size=900):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (size, size), "#ffffff")
    draw = ImageDraw.Draw(image)
    room_x, room_y = scene["room_dimensions"][:2]
    scale = (size - 92) / max(room_x, room_y)
    xpad = (size - room_x * scale) / 2
    ypad = (size - room_y * scale) / 2
    def pixel(xy):
        return xpad + xy[0] * scale, size - ypad - xy[1] * scale
    for box in sorted(scene["boxes"], key=lambda b: b["center"][2]):
        x, y = box["center"][:2]
        hx, hy = box["half_size"][:2]
        c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
        points = [pixel((x + c*dx*hx - s*dy*hy, y + s*dx*hx + c*dy*hy))
                  for dx, dy in ((-1,-1), (1,-1), (1,1), (-1,1))]
        draw.polygon(points, fill=PALETTE.get(box["category"], "#83939a"), outline="#556673")
    route = [pixel(point) for point in scene["route"]]
    draw.line(route, fill="#0e9066", width=5, joint="curve")
    for label, point, fill in (("A", route[0], "#165ac2"), ("B", route[-1], "#dc601c")):
        radius = 19
        draw.ellipse((point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius), fill=fill)
        draw.text(point, label, fill="white", font=font(23, True), anchor="mm")
    return image


def main():
    from PIL import Image, ImageDraw
    from cat_ppo.furniture.scenes import validate_scene
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    cards, records = [], []
    for entry in sorted(source["scenes"], key=lambda entry: entry["scene"]["seed"]):
        scene, record = entry["scene"], entry["record"]
        validate_scene(scene)
        if scene["geometry_hash"] != record["source"]["geometry_sha256"]:
            raise ValueError("Scene does not match active bank geometry")
        seed = scene["seed"]
        iso, plan = render_3d(scene), render_plan(scene)
        iso.save(args.output / f"scene-{seed}-3d.png")
        plan.save(args.output / f"scene-{seed}-plan.png")
        title = f"Furniture {seed}" if scene["family"] == "furniture" else f"Mixed clutter {seed}"
        counts = scene["counts"]
        objects = (f"{counts['tables']} tables + {counts['chairs']} chairs" if scene["family"] == "furniture"
                   else f"{counts['generic_objects']} crates, shelves, blocks and partitions")
        room = scene["room_dimensions"]
        card = Image.new("RGB", (850, 1560), "white")
        draw = ImageDraw.Draw(card)
        draw.text((28, 26), title, font=font(36, True), fill="#173346")
        draw.text((28, 77), objects, font=font(23), fill="#425a6d")
        draw.text((28, 112), f"{room[0]:.1f} × {room[1]:.1f} m  |  A–B route: {scene['route_length_m']:.1f} m",
                  font=font(24), fill="#425a6d")
        card.paste(iso.resize((850, 668), Image.Resampling.LANCZOS), (0, 158))
        draw.text((28, 826), "Floor plan · same geometry", font=font(25, True), fill="#173346")
        card.paste(plan.resize((700, 700), Image.Resampling.LANCZOS), (75, 855))
        card.save(args.output / f"scene-{seed}.png")
        cards.append(card)
        records.append(dict(seed=seed, scene_id=scene["scene_id"],
                            scene_sha256=record["scene_sha256"], geometry_hash=scene["geometry_hash"],
                            counts=counts, route_length_m=scene["route_length_m"]))
        print(f"Rendered exact scene {seed}", flush=True)
    sheet = Image.new("RGB", (2670, 1840), "#edf2f5")
    draw = ImageDraw.Draw(sheet)
    draw.text((36, 28), "Clutter scenes from the active CAT training bank", font=font(48, True), fill="#173346")
    draw.text((36, 98), "Random positions and orientations · G1 shown at its starting pose · static geometry previews",
              font=font(28), fill="#425a6d")
    for i, card in enumerate(cards):
        sheet.paste(card, (30 + i * 880, 164))
    draw.text((36, 1750), "Blue A: start   ·   Orange B: goal   ·   Green: geometrically checked root route, not a policy rollout",
              font=font(27), fill="#425a6d")
    draw.text((36, 1798), "Walls are cut away in 3D. The route checks a 23 cm root radius; whole-body traversal is evaluated separately.",
              font=font(26), fill="#425a6d")
    destination = args.output / "clutter-examples.png"
    sheet.save(destination)
    provenance = dict(active_bank_sha256=source["active_bank_sha256"], scenes=records,
                      renderer="MuJoCo static view of canonical boxes and current CAT G1 with fixed Dex3 hands",
                      simulation_steps=0, learned_policy=False,
                      visual_changes=["wall cutaway", "materials/colors", "lighting", "route/start/goal markers"],
                      source_bundle_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      image_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    (args.output / "render-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
