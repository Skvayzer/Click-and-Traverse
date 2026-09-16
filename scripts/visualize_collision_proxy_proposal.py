"""Render a mesh-fitted collision proposal for review; never change training.

The fitted shapes are private preview geometry. This script runs forward
kinematics only and saves an auditable proposal plus a single review image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")


def main():
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from cat_ppo.envs.g1 import constants
    from cat_ppo.envs.g1.env_cat_wholebody import assemble_training_xml
    from collision_proxy_preview_geometry import build_proposal, validate_proposal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "docs/assets/collision-proxy-proposal-20260916")
    parser.add_argument("--copy-to", type=Path,
                        default=Path.home() / "Downloads/CAT-collision-shapes-proposal.png")
    args = parser.parse_args()
    xml = assemble_training_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:] = constants.DEFAULT_QPOS
    mujoco.mj_forward(model, data)
    proposal = build_proposal(model, data)

    # Only move joints for the review view; no integration or policy inference.
    for side, sign in (("left", 1), ("right", -1)):
        for name, value in (("shoulder_roll", sign * .42),
                            ("shoulder_pitch", 0.), ("elbow", .20)):
            address = int(model.joint(f"{side}_{name}_joint").qposadr[0])
            data.qpos[address] = value
    mujoco.mj_forward(model, data)
    proposal["review_pose_validation"] = validate_proposal(model, data, proposal)
    review_qpos = data.qpos.copy()
    proposal["additional_pose_validation"] = []
    for name, changes in (
        ("raised_arms", {"left_shoulder_pitch_joint": -.8,
                         "right_shoulder_pitch_joint": -.8,
                         "left_elbow_joint": 1.1, "right_elbow_joint": 1.1}),
        ("step_and_waist_turn", {"waist_yaw_joint": .35, "left_hip_pitch_joint": -.5,
                                 "left_knee_joint": .7, "left_ankle_pitch_joint": -.2}),
    ):
        data.qpos[:] = review_qpos
        for joint_name, value in changes.items():
            data.qpos[int(model.joint(joint_name).qposadr[0])] = value
        mujoco.mj_forward(model, data)
        result = validate_proposal(model, data, proposal)
        proposal["additional_pose_validation"].append({"name": name,
            "qpos": data.qpos.tolist(), "validation": result})
    data.qpos[:] = review_qpos
    mujoco.mj_forward(model, data)
    proposal["preview"] = {
        "status": "Proposal for visual review; not active in training",
        "training_changed": False,
        "actor_observations": 222,
        "critic_observations": 310,
        "source_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "review_qpos": data.qpos.tolist(),
        "rendering": "Native MuJoCo forward kinematics; original meshes and mesh-fitted primitives",
    }

    colors = {
        "feet": (0.96, .43, .10), "legs": (.05, .60, .63),
        "body": (.16, .42, .83), "arms": (.61, .34, .78),
        "hands": (.20, .63, .32),
    }

    def region(shape):
        name = shape["body_name"]
        if shape.get("group") == "hands" or "hand" in shape.get("id", ""):
            return "hands"
        if "ankle" in name:
            return "feet"
        if any(word in name for word in ("hip", "knee")):
            return "legs"
        if any(word in name for word in ("shoulder", "elbow", "wrist")):
            return "arms"
        return "body"

    # Display each visual mesh once; hide physical-contact/debug primitives.
    base_groups = np.full(model.ngeom, 5, dtype=np.int32)
    seen = set()
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        key = (int(model.geom_bodyid[geom]), int(model.geom_dataid[geom]),
               tuple(model.geom_pos[geom]), tuple(model.geom_quat[geom]))
        if key in seen:
            continue
        seen.add(key)
        base_groups[geom] = 2
        model.geom_matid[geom] = -1
        model.geom_rgba[geom] = [.53, .56, .61, 1.]
    model.geom_group[:] = base_groups
    model.vis.global_.offwidth = 900
    model.vis.global_.offheight = 1150
    model.vis.global_.fovy = 32
    model.vis.headlight.ambient[:] = [.65] * 3
    model.vis.headlight.diffuse[:] = [.55] * 3
    model.vis.headlight.specular[:] = [.12] * 3
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[2] = 1
    option.sitegroup[:] = 0

    def geometry(shape):
        body = model.body(shape["body_name"]).id
        rotation = data.xmat[body].reshape(3, 3)
        origin = data.xpos[body]
        if shape["kind"] == "capsule":
            ends = np.asarray(shape["endpoints"])
            world = np.einsum("ij,nj->ni", rotation, ends) + origin
            delta = world[1] - world[0]
            length = float(np.linalg.norm(delta))
            if length < 1e-10:
                return (mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([shape["radius"], 0., 0.]), world.mean(0), rotation)
            z = delta / max(length, 1e-12)
            ref = np.array([1., 0., 0.]) if abs(z[0]) < .9 else np.array([0., 1., 0.])
            x = np.cross(ref, z)
            x /= np.linalg.norm(x)
            mat = np.column_stack((x, np.cross(z, x), z))
            return (mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.array([shape["radius"], length / 2, 0.]), world.mean(0), mat)
        center = np.asarray(shape["center"])
        pos = origin + rotation @ center
        if shape["kind"] == "sphere":
            return (mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([shape["radius"], 0., 0.]), pos, rotation)
        local = np.empty(9)
        mujoco.mju_quat2Mat(local, np.asarray(shape.get("quat", [1., 0., 0., 0.])))
        return (mujoco.mjtGeom.mjGEOM_BOX, np.asarray(shape["half_size"]),
                pos, rotation @ local.reshape(3, 3))

    def render(width, height, azimuth, elevation, distance, target, foot_only=False):
        model.geom_group[:] = base_groups
        if foot_only:
            for geom in range(model.ngeom):
                body = model.body(int(model.geom_bodyid[geom])).name
                if body not in ("left_ankle_roll_link", "left_ankle_pitch_link"):
                    model.geom_group[geom] = 5
        shapes = [s for s in proposal["shapes"] if not foot_only or
                  s["body_name"] in ("left_ankle_roll_link", "left_ankle_pitch_link")]
        camera = mujoco.MjvCamera()
        camera.lookat[:] = target
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            for shape in shapes:
                kind, size, pos, mat = geometry(shape)
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom, kind, size, pos, mat.ravel(),
                                   np.array([*colors[region(shape)], .23]))
                geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                geom.objtype = mujoco.mjtObj.mjOBJ_GEOM
                geom.objid = 0  # A valid foreground id for segmentation masking.
                geom.segid = renderer.scene.ngeom
                renderer.scene.ngeom += 1
            rgb = renderer.render().copy()
            cameras = renderer.scene.camera
            eye = (np.asarray(cameras[0].pos) + np.asarray(cameras[1].pos)) / 2
            forward = np.asarray(cameras[0].forward).copy()
            up = np.asarray(cameras[0].up).copy()
            right = np.cross(forward, up)
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render()
            rgb[segmentation[:, :, 0] < 0] = 255
        picture = Image.fromarray(rgb)
        draw = ImageDraw.Draw(picture)
        focal = height / (2 * np.tan(np.deg2rad(float(model.vis.global_.fovy)) / 2))

        def project(points):
            delta = np.asarray(points) - eye
            depth = np.einsum("ni,i->n", delta, forward)
            return np.column_stack((width / 2 + focal * np.einsum("ni,i->n", delta, right) / depth,
                                    height / 2 - focal * np.einsum("ni,i->n", delta, up) / depth))

        # Draw precise box edges on top so sole thickness can be judged.
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        for shape in shapes:
            if shape["kind"] != "box":
                continue
            _, half, pos, mat = geometry(shape)
            corners = np.einsum("ij,nj->ni", mat, signs * half) + pos
            p = project(corners)
            color = tuple(int(255 * c) for c in colors[region(shape)])
            for a in range(8):
                for b in range(a + 1, 8):
                    if np.count_nonzero(signs[a] != signs[b]) == 1:
                        draw.line([tuple(p[a]), tuple(p[b])], fill=color, width=3 if foot_only else 2)
        return picture

    canvas = Image.new("RGB", (2200, 1530), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = next(p for p in (Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) if p.is_file())
    font = lambda size: ImageFont.truetype(str(font_path), size)
    ink, muted = "#19354A", "#536776"
    draw.text((45, 28), "G1 collision geometry — proposal for review", font=font(44), fill=ink)
    draw.text((45, 90), "Actual robot meshes in grey; proposed collision volumes in color. Feet use flat boxes.",
              font=font(27), fill=muted)
    canvas.paste(render(680, 1130, 0, -4, 2.85, [0., 0., .77]), (10, 160))
    canvas.paste(render(680, 1130, 90, -4, 2.85, [0., 0., .77]), (680, 160))
    draw.text((45, 156), "FRONT", font=font(25), fill=ink)
    draw.text((730, 156), "SIDE", font=font(25), fill=ink)
    foot = data.xpos[model.body("left_ankle_roll_link").id]
    foot_target = foot + data.xmat[model.body("left_ankle_roll_link").id].reshape(3, 3) @ np.array([.035, 0., -.01])
    canvas.paste(render(800, 500, 90, -3, .49, foot_target, True), (1380, 180))
    canvas.paste(render(800, 500, 0, -85, .49, foot_target, True), (1380, 770))
    draw.text((1410, 156), "LEFT FOOT — SIDE", font=font(25), fill=ink)
    draw.text((1410, 740), "LEFT FOOT — TOP", font=font(25), fill=ink)
    draw.text((1410, 685), "Flat sole + raised ankle; toe and heel included", font=font(25), fill=muted)
    sole = next(s for s in proposal["shapes"] if s["id"] == "left_sole")
    dimensions = np.asarray(sole["half_size"]) * 200
    draw.text((1410, 716),
              f"Sole box: {dimensions[0]:.1f} × {dimensions[1]:.1f} × {dimensions[2]:.1f} cm; 3 mm padding",
              font=font(21), fill=muted)

    labels = [("feet", "Feet / ankles: boxes"), ("legs", "Legs"), ("body", "Trunk / head"),
              ("arms", "Arms / wrists"), ("hands", "Hands: existing spheres")]
    x = 45
    for key, label in labels:
        color = tuple(int(c * 255) for c in colors[key])
        draw.rounded_rectangle((x, 1330, x + 25, 1355), radius=5, fill=color)
        draw.text((x + 37, 1325), label, font=font(25), fill=ink)
        x += int(draw.textlength(label, font=font(25))) + 86
    counts = {kind: sum(s["kind"] == kind for s in proposal["shapes"])
              for kind in ("box", "capsule", "sphere")}
    summary = (f"{len(proposal['shapes'])} proposed shapes: {counts['box']} boxes, "
               f"{counts['capsule']} capsules, {counts['sphere']} spheres")
    draw.text((45, 1392), summary, font=font(27), fill=ink)
    draw.text((45, 1440), "Geometry preview only. Training unchanged. No new policy observations. GPU cost has not been measured.",
              font=font(25), fill=muted)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "collision-shapes-proposal.png"
    canvas.save(output)
    (args.output_dir / "proposal.json").write_text(json.dumps(proposal, indent=2, allow_nan=False) + "\n")
    args.copy_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, args.copy_to)
    print(output)
    print(args.copy_to)
    print(summary)


if __name__ == "__main__":
    main()
