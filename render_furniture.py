"""Render the native MuJoCo furniture scene at a fixed pose, without a rollout.

The overview and robot inset use the exact physical model and scene start.
Colors, translucent walls, route markers and highlighted hand envelopes are
visual aids only. A fixed kinematic posture is not learned or dynamically held.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scene-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True, help="PNG; provenance is written alongside it")
    result.add_argument("--posture", choices=("nominal", "raised", "tucked", "contextual"), default="nominal")
    return result


def render(scene_dir, output, posture="nominal"):
    # Importing the native assembly helper should not acquire a compute GPU.
    # EGL still renders through OpenGL; there is no physics rollout or policy.
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from cat_ppo.envs.g1 import constants
    from cat_ppo.envs.g1.env_furniture import assemble_scene_xml, default_config
    from cat_ppo.furniture import control
    from cat_ppo.furniture.scenes import load_scene

    started = time.monotonic()
    scene_dir, output = Path(scene_dir).resolve(), Path(output).absolute()
    if output.suffix.lower() != ".png":
        raise ValueError("Output must be a PNG path")
    if output.is_symlink() or output.with_suffix(".json").is_symlink():
        raise ValueError("Refusing a symlink output")
    scene = load_scene(scene_dir)
    xml = assemble_scene_xml(scene)
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1400, 1040
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(constants.DEFAULT_QPOS)
    data.qpos[:2] = scene["start"][:2]
    yaw = float(scene["start"][2])
    data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    foot_ids = {model.geom(name).id for name in constants.FEET_GEOMS}
    floor_id = model.geom("floor").id

    def contacts():
        result = []
        for contact in data.contact:
            a, b = (int(value) for value in contact.geom)
            allowed = (a == floor_id and b in foot_ids) or (b == floor_id and a in foot_ids)
            if contact.dist <= 0 and not allowed:
                result.append({"geoms": [model.geom(a).name, model.geom(b).name],
                               "distance_m": float(contact.dist)})
        return result

    nominal_contacts = contacts()
    if nominal_contacts:
        raise ValueError(f"Native scene start has forbidden contacts: {nominal_contacts}")
    config = default_config()
    features = None
    if posture == "contextual":
        from cat_ppo.furniture.perception import probe_features, apply_uncertainty_margin
        positions = np.asarray([data.site_xpos[model.site(f"furniture_probe_{name}").id]
                                for name, _, _, _ in control.PROBE_SPECS])
        features = probe_features(np.load(scene_dir / "sdf.npy", allow_pickle=False),
            np.load(scene_dir / "bf.npy", allow_pickle=False), positions,
            np.zeros_like(positions), np.asarray([probe[3] for probe in control.PROBE_SPECS]),
            np.asarray(scene["grid"]["sample_origin"]), scene["grid"]["voxel_size"])
        features = np.asarray(apply_uncertainty_margin(features,
            fixed_margin=config.fixed_uncertainty_margin,
            resolution_error_bound=scene["grid"]["resolution_error_bound_m"]))
    action = control.posture_actions(np.zeros(29), control.JOINT_NAMES, posture,
                                    probe_features=features, upper_scale=config.upper_action_scale)
    nominal = np.asarray(constants.DEFAULT_QPOS[7:])
    lower, upper = model.jnt_range[1:].T
    center, radius = (lower + upper) / 2, (upper - lower) / 2 * config.soft_joint_pos_limit_factor
    targets = nominal.copy()
    # Kinematic preset target, using exactly the environment's bounded command
    # mapping. Repeated slew increments are algebra only, never mj_step/MJX.
    for _ in range(64):
        targets = control.motor_targets(action, targets, nominal, center-radius, center+radius,
            np.arange(29), leg_scale=config.action_scale, upper_scale=config.upper_action_scale,
            upper_rate=config.upper_target_rate, dt=config.ctrl_dt)
    data.qpos[7:] = targets
    mujoco.mj_forward(model, data)
    posture_contacts = contacts()
    physical_model = {"nq": model.nq, "nv": model.nv, "nu": model.nu,
                      "ngeom": model.ngeom, "furniture_boxes": len(scene["boxes"])}

    palette = {"tabletop": (.65, .43, .22, 1), "table_leg": (.30, .25, .21, 1),
        "chair_seat": (.07, .39, .44, 1), "chair_back": (.09, .43, .48, 1),
        "chair_leg": (.15, .23, .27, 1), "chair_armrest": (.12, .31, .34, 1),
        "overhead": (.86, .56, .18, 1), "overhead_support": (.51, .37, .20, 1),
        "wall": (.52, .60, .65, .16)}
    for i, box in enumerate(scene["boxes"]):
        model.geom_rgba[model.geom(f"furniture_object_{i}").id] = palette.get(box["category"], (.43,.47,.51,1))
    model.geom_rgba[floor_id] = (.91, .93, .94, 1)
    for side in ("left", "right"):
        geom = model.geom(f"furniture_{side}_hand_envelope").id
        model.geom_group[geom] = 4
        model.geom_rgba[geom] = (.96, .57, .16, .66)
    model.vis.headlight.ambient[:] = [.5] * 3
    model.vis.headlight.diffuse[:] = [.7] * 3
    model.vis.headlight.specular[:] = [.15] * 3
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0  # Hide collision overlays except highlighted hands.
    options.geomgroup[4] = 1
    options.sitegroup[:] = 0
    width, depth = scene["room_dimensions"][:2]

    def render_view(image_width, image_height, camera, draw_route=False):
        with mujoco.Renderer(model, height=image_height, width=image_width) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=options)
            if draw_route:
                for first, second in zip(scene["route"], scene["route"][1:]):
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), np.array([.0,.53,.55,1]))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .026,
                        np.array([*first,.018]), np.array([*second,.018]))
                    renderer.scene.ngeom += 1
                for point, rgba in ((scene["start"][:2], [.15,.42,.83,1]), (scene["goal"], [.20,.65,.32,1])):
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.11]*3),
                        np.array([*point,.10]), np.eye(3).reshape(-1), np.asarray(rgba))
                    renderer.scene.ngeom += 1
            return Image.fromarray(renderer.render())

    overview = mujoco.MjvCamera()
    overview.lookat[:] = [width/2, depth/2, .35]
    overview.distance, overview.azimuth, overview.elevation = max(width,depth)*1.60, -90, -60
    detail = mujoco.MjvCamera()
    detail.lookat[:] = [*scene["start"][:2], .85]
    detail.distance, detail.azimuth, detail.elevation = 2.8, -125 + np.degrees(yaw), -15
    main_image = render_view(1360, 1010, overview, True)
    detail_image = render_view(520, 1010, detail)
    canvas = Image.new("RGB", (1920, 1200), "white")
    canvas.paste(main_image, (20, 92));canvas.paste(detail_image, (1380, 92))
    draw = ImageDraw.Draw(canvas)

    def font(size):
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
            if Path(path).is_file():
                return ImageFont.truetype(path, size)
        return ImageFont.load_default(size=size)

    counts = scene.get("counts", {})
    draw.text((25, 14), "Native CAT whole-body furniture environment", font=font(32), fill="#16324F")
    draw.text((25, 55), f"{counts.get('tables', 0)} tables  |  {counts.get('chairs', 0)} chairs  |  {counts.get('bottlenecks', 0)} route bottlenecks", font=font(23), fill="#385368")
    draw.text((1390, 54), f"Start pose: {posture}", font=font(23), fill="#16324F")
    draw.text((25, 1120), "STATIC MUJOCO PREVIEW — fixed posture; no learned policy or rollout.", font=font(26), fill="#16324F")
    draw.text((25, 1162), "Teal: proposed root route. Amber: hand envelopes / overhead bars. Walls translucent for visibility.", font=font(22), fill="#385368")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    record = {"schema": "cat-furniture-static-render-v1", "output": str(output),
        "scene_dir": str(scene_dir), "scene_id": scene["scene_id"],
        "geometry_hash": scene["geometry_hash"],
        "scene_json_sha256": hashlib.sha256((scene_dir/"scene.json").read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assembled_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "png_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__, "physics_source": "assemble_scene_xml(scene); native MuJoCo model",
        "physical_model": physical_model, "posture": posture,
        "nominal_start_forbidden_contacts": nominal_contacts,
        "displayed_pose_forbidden_contacts": posture_contacts,
        "joint_targets_rad": targets.tolist(), "policy_loaded": False, "simulation_steps": 0,
        "posture_dynamically_validated": False, "root_route_dynamically_validated": False,
        "visual_only_changes": ["colors", "wall transparency", "hand envelope highlighting", "route/start/goal markers"],
        "elapsed_seconds": time.monotonic()-started}
    output.with_suffix(".json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    return record


if __name__ == "__main__":
    args = parser().parse_args()
    result = render(args.scene_dir, args.output, args.posture)
    print(json.dumps({key: value for key, value in result.items() if key != "joint_targets_rad"}, indent=2))
