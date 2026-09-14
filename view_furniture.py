"""Inspect the actual furniture geometry in Viser, with a fixed native robot pose.

Prepare assets using the training environment, then serve them using the small
separate visualization environment. No policy inference or physics steps run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(scene_dir, output):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import mujoco
    import numpy as np
    import trimesh
    from cat_ppo.envs.g1 import constants
    from cat_ppo.envs.g1.env_furniture import assemble_scene_xml
    from cat_ppo.furniture.control import observation_contract
    from cat_ppo.furniture.grippers import validate_hand_envelopes
    from cat_ppo.furniture.scenes import load_scene

    scene_dir, output = Path(scene_dir).resolve(), Path(output).resolve()
    scene = load_scene(scene_dir)
    xml = assemble_scene_xml(scene)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(constants.DEFAULT_QPOS)
    data.qpos[:2] = scene["start"][:2]
    yaw = scene["start"][2]
    data.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    mujoco.mj_forward(model, data)
    containment = validate_hand_envelopes(model, data)
    mesh_scene = trimesh.Scene()
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] == 0 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        vertices = np.asarray(model.mesh_vert[va:va+vn], dtype=float)
        vertices = np.einsum("ij,kj->ik", vertices, data.geom_xmat[geom].reshape(3,3)) + data.geom_xpos[geom]
        if not np.isfinite(vertices).all():
            raise ValueError("Native robot mesh contains nonfinite transformed vertices")
        geometry = trimesh.Trimesh(vertices=vertices, faces=model.mesh_face[fa:fa+fn], process=False)
        geometry.visual.face_colors = np.asarray(np.clip(model.geom_rgba[geom] * 255, 0, 255), dtype=np.uint8)
        mesh_scene.add_geometry(geometry, node_name=f"robot_visual_{geom}")
    hands = []
    for side in ("left", "right"):
        geom = model.geom(f"furniture_{side}_hand_envelope").id
        hands.append(dict(side=side, center=data.geom_xpos[geom].tolist(),
            dimensions=(2*model.geom_size[geom]).tolist(), rotation=data.geom_xmat[geom].reshape(3,3).tolist()))
    output.mkdir(parents=True, exist_ok=False)
    mesh_scene.export(output / "robot.glb")
    (output / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    (output / "hands.json").write_text(json.dumps(hands, indent=2) + "\n")
    manifest = dict(schema="cat-furniture-viser-v1", geometry_hash=scene["geometry_hash"],
        native_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        robot_mesh_source="compiled MuJoCo mesh vertices and native FK geom poses",
        hand_geometry=observation_contract()["hand_geometry"],
        hand_envelope_check=containment,
        simulation_steps=0, policy_loaded=False,
        files={name: sha256(output/name) for name in ("scene.json", "robot.glb", "hands.json")})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def serve(bundle, *, port=8085):
    import numpy as np
    import viser
    import viser.transforms as tf
    from cat_ppo.furniture.scenes import validate_scene

    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(bundle/name) != digest:
            raise ValueError(f"Viewer asset hash mismatch: {name}")
    scene = json.loads((bundle / "scene.json").read_text())
    validate_scene(scene)
    if scene["geometry_hash"] != manifest["geometry_hash"]:
        raise ValueError("Viewer geometry differs from prepared native model")
    width, depth, _ = scene["room_dimensions"]
    server = viser.ViserServer(host="127.0.0.1", port=port, label="CAT · Furniture traversal")
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    server.gui.configure_theme(control_layout="floating", control_width="medium", dark_mode=False,
        show_logo=False, show_share_button=False, brand_color=(16, 114, 122))
    server.initial_camera.position = (14.5, -8.2, 12.5)
    server.initial_camera.look_at = (width/2, depth/2, .35)
    server.initial_camera.up_direction = (0, 0, 1)
    server.initial_camera.fov = .76
    server.scene.add_box("/floor", dimensions=(width+.2, depth+.2, .04), position=(width/2, depth/2, -.03),
        color=(229, 234, 237), receive_shadow=True)
    walls = server.scene.add_frame("/walls", show_axes=False)
    room = server.scene.add_frame("/furniture", show_axes=False)
    palette = {"tabletop": (177, 124, 67), "table_leg": (77, 69, 62),
        "chair_seat": (26, 116, 130), "chair_back": (27, 121, 134), "chair_leg": (42, 66, 74),
        "chair_armrest": (32, 91, 101), "wall": (125, 144, 155), "overhead": (228, 161, 62)}
    for box in scene["boxes"]:
        is_wall = box["category"] == "wall"
        server.scene.add_box(f"/{'walls' if is_wall else 'furniture'}/{box['name']}",
            dimensions=tuple(2*np.asarray(box["half_size"])), position=tuple(box["center"]),
            wxyz=tf.SO3.from_z_radians(box["yaw"]).wxyz,
            color=palette.get(box["category"], (115, 102, 84)),
            opacity=.15 if is_wall else 1., cast_shadow=not is_wall, receive_shadow=True)
    server.scene.add_glb("/robot", (bundle / "robot.glb").read_bytes())
    hand_group = server.scene.add_frame("/hand_envelopes", show_axes=False)
    hands = json.loads((bundle / "hands.json").read_text())
    for hand in hands:
        rotation = np.asarray(hand["rotation"])
        center = np.asarray(hand["center"])
        server.scene.add_box("/hand_envelopes/"+hand["side"], dimensions=tuple(hand["dimensions"]),
            position=tuple(center), wxyz=tf.SO3.from_matrix(rotation).wxyz,
            color=(243, 165, 55), opacity=.10, cast_shadow=False)
        # Wire edges keep the thumb and fingertips visible through the enclosure.
        from itertools import product
        signs = np.asarray(list(product((-1, 1), repeat=3)))
        corners = np.einsum("ij,kj->ik", signs*np.asarray(hand["dimensions"])/2, rotation)+center
        edges = [(a,b) for a in range(8) for b in range(a+1,8)
                 if np.count_nonzero(signs[a] != signs[b]) == 1]
        server.scene.add_line_segments("/hand_envelopes/"+hand["side"]+"_edges",
            corners[np.asarray(edges)], colors=(220, 135, 24), thickness=.0015)
    route = np.column_stack([scene["route"], np.full(len(scene["route"]), .026)])
    route_handle = server.scene.add_line_segments("/route", np.stack([route[:-1], route[1:]], axis=1),
        colors=(13, 157, 151), thickness=.033)
    gate_group = server.scene.add_frame("/gates", show_axes=False)
    for i, gate in enumerate(scene["bottlenecks"]):
        x, y = gate["center"]
        half = gate["width_m"] / 2
        points = np.array([[[x,y-half,.08],[x,y+half,.08]]])
        server.scene.add_line_segments(f"/gates/{i}/width", points, colors=(211,121,30), thickness=.022)
        server.scene.add_label(f"/gates/{i}/label", f"{i+1} · {gate['width_m']*100:.0f} cm",
            position=(x,y,1.12), font_screen_scale=.9)
    server.scene.add_label("/start", "START", position=tuple(route[0]+[0,0,.12]))
    server.scene.add_label("/goal", "GOAL", position=tuple(route[-1]+[0,0,.12]))
    counts = scene.get("counts", {})
    server.gui.add_markdown(f"## Whole-body furniture traversal\n**{counts.get('tables',0)} tables · {counts.get('chairs',0)} chairs · {len(scene.get('bottlenecks',[]))} narrow passages**\n\nDrag to orbit; scroll to zoom.\n\nCamera inspection of the actual scene. The robot is in a fixed pose; this is not a policy rollout.")
    views = {
        "Room overview": ((14.5,-8.2,12.5), (width/2,depth/2,.35), (0,0,1), .76),
        "Overhead layout": ((width/2,depth/2,15.8), (width/2,depth/2,0), (0,1,0), .70),
        "Aisle at robot height": ((8.05, scene["bottlenecks"][0]["center"][1],1.22),
            (2.2,scene["bottlenecks"][0]["center"][1],.94), (0,0,1), .97),
        "Robot and hand clearance": ((2.25,-1.15,1.8), (*scene["start"][:2],.8), (0,0,1), .78),
    }
    for hand in hands:
        center, rotation = np.asarray(hand["center"]), np.asarray(hand["rotation"])
        eye = center + rotation @ np.array([.015, .16 if hand["side"] == "left" else -.16, .42])
        views[hand["side"].capitalize()+" gripper envelope"] = (tuple(eye), tuple(center), (0,0,1), .68)
    for name, (position, target, up, fov) in views.items():
        button = server.gui.add_button(name)
        @button.on_click
        def change_view(event, position=position, target=target, up=up, fov=fov):
            if event.client is not None:
                with event.client.atomic():
                    event.client.camera.position = position
                    event.client.camera.look_at = target
                    event.client.camera.up_direction = up
                    event.client.camera.fov = fov
    for label, handle in (("Show walls", walls), ("Show route", route_handle),
                          ("Show passage widths", gate_group), ("Show hand envelopes", hand_group)):
        checkbox = server.gui.add_checkbox(label, initial_value=True)
        @checkbox.on_update
        def update_visibility(event, handle=handle):
            handle.visible = event.target.value
    server.gui.add_markdown("Teal: intended root route. Amber: passage-width markers, hand envelopes and overhead constraints. Widths describe the generator's root-height clearance band, not certified full-body clearance.")
    print(json.dumps({"url": f"http://127.0.0.1:{server.get_port()}", "geometry_hash": scene["geometry_hash"],
                      "simulation_steps": 0, "policy_loaded": False}), flush=True)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--scene-dir", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    view = commands.add_parser("serve")
    view.add_argument("--bundle", type=Path, required=True)
    view.add_argument("--port", type=int, default=8085)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.scene_dir, args.output), indent=2))
    else:
        server = serve(args.bundle, port=args.port)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.stop()


if __name__ == "__main__":
    main()
