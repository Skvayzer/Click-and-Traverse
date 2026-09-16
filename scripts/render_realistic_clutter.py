"""Render an existing CAT room with downloaded furniture assets in Blender.

Run with the isolated Python environment containing bpy. The visual meshes are
fitted to each generated object's bounding dimensions and yaw. This is a static
appearance study; it does not replace the training obstacle fields.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import bpy
from mathutils import Matrix, Vector


def material(name, rgb, roughness=.6, metallic=0.):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*rgb, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return mat


def box(name, center, dimensions, mat, yaw=0., bevel=0.):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.rotation_euler.z = yaw
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new("soft manufactured edges", "BEVEL")
        mod.width, mod.segments = bevel, 3
        obj.modifiers.new("weighted normals", "WEIGHTED_NORMAL")
    return obj


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def import_join(path, name):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    created = set(bpy.data.objects) - before
    meshes = [obj for obj in created if obj.type == "MESH"]
    non_meshes = [obj for obj in created if obj.type != "MESH"]
    if not meshes:
        raise ValueError(f"No meshes in {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        # glTF can parent meshes. Preserve their world transform when joining.
        transform = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = transform
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = name
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # glTF imports retain quaternion mode, which ignores later Euler yaw edits.
    obj.rotation_mode = "XYZ"
    for old in non_meshes:
        if old.name in bpy.data.objects:
            bpy.data.objects.remove(old, do_unlink=True)
    return obj


def normalize_template(obj, kind):
    coords = [v.co.copy() for v in obj.data.vertices]
    lo = Vector([min(v[i] for v in coords) for i in range(3)])
    hi = Vector([max(v[i] for v in coords) for i in range(3)])
    rotation = 0.
    if kind == "table" and hi.y-lo.y > hi.x-lo.x:
        rotation = math.pi / 2
    if kind == "chair":
        top = [v for v in coords if v.z > lo.z + .73 * (hi.z-lo.z)]
        # CAT's chair backs are on local negative Y.
        if sum(v.y for v in top) / len(top) > (lo.y+hi.y)/2:
            rotation = math.pi
    if rotation:
        obj.data.transform(Matrix.Rotation(rotation, 4, "Z"))
    coords = [v.co.copy() for v in obj.data.vertices]
    lo = Vector([min(v[i] for v in coords) for i in range(3)])
    hi = Vector([max(v[i] for v in coords) for i in range(3)])
    extent = hi-lo
    for vertex in obj.data.vertices:
        vertex.co.x = (vertex.co.x-(lo.x+hi.x)/2) / extent.x
        vertex.co.y = (vertex.co.y-(lo.y+hi.y)/2) / extent.y
        vertex.co.z = (vertex.co.z-lo.z) / extent.z
    obj.hide_render = True
    obj.hide_viewport = True
    return dict(native_dimensions_m=list(extent), orientation_adjustment_rad=rotation)


def fit_target(parts):
    anchor = next(part for part in parts if part["category"] in ("tabletop", "chair_seat"))
    yaw = anchor["yaw"]
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for part in parts:
        if part["category"] == "chair_armrest":
            continue
        dx, dy = part["center"][0]-anchor["center"][0], part["center"][1]-anchor["center"][1]
        center = (c*dx+s*dy, -s*dx+c*dy, part["center"][2])
        for a in (-1, 1):
            for b in (-1, 1):
                for z in (-1, 1):
                    corners.append([center[0]+a*part["half_size"][0],
                                    center[1]+b*part["half_size"][1],
                                    center[2]+z*part["half_size"][2]])
    lo = [min(p[i] for p in corners) for i in range(3)]
    hi = [max(p[i] for p in corners) for i in range(3)]
    x, y = (lo[0]+hi[0])/2, (lo[1]+hi[1])/2
    position = [anchor["center"][0]+c*x-s*y, anchor["center"][1]+s*x+c*y, lo[2]]
    return position, [hi[i]-lo[i] for i in range(3)], yaw


def floor_material(asset_dir, room):
    mat = material("Actual 2K scanned tile material", (.6, .6, .6))
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get("Principled BSDF")
    uv = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeVectorMath")
    mapping.operation = "SCALE"
    mapping.inputs[3].default_value = .5  # One texture repeat per two metres.
    links.new(uv.outputs["Object"], mapping.inputs[0])
    # A world-space empty makes metre scale independent of slab dimensions.
    empty = bpy.data.objects.new("Floor texture metre reference", None)
    bpy.context.collection.objects.link(empty)
    uv.object = empty
    textures = {}
    for channel in ("diff", "nor_gl", "rough"):
        node = nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(str(asset_dir / f"floor_tiles_02_{channel}_2k.jpg"))
        node.extension = "REPEAT"
        if channel != "diff":
            node.image.colorspace_settings.name = "Non-Color"
        links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        textures[channel] = node
    links.new(textures["diff"].outputs["Color"], shader.inputs["Base Color"])
    links.new(textures["rough"].outputs["Color"], shader.inputs["Roughness"])
    normal = nodes.new("ShaderNodeNormalMap")
    normal.inputs["Strength"].default_value = .55
    links.new(textures["nor_gl"].outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], shader.inputs["Normal"])
    return mat


def make_camera(name, position, target, lens):
    data = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    data.lens = lens
    data.clip_end = 200
    aim(obj, target)
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--robot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4008)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.source.read_text())
    entry = next(e for e in source["scenes"] if e["scene"]["seed"] == args.seed)
    room = entry["scene"]
    assets = json.loads((args.assets/"manifest.json").read_text())["assets"]
    # Verify files before loading external meshes/textures.
    for record in assets.values():
        for file in record["files"]:
            path = args.assets/file["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != file["sha256"]:
                raise ValueError(f"Asset checksum differs: {path}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24 if args.preview else args.samples
    scene.cycles.use_denoising = True
    scene.cycles.adaptive_threshold = .03 if args.preview else .015
    scene.cycles.max_bounces = 8
    devices = []
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "METAL"
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == "METAL"
            if device.use:
                devices.append(device.name)
        scene.cycles.device = "GPU" if devices else "CPU"
    except Exception:
        scene.cycles.device = "CPU"
    scene.render.resolution_x, scene.render.resolution_y = ((1280, 900) if args.preview else (2400, 1688))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = .45
    scene.world = bpy.data.worlds.new("Soft daylight")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.72, .80, 1., 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .35
    templates, template_info = {}, {}
    for kind, asset in (("table", "wooden_table_02"), ("chair", "SchoolChair_01")):
        obj = import_join(args.assets/assets[asset]["main_path"], "asset_template_"+kind)
        template_info[asset] = normalize_template(obj, kind)
        templates[kind] = obj
    groups = defaultdict(list)
    for part in room["boxes"]:
        if "furniture_id" in part:
            groups[part["furniture_id"]].append(part)
    arm_material = material("Dark padded armrests", (.026, .038, .044), .62)
    support_material = material("Armrest metal", (.022, .026, .03), .28, .72)
    placements = []
    for name, parts in groups.items():
        kind = parts[0]["furniture_type"]
        position, dimensions, yaw = fit_target(parts)
        obj = templates[kind].copy()
        obj.data = templates[kind].data
        bpy.context.collection.objects.link(obj)
        obj.name, obj.hide_render, obj.hide_viewport = name, False, False
        obj.location, obj.scale, obj.rotation_euler.z = position, dimensions, yaw
        placements.append(dict(name=name, kind=kind, position=position, dimensions=dimensions, yaw=yaw))
        for part in parts:
            if part["category"] == "chair_armrest":
                box(part["name"], part["center"], [2*v for v in part["half_size"]], arm_material,
                    yaw=part["yaw"], bevel=.012)
                for dy in (-.12, .12):
                    center = [part["center"][0]-math.sin(yaw)*dy,
                              part["center"][1]+math.cos(yaw)*dy, .555]
                    box(part["name"]+f" support {dy}", center, (.018, .018, .21),
                        support_material, yaw=yaw, bevel=.006)
    x, y, h = room["room_dimensions"]
    floor = floor_material(args.assets/"floor_tiles_02", room)
    slab = box("Tiled floor", (x/2, y/2, -.065), (x+.24, y+.24, .13), floor, bevel=.016)
    paint = material("Warm plaster", (.76, .735, .69), .83)
    trim = material("Painted baseboards", (.34, .37, .36), .48)
    # North and east walls establish an interior, with a cutaway front/left for
    # the overview. Architectural additions are visual staging only.
    box("North wall", (x/2, y+.055, h/2), (x+.24, .11, h), paint, bevel=.01)
    box("East wall", (x+.055, y/2, h/2), (.11, y+.24, h), paint, bevel=.01)
    box("North baseboard", (x/2, y-.012, .05), (x, .04, .10), trim, bevel=.004)
    box("East baseboard", (x-.012, y/2, .05), (.04, y, .10), trim, bevel=.004)
    # Restrained wall details provide scale without adding traversal clutter.
    dark = material("Noticeboard charcoal", (.038, .063, .066), .7)
    frame = material("Noticeboard frame", (.2, .23, .22), .4, .25)
    box("Noticeboard frame", (x*.48, y-.025, 1.62), (2.45, .055, .85), frame, bevel=.016)
    box("Noticeboard", (x*.48, y-.060, 1.62), (2.35, .025, .75), dark, bevel=.01)
    paper = material("Paper", (.80, .78, .70), .9)
    for i, (px, pz) in enumerate(((.15,.04), (-.56,-.08), (.64,-.1))):
        box(f"Notice paper {i}", (x*.48+px, y-.077, 1.62+pz), (.28,.005,.38), paper,
            yaw=0., bevel=.003)
    robot = import_join(args.robot, "G1 at actual scene start")
    robot.location = (*room["start"][:2], 0.)
    robot.rotation_euler.z = room["start"][2]
    # Validate applied transforms rather than trusting recorded target poses.
    bpy.context.view_layer.update()
    for placement in placements:
        obj = bpy.data.objects[placement["name"]]
        matrix = obj.matrix_world
        actual_yaw = math.atan2(matrix[1][0], matrix[0][0])
        yaw_error = math.atan2(math.sin(actual_yaw-placement["yaw"]),
                               math.cos(actual_yaw-placement["yaw"]))
        assert abs(yaw_error) < 1e-5, (obj.name, actual_yaw, placement["yaw"])
        assert (matrix.translation-Vector(placement["position"])).length < 1e-5
        assert (matrix.to_scale()-Vector(placement["dimensions"])).length < 1e-5
        placement["applied_yaw"] = actual_yaw
    robot_yaw = math.atan2(robot.matrix_world[1][0], robot.matrix_world[0][0])
    assert abs(math.atan2(math.sin(robot_yaw-room["start"][2]),
                          math.cos(robot_yaw-room["start"][2]))) < 1e-5
    for label, position, rgb in (("A", room["start"][:2], (.07,.29,.42)),
                                  ("B", room["goal"], (.64,.21,.045))):
        mat = material(label+" floor marker", rgb, .65)
        bpy.ops.mesh.primitive_torus_add(major_radius=.23, minor_radius=.012,
                                        location=(*position, .016), major_segments=64, minor_segments=10)
        bpy.context.object.name = label+" floor ring"
        bpy.context.object.data.materials.append(mat)
    for name, location, target, energy, size, rgb in (
        ("Daylight", (-1.5, y*.35, 5.5), (x*.55,y*.55,0), 1900, 5.0, (1., .92, .81)),
        ("Front bounce", (x*.65,-2.,4.), (x*.5,y*.5,1), 500, 5., (.78,.88,1.)),
        ("Ceiling soft fill", (x*.6,y*.5,6.), (x*.5,y*.5,0), 600, 6., (1.,1.,1.))):
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.shape, data.size, data.color = energy, "DISK", size, rgb
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        aim(obj, target)
    sun_data = bpy.data.lights.new("Low afternoon sun", "SUN")
    sun_data.energy, sun_data.angle = 1.6, math.radians(6)
    sun_data.color = (1., .91, .80)
    sun = bpy.data.objects.new("Low afternoon sun", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(30), math.radians(-20), math.radians(-35))
    cameras = {
        "overview": make_camera("Room overview", (-4.4,-7.2,8.8), (x*.48,y*.49,.45), 42),
        "interior": make_camera("Across the clutter", (.65,.65,1.75), (x*.56,y*.69,1.0), 22),
    }
    scene.camera = cameras["overview"]
    # Pack texture images so the project can be opened independently.
    bpy.ops.file.pack_all()
    blend = args.output/f"realistic-clutter-{args.seed}.blend"
    if not args.preview:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend), check_existing=False)
    outputs = {}
    for name, camera in cameras.items():
        scene.camera = camera
        scene.render.filepath = str(args.output/(f"preview-{name}.png" if args.preview else f"realistic-clutter-{name}.png"))
        started = time.monotonic()
        bpy.ops.render.render(write_still=True)
        outputs[name] = dict(path=scene.render.filepath, seconds=time.monotonic()-started,
                             sha256=hashlib.sha256(Path(scene.render.filepath).read_bytes()).hexdigest())
    provenance = dict(scene_id=room["scene_id"], geometry_hash=room["geometry_hash"],
                      active_bank_sha256=source["active_bank_sha256"], source_scene_sha256=entry["record"]["scene_sha256"],
                      blender_version=bpy.app.version_string, engine="CYCLES", devices=devices,
                      resolution=[scene.render.resolution_x,scene.render.resolution_y],samples=scene.cycles.samples,
                      object_counts=room["counts"], asset_templates=template_info,placements=placements,
                      asset_manifest_sha256=hashlib.sha256((args.assets/"manifest.json").read_bytes()).hexdigest(),
                      robot_glb_sha256=hashlib.sha256(args.robot.read_bytes()).hexdigest(),
                      static_pose=True, simulation_steps=0, policy_rollout=False,
                      geometry_note="Mesh appearance fitted to original object OBB dimensions and yaw; detailed surfaces differ from training SDF. Existing armrests retained with visual support posts.",
                      architectural_staging="Cutaway walls, baseboards, noticeboard, daylight, floor surface", outputs=outputs)
    (args.output/("preview.json" if args.preview else "render-provenance.json")).write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps(dict(outputs=outputs,devices=devices,object_count=len(placements)),indent=2),flush=True)


if __name__ == "__main__":
    main()
