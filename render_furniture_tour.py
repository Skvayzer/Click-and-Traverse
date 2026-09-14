"""Native MuJoCo camera tour of an existing scene; no simulation or policy.

Example (Linux EGL):
  MUJOCO_GL=egl python render_furniture_tour.py --scene-dir generated_scenes/verified_v1/dense --output-dir /tmp/cat-visuals

The robot stays at the real scene start. Camera motion, colors, translucent
walls and guide markers are presentation aids; collision geometry is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--duration", type=float, default=27.0)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--stills-only", action="store_true")
    p.add_argument("--ffmpeg", type=Path, help="Optional encoder binary from a separate visualization environment")
    return p


def render_tour(args):
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from cat_ppo.envs.g1 import constants
    from cat_ppo.envs.g1.env_furniture import assemble_scene_xml
    from cat_ppo.furniture.scenes import load_scene

    if not 20 <= args.duration <= 35 or not 12 <= args.fps <= 60:
        raise ValueError("Use duration 20–35 seconds and fps 12–60")
    if args.width < 640 or args.height < 360 or args.width % 2 or args.height % 2:
        raise ValueError("Video dimensions must be even and at least 640x360")
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_dir = args.scene_dir.resolve()
    scene = load_scene(scene_dir)
    xml = assemble_scene_xml(scene)
    display_xml = ET.fromstring(xml)
    ET.SubElement(display_xml.find("asset"), "texture", name="tour_sky", type="skybox",
        builtin="gradient", rgb1="0.83 0.89 0.94", rgb2="0.97 0.98 0.99", width="512", height="3072")
    model = mujoco.MjModel.from_xml_string(ET.tostring(display_xml,encoding="unicode"))
    model.vis.global_.offwidth = max(args.width, 1920)
    model.vis.global_.offheight = max(args.height, 1080)
    model.vis.quality.shadowsize = 4096
    model.vis.headlight.ambient[:] = [.48] * 3
    model.vis.headlight.diffuse[:] = [.65] * 3
    model.vis.headlight.specular[:] = [.10] * 3
    model.vis.map.znear = .003
    data = mujoco.MjData(model)
    data.qpos[:] = constants.DEFAULT_QPOS
    data.qpos[:2] = scene["start"][:2]
    yaw = scene["start"][2]
    data.qpos[3:7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
    mujoco.mj_forward(model, data)
    original_pose = data.qpos.copy()
    foot_ids = {model.geom(name).id for name in constants.FEET_GEOMS}
    floor_id = model.geom("floor").id
    forbidden = []
    for contact in data.contact:
        a, b = map(int, contact.geom)
        allowed = (a == floor_id and b in foot_ids) or (b == floor_id and a in foot_ids)
        if contact.dist <= 0 and not allowed:
            forbidden.append([model.geom(a).name, model.geom(b).name])
    if forbidden:
        raise ValueError(f"Scene start has forbidden physical contacts: {forbidden}")

    colors = {"tabletop": (.66,.43,.22,1), "table_leg": (.31,.26,.22,1),
        "chair_seat": (.04,.36,.41,1), "chair_back": (.07,.42,.46,1),
        "chair_leg": (.13,.22,.25,1), "chair_armrest": (.10,.30,.33,1),
        "overhead": (.89,.58,.19,1), "overhead_support": (.51,.37,.20,1),
        "wall": (.57,.65,.70,.15)}
    for i, box in enumerate(scene["boxes"]):
        model.geom_rgba[model.geom(f"furniture_object_{i}").id] = colors.get(box["category"], (.40,.46,.50,1))
    model.geom_rgba[floor_id] = (.91,.93,.94,1)
    for side in ("left", "right"):
        geom = model.geom(f"furniture_{side}_hand_envelope").id
        model.geom_group[geom] = 4
        model.geom_rgba[geom] = (.99,.60,.13,.65)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.geomgroup[4] = 1
    options.sitegroup[:] = 0
    room_x, room_y = scene["room_dimensions"][:2]
    span = max(room_x, room_y)
    gates = sorted(scene.get("bottlenecks", []), key=lambda g:g["route_distance_m"])
    aisle_y = gates[0]["center"][1] if gates else scene["route"][1][1]
    counts = scene.get("counts", {})
    font_path = next((path for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf") if Path(path).is_file()), None)

    def font(size):
        return ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default(size=size)

    def smooth(t):
        return t*t*(3-2*t)

    def camera(shot, fraction):
        u = smooth(float(np.clip(fraction, 0, 1)))
        c = mujoco.MjvCamera()
        if shot == "overview":
            c.lookat[:] = [room_x/2, room_y/2, .35]
            c.distance = span * 1.80
            c.azimuth = -100 + 36*u
            c.elevation = -54 - 6*u
        elif shot == "overhead":
            c.lookat[:] = [room_x/2, room_y/2, 0]
            c.distance = span * (1.74 - .05*u)
            c.azimuth = -90
            c.elevation = -89.9
        else:
            # View down an actual tight aisle at G1-scale eye height. Only the
            # camera moves; there is no robot traversal or collision claim.
            eye = np.array([room_x - .85 - .9*u, aisle_y, 1.15])
            target = np.array([.8, aisle_y + .025*math.sin(math.pi*u), .86])
            offset = eye - target
            c.lookat[:] = target
            c.distance = float(np.linalg.norm(offset))
            c.azimuth = math.degrees(math.atan2(-offset[1], -offset[0]))
            c.elevation = -math.degrees(math.asin(offset[2]/c.distance))
        return c

    def annotations(image, shot):
        image = image.convert("RGBA")
        layer = Image.new("RGBA", image.size)
        d = ImageDraw.Draw(layer)
        scale = image.width / 1280
        f = lambda n: font(max(12, int(n*scale)))
        margin = int(26*scale)
        header_h, footer_h = int(98*scale), int(51*scale)
        d.rectangle((0,0,image.width,header_h), fill=(255,255,255,241))
        d.rectangle((0,image.height-footer_h,image.width,image.height), fill=(22,50,79,241))
        titles = {"overview": "Dense furniture room · oblique overview",
                  "overhead": "Winding route · overhead view",
                  "aisle": "Between chair backs · camera height 1.15 m"}
        d.text((margin,int(14*scale)), titles[shot], fill="#16324F", font=f(27))
        subtitle = (f"{counts.get('tables',0)} tables  /  {counts.get('chairs',0)} chairs  /  "
                    f"{counts.get('bottlenecks',0)} route bottlenecks   ·   native MuJoCo geometry")
        d.text((margin,int(55*scale)), subtitle, fill="#486172", font=f(19))
        d.text((margin,image.height-footer_h+int(14*scale)),
            "CAMERA-ONLY SCENE TOUR  ·  robot fixed at start  ·  no learned rollout", fill="white", font=f(19))
        if shot != "aisle":
            label = "Teal: proposed root route   |   Amber: overhead constraints"
            box_y = image.height-footer_h-int(43*scale)
            bounds = d.textbbox((0,0), label, font=f(16))
            d.rounded_rectangle((margin-9,box_y-7,margin+bounds[2]+12,box_y+int(29*scale)),
                radius=int(6*scale), fill=(255,255,255,230))
            d.text((margin,box_y),label,fill="#385368",font=f(16))
        return Image.alpha_composite(image,layer).convert("RGB")

    def frame(renderer, shot, fraction):
        renderer.update_scene(data, camera=camera(shot, fraction), scene_option=options)
        for first, second in zip(scene["route"], scene["route"][1:]):
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),
                               np.eye(3).reshape(-1),np.array([0,.52,.54,1]))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.022,
                                np.array([*first,.016]),np.array([*second,.016]))
            renderer.scene.ngeom += 1
        for point, rgba in ((scene["start"][:2],[.15,.43,.83,1]),(scene["goal"],[.20,.64,.33,1])):
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.08]*3),
                np.array([*point,.07]),np.eye(3).reshape(-1),np.asarray(rgba))
            renderer.scene.ngeom += 1
        return annotations(Image.fromarray(renderer.render()),shot)

    outputs = {}
    with mujoco.Renderer(model,height=1080,width=1920) as renderer:
        for shot, name in (("overview","01-oblique-overview.png"),
                           ("overhead","02-overhead-route.png"),("aisle","03-tight-aisle.png")):
            path = output/name
            if path.is_symlink():
                raise ValueError(f"Refusing symlink output: {path}")
            frame(renderer,shot,.5).save(path)
            outputs[shot] = {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    print("Three stills rendered",flush=True)

    if not args.stills_only:
        ffmpeg = str(args.ffmpeg.resolve()) if args.ffmpeg else shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError as error:
                raise RuntimeError("ffmpeg is required for MP4; stills have been rendered") from error
        movie = output/"dense-furniture-camera-tour.mp4"
        if movie.is_symlink():
            raise ValueError("Refusing a symlink video output")
        temporary = output/".dense-furniture-camera-tour.partial.mp4"
        frame_count = round(args.duration*args.fps)
        process = subprocess.Popen([ffmpeg,"-y","-loglevel","error","-f","rawvideo","-vcodec","rawvideo",
            "-pix_fmt","rgb24","-s",f"{args.width}x{args.height}","-r",str(args.fps),"-i","-",
            "-an","-c:v","libx264","-preset","medium","-crf","18","-pix_fmt","yuv420p",
            "-movflags","+faststart",str(temporary)],stdin=subprocess.PIPE)
        try:
            with mujoco.Renderer(model,height=args.height,width=args.width) as renderer:
                for index in range(frame_count):
                    progress = index/max(frame_count-1,1)*3
                    number = min(int(progress),2)
                    fraction = progress-number
                    shot = ("overview","overhead","aisle")[number]
                    image = frame(renderer,shot,fraction)
                    # Brief fades separate camera positions without suggesting
                    # that the fixed robot moved between those viewpoints.
                    edge = min(fraction,1-fraction)
                    fade = min(1.0,edge/.035)
                    if fade < 1:
                        image = Image.blend(Image.new("RGB",image.size,"#16324F"),image,fade)
                    process.stdin.write(image.tobytes())
                    if index % args.fps == 0:
                        print(f"Rendered {index//args.fps}/{args.duration:g} video seconds",flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg failed while encoding the native tour")
            temporary.replace(movie)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            raise
        outputs["video"] = {"path":str(movie),"sha256":hashlib.sha256(movie.read_bytes()).hexdigest(),
                            "codec":"H.264","width":args.width,"height":args.height,
                            "fps":args.fps,"frames":frame_count,"duration_seconds":frame_count/args.fps}
    if not np.array_equal(data.qpos,original_pose):
        raise RuntimeError("The fixed robot pose unexpectedly changed")
    record = {"schema":"cat-native-camera-tour-v1","scene_id":scene["scene_id"],
        "geometry_hash":scene["geometry_hash"],"counts":counts,"scene_dir":str(scene_dir),
        "scene_json_sha256":hashlib.sha256((scene_dir/"scene.json").read_bytes()).hexdigest(),
        "assembled_xml_sha256":hashlib.sha256(xml.encode()).hexdigest(),
        "renderer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mujoco_version":mujoco.__version__,"simulation_steps":0,"policy_loaded":False,
        "robot_fixed_at_scene_start":True,"native_start_forbidden_contacts":forbidden,
        "visual_only_changes":["colors","skybox","translucent walls","route/start/goal markers","camera motion","text overlays"],
        "geometry_or_dynamics_modified":False,"traversal_demonstrated":False,
        "elapsed_seconds":time.monotonic()-started,"outputs":outputs}
    (output/"render-manifest.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    print(json.dumps(record,indent=2),flush=True)
    return record


if __name__ == "__main__":
    render_tour(parser().parse_args())
