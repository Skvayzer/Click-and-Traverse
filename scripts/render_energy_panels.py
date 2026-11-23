#!/usr/bin/env python3
"""Make aligned 720p panels from the CPU reference or matched policy recordings."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cat_mjlab.model import assemble_training_xml
from energy_optimal_reference import InverseReference, STRATEGIES
from scripts.render_clutter_rollouts import resolve_xml_assets

WIDTH, HEIGHT, FPS = 1280, 720, 25
PANEL_TOP, PANEL_HEIGHT = 128, 565
FONT = Path("/usr/share/fonts/truetype/dejavu")
NAMES = {"arm": "Arm only", "body": "Lean/twist", "step": "Side step"}


def rgba(*items):
    return " ".join(str(x) for x in items)


def visual_model(xml, kinds, sizes):
    tree = ET.fromstring(xml)
    world = tree.find("worldbody")
    for i, (kind, size) in enumerate(zip(kinds, sizes, strict=True)):
        name = f"energy_vis_object_{i}"
        body = ET.SubElement(world, "body", name=name, mocap="true", pos="0 0 -10")
        typ, width = (("sphere", 1), ("box", 3), ("cylinder", 2))[int(kind)]
        ET.SubElement(body, "geom", type=typ, size=rgba(*size[:width]),
                      contype="0", conaffinity="0", rgba=".97 .35 .079 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    model.vis.global_.offheight = 720
    model.vis.global_.offwidth = 1280
    model.vis.global_.fovy = 39
    model.vis.quality.shadowsize = 2048
    model.vis.headlight.ambient[:] = [.52] * 3
    model.vis.headlight.diffuse[:] = [.65] * 3
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor >= 0:
        model.geom_matid[floor] = -1
        model.geom_rgba[floor] = [.89, .91, .93, 1.]
    mocap = [model.body(f"energy_vis_object_{i}").mocapid[0] for i in range(len(kinds))]
    return model, mocap


def panel_state(panel, time):
    times = panel["time"]
    n = len(times)
    ix = int(np.clip(np.searchsorted(times, time), 1, n-1)) if n > 1 else 0
    prev = max(ix-1, 0)
    frac = np.clip((time-times[prev]) / max(times[ix]-times[prev], 1e-9), 0., 1.)
    state = (1-frac)*panel["qpos"][prev] + frac*panel["qpos"][ix]
    state[3:7] /= np.linalg.norm(state[3:7])
    positions = (1-frac)*panel["position"][prev] + frac*panel["position"][ix]
    return state, positions, panel["rotation"][ix], panel["valid"][ix]


def composition(output, title, panels, *, duration, view):
    output.parent.mkdir(parents=True, exist_ok=True)
    destination = output.with_name(f".{output.stem}.tmp.mp4")
    borders = np.linspace(0, WIDTH, len(panels)+1, dtype=int)
    camera = mujoco.MjvCamera()
    qw, qx, qy, qz = panels[0]["qpos"][0, 3:7]
    yaw = math.degrees(math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))
    camera.azimuth = yaw + 180. + view
    camera.elevation = -11.
    camera.distance = 2.80
    camera.lookat[:] = [*panels[0]["qpos"][0, :2], .63]
    fonts = [ImageFont.truetype(str(FONT / name), px) for name, px in
             (("DejaVuSans-Bold.ttf", 24), ("DejaVuSans-Bold.ttf", 19), ("DejaVuSans.ttf", 16))]
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[:3] = 1
    from contextlib import ExitStack
    with ExitStack() as stack:
        renderers = []
        for j, p in enumerate(panels):
            model, mocap = visual_model(p["xml"], p["kinds"], p["sizes"])
            renderer = stack.enter_context(mujoco.Renderer(model, height=PANEL_HEIGHT,
                                                            width=int(borders[j+1]-borders[j])-6))
            renderers.append((model, mujoco.MjData(model), mocap, renderer))
        proc = subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo",
                                 "-pixel_format","rgb24","-video_size",f"{WIDTH}x{HEIGHT}","-framerate",str(FPS),
                                 "-i","-","-an","-c:v","libx264","-threads","3","-preset","fast",
                                 "-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(destination)],
                                stdin=subprocess.PIPE)
        try:
            for f in range(round(FPS * min(duration, 9.8))):
                t = f/FPS
                canvas = Image.new("RGB", (WIDTH, HEIGHT), "#111c2a")
                draw = ImageDraw.Draw(canvas)
                draw.text((WIDTH/2, 25), title, fill="#f4f8fc", font=fonts[0], anchor="mm")
                for i, (p, (model, data, mocap, renderer)) in enumerate(zip(panels, renderers, strict=True)):
                    c = (borders[i]+borders[i+1])/2
                    draw.text((c, 72), p["name"], fill="#f4f8fc", font=fonts[1], anchor="mm")
                    if p.get("subtitle"):
                        draw.text((c, 100), p["subtitle"], fill="#a9d5f1", font=fonts[2], anchor="mm")
                    qpos, objects, rotations, valid = panel_state(p, t)
                    data.qpos[:] = qpos
                    for j, mid in enumerate(mocap):
                        data.mocap_pos[mid] = objects[j] if valid[j] else [0.,0.,-10.]
                        mujoco.mju_mat2Quat(data.mocap_quat[mid], rotations[j].reshape(9))
                    mujoco.mj_forward(model,data)
                    renderer.update_scene(data,camera=camera,scene_option=options)
                    canvas.paste(Image.fromarray(renderer.render()), (int(borders[i])+3,PANEL_TOP))
                proc.stdin.write(canvas.tobytes())
        finally:
            proc.stdin.close()
            status = proc.wait()
        if status:
            raise RuntimeError(f"Encoder exited with status {status}")
    destination.replace(output)
    print(output, flush=True)


def representative(events):
    def supports(e, names):
        return all(e["optimal"]["energies"][s]["heat"] is not None for s in names)

    arm_body = [e for e in events if supports(e, ("arm", "body"))]
    all_three = [e for e in arm_body if supports(e, STRATEGIES)]
    chosen = []
    if all_three:
        middle = np.median([e["reference"]["seconds"] for e in all_three])
        chosen.append(min(all_three,key=lambda e:abs(e["reference"]["seconds"]-middle)))
    for label in ("danger", "anticipation", "danger"):
        if len(chosen) == 3:
            break
        candidates = [e for e in arm_body if e["bucket"]==label and e["row"] not in {x["row"] for x in chosen}]
        if candidates:
            left = [e for e in candidates if e["hand_side"]=="left"]
            candidates = left or candidates
            mid = np.median([e["reference"]["seconds"] for e in candidates])
            chosen.append(min(candidates,key=lambda e:abs(e["reference"]["seconds"]-mid)))
    if len(chosen) != 3:
        raise ValueError("The reference does not contain three feasible comparison threats")
    return chosen


def reference(args):
    events = json.loads(args.reference.read_text())["events"]
    manifest = json.loads(args.bank.read_text())["scenes"]
    solver = InverseReference(samples=25)
    xml = assemble_training_xml()
    for e in representative(events):
        ref = e["reference"]
        row = manifest[e["row"]]
        result = solver.solve(e, keep_trajectories=True)
        trajectories = result["trajectories"]
        panels = []
        for name in STRATEGIES:
            originally = e["optimal"]["energies"][name]["heat"]
            if originally is None:
                continue
            selected = trajectories[name]
            energy = selected["energy"]
            if not math.isclose(energy["heat"], originally,rel_tol=.002,abs_tol=.02):
                raise ValueError(f"Reference energy changed for {name} in benchmark row {e['row']}")
            q = selected["qpos"]
            time = np.linspace(0., ref["seconds"], len(q))
            start, end = np.array(ref["sweep_start"]), np.array(ref["sweep_end"])
            center = start + np.linspace(0.,1.,len(q))[:,None]*(end-start)
            panels.append(dict(xml=xml,qpos=q,time=time,position=center[:,None],kinds=row["kinds"][:1],
                               sizes=row["sizes"][:1],rotation=np.tile(row["rotations"][:1],(len(q),1,1,1)),
                               valid=np.ones((len(q),1),bool),name=f"{NAMES[name]} — {energy['mech']:.2f} J mech",
                               subtitle=f"{energy['heat']:.2f} J size-weighted heat"))
        tail = "arm-lean-step" if len(panels)==3 else "arm-lean"
        out = args.out_dir/f"reference-{e['bucket']}-row-{e['row']:04d}-{tail}.mp4"
        composition(out,f"{e['bucket'].capitalize()} threat · row {e['row']} · {ref['seconds']:.2f} s approach",
                    panels,duration=max(3.,ref["seconds"]+1.),view=43. if e["hand_side"]=="right" else -43.)


def recordings(args):
    panels = []
    for label,path in args.panel:
        d = Path(path)
        xml,_ = resolve_xml_assets(d/"model.xml")
        with np.load(d/"trajectory.npz") as z:
            panels.append(dict(xml=xml,name=label,qpos=z["qpos"],time=z["time"]-z["time"][0],
                               position=z["object_position"],rotation=z["object_rotation"],
                               valid=z["object_valid"],kinds=z["object_kinds"][0],sizes=z["object_sizes"][0]))
    if len(panels) != 3:
        raise ValueError("Expected the three policy recordings")
    composition(args.output,args.title,panels,duration=8.,view=-30.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode",required=True)
    opt = sub.add_parser("reference")
    for flag in ("reference","bank","out-dir"):
        opt.add_argument("--"+flag,type=Path,required=True)
    match = sub.add_parser("recordings")
    match.add_argument("--panel",nargs=2,action="append",required=True,metavar=("LABEL","DIRECTORY"))
    match.add_argument("--output",type=Path,required=True)
    match.add_argument("--title",required=True)
    args = parser.parse_args()
    (reference if args.mode=="reference" else recordings)(args)


if __name__ == "__main__":
    main()
