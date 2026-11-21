#!/usr/bin/env python3
"""Clean PNG stills for paper figures from record_mjlab_rollout.py recordings (traversal or standing).

The robot at --time is drawn solid. Optional earlier poses (--trail, seconds) are drawn as faded copies, so a single
image shows the motion. Scene boxes are opaque (--obstacle-alpha). The camera is fixed relative to the robot's start
heading. Approaching objects of standing recordings are orange, and green discs mark where the feet settled 1 s in
(as in render_standing_hands.py). No text is drawn; the figure adds its own labels.

  render_paper_stills.py --input-dir REC --out still.png --time 6.0 [--trail 1.5 0.8] [--camera 3.0 -20 150]
  render_paper_stills.py --input-dir STANDING_REC --out still.png --auto-time
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ.setdefault("MUJOCO_GL", "egl")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--time", type=float, nargs="+", help="episode times (s) of the solid robot, one PNG each")
    p.add_argument("--auto-time", action="store_true",
                   help="standing recordings: the moment an object comes closest to a palm (printed with the foot drift)")
    p.add_argument("--trail", type=float, nargs="*", default=[], help="seconds before each time drawn as faded robots")
    p.add_argument("--camera", type=float, nargs=3, default=(3.0, -20., 150.), metavar=("DIST", "ELEV", "AZIM_OFFSET"),
                   help="distance (m), elevation (deg), azimuth relative to the start heading (deg)")
    p.add_argument("--lookat", type=float, nargs=3, metavar=("DX", "DY", "Z"),
                   help="look-at point: offset from the robot's pelvis at --time (m, world frame) and height; "
                        "default: the pelvis, 0.7 m high")
    p.add_argument("--size", default="1600x1200")
    p.add_argument("--obstacle-alpha", type=float, default=1.0)
    p.add_argument("--hide-above", type=float, help="hide scene boxes whose bottom is above this height (m), e.g. ceilings")
    p.add_argument("--hide-geoms", nargs="*", default=[], help="scene geom names to hide (occluders)")
    a = p.parse_args(argv)
    import mujoco
    from planner_video import add_ghost
    from render_clutter_rollouts import resolve_xml_assets
    width, height = (int(x) for x in a.size.split("x"))
    z = np.load(a.input_dir / "trajectory.npz")
    qpos, times = z["qpos"], z["time"]
    xml, _ = resolve_xml_assets(a.input_dir / "model.xml")
    standing = "object_position" in z.files
    if standing:
        bodies = []
        kinds, sizes = z["object_kinds"][0], z["object_sizes"][0]
        for m in range(z["object_position"].shape[1]):
            k, s = int(kinds[m]), sizes[m]
            geom = {0: f'type="sphere" size="{s[0]:.4f}"', 1: f'type="box" size="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}"',
                    2: f'type="cylinder" size="{s[0]:.4f} {s[1]:.4f}"'}[k]
            bodies.append(f'<body name="obj{m}" mocap="true" pos="0 0 -10"><geom {geom} contype="0" conaffinity="0" '
                          f'rgba="1 0.42 0.05 1"/></body>')
        for f in range(2):
            bodies.append(f'<body name="foot{f}" mocap="true" pos="0 0 -10"><geom type="cylinder" size="0.07 0.0015" '
                          f'contype="0" conaffinity="0" rgba="0.1 0.75 0.3 0.7"/></body>')
        xml = xml.replace("</worldbody>", "".join(bodies) + "</worldbody>", 1)
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.global_.fovy = 35
    model.vis.quality.shadowsize = 4096
    model.vis.headlight.ambient[:] = [.45] * 3
    model.vis.headlight.diffuse[:] = [.55] * 3
    data, ghost = mujoco.MjData(model), mujoco.MjData(model)
    hide = set(a.hide_geoms)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != 0 or model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        bottom = model.geom_pos[g, 2] - (model.geom_size[g, 2] if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else 0.)
        if name in hide or (a.hide_above is not None and bottom > a.hide_above):
            model.geom_rgba[g, 3] = 0.
        else:
            model.geom_rgba[g, 3] = a.obstacle_alpha
    if a.auto_time:
        if not standing:
            raise SystemExit("--auto-time needs a standing recording")
        palms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_palm", "right_palm")]
        feet_sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]
        radius = z["object_sizes"][0][:, 0]
        best, best_t, feet0, drift = 1e9, 0., None, 0.
        for i in range(len(qpos)):
            data.qpos[:] = qpos[i]; mujoco.mj_kinematics(model, data)
            feet = np.array([data.site_xpos[f][:2] for f in feet_sites])
            if times[i] >= 1.0:
                feet0 = feet if feet0 is None else feet0
                drift = max(drift, float(np.linalg.norm(feet - feet0, axis=-1).max()))
            for m in range(z["object_position"].shape[1]):
                if not z["object_valid"][i, m]:
                    continue
                d = min(float(np.linalg.norm(data.site_xpos[h] - z["object_position"][i, m])) for h in palms) - float(radius[m])
                if d < best:
                    best, best_t = d, float(times[i])
        print(f"AUTO {a.input_dir.name} closest_palm_surface_m={best:.3f} at t={best_t:.2f}s max_foot_drift_m={drift:.3f}", flush=True)
        a.time = [best_t]
    if not a.time:
        raise SystemExit("give --time or --auto-time")
    for t_main in a.time:
        render_one(a, mujoco, model, data, ghost, z, qpos, times, standing, t_main, width, height, add_ghost)


def render_one(a, mujoco, model, data, ghost, z, qpos, times, standing, t_main, width, height, add_ghost):
    step = int(np.clip(np.searchsorted(times, t_main), 0, len(qpos) - 1))
    if standing:
        mocap = {f"obj{m}": int(model.body_mocapid[model.body(f"obj{m}").id]) for m in range(z["object_position"].shape[1])}
        feet = [int(model.body_mocapid[model.body(f"foot{f}").id]) for f in range(2)]
        settle = min(int(np.searchsorted(times, 1.0)), len(qpos) - 1)
        data.qpos[:] = qpos[settle]; mujoco.mj_forward(model, data)
        for f, site in enumerate(("left_foot", "right_foot")):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
            if sid >= 0:
                data.mocap_pos[feet[f]] = [data.site_xpos[sid][0], data.site_xpos[sid][1], .0015]
        for m, mid in mocap.items():
            i = int(m[3:])
            if z["object_valid"][step, i]:
                data.mocap_pos[mid] = z["object_position"][step, i]
                quat = np.empty(4); mujoco.mju_mat2Quat(quat, z["object_rotation"][step, i].reshape(9))
                data.mocap_quat[mid] = quat
            else:
                data.mocap_pos[mid] = [0, 0, -10]
    data.qpos[:] = qpos[step]; mujoco.mj_forward(model, data)
    q0 = qpos[0]
    heading = math.degrees(math.atan2(2 * (q0[3] * q0[6] + q0[4] * q0[5]), 1 - 2 * (q0[5] ** 2 + q0[6] ** 2)))
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = a.camera[0], a.camera[1], heading + a.camera[2]
    if a.lookat:
        camera.lookat[:] = [qpos[step, 0] + a.lookat[0], qpos[step, 1] + a.lookat[1], a.lookat[2]]
    else:
        camera.lookat[:] = [qpos[step, 0], qpos[step, 1], .7]
    option, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    option.geomgroup[:] = 0; option.geomgroup[:3] = 1; option.sitegroup[:] = 0
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        for k, back in enumerate(sorted(a.trail, reverse=True)):
            i = int(np.clip(np.searchsorted(times, t_main - back), 0, len(qpos) - 1))
            shade = 0.35 + 0.3 * k / max(len(a.trail), 1)
            add_ghost(renderer.scene, model, ghost, mujoco, option, perturb, qpos[i], (shade, shade + .05, shade + .12, .30))
        frame = renderer.render().copy()
    from PIL import Image
    out = a.out if len(a.time) == 1 else a.out.with_name(f"{a.out.stem}_t{times[step]:05.2f}{a.out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(out)
    print("STILL", out, f"t={times[step]:.2f}s", flush=True)


if __name__ == "__main__":
    main()
