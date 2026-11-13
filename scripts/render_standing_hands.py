#!/usr/bin/env python3
"""Render a standing hand-protection recording (record_mjlab_rollout.py --reactive-row) for side-by-side comparison.

Fixed camera on the robot's starting spot (a moving camera would hide stepping), green floor discs where the feet first
settled (1 s in), approaching objects in orange, and only a label plus the clock -- no traversal overlays.

  render_standing_hands.py --input-dir REC --output out.mp4 --label "S2 energy"
"""
import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
os.environ.setdefault('MUJOCO_GL', 'egl')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', default='')
    p.add_argument('--fps', type=int, default=25)
    p.add_argument('--distance', type=float, default=2.6)
    p.add_argument('--elevation', type=float, default=-14.)
    p.add_argument('--azimuth-offset', type=float, default=-35., help='camera azimuth relative to the initial heading (deg)')
    p.add_argument('--size', default='960x720')
    a = p.parse_args()
    import mujoco
    from scripts.render_clutter_rollouts import resolve_xml_assets
    width, height = (int(x) for x in a.size.split('x'))
    z = np.load(a.input_dir / 'trajectory.npz')
    qpos, times = z['qpos'], z['time']
    xml, _ = resolve_xml_assets(a.input_dir / 'model.xml')
    bodies = []
    kinds, sizes = z['object_kinds'][0], z['object_sizes'][0]
    for m in range(z['object_position'].shape[1]):
        k, s = int(kinds[m]), sizes[m]
        geom = {0: f'type="sphere" size="{s[0]:.4f}"', 1: f'type="box" size="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}"',
                2: f'type="cylinder" size="{s[0]:.4f} {s[1]:.4f}"'}[k]
        bodies.append(f'<body name="obj{m}" mocap="true" pos="0 0 -10"><geom {geom} contype="0" conaffinity="0" rgba="1 0.42 0.05 1"/></body>')
    for f in range(2):
        bodies.append(f'<body name="foot{f}" mocap="true" pos="0 0 -10"><geom type="cylinder" size="0.07 0.0015" contype="0" '
                      f'conaffinity="0" rgba="0.1 0.75 0.3 0.55"/></body>')
    xml = xml.replace('</worldbody>', ''.join(bodies) + '</worldbody>', 1)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.global_.fovy = 40
    model.vis.quality.shadowsize = 2048
    model.vis.headlight.ambient[:] = [.5] * 3
    model.vis.headlight.diffuse[:] = [.6] * 3
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    if floor >= 0:
        model.geom_matid[floor] = -1
        model.geom_rgba[floor] = [.9, .92, .93, 1.]
    mocap = {name: int(model.body_mocapid[model.body(name).id]) for name in
             [f'obj{m}' for m in range(z['object_position'].shape[1])] + ['foot0', 'foot1']}
    feet_sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in ('left_foot', 'right_foot')]
    settle = min(50, len(qpos) - 1)              # where the feet stand 1 s in: the reference discs
    data.qpos[:] = qpos[settle]
    mujoco.mj_forward(model, data)
    for f, sid in enumerate(feet_sites):
        if sid >= 0:
            data.mocap_pos[mocap[f'foot{f}']] = [data.site_xpos[sid][0], data.site_xpos[sid][1], .0015]
    foot_ref = data.mocap_pos[[mocap['foot0'], mocap['foot1']]].copy()
    qw, qx, qy, qz = qpos[0, 3:7]
    heading = math.degrees(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = heading + 180 + a.azimuth_offset, a.elevation, a.distance
    camera.lookat[:] = [qpos[0, 0], qpos[0, 1], .62]
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[:3] = 1
    step = max(1, round(1 / (a.fps * float(np.median(np.diff(times))))))
    frames = range(0, len(qpos), step)
    font = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    label = a.label.replace(':', r'\:').replace("'", '')
    draw = (f"drawtext=fontfile={font}:text='{label}':x=18:y=16:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8,"
            f"drawtext=fontfile={font}:text='%{{eif\\:t\\:d}}.%{{eif\\:mod(t*10\\,10)\\:d}} s':x=w-tw-18:y=16:fontsize=30:"
            f"fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
                             '-r', str(a.fps), '-i', '-', '-vf', draw, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '21',
                             str(a.output)], stdin=subprocess.PIPE)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for i in frames:
            data.qpos[:] = qpos[i]
            for m in range(z['object_position'].shape[1]):
                mid = mocap[f'obj{m}']
                if z['object_valid'][i, m]:
                    data.mocap_pos[mid] = z['object_position'][i, m]
                    quat = np.empty(4); mujoco.mju_mat2Quat(quat, z['object_rotation'][i, m].reshape(9))
                    data.mocap_quat[mid] = quat
                else:
                    data.mocap_pos[mid] = [0, 0, -10]
            data.mocap_pos[[mocap['foot0'], mocap['foot1']]] = foot_ref
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=options)
            proc.stdin.write(renderer.render().tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')
    print(a.output)


if __name__ == '__main__':
    main()
