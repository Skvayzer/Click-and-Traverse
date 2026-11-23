#!/usr/bin/env python3
"""Compare two exact, time-aligned plan-blind benchmark recordings."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco

from cat_mjlab.model import ARM_LINKS
from cat_mjlab.plan_blind import BOX_HALF
from planner_render import label, scene_geometry
from planner_video import add_ghost, add_line
from render_clutter_rollouts import resolve_xml_assets
from render_plan_reflex import furnish_room, plan_pose


def read_trace(path):
    with np.load(path, allow_pickle=False) as data:
        result = dict(data)
    return result, json.loads(str(result['meta']))


def scene_model(bank, row):
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        scene_geometry(directory, bank, row)
        xml, _ = resolve_xml_assets(directory / 'model.xml')
        if row.get('task_kind') == 'room':
            xml = furnish_room(xml, json.loads((directory / 'scene.json').read_text()))
    root = ET.fromstring(xml)
    body = ET.SubElement(root.find('worldbody'), 'body', name='replay_probe', mocap='true', pos='0 0 -20')
    ET.SubElement(body, 'geom', type='box', size=' '.join(map(str, BOX_HALF)),
                  rgba='.91 .13 .31 1', contype='0', conaffinity='0')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 720
    model.vis.quality.shadowsize = 2048
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('on', 'off', 'out'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--bank', default='data/furniture/tablegap_v1')
    p.add_argument('--clip', type=int, default=0, help='index in the trace list')
    a = p.parse_args()
    on, on_meta = read_trace(a.on)
    off, off_meta = read_trace(a.off)
    for k in ('scenes','rows','dt','speed','seed','variant','bank'):
        if on_meta[k] != off_meta[k]:
            raise ValueError(f'Paired traces differ on {k}')
    if on_meta['band'] <= 0. or off_meta['band'] != 0.:
        raise ValueError('The left clip must use an active reflex and the right clip must use a zero band')
    w = a.clip
    if not np.allclose(on['qpos'][0, w], off['qpos'][0, w], atol=2e-6, rtol=0.):
        raise ValueError('Paired starts differ')
    n = min(on_meta['end'][w], off_meta['end'][w]) + 1
    if not np.array_equal(on['active'][:n,w],off['active'][:n,w]) or not np.allclose(on['position'][:n,w],off['position'][:n,w]):
        raise ValueError('Paired obstacle motion differs')
    bank = Path(a.bank + '_packed/manifest.json')
    manifests = json.loads(bank.read_text())['scenes']
    scene = next(r for r in manifests if r['scene_id'] == on_meta['scenes'][w])
    model = scene_model(bank, scene)
    mocap = model.body_mocapid[model.body('replay_probe').id]
    data, ghost = mujoco.MjData(model), mujoco.MjData(model)
    camera = mujoco.MjvCamera(); camera.elevation, camera.distance = -30., 3.6
    qw,qx,qy,qz = on['qpos'][0,w,3:7]
    camera.azimuth = math.degrees(math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))) + 65.
    opt, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    opt.geomgroup[3:] = 0; opt.sitegroup[:] = 0
    arm_bodies = {model.body(f'{side}_{link}').id: h for h,side in enumerate(('left','right')) for link in ARM_LINKS}
    arms = [arm_bodies.get(b) for b in model.geom_bodyid]
    font = ImageFont.truetype('DejaVuSans.ttf', 17)
    dt = on_meta['dt']; ends = [on_meta['end'][w],off_meta['end'][w]]
    steps = range(0,min(max(ends)+1,750),round(1/(25*dt)))
    look = on['qpos'][0,w,:2]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    cmd = ['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','25','-i','-',
           '-f','lavfi','-i','anullsrc=r=48000:cl=stereo','-map','0:v:0','-map','1:a:0','-t',f'{len(steps)/25:.2f}',
           '-c:a','aac','-b:a','96k','-c:v','libx264','-threads','4','-pix_fmt','yuv420p','-crf','21',
           '-movflags','+faststart',str(a.out)]
    proc = subprocess.Popen(cmd,stdin=subprocess.PIPE)
    with mujoco.Renderer(model,width=640,height=720) as renderer:
        for t in steps:
            sample = [min(t,e) for e in ends]
            center = (on['qpos'][sample[0],w,:2]+off['qpos'][sample[1],w,:2])/2
            look = .90*look+.10*center
            camera.lookat[:] = [*look,.63]
            frame = np.empty((720,1280,3),dtype=np.uint8)
            for i,(trace,title) in enumerate(((on,'Reflex on'),(off,'Reflex off'))):
                j = sample[i]; data.qpos[:] = trace['qpos'][j,w]
                visible = trace['active'][j,w] and j == t
                data.mocap_pos[mocap] = trace['position'][j,w] if visible else (0.,0.,-20.)
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera,scene_option=opt)
                sc = renderer.scene
                released = trace['released'][j,w]
                if released.any():
                    for k in range(sc.ngeom):
                        g=sc.geoms[k]
                        arm=arms[g.objid] if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0 else None
                        if arm is not None and released[arm]:
                            g.rgba[:3]=.22*g.rgba[:3]+.78*np.array([1.,.40,.055])
                if trace['plan_active'][j,w]:
                    plan = trace['plan'][j,w]; clock=float(trace['plan_t'][j,w]); k0=min(20,int(clock/.1))
                    add_ghost(sc,model,ghost,mujoco,opt,perturb,plan_pose(plan,clock+.3),(.22,.58,1.,.25))
                    if k0 < 20:
                        path=np.column_stack((plan[k0:,:2],np.full(21-k0,.025)))
                        add_line(sc,mujoco,path,(.13,.50,1.,.96),.013)
                text=[f"{title}   {on_meta['variant'].capitalize()} obstacle   {t*dt:.1f} s"]
                if trace['hand_contact'][:j+1,w].any(): text.append('Hand contacted the inserted obstacle')
                elif released.any(): text.append('Orange arm released from plan tracking')
                view=label(renderer.render(),text,font)
                frame[:,640*i:640*(i+1)]=view
            frame[:,638:642]=0
            proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('FFmpeg failed to encode paired replay')
    print(a.out)


if __name__ == '__main__':
    main()
