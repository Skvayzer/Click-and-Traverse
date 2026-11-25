"""Rendering helpers for COS: training model + scene boxes (visual only), ghosts of a second motion, labels, H.264 output.
Use MUJOCO_GL=egl under `flock -x /tmp/cat-energy-dep1-gpu.lock` on dep-1."""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def quat_from_mat(R):
    import mujoco
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.asarray(R, np.float64).ravel()); return q


def model_with_boxes(boxes, colors=None, width=1280, height=720):
    """MjModel of the G1 training model plus visual-only boxes {box_center, box_rot, box_half} (contype 0)."""
    import mujoco
    from ..model import assemble_training_xml
    root = ET.fromstring(assemble_training_xml())
    wb = root.find("worldbody")
    n = len(boxes["box_center"]) if boxes is not None else 0
    for j in range(n):
        q = quat_from_mat(boxes["box_rot"][j])
        rgba = colors[j] if colors is not None else (0.75, 0.55, 0.35, 0.85)
        ET.SubElement(wb, "geom", name=f"cos_box_{j}", type="box", size=" ".join(f"{v:.4f}" for v in boxes["box_half"][j]),
                      pos=" ".join(f"{v:.4f}" for v in boxes["box_center"][j]), quat=" ".join(f"{v:.6f}" for v in q),
                      rgba=" ".join(f"{v:.3f}" for v in rgba), contype="0", conaffinity="0", group="1")
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    g = vis.find("global")
    if g is None:
        g = ET.SubElement(vis, "global")
    g.set("offwidth", str(width)); g.set("offheight", str(height))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def add_ghost(scene, model, ghost_data, mujoco, option, perturb, q, rgba):
    ghost_data.qpos[:] = q; mujoco.mj_kinematics(model, ghost_data)
    before = scene.ngeom
    mujoco.mjv_addGeoms(model, ghost_data, option, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
    for g in range(before, scene.ngeom):
        scene.geoms[g].rgba[:] = rgba
        scene.geoms[g].matid = -1


def add_line(scene, mujoco, pts, rgba, width):
    for a, b in zip(pts[:-1], pts[1:]):
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(b - a) < 1e-4:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scene.geoms[scene.ngeom].matid = -1
        scene.ngeom += 1


def label(frame, lines, font, xy=(14, 10), fill=(255, 255, 255)):
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame); d = ImageDraw.Draw(img); x, y = xy
    for text in lines:
        box = d.textbbox((x, y), text, font=font)
        d.rectangle((box[0] - 6, box[1] - 3, box[2] + 6, box[3] + 3), fill=(0, 0, 0))
        d.text((x, y), text, font=font, fill=fill)
        y = box[3] + 8
    return np.asarray(img)


def fonts(size=18):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size), ImageFont.truetype("DejaVuSans-Bold.ttf", size + 4)
    except OSError:
        f = ImageFont.load_default(); return f, f


class Writer:
    """H.264 yuv420p faststart writer with a silent audio track (plays everywhere)."""

    def __init__(self, path, w, h, fps):
        ff = shutil.which("ffmpeg")
        if ff is None:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        self.cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-r", str(fps), "-i", "-", "-c:v", "libx264", "-threads", "4", "-pix_fmt", "yuv420p", "-crf", "21",
                    "-movflags", "+faststart", str(path)]
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError("ffmpeg failed")


def yaw(q):
    return math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
