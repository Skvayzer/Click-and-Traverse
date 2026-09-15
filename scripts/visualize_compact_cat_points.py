"""One model-derived picture of the compact CAT obstacle measurement sites."""
from pathlib import Path
import os
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'glfw' if sys.platform == 'darwin' else 'egl')


def main():
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from cat_ppo.envs.g1.env_cat_wholebody import assemble_training_xml
    from cat_ppo.envs.g1 import constants
    from cat_ppo.furniture.control import ELBOW_RADIUS_M
    from cat_ppo.furniture.grippers import validate_hand_spheres

    xml = ET.fromstring(assemble_training_xml())
    colors = {'point': (.93, .40, .08, 1.), 'hand': (.08, .49, .83, .32), 'elbow': (.58, .24, .77, .48)}
    groups = {
        'Head (1)': ['head'], 'Torso (1)': ['imu_in_torso'], 'Pelvis (1)': ['imu_in_pelvis'],
        'Shoulders (2)': ['left_shoulder', 'right_shoulder'],
        'Elbows (2)': ['left_elbow_probe', 'right_elbow_probe'],
        'Hands (2)': ['left_palm', 'right_palm'],
        'Knees (2)': ['left_knee', 'right_knee'], 'Feet (2)': ['left_foot', 'right_foot'],
    }
    for body in xml.iter('body'):
        for site in list(body.findall('site')):
            name = site.get('name')
            if name not in {n for names in groups.values() for n in names}:
                continue
            kind = 'hand' if 'palm' in name else 'elbow' if 'elbow_probe' in name else 'point'
            if kind == 'hand':
                continue  # Actual enclosing spheres already exist in training XML.
            ET.SubElement(body, 'geom', name='measurement_' + name, type='sphere',
                pos=site.get('pos', '0 0 0'), size=str(ELBOW_RADIUS_M if kind == 'elbow' else .018),
                contype='0', conaffinity='0', density='0', group='4',
                rgba=' '.join(map(str, colors[kind])))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    data = mujoco.MjData(model)
    data.qpos[:] = constants.DEFAULT_QPOS
    # Spread the arms slightly to expose both elbow and hand measurement sites.
    for side, sign in [('left', 1), ('right', -1)]:
        for joint, value in [('shoulder_roll', sign * .35), ('shoulder_pitch', .0), ('elbow', .15)]:
            data.qpos[int(model.joint(f'{side}_{joint}_joint').qposadr[0])] = value
    mujoco.mj_forward(model, data)
    validate_hand_spheres(model, data)
    for i in range(model.ngeom):
        name = model.geom(i).name or ''
        if name == 'floor':
            model.geom_group[i] = 5
        elif name.endswith('hand_sphere'):
            model.geom_rgba[i] = colors['hand']
        elif not name.startswith('measurement_') and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            model.geom_matid[i] = -1
            model.geom_rgba[i] = [.50, .55, .61, .58]
    model.vis.global_.offwidth = 1000
    model.vis.global_.offheight = 1150
    model.vis.global_.fovy = 38
    model.vis.headlight.ambient[:] = [.65] * 3
    model.vis.headlight.diffuse[:] = [.6] * 3
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0., 0., .88]
    camera.distance = 2.7
    camera.azimuth = 0
    camera.elevation = -3
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.geomgroup[4] = 1
    option.geomgroup[5] = 0
    option.sitegroup[:] = 0
    with mujoco.Renderer(model, height=1150, width=1000) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        rendered = renderer.render().copy()
        cameras = renderer.scene.camera
        cam_pos = (np.asarray(cameras[0].pos) + np.asarray(cameras[1].pos)) / 2
        forward, up = np.asarray(cameras[0].forward), np.asarray(cameras[0].up)
        right = np.cross(forward, up)
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render()
        foreground = segmentation[:, :, 0] >= 0
        rendered[~foreground] = 255
    picture = Image.new('RGB', (1400, 1400), 'white')
    picture.paste(Image.fromarray(rendered), (200, 135))
    draw = ImageDraw.Draw(picture)
    font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    title = ImageFont.truetype(font_path, 36)
    label_font = ImageFont.truetype(font_path, 27)
    note_font = ImageFont.truetype(font_path, 23)

    def centered(y, text, font, color):
        draw.text(((1400 - draw.textlength(text, font=font))/2, y), text, font=font, fill=color)

    centered(27, 'Corrected CAT: 13 obstacle measurements', title, '#15354D')
    centered(80, '9 points + 2 hand spheres + 2 elbow spheres', note_font, '#486377')
    focal = 1150 / (2 * np.tan(np.deg2rad(float(model.vis.global_.fovy)) / 2))

    def project(position):
        delta = position - cam_pos
        depth = float(np.dot(delta, forward))
        return np.array([700 + focal*np.dot(delta, right)/depth,
                         135 + 575 - focal*np.dot(delta, up)/depth])

    points = {name: [project(data.site_xpos[model.site(site).id]) for site in sites]
              for name, sites in groups.items()}
    # Overlay every center so the robot surface cannot hide a measurement.
    for name, centers in points.items():
        color = '#1677B9' if 'Hands' in name else '#8740B4' if 'Elbows' in name else '#C45413'
        for x, y in centers:
            draw.ellipse((x-7, y-7, x+7, y+7), fill=color, outline='white', width=2)
    assignments = {'left': ['Shoulders (2)', 'Elbows (2)', 'Hands (2)', 'Knees (2)'],
                   'right': ['Head (1)', 'Torso (1)', 'Pelvis (1)', 'Feet (2)']}
    for side, labels in assignments.items():
        previous_y = 170
        for name in labels:
            anchor = sorted(points[name], key=lambda p: p[0])[0 if side == 'left' else -1]
            y = max(float(anchor[1]), previous_y + 92)
            previous_y = y
            kind = 'hand' if 'Hands' in name else 'elbow' if 'Elbows' in name else 'point'
            color = {'point': '#C45413', 'hand': '#1677B9', 'elbow': '#8740B4'}[kind]
            x = 30 if side == 'left' else 1120
            edge = 230 if side == 'left' else 1100
            elbow_x = 310 if side == 'left' else 1060
            draw.text((x, y-16), name, font=label_font, fill=color)
            draw.line([(edge, y+1), (elbow_x, y+1), tuple(anchor)], fill=color, width=2)
            draw.ellipse((anchor[0]-4, anchor[1]-4, anchor[0]+4, anchor[1]+4), fill=color)
    centered(1285, 'Hand spheres: fixed radius 10.35 cm  |  Elbow spheres: radius 5 cm', note_font, '#244961')
    centered(1330, 'Orange dots mark point samples; their drawn size is only for visibility.', note_font, '#647482')
    output = ROOT / 'docs/assets/compact-cat-points'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'CAT-obstacle-measurement-points.png'
    picture.save(path)
    download = Path('/Users/konstantinsmirnov/Downloads/CAT-obstacle-measurement-points.png')
    shutil.copy2(path, download)
    print(path)
    print(download)


if __name__ == '__main__':
    main()
