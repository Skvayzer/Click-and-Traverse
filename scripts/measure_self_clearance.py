"""Ground-truth hand/finger vs own-body distances for recorded rollouts (B4).

Uses MuJoCo's exact geom distance between every hand/finger geom and every leg geom of the
recorded model, independent of any reward term. Reports per recording: fraction of frames
with a hand within 2 cm of a leg, fraction penetrating, deepest penetration, plus posture
(pelvis height, torso pitch, head height). Usage: measure_self_clearance.py REC_DIR [REC_DIR...]
"""
import sys
import numpy as np
import mujoco


def measure(rec, stride=4):
    m = mujoco.MjModel.from_xml_path(f'{rec}/model.xml'); d = mujoco.MjData(m)
    q = np.load(f'{rec}/trajectory.npz')['qpos']
    body = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ''
    gb = [m.geom_bodyid[g] for g in range(m.ngeom)]
    hand = [g for g in range(m.ngeom) if any(t in body(gb[g]) for t in ('hand', 'thumb', 'index', 'middle', 'finger'))]
    leg = [g for g in range(m.ngeom) if any(t in body(gb[g]) for t in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch'))]
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
    head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'head')
    ft = np.zeros(6); mind, pitch, pz, hz = [], [], [], []
    for t in range(0, len(q), stride):
        d.qpos[:] = q[t]; mujoco.mj_kinematics(m, d)
        mind.append(min(mujoco.mj_geomDistance(m, d, a, b, .5, ft) for a in hand for b in leg))
        pitch.append(np.degrees(np.arcsin(-d.xmat[torso].reshape(3, 3)[2, 0]))); pz.append(q[t, 2]); hz.append(d.site_xpos[head, 2])
    mind = np.array(mind)
    return dict(frames=len(mind), near_2cm=float((mind < .02).mean()), penetrating=float((mind < 0).mean()), deepest_m=float(mind.min()),
                pelvis_z=float(np.mean(pz)), torso_pitch_deg=float(np.mean(pitch)), torso_pitch_max=float(np.max(pitch)), head_z=float(np.mean(hz)))


if __name__ == '__main__':
    print(f"{'recording':28s} {'near<2cm':>8s} {'penetr':>7s} {'deepest':>8s} {'pelvis':>7s} {'pitch':>6s} {'pitchmax':>8s} {'head':>6s}")
    for rec in sys.argv[1:]:
        r = measure(rec.rstrip('/'))
        print(f"{rec.rstrip('/').split('/')[-1]:28s} {r['near_2cm']:8.2f} {r['penetrating']:7.2f} {r['deepest_m']:+8.3f} {r['pelvis_z']:7.2f} {r['torso_pitch_deg']:6.1f} {r['torso_pitch_max']:8.1f} {r['head_z']:6.2f}")
