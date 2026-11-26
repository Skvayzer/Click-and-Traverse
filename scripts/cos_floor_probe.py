#!/usr/bin/env python3
"""C0 diagnosis: per source, distribution of each window's floor level (p5 of the per-frame min foot z of the proxy) and of
the minimum foot z, for M (retargeted) and the SONIC-tracked rollout (physical, on the MuJoCo floor), plus the proxy-vs-
MuJoCo foot offset in a physical standing pose. Also the whole-body lowest point.
  python scripts/cos_floor_probe.py --root /l/.../cos/scenes_v1 --per-source 300 --out probe.json
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_mjlab.cos.features import FrameFeatures, _extent

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--per-source", type=int, default=300)
ap.add_argument("--out", required=True); a = ap.parse_args()
torch.set_num_threads(4)
FF = FrameFeatures(); G = FF.g
feet = FF.left_foot + FF.right_foot


def stats(q):
    w, x = G.primitives(torch.as_tensor(q, dtype=torch.float32)); lo, _ = _extent(w, G.compiled); lo = lo.numpy()
    fm = lo[:, feet, 2].min(1); body = lo[:, :, 2].min(1)
    return float(np.percentile(fm, 5)), float(fm.min()), float(np.percentile(fm, 50)), float(np.percentile(body, 5))


def pct(v):
    v = np.asarray(v); return {f"p{q}": round(float(np.percentile(v, q)), 4) for q in (1, 5, 25, 50, 75, 95, 99)} | {"n": len(v)}


# physical reference: MuJoCo standing pose settled on the floor (training model), proxy lowest foot point vs MuJoCo geoms
import mujoco
m = G.model; d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)
if m.nkey: mujoco.mj_resetDataKeyframe(m, d, 0)
for _ in range(1):
    mujoco.mj_forward(m, d)
lowest = []
for g in range(m.ngeom):
    if m.geom_bodyid[g] == 0: continue
    b = m.geom_bodyid[g]; nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    if "ankle" in nm or "foot" in nm:
        # conservative lowest point: geom centre z - bounding radius for spheres/capsules, aabb for boxes/meshes
        lowest.append((nm, int(m.geom_type[g]), int(m.geom_contype[g]), int(m.geom_conaffinity[g]),
                       float(d.geom_xpos[g][2] - (m.geom_rbound[g] if m.geom_type[g] != 6 else abs(d.geom_xmat[g].reshape(3, 3) @ m.geom_size[g])[2]))))
proxy = stats(d.qpos[None].copy())
res = dict(stand_qpos_pelvis_z=float(d.qpos[2]), stand_proxy_foot_min=proxy[1], stand_mujoco_foot_geoms=lowest)
for src in sorted(glob.glob(f"{a.root}/*/motions")):
    name = Path(src).parent.name
    if name == "phuma_subset" or name == "retarget_matched": continue
    fs = sorted(glob.glob(f"{src}/*.npz")); fs = fs[::max(1, len(fs) // a.per_source)][:a.per_source]
    R = {k: [] for k in ("M_floor", "M_min", "M_med", "M_body_p5", "T_floor", "T_min", "T_med", "T_body_p5", "M_minus_floor")}
    for f in fs:
        z = np.load(f)
        s = stats(z["M"]); R["M_floor"].append(s[0]); R["M_min"].append(s[1]); R["M_med"].append(s[2]); R["M_body_p5"].append(s[3])
        R["M_minus_floor"].append(s[1] - s[0])
        if "tracked" in z.files:
            s = stats(z["tracked"]); R["T_floor"].append(s[0]); R["T_min"].append(s[1]); R["T_med"].append(s[2]); R["T_body_p5"].append(s[3])
    res[name] = {k: pct(v) for k, v in R.items() if v}
    print(name, json.dumps({k: res[name][k]["p50"] for k in res[name]}), flush=True)
json.dump(res, open(a.out, "w"), indent=1)
