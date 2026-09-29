#!/usr/bin/env python3
"""Offline check of the style discriminators BEFORE any RL (pipeline step I-2).

Robot data: recorded rollouts of the current policy (qpos/qvel at the 50 Hz control rate).
Human data: library transitions (locomotion group). A discriminator is trained exactly as in
training (least-squares GAN loss + gradient penalty) and evaluated on held-out clips/recordings.

Pass criteria:
  * held-out human transitions score high, robot transitions low (the robot IS distinguishable);
  * the separation must NOT come only from velocities: a position-only discriminator must also
    separate (otherwise the style reward would mostly reward smoothness, not posture/shape).
Also reports which features differ most (robot vs human), as a sanity readout.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def robot_transitions(recordings):
    import mujoco
    from cat_mjlab.model import assemble_training_xml
    from cat_mjlab.style_prior import robot_features, KEY_SITES, KEY_BODIES
    m = mujoco.MjModel.from_xml_string(assemble_training_xml()); d = mujoco.MjData(m)
    sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in KEY_SITES]
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in KEY_BODIES]
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    per_rec = {}
    for rec in recordings:
        tr = np.load(Path(rec) / "trajectory.npz"); rows = []
        for qp, qv in zip(tr["qpos"], tr["qvel"]):
            d.qpos[:] = qp[:m.nq]; mujoco.mj_kinematics(m, d)
            rows.append(robot_features(torch.tensor(qp[None, :36], dtype=torch.float32), torch.tensor(qv[None, :35], dtype=torch.float32),
                                       torch.tensor(d.site_xpos[sid][None], dtype=torch.float32), torch.tensor(d.xpos[bid][None], dtype=torch.float32),
                                       torch.tensor(d.xmat[pelvis].reshape(1, 3, 3), dtype=torch.float32))[0])
        f = torch.stack(rows); per_rec[Path(rec).name] = torch.cat((f[:-1], f[1:]), -1)
    return per_rec


def train_eval(human_tr, human_te, robot_tr, robot_te, mask, steps=3000, gp=5.):
    from cat_mjlab.style_prior import Discriminator
    torch.manual_seed(0)
    net = Discriminator(dim=int(mask.sum())); opt = torch.optim.Adam(net.parameters(), lr=5e-5)
    for _ in range(steps):
        h = human_tr[torch.randint(len(human_tr), (1024,))][:, mask].requires_grad_(True)
        r = robot_tr[torch.randint(len(robot_tr), (1024,))][:, mask]
        dh, dr = net(h), net(r)
        g = torch.autograd.grad(dh.sum(), h, create_graph=True)[0]
        loss = .5 * (dh - 1).square().mean() + .5 * (dr + 1).square().mean() + gp * g.square().sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        score = lambda x: (1 - .25 * (net(x[:, mask]) - 1).square()).clamp_min(0)
        return float(score(human_te).mean()), float(score(robot_te).mean())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library", default="data/motion_library/library_v1")
    p.add_argument("--recordings", default="outputs/videos_20260925/rec/*")
    args = p.parse_args(argv)
    L = np.load(ROOT / (args.library + ".npz")); meta = json.loads((ROOT / (args.library + ".json")).read_text())
    feats = torch.tensor(L["features"].astype(np.float32)); mean, std = feats.mean(0), feats.std(0).clamp_min(1e-3)
    norm = lambda x: (x - mean) / std
    ok = torch.tensor(L["next_ok"]) & (torch.tensor(L["group"].astype(np.int64)) == 0)
    clip = torch.tensor(L["clip"].astype(np.int64)); idx = torch.nonzero(ok).flatten()
    held = (clip[idx] % 10 == 0)                                      # 10% of clips held out
    pair = lambda i: torch.cat((norm(feats[i]), norm(feats[i + 1])), -1)
    human_tr, human_te = pair(idx[~held]), pair(idx[held][:20000])
    recs = sorted(glob.glob(str(ROOT / args.recordings)))
    per_rec = robot_transitions(recs)
    names = sorted(per_rec); test_names = names[::4]
    to_norm = lambda x: torch.cat((norm(x[:, :89]), norm(x[:, 89:])), -1)
    robot_tr = to_norm(torch.cat([per_rec[n] for n in names if n not in test_names]))
    robot_te = to_norm(torch.cat([per_rec[n] for n in test_names]))
    fn = meta["summary"]["feature_names"]
    vel = np.array([n.startswith(("lin_vel", "ang_vel", "qd_")) for n in fn])
    full = torch.ones(178, dtype=torch.bool)
    posonly = torch.tensor(np.r_[~vel, ~vel])
    report = {}
    for name, mask in (("all_features", full), ("positions_only", posonly)):
        h, r = train_eval(human_tr, human_te, robot_tr, robot_te, mask)
        report[name] = dict(human_heldout_reward=round(h, 3), robot_heldout_reward=round(r, 3), separation=round(h - r, 3))
    diff = (robot_tr[:, :89].mean(0) - human_tr[:, :89].mean(0)).abs()
    report["largest_robot_vs_human_differences_in_std_units"] = {fn[i]: round(float(diff[i]), 2) for i in diff.argsort(descending=True)[:10]}
    report["robot_transitions"] = dict(train=len(robot_tr), test=len(robot_te), recordings=len(names))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
