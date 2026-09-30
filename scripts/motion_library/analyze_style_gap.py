#!/usr/bin/env python3
"""Why does the style reward not rise? Compare robot rollouts with the motion library.

Uses the discriminators saved in a training checkpoint (style_prior.nets) and recorded rollouts
(trajectory.npz from record_mjlab_rollout.py). Reports:
  1. trained locomotion D: style reward on held-out human transitions vs robot transitions;
  2. the same D on human transitions binned by planar speed (does it simply prefer standing?);
  3. per-feature-group gradient saliency of D on robot transitions (what it looks at);
  4. per-feature separability robot vs human (AUC of each single feature), overall and restricted
     to human frames at the robot's own speed (0.3-0.8 m/s) -- the speed-matched comparison;
  5. fresh offline discriminators on feature subsets, speed-matched human data (which groups alone
     are enough to tell robot from human).
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/motion_library"))


def auc(a, b):
    """P(a > b) for 1-D samples (0.5 = indistinguishable, 0/1 = perfectly separated)."""
    x = np.concatenate((a, b)); r = x.argsort().argsort().astype(np.float64) + 1
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def groups(fn):
    def pick(f):
        return np.array([f(n) for n in fn])
    return {
        "root height+gravity": pick(lambda n: n in ("root_z", "gravity_x", "gravity_y", "gravity_z")),
        "root lin vel": pick(lambda n: n.startswith("lin_vel")),
        "root ang vel": pick(lambda n: n.startswith("ang_vel")),
        "leg joints": pick(lambda n: n.startswith("q_") and any(s in n for s in ("hip", "knee", "ankle"))),
        "waist joints": pick(lambda n: n.startswith("q_waist")),
        "arm joints (sh/elbow)": pick(lambda n: n.startswith("q_") and any(s in n for s in ("shoulder", "elbow"))),
        "wrist joints": pick(lambda n: n.startswith("q_") and "wrist" in n),
        "leg joint vel": pick(lambda n: n.startswith("qd_") and any(s in n for s in ("hip", "knee", "ankle"))),
        "arm+waist joint vel": pick(lambda n: n.startswith("qd_") and not any(s in n for s in ("hip", "knee", "ankle"))),
        "key points (head/palms/feet/elbows)": pick(lambda n: n.split("_")[0] in ("head", "left", "right") and n[-2:] in ("_x", "_y", "_z")
                                                    and not n.startswith(("q_", "qd_"))),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library", default="data/motion_library/library_v1")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--recordings", nargs="+", required=True)
    p.add_argument("--offline-steps", type=int, default=1500)
    p.add_argument("--output")
    args = p.parse_args(argv)
    from offline_discriminator_check import robot_transitions, train_eval
    from cat_mjlab.style_prior import Discriminator, GROUPS
    L = np.load(ROOT / (args.library + ".npz")); meta = json.loads((ROOT / (args.library + ".json")).read_text())
    fn = meta["summary"]["feature_names"]; names = meta["summary"]["groups"]
    feats = torch.tensor(L["features"].astype(np.float32)); mean, std = feats.mean(0), feats.std(0).clamp_min(1e-3)
    norm = lambda x: (x - mean) / std
    ok = torch.tensor(L["next_ok"]) & (torch.tensor(L["group"].astype(np.int64)) == names.index("locomotion"))
    idx = torch.nonzero(ok).flatten(); clip = torch.tensor(L["clip"].astype(np.int64))
    held = clip[idx] % 10 == 0
    human_raw = feats[idx]; human = torch.cat((norm(feats[idx]), norm(feats[idx + 1])), -1)
    speed_h = torch.linalg.vector_norm(human_raw[:, 4:6], dim=-1).numpy()

    recs = sorted(r for pat in args.recordings for r in glob.glob(str(ROOT / pat)))
    per_rec = robot_transitions(recs)
    robot_raw = torch.cat([v[:, :89] for v in per_rec.values()])
    robot = torch.cat((norm(robot_raw), norm(torch.cat([v[:, 89:] for v in per_rec.values()]))), -1)
    speed_r = torch.linalg.vector_norm(robot_raw[:, 4:6], dim=-1).numpy()
    report = dict(recordings=len(recs), robot_transitions=len(robot), human_transitions=len(human),
                  robot_speed_pct_10_50_90=np.percentile(speed_r, [10, 50, 90]).round(2).tolist(),
                  human_speed_pct_10_50_90=np.percentile(speed_h, [10, 50, 90]).round(2).tolist(),
                  human_share_standing_lt_0_15=round(float((speed_h < .15).mean()), 3),
                  human_share_0_3_to_0_8=round(float(((speed_h > .3) & (speed_h < .8)).mean()), 3))

    # 1-3: the trained discriminator
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)["style_prior"]
    nets = torch.nn.ModuleList([Discriminator() for _ in GROUPS]); nets.load_state_dict(state["nets"]); nets.eval()
    D = nets[GROUPS.index("locomotion")]
    reward = lambda x: (1 - .25 * (D(x) - 1).square()).clamp_min(0)
    with torch.no_grad():
        h_te = human[held][:50000]
        report["trained_D_reward"] = dict(human_heldout=round(float(reward(h_te).mean()), 3), robot=round(float(reward(robot).mean()), 3))
        bins = [(0, .15), (.15, .3), (.3, .5), (.5, .8), (.8, 1.2), (1.2, 9)]
        sh = speed_h[held.numpy()][:50000]; rh = reward(h_te).numpy(); rr = reward(robot).numpy()
        report["trained_D_reward_by_speed"] = {f"{a}-{b} m/s": dict(human=round(float(rh[(sh >= a) & (sh < b)].mean()), 3) if ((sh >= a) & (sh < b)).any() else None,
                                                                    robot=round(float(rr[(speed_r >= a) & (speed_r < b)].mean()), 3) if ((speed_r >= a) & (speed_r < b)).any() else None,
                                                                    robot_share=round(float(((speed_r >= a) & (speed_r < b)).mean()), 3))
                                              for a, b in bins}
    x = robot[torch.randperm(len(robot))[:4000]].clone().requires_grad_(True)
    g = torch.autograd.grad(D(x).sum(), x)[0].abs().mean(0).numpy()
    g = g[:89] + g[89:]
    G = groups(fn)
    report["trained_D_saliency_share_by_group"] = {k: round(float(g[m].sum() / g.sum()), 3) for k, m in G.items()}
    top = np.argsort(-g)[:12]
    report["trained_D_top_features"] = {fn[i]: round(float(g[i] / g.sum()), 3) for i in top}

    # 4: single-feature separability, overall and speed-matched
    matched = (speed_h > .3) & (speed_h < .8)
    hr = human_raw.numpy(); rr_ = robot_raw.numpy()
    sub = np.random.default_rng(0).choice(len(hr), min(len(hr), 60000), replace=False)
    subm = np.random.default_rng(0).choice(np.nonzero(matched)[0], min(int(matched.sum()), 60000), replace=False)
    sep = {}
    for i, n in enumerate(fn):
        a_all = auc(rr_[:, i], hr[sub, i]); a_m = auc(rr_[:, i], hr[subm, i])
        sep[n] = (round(abs(a_all - .5) * 2, 3), round(abs(a_m - .5) * 2, 3), round(float(rr_[:, i].mean()), 3), round(float(hr[subm, i].mean()), 3))
    order = sorted(sep, key=lambda n: -sep[n][1])[:15]
    report["most_separable_features_speed_matched"] = {n: dict(separability_all=sep[n][0], separability_matched=sep[n][1],
                                                               robot_mean=sep[n][2], human_mean_at_robot_speed=sep[n][3]) for n in order}

    # 5: offline discriminators on subsets, human restricted to the robot's speed range
    h_idx = torch.nonzero(torch.tensor(matched)).flatten()
    hm_held = held[h_idx]
    h_tr, h_te = human[h_idx][~hm_held], human[h_idx][hm_held][:20000]
    names_r = sorted(per_rec); test = names_r[::4]
    to_norm = lambda v: torch.cat((norm(v[:, :89]), norm(v[:, 89:])), -1)
    r_tr = to_norm(torch.cat([per_rec[n] for n in names_r if n not in test])); r_te = to_norm(torch.cat([per_rec[n] for n in test]))
    subsets = {"all features": np.ones(89, bool)}
    subsets.update({f"only {k}": m for k, m in G.items()})
    subsets["all except arm+wrist joints/vel and palms/elbows"] = ~(G["arm joints (sh/elbow)"] | G["wrist joints"] | G["arm+waist joint vel"]
                                                                   | np.array([("palm" in n) or ("elbow_link" in n) or n.startswith("left_elbow") or n.startswith("right_elbow") for n in fn]))
    off = {}
    for k, m in subsets.items():
        mask = torch.tensor(np.r_[m, m])
        h, r = train_eval(h_tr, h_te, r_tr, r_te, mask, steps=args.offline_steps)
        off[k] = dict(human=round(h, 3), robot=round(r, 3), gap=round(h - r, 3))
        print(k, off[k], flush=True)
    report["offline_D_speed_matched_by_subset"] = off
    text = json.dumps(report, indent=1); print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
