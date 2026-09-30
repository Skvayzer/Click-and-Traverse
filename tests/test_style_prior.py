"""Checks for the context-gated style prior."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "data/motion_library/library_v1"


def test_gate_weights_sum_to_one_and_open_in_context():
    from cat_mjlab.style_prior import gate_weights
    open_space = torch.full((1, 4), 1.0)
    narrow = torch.full((1, 4), .04)                  # 0.40 m corridor: 0.32 + 0.04 + 0.04
    far = torch.tensor([1., 1.])
    w, gap = gate_weights(torch.cat((open_space, narrow)), far, far)
    assert torch.allclose(w.sum(-1), torch.ones(2))
    assert w[0, 0] == 1 and w[1, 1] == 1 and abs(float(gap[1]) - .40) < 1e-6
    w, _ = gate_weights(open_space.repeat(2, 1), torch.tensor([.02, 1.]), torch.tensor([1., .02]))   # beam overhead / box ahead
    assert w[0, 2] == 1 and w[1, 2] == 1
    w, _ = gate_weights(narrow, torch.tensor([.02]), torch.tensor([1.]))                               # both: capped at 1
    assert torch.allclose(w.sum(-1), torch.ones(1)) and w[0, 0] == 0


@pytest.mark.skipif(not (Path(str(LIB) + ".npz")).exists(), reason="motion library not built")
def test_simulator_features_match_library_positions():
    """Recompute features for library frames with MuJoCo and compare the non-velocity parts."""
    import mujoco
    from cat_mjlab.model import assemble_training_xml
    from cat_mjlab.style_prior import robot_features, KEY_SITES, KEY_BODIES
    m = mujoco.MjModel.from_xml_string(assemble_training_xml()); d = mujoco.MjData(m)
    L = np.load(str(LIB) + ".npz")
    rows = np.linspace(0, len(L["qpos"]) - 1, 25).astype(int)
    sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in KEY_SITES]
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in KEY_BODIES]
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    for r in rows:
        d.qpos[:] = L["qpos"][r]; d.qvel[:] = 0; mujoco.mj_kinematics(m, d)
        f = robot_features(torch.tensor(d.qpos[None], dtype=torch.float32), torch.zeros(1, 35),
                           torch.tensor(d.site_xpos[sid][None], dtype=torch.float32), torch.tensor(d.xpos[bid][None], dtype=torch.float32),
                           torch.tensor(d.xmat[pelvis].reshape(1, 3, 3), dtype=torch.float32))[0].numpy()
        lib = L["features"][r].astype(np.float32)
        pos = np.r_[0:4, 10:39, 68:89]                   # root z, gravity, joint angles, key positions
        assert np.abs(f[pos] - lib[pos]).max() < 5e-3, (r, np.abs(f[pos] - lib[pos]).max())


def test_style_lambda_zero_leaves_the_loss_unchanged():
    from cat_mjlab.learning import ActorCritic, LearnerConfig, compute_loss
    torch.manual_seed(0)
    cfg = LearnerConfig(algorithm="ppo", actor_obs=6, critic_obs=8, action_size=3, actor_hidden=(8,), critic_hidden=(8,), style_critic=True)
    model = ActorCritic(cfg); B, T = 4, 5
    data = dict(state=torch.randn(B, T, 6), privileged_state=torch.randn(B, T, 8), next_state=torch.randn(B, T, 6),
                next_privileged_state=torch.randn(B, T, 8), raw_action=torch.randn(B, T, 3), log_prob=torch.randn(B, T),
                reward=torch.randn(B, T), discount=torch.ones(B, T), truncation=torch.zeros(B, T),
                policy_id=torch.zeros(B, T, dtype=torch.long), style_reward=torch.rand(B, T))
    with_style, m1 = compute_loss(model, data, cfg, style_lambda=0.)
    without = {k: v for k, v in data.items() if k != "style_reward"}
    base, m0 = compute_loss(model, without, cfg, style_lambda=0.)
    assert torch.allclose(m1["policy_loss"], m0["policy_loss"])
    assert "style_v_loss" in m1 and float(m1["style_v_loss"]) > 0
    _, m2 = compute_loss(model, data, cfg, style_lambda=.5)
    assert not torch.allclose(m2["policy_loss"], m0["policy_loss"])


def test_pose_feature_set_and_strided_pairs():
    from cat_mjlab.style_prior import feature_mask, strided_pairs
    names = ["root_z", "qd_left_knee", "q_left_knee", "q_left_wrist_roll", "head_x"]
    assert feature_mask(names, "all").all()
    assert feature_mask(names, "pose").tolist() == [True, False, True, False, True]
    next_ok = torch.tensor([True, True, True, False, True, True, True, True, False])   # clips end at 3 and 8
    assert strided_pairs(next_ok, 1).tolist() == next_ok.tolist()
    # a stride-3 pair starting at i needs next_ok[i], [i+1], [i+2]
    assert strided_pairs(next_ok, 3).tolist() == [True, False, False, False, True, True, False, False, False]
