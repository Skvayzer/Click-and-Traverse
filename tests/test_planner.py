"""Whole-body planner v1 (cat_mjlab/planner): kinematics, scene grid, windows, features, model (2026-11-18)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def model():
    import mujoco
    from cat_mjlab.model import assemble_training_xml
    return mujoco.MjModel.from_xml_string(assemble_training_xml())


def random_qpos(model, n, seed=0):
    rng = np.random.default_rng(seed)
    q = np.tile(model.qpos0, (n, 1))
    q[:, :3] = rng.uniform(-2, 2, (n, 3))
    quat = rng.normal(size=(n, 4)); q[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    q[:, 7:] = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1], (n, model.nq - 7))
    return q


def test_fk_matches_mujoco(model):
    import mujoco
    from cat_mjlab.planner.fk import TorchFK
    data = mujoco.MjData(model)
    q = random_qpos(model, 16)
    xpos, xmat = TorchFK(model, dtype=torch.float64)(torch.as_tensor(q))
    for i in range(len(q)):
        data.qpos[:] = q[i]; mujoco.mj_kinematics(model, data)
        assert np.abs(data.xpos - xpos[i].numpy()).max() < 1e-9
        assert np.abs(data.xmat.reshape(-1, 3, 3) - xmat[i].numpy()).max() < 1e-9


def test_body_spheres_match_envelope(model):
    import mujoco
    from cat_mjlab.planner.fk import BodySpheres
    from cat_mjlab.reactive import BodyEnvelope
    data = mujoco.MjData(model)
    q = random_qpos(model, 1, seed=3)[0]
    data.qpos[:] = q; mujoco.mj_kinematics(model, data)
    env = BodyEnvelope(model, "cpu")
    ref = env.spheres(SimpleNamespace(xpos=torch.tensor(data.xpos[None], dtype=torch.float32),
                                      xmat=torch.tensor(data.xmat.reshape(1, -1, 3, 3), dtype=torch.float32)), torch.tensor([0]))[0]
    assert torch.allclose(BodySpheres(model)(torch.as_tensor(q[None], dtype=torch.float32))[0], ref, atol=1e-5)


def fake_bank(sdf, origin, dx):
    nx, ny, nz = sdf.shape
    return SimpleNamespace(fields=dict(sdf=torch.as_tensor(sdf.reshape(-1, 1), dtype=torch.float32)),
                           origins=torch.tensor([origin], dtype=torch.float32), dxs=torch.tensor([dx]),
                           shapes=torch.tensor([[nx, ny, nz]]), offsets=torch.tensor([0]))


def test_bank_sampler_is_standard_trilinear():
    from cat_mjlab.planner.grid import sample_bank_sdf
    rng = np.random.default_rng(1)
    sdf = rng.normal(size=(7, 6, 5))
    bank = fake_bank(sdf, [0.1, -0.2, 0.02], 0.04)
    # a linear field is reproduced exactly by trilinear interpolation
    xs, ys, zs = (np.asarray(bank.origins[0][i]) + 0.04 * np.arange(n) for i, n in enumerate(sdf.shape))
    lin = 2 * xs[:, None, None] - 3 * ys[None, :, None] + 0.5 * zs[None, None, :]
    bank_lin = fake_bank(lin, [0.1, -0.2, 0.02], 0.04)
    pts = torch.tensor(rng.uniform([0.12, -0.18, 0.03], [0.30, -0.02, 0.15], (1, 50, 3)), dtype=torch.float32)
    got = sample_bank_sdf(bank_lin, pts, torch.tensor([0]))[0]
    want = 2 * pts[0, :, 0] - 3 * pts[0, :, 1] + 0.5 * pts[0, :, 2]
    assert torch.allclose(got, want, atol=1e-5)
    # nodes are reproduced exactly
    node = torch.tensor([[[0.1 + 0.04 * 3, -0.2 + 0.04 * 2, 0.02 + 0.04 * 1]]], dtype=torch.float32)
    assert abs(float(sample_bank_sdf(bank, node, torch.tensor([0]))[0, 0]) - sdf[3, 2, 1]) < 1e-5


def test_local_grid_roundtrip_and_frames():
    from cat_mjlab.planner.grid import cell_centres, dequantize, quantize, sample_local_grid, to_local, to_world, SHAPE
    c = cell_centres()
    sdf = (c[..., 0] - 0.7).clamp(-0.1, 0.4)                       # a wall plane at x = 0.7 m (chunk frame)
    grid = quantize(sdf)[None]
    pts = torch.tensor([[[0.3, 0.0, 1.0], [0.68, 0.1, 0.5], [3.0, 0.0, 0.5]]])
    got = sample_local_grid(grid, pts)[0]
    assert float(got[0]) == pytest.approx(-0.1, abs=0.006)              # clipped at SDF_LO inside the obstacle
    assert float(got[1]) == pytest.approx(-0.02, abs=0.006)
    assert float(got[2]) == pytest.approx(0.4)                      # outside the grid reads free
    origin, yaw = torch.tensor([[1.0, -2.0]]), torch.tensor([0.7])
    p = torch.tensor([[[0.4, 0.2, 0.9]]])
    assert torch.allclose(to_local(to_world(p, origin, yaw), origin, yaw), p, atol=1e-6)
    assert dequantize(quantize(torch.tensor(0.123))).item() == pytest.approx(0.123, abs=0.002)


def test_euler_roundtrip():
    from cat_mjlab.planner.data import euler_zyx, quat_from_euler
    rng = np.random.default_rng(2)
    r, p, y = (torch.tensor(rng.uniform(lo, hi, 64), dtype=torch.float64) for lo, hi in ((-1, 1), (-1.2, 1.2), (-3, 3)))
    rr, pp, yy = euler_zyx(quat_from_euler(r, p, y))
    assert torch.allclose(rr, r, atol=1e-9) and torch.allclose(pp, p, atol=1e-9) and torch.allclose(yy, y, atol=1e-9)


def test_valid_starts_rules():
    from cat_mjlab.planner.data import valid_starts
    T, N = 400, 3
    ended = np.zeros((T, N), bool); bad = np.zeros((T, N), bool)
    epstep = np.tile(np.arange(T + 1)[:, None], (1, N)).astype(np.int32)
    ended[199, 1] = True; epstep[200:, 1] = np.arange(T + 1 - 200)      # world 1: episode ends cleanly at step 199
    ended[299, 2] = bad[299, 2] = True; epstep[300:, 2] = np.arange(T + 1 - 300)   # world 2: contact at step 299
    s = valid_starts(ended, bad, epstep, horizon=100, settle=25, margin=25, every=10)
    w0, w1, w2 = (s[s[:, 1] == k, 0] for k in range(3))
    assert w0.min() == 30 and w0.max() == 300
    assert not any((t < 200 < t + 100) for t in w1)                       # no window across the reset
    assert w1.min() == 30 and 230 in w1                                   # restarts after the settle period
    assert w2.max() == 170                                                # t0 + 100 + 25 <= 299 (no contact before the margin ends)


def test_features_frame_and_chunk_fk(model):
    """The chunk frame is the pelvis floor frame at t0: FK of the rebuilt chunk-frame qpos reproduces the bodies."""
    from cat_mjlab.planner.data import chunk_qpos, features, state_qpos
    from cat_mjlab.planner.fk import TorchFK
    from cat_mjlab.planner.grid import to_local
    T, N = 120, 2
    rng = np.random.default_rng(4)
    q = torch.as_tensor(np.stack([random_qpos(model, N, seed=10 + t) for t in range(T + 1)]), dtype=torch.float32)
    q[..., 3:7] = torch.tensor([math.cos(0.3), 0, 0, math.sin(0.3)])   # upright, heading 0.6 rad
    v = torch.zeros(T + 1, N, 35)
    starts = np.array([[0, 0], [10, 1]])
    state, chunk, goal, origin, yaw = features(q, v, starts)
    assert state.shape == (2, 67) and chunk.shape == (2, 20, 35) and goal.shape == (2, 5)
    fk = TorchFK(model)
    robot = torch.as_tensor(model.body_rootid == 1)                      # bodies of the robot (not world-fixed ones)
    xpos_world, _ = fk(q[5 * 20, 0][None])
    xpos_local, _ = fk(chunk_qpos(chunk[0, -1])[None])
    want = to_local(xpos_world, origin[:1], yaw[:1])
    assert torch.allclose(xpos_local[:, robot], want[:, robot], atol=1e-4)
    start_local, _ = fk(state_qpos(state[:1]))
    assert torch.allclose(start_local[:, robot], to_local(fk(q[0, 0][None])[0], origin[:1], yaw[:1])[:, robot], atol=1e-4)
    assert float(goal[0, 4]) == pytest.approx(float(torch.linalg.vector_norm(chunk[0, -1, :2])) / 2.0, rel=1e-5)


def test_model_shapes_and_learning():
    from cat_mjlab.planner.model import FlowPlanner, flow_loss, sample
    torch.manual_seed(0)
    m = FlowPlanner(width=64, layers=2, heads=4)
    grid = torch.randint(0, 255, (8, 40, 32, 36), dtype=torch.uint8)
    state, goal = torch.randn(8, 67), torch.randn(8, 5)
    x1 = torch.randn(8, 20, 35) * 0.1 + 1.0
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    first = None
    for i in range(60):
        loss = flow_loss(m, x1, state, goal, grid)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < first
    out = sample(m, state, goal, grid, samples=3, steps=4)
    assert out.shape == (8, 3, 20, 35)
    assert sample(m, state, goal, None, use_scene=False, samples=2, steps=2).shape == (8, 2, 20, 35)
