"""Ray casting and voxelisation of the simulated head Mid-360 (cat_mjlab/lidar.py)."""
import math
import torch
from cat_mjlab.lidar import ray_box, ray_sphere, VoxelMemory, SHAPE


def test_ray_sphere_and_box():
    o = torch.zeros(1, 3); d = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    t = ray_sphere(o, d, torch.tensor([[[2., 0., 0.]]]), torch.tensor([.5]))
    assert abs(float(t[0, 0, 0]) - 1.5) < 1e-6 and math.isinf(float(t[0, 1, 0]))
    t = ray_box(o, d, torch.tensor([[[2., 0., 0.]]]), torch.eye(3)[None, None], torch.tensor([[.5, .5, .5]]))
    assert abs(float(t[0, 0, 0]) - 1.5) < 1e-6 and math.isinf(float(t[0, 1, 0]))
    inside = ray_box(o, d, torch.zeros(1, 1, 3), torch.eye(3)[None, None], torch.tensor([[.5, .5, .5]]))
    assert abs(float(inside[0, 0, 0]) - .5) < 1e-6                         # starting inside: exit distance
    yawed = torch.tensor([[[math.cos(.3), -math.sin(.3), 0.], [math.sin(.3), math.cos(.3), 0.], [0., 0., 1.]]])[None]
    t = ray_box(o, d[:, :1], torch.tensor([[[2., 0., 0.]]]), yawed, torch.tensor([[.5, .5, .5]]))
    assert 1.4 < float(t[0, 0, 0]) < 1.5                                    # rotated box is hit earlier


def test_voxel_memory_robot_frame_and_latency():
    m = VoxelMemory(1, 2, "cpu", scans=3)
    pts = torch.tensor([[[1.325, .025, .775], [1.0, .5, .0]]])             # world points (cell centres); robot at (1, 0, .75)
    m.add(pts, torch.ones(1, 2, dtype=torch.bool))
    g = m.grid(torch.tensor([[1., 0., .75]]), torch.zeros(1), dropout=0)
    assert g.shape == (1, *SHAPE) and int(g.sum()) == 2
    assert g[0, 20, 16, 22] == 1                                            # facing +x: 0.325 m ahead, pelvis height
    turned = m.grid(torch.tensor([[1., 0., .75]]), torch.tensor([math.pi / 2]), dropout=0)
    assert turned[0, 20, 9, 16] == 1                                        # facing +y: the point is now 0.325 m to the right
    assert int(m.grid(torch.tensor([[1., 0., .75]]), torch.zeros(1), latency_scans=torch.tensor([1]), dropout=0).sum()) == 0
    m.clear(torch.tensor([0]))
    assert int(m.grid(torch.tensor([[1., 0., .75]]), torch.zeros(1), dropout=0).sum()) == 0


def test_voxel_packing_and_student_shapes():
    from cat_mjlab.student import VoxelStudent, pack_voxels, unpack_voxels, PROPRIO
    g = (torch.rand(3, 40, 32, 32) < .05).to(torch.uint8)
    assert torch.equal(unpack_voxels(pack_voxels(g)), g.float())
    s = VoxelStudent()
    out = s(torch.randn(3, PROPRIO), g)
    assert out.shape == (3, 29)
    n = sum(p.numel() for p in s.parameters())
    assert 300_000 < n < 2_000_000, n
