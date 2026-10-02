"""Deployable student: proprioception + command + Gallant-style voxel grid -> 29 joint actions.

Inputs
  proprio [N, 131]: the teacher's actor features 0..130 (gyro, gravity, joint positions/velocities, last
    action, motor targets, command, foot height, gait phase) -- everything except the scene fields, which
    need the map. command[3] is the RAW heading error (reference heading - facing): the teacher's gate
    (would a body facing it fit here?) is computed from the map, the student decides from the voxels.
  voxels [N, 40, 32, 32] uint8 (z slices as channels; cat_mjlab.lidar).
Network (Gallant, CVPR 2026, widened): z-grouped 2D CNN 40 -> 32 -> 32 -> 32 channels, stride 2 each
(32x32 -> 4x4), linear to 128; proprio -> 256; concat -> 512 -> 256 -> 128 -> 29 (Mish, LayerNorm).
Output: the pre-squash action mean, like the teachers (executed action = tanh(mean)).
"""
from __future__ import annotations

import torch
import torch.nn as nn

PROPRIO = 131                     # teacher actor features 0..130 (see observation_contract)
HEADING_INDEX = 125               # command[3] inside the proprio block
VOXEL_SHAPE = (40, 32, 32)
PACKED = 40 * 32 * 32 // 8


def pack_voxels(grid):
    """uint8 [N,40,32,32] (0/1) -> uint8 [N,5120]."""
    bits = grid.reshape(len(grid), PACKED, 8).to(torch.uint8)
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=grid.device)
    return (bits * weights).sum(-1).to(torch.uint8)


def unpack_voxels(packed):
    """uint8 [N,5120] -> float [N,40,32,32]."""
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = (packed[..., None] >> shifts) & 1
    return bits.reshape(len(packed), *VOXEL_SHAPE).float()


class VoxelEncoder(nn.Module):
    def __init__(self, channels=(32, 32, 32), out=128):
        super().__init__()
        layers, last = [], VOXEL_SHAPE[0]
        for c in channels:
            layers += [nn.Conv2d(last, c, 3, stride=2, padding=1), nn.Mish()]
            last = c
        self.conv = nn.Sequential(*layers)
        side = VOXEL_SHAPE[1] // 2 ** len(channels)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(last * side * side, out), nn.LayerNorm(out), nn.Mish())

    def forward(self, voxels):
        return self.head(self.conv(voxels))


class VoxelStudent(nn.Module):
    def __init__(self, action_size=29, hidden=(512, 256, 128), proprio_hidden=256, voxel_features=128):
        super().__init__()
        self.voxels = VoxelEncoder(out=voxel_features)
        self.proprio = nn.Sequential(nn.LayerNorm(PROPRIO), nn.Linear(PROPRIO, proprio_hidden), nn.Mish())
        layers, last = [], proprio_hidden + voxel_features
        for h in hidden:
            layers += [nn.Linear(last, h), nn.LayerNorm(h), nn.Mish()]
            last = h
        layers.append(nn.Linear(last, action_size))
        self.policy = nn.Sequential(*layers)
        self.config = dict(action_size=action_size, hidden=list(hidden), proprio_hidden=proprio_hidden, voxel_features=voxel_features)

    def forward(self, proprio, voxels):
        return self.policy(torch.cat((self.proprio(proprio), self.voxels(voxels.float())), -1))


def student_proprio(task):
    """The student's proprio block from the task's teacher observation: features 0..130, raw heading error."""
    proprio = task.obs["state"][:, :PROPRIO].clone()
    error = getattr(task, "_heading_error", None)
    commanded = getattr(task, "_heading_commanded", None)
    if error is not None and commanded is not None:
        proprio[:, HEADING_INDEX] = torch.where(commanded, error, torch.zeros_like(error))
    return proprio
