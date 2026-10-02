#!/usr/bin/env python3
"""Look at what the simulated head Mid-360 sees: walk a scene with a policy, scan at 10 Hz, save images.

Per scene: (left) top view -- obstacle cross-section from the scene SDF at 0.1/0.5/1.0 m, the
accumulated LiDAR points (memory), the robot path and the +-0.8 m grid square; (right) the robot-centric
32x32x40 voxel grid in 3D. Also prints scan timing and how many voxels are occupied.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/../outputs/saved_checkpoints/cat_g1_rooms_keyboard_v9_u492_full.pt")
    p.add_argument("--scenes", nargs="+", default=["random-furniture-dense-train-004003", "table-contrast-train-20260924-forward_protected",
                                                    "D8G0L1O0S3"])
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--output", default="outputs/lidar_preview")
    args = p.parse_args()
    import numpy as np
    import torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = importlib.util.spec_from_file_location("demo", ROOT / "scripts/demo/teleop_demo.py")
    demo = importlib.util.module_from_spec(spec); spec.loader.exec_module(demo)
    from cat_mjlab.lidar import Mid360, VoxelMemory, GRID, SHAPE, sensor_pose_noise
    s = demo.Session(argparse.Namespace(checkpoint=args.checkpoint, bank="data/furniture/table_edges_v1", scenes=args.scenes, speed=.5))
    task, sim = s.task, s.sim
    lidar = Mid360(sim.model, "cpu")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    for scene in s.scenes:
        task.joystick = torch.zeros((1, 2)); task.joystick_body = torch.zeros((1, 3)) if s.body_mode else None
        s.load(scene); task.joystick = None; task.joystick_body = None   # follow the scene's own route / CAT field
        memory = VoxelMemory(1, lidar.rays, "cpu", scans=10)
        noise = sensor_pose_noise(1, "cpu", gen)
        path, scan_ms = [], []
        for k in range(args.steps):
            with torch.no_grad():
                task.step(s.learner.act(task.obs, policy_ids=0, deterministic=True)["action"])
            path.append(sim.raw.qpos[:3].copy())
            if k % 5 == 0:
                d = sim.data; t0 = time.time()
                pts, valid = lidar.scan(d.site_xpos[:, lidar.site], d.site_xmat[:, lidar.site], task.scene_ids, task.bank,
                                        d.xpos, d.xmat, pose_noise=noise, generator=gen)
                scan_ms.append((time.time() - t0) * 1000)
                memory.add(pts, valid)
        q = sim.raw.qpos; yaw = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
        grid = memory.grid(torch.tensor(q[None, :3], dtype=torch.float32), torch.tensor([yaw]), dropout=.02, generator=gen)[0]
        occupied = torch.nonzero(grid).numpy()
        fig = plt.figure(figsize=(14, 6.5))
        ax = fig.add_subplot(1, 2, 1)
        path = np.array(path); c = path[-1]
        xs = np.linspace(c[0] - 2.5, c[0] + 2.5, 200); ys = np.linspace(c[1] - 2.5, c[1] + 2.5, 200)
        X, Y = np.meshgrid(xs, ys)
        for h, col in ((.1, "#d9b38c"), (.5, "#b07a4a"), (1.0, "#6b4226")):
            P = torch.tensor(np.stack((X, Y, np.full_like(X, h)), -1).reshape(1, -1, 3), dtype=torch.float32)
            sd = task.bank.sample("sdf", P, task.scene_ids).reshape(X.shape).numpy()
            ax.contourf(X, Y, (sd < 0).astype(float), levels=[.5, 1.5], colors=[col], alpha=.35)
        mp = memory.points[0][memory.valid[0]].numpy()
        if len(mp):
            ax.scatter(mp[:, 0], mp[:, 1], s=1, c=mp[:, 2], cmap="viridis", vmin=0, vmax=1.8)
        ax.plot(path[:, 0], path[:, 1], "r-", lw=1.5)
        sq = np.array([[-.8, -.8], [.8, -.8], [.8, .8], [-.8, .8], [-.8, -.8]])
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        sq = sq @ R.T + c[:2]; ax.plot(sq[:, 0], sq[:, 1], "k--", lw=1)
        ax.set_xlim(c[0] - 2.5, c[0] + 2.5); ax.set_ylim(c[1] - 2.5, c[1] + 2.5); ax.set_aspect("equal")
        ax.set_title(f"{scene[:40]}\nobstacles at 0.1/0.5/1.0 m (light->dark), LiDAR memory points (colour = height), path, grid square")
        ax3 = fig.add_subplot(1, 2, 2, projection="3d")
        if len(occupied):
            z, y, x = occupied.T
            cx = GRID["x"][0] + (x + .5) * GRID["cell"]; cy = GRID["y"][0] + (y + .5) * GRID["cell"]; cz = GRID["z"][0] + (z + .5) * GRID["cell"]
            ax3.scatter(cx, cy, cz, s=3, c=cz, cmap="viridis")
        ax3.set_xlim(-.8, .8); ax3.set_ylim(-.8, .8); ax3.set_zlim(-1, 1)
        ax3.set_xlabel("forward"); ax3.set_ylabel("left"); ax3.set_zlabel("up (from pelvis)")
        ax3.set_title(f"robot-centric voxel grid 32x32x40 ({len(occupied)} occupied)")
        fig.tight_layout(); name = out / f"{scene[:48]}.png"; fig.savefig(name, dpi=90); plt.close(fig)
        print(json.dumps(dict(scene=scene, image=str(name), scans=len(scan_ms), scan_ms_cpu=round(float(np.mean(scan_ms)), 1),
                              memory_points=int(memory.valid.sum()), occupied_voxels=len(occupied))), flush=True)


if __name__ == "__main__":
    main()
