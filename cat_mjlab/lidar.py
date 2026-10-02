"""Simulated head Livox Mid-360 and a Gallant-style robot-centric voxel grid.

Sensor (the G1 head unit, MJCF site ``mid360_site``, mounted upside down: native vertical field of view
-7..+52 deg becomes 52 deg down .. 7 deg up). Each 10 Hz scan casts ``rays`` directions drawn at random
inside the field of view (the Mid-360 pattern is non-repetitive), against:
  * the scene: sphere tracing on the scene bank's signed distance field (rooms and CAT scenes alike);
  * the floor plane z = 0 (not part of the obstacle SDF);
  * the robot itself: the approved collision-proxy primitives occlude rays (capsules approximated by
    three spheres); the head box that houses the sensor is excluded. Body hits are dropped, as the
    real pipeline filters points on the robot with its kinematics.
Randomisation (Gallant): sensor pose per episode (N(0, 1 cm), N(0, 1 deg)), hit position N(0, 1 cm),
latency 100-200 ms (the newest one or two scans are withheld), 2 % of voxels dropped.

With a single downward-tilted head sensor the floor is visible only beyond ~0.96 m and shin-height
obstacles only from ~0.7 m, so points of the last ``memory_scans`` scans are kept in WORLD coordinates
(on the robot: FAST-LIO2 odometry) and voxelised in the current robot frame every control step.

Grid (Gallant): x, y in [-0.8, 0.8] m, z in [-1.0, 1.0] m around the pelvis, 0.05 m cells, yaw-aligned
with the body -> uint8 [N, 40, 32, 32] (height slices as channels).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

GRID = dict(x=(-.8, .8), y=(-.8, .8), z=(-1., 1.), cell=.05)
SHAPE = (40, 32, 32)            # z, y, x


def _quat_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(*q.shape[:-1], 3, 3)


def ray_sphere(origin, direction, center, radius):
    """Rays from origin [N,3] along direction [N,R,3] vs spheres center [N,S,3], radius [S] -> [N,R,S] (inf: miss)."""
    oc = origin[:, None, :] - center                                   # [N,S,3]
    b = torch.einsum("nsk,nrk->nrs", oc, direction)                    # [N,R,S]
    c = (oc * oc).sum(-1) - radius.square()                            # [N,S]
    disc = b * b - c[:, None, :]
    root = disc.clamp_min(0).sqrt()
    t = torch.where(-b - root > 0, -b - root, -b + root)
    return torch.where((disc >= 0) & (t > 0), t, torch.full_like(t, float("inf")))


def ray_box(origin, direction, center, rotation, half):
    """Ray vs oriented boxes. origin [N,3], direction [N,R,3], center [N,B,3], rotation [N,B,3,3] (local->world),
    half [B,3] -> [N,R,B]. Rays starting inside a box report the exit distance."""
    o = torch.einsum("nbji,nbj->nbi", rotation, origin[:, None] - center)          # [N,B,3] local origin
    d = torch.einsum("nbji,nrj->nrbi", rotation, direction)                        # [N,R,B,3]
    inv = 1. / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    t1 = (-half - o[:, None]) * inv; t2 = (half - o[:, None]) * inv
    near = torch.minimum(t1, t2).amax(-1); far = torch.maximum(t1, t2).amin(-1)
    t = torch.where(near > 0, near, far)
    return torch.where((far >= near) & (far > 0), t, torch.full_like(t, float("inf")))


class RobotOccluder:
    """Collision-proxy primitives of the robot, posed from MuJoCo body frames each scan."""

    def __init__(self, model, device, proposal=None):
        import mujoco
        from .collision import PROPOSAL
        shapes = json.loads(Path(proposal or PROPOSAL).read_text())["shapes"]
        body = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        boxes = [s for s in shapes if s["kind"] == "box" and s["group"] != "head"]
        spheres = []
        for s in shapes:
            if s["kind"] == "sphere":
                spheres.append((body(s["body_name"]), s["center"], s["radius"]))
            elif s["kind"] == "capsule":
                a, b = torch.tensor(s["endpoints"][0]), torch.tensor(s["endpoints"][1])
                for f in (0., .5, 1.):
                    spheres.append((body(s["body_name"]), (a + f * (b - a)).tolist(), s["radius"]))
        t = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=device)
        self.box_body = t([body(s["body_name"]) for s in boxes], torch.long)
        self.box_center = t([s["center"] for s in boxes]); self.box_half = t([s["half_size"] for s in boxes])
        self.box_rot = _quat_to_matrix(t([s["quat"] for s in boxes]))
        self.sphere_body = t([b for b, _, _ in spheres], torch.long)
        self.sphere_center = t([c for _, c, _ in spheres]); self.sphere_radius = t([r for _, _, r in spheres])

    def first_hit(self, origin, direction, xpos, xmat):
        """[N,R] distance to the first robot surface along each ray (inf if none)."""
        bp, bm = xpos[:, self.box_body], xmat[:, self.box_body]
        center = bp + torch.einsum("nbij,bj->nbi", bm, self.box_center)
        rotation = bm @ self.box_rot
        t_box = ray_box(origin, direction, center, rotation, self.box_half).amin(-1)
        sp, sm = xpos[:, self.sphere_body], xmat[:, self.sphere_body]
        centers = sp + torch.einsum("nsij,sj->nsi", sm, self.sphere_center)
        t_sph = ray_sphere(origin, direction, centers, self.sphere_radius).amin(-1)
        return torch.minimum(t_box, t_sph)


class Mid360:
    def __init__(self, model, device, *, rays=2048, min_range=.1, max_range=4., native_fov=(-7., 52.),
                 trace_steps=40, hit_eps=.02, min_step=.02, site="mid360_site"):
        import mujoco
        self.device, self.rays = torch.device(device), int(rays)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site < 0:
            raise ValueError(f"robot model has no site {site}")
        self.min_range, self.max_range = min_range, max_range
        lo, hi = (math.radians(a) for a in native_fov)
        self.sin_lo, self.sin_hi = math.sin(lo), math.sin(hi)
        self.trace_steps, self.hit_eps, self.min_step = trace_steps, hit_eps, min_step
        self.occluder = RobotOccluder(model, device)

    def directions(self, n, generator=None):
        """Uniform over the native field-of-view band, in the sensor frame: [n, rays, 3]."""
        u = torch.rand((n, self.rays, 2), device=self.device, generator=generator)
        az = 2 * math.pi * u[..., 0]; s = self.sin_lo + (self.sin_hi - self.sin_lo) * u[..., 1]
        c = (1 - s * s).clamp_min(0).sqrt()
        return torch.stack((c * az.cos(), c * az.sin(), s), -1)

    def scan(self, site_pos, site_rot, scene_ids, bank, xpos, xmat, *, pose_noise=None, hit_noise=.01, generator=None):
        """One scan. site_pos [N,3], site_rot [N,3,3] -> world points [N,R,3], valid [N,R]."""
        n = len(site_pos)
        if pose_noise is not None:
            site_pos = site_pos + pose_noise[0]; site_rot = site_rot @ pose_noise[1]
        d = torch.einsum("nij,nrj->nri", site_rot, self.directions(n, generator))
        o = site_pos
        t = torch.full((n, self.rays), self.min_range, device=self.device)
        alive = torch.ones((n, self.rays), dtype=torch.bool, device=self.device)
        hit = torch.zeros_like(alive)
        for _ in range(self.trace_steps):
            p = o[:, None] + t[..., None] * d
            s = bank.sample("sdf", p, scene_ids).reshape(n, self.rays)
            now = alive & (s < self.hit_eps)
            hit |= now
            alive = alive & ~now & (t < self.max_range)
            t = torch.where(alive, t + s.clamp_min(self.min_step), t)
            if not bool(alive.any()):
                break
        t_scene = torch.where(hit, t, torch.full_like(t, float("inf")))
        t_floor = torch.where(d[..., 2] < -1e-6, -o[:, None, 2] / d[..., 2].clamp_max(-1e-6), torch.full_like(t, float("inf")))
        t_world = torch.minimum(t_scene, t_floor)
        t_self = self.occluder.first_hit(o, d, xpos, xmat)
        valid = (t_world < self.max_range) & (t_world < t_self)
        points = o[:, None] + t_world.clamp_max(self.max_range)[..., None] * d
        if hit_noise:
            points = points + hit_noise * torch.randn(points.shape, device=self.device, generator=generator)
        return points, valid


class VoxelMemory:
    """Points of the last ``scans`` scans in world coordinates -> robot-centric occupancy grid."""

    def __init__(self, n, rays, device, *, scans=10):
        self.points = torch.zeros((n, scans, rays, 3), device=device, dtype=torch.float16 if str(device).startswith("cuda") else torch.float32)
        self.valid = torch.zeros((n, scans, rays), dtype=torch.bool, device=device)
        self.age = torch.full((n, scans), 10 ** 6, dtype=torch.long, device=device)   # scans since recorded
        self.slot = 0

    def clear(self, ids):
        self.valid[ids] = False; self.age[ids] = 10 ** 6

    def add(self, points, valid):
        self.age += 1
        self.points[:, self.slot] = points.to(self.points.dtype); self.valid[:, self.slot] = valid; self.age[:, self.slot] = 0
        self.slot = (self.slot + 1) % self.points.shape[1]

    def grid(self, base_pos, base_yaw, *, latency_scans=None, dropout=.02, generator=None):
        """uint8 [N,40,32,32]. latency_scans [N] (0..2): withhold that many newest scans."""
        n = len(base_pos)
        keep = self.valid.clone()
        if latency_scans is not None:
            keep &= (self.age >= latency_scans[:, None])[..., None]
        rel = self.points.float() - base_pos[:, None, None]
        c, s = base_yaw.cos()[:, None, None], base_yaw.sin()[:, None, None]
        x = c * rel[..., 0] + s * rel[..., 1]; y = -s * rel[..., 0] + c * rel[..., 1]; z = rel[..., 2]
        cell = GRID["cell"]
        ix = ((x - GRID["x"][0]) / cell).floor().long(); iy = ((y - GRID["y"][0]) / cell).floor().long()
        iz = ((z - GRID["z"][0]) / cell).floor().long()
        inside = keep & (ix >= 0) & (ix < SHAPE[2]) & (iy >= 0) & (iy < SHAPE[1]) & (iz >= 0) & (iz < SHAPE[0])
        flat = torch.where(inside, (iz * SHAPE[1] + iy) * SHAPE[2] + ix, torch.full_like(ix, SHAPE[0] * SHAPE[1] * SHAPE[2]))
        occupancy = torch.zeros((n, SHAPE[0] * SHAPE[1] * SHAPE[2] + 1), dtype=torch.uint8, device=base_pos.device)
        occupancy.scatter_(1, flat.reshape(n, -1), 1)
        occupancy = occupancy[:, :-1]
        if dropout:
            occupancy = occupancy * (torch.rand(occupancy.shape, device=base_pos.device, generator=generator) >= dropout)
        return occupancy.reshape(n, *SHAPE)


def sensor_pose_noise(n, device, generator=None, *, position=.01, angle=math.radians(1.)):
    """Per-episode sensor mounting error: (offset [n,3], rotation [n,3,3])."""
    offset = position * torch.randn((n, 3), device=device, generator=generator)
    a = angle * torch.randn((n, 3), device=device, generator=generator)
    rx, ry, rz = a.unbind(-1)
    one, zero = torch.ones_like(rx), torch.zeros_like(rx)
    Rx = torch.stack((one, zero, zero, zero, rx.cos(), -rx.sin(), zero, rx.sin(), rx.cos()), -1).reshape(n, 3, 3)
    Ry = torch.stack((ry.cos(), zero, ry.sin(), zero, one, zero, -ry.sin(), zero, ry.cos()), -1).reshape(n, 3, 3)
    Rz = torch.stack((rz.cos(), -rz.sin(), zero, rz.sin(), rz.cos(), zero, zero, zero, one), -1).reshape(n, 3, 3)
    return offset, Rz @ Ry @ Rx
