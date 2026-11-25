"""Per-episode failure diagnostics for the static narrow-scene benchmark (eval_per_scene.py --diagnose).

Records, for every finished episode: outcome; the FIRST physical contact (step, regions, robot shapes, the obstacle box and its
type); progress along the route and pelvis velocity/heading at that contact; how long the robot had stalled; hand and head
clearance minima; and crouch / foot-lift evidence within 1 m of low obstacles (hurdles) and overhead obstacles (beams).

Obstacle boxes come from the scene's body-collision bank (the same boxes that end episodes). A box is
  overhead  if its lower face is at least 0.15 m above the floor (beam, bar, table top),
  low       if it touches the floor and its top is at most 0.5 m (hurdle, kerb, low bar),
  wall      otherwise (wall, pillar, cabinet side).
Routes in these scenes run along +x from the start pose to the goal, so progress is the pelvis x relative to its start.
"""
from __future__ import annotations

import numpy as np
import torch

from .collision import REGIONS, box_box_separation, capsule_box_separation, lookup_collision_candidates, \
    sphere_box_separation, transform_primitives

TYPES = ("low", "wall", "overhead")
OVERHEAD_Z, LOW_TOP = 0.15, 0.5
WINDOW = 1.0          # m around an obstacle for crouch / foot-lift evidence
STALL_SPEED = 0.03    # m/s
GOAL_RADIUS = 0.3


def classify(centers, half):
    lo, hi = centers[..., 2] - half[..., 2], centers[..., 2] + half[..., 2]
    kind = np.full(lo.shape, 1, dtype=np.int64)           # wall
    kind[hi <= LOW_TOP] = 0                                # low (floating-free)
    kind[lo >= OVERHEAD_Z] = 2                             # overhead
    return kind


def colliding_boxes(compiled, bank, scene_id, xpos, xmat, max_candidates):
    """Like body_collisions, but returns [B, shapes] ids of the first colliding box (-1 for none) and shape centres."""
    world = transform_primitives(xpos, xmat, compiled)
    result = torch.full((len(scene_id), len(compiled["body_ids"])), -1, dtype=torch.long, device=xpos.device)
    for kind, indices in compiled["indices"].items():
        if not len(indices):
            continue
        centers = world["centers"][:, indices]
        ids, valid = lookup_collision_candidates(bank, scene_id, centers, max_candidates=max_candidates)
        found = torch.full(centers.shape[:2], -1, dtype=torch.long, device=xpos.device)
        for begin in range(0, max_candidates, 8):
            item, mask = ids[..., begin:begin + 8], valid[..., begin:begin + 8]
            obstacle = (bank["centers"][item], bank["rotations"][item], bank["half_sizes"][item])
            if kind == "box":
                separation = box_box_separation(centers[:, :, None], world["rotations"][:, indices, None],
                                                compiled["half_sizes"][indices][None, :, None], *obstacle)
            elif kind == "capsule":
                ends = world["endpoints"][:, indices]
                separation = capsule_box_separation(ends[:, :, None, 0], ends[:, :, None, 1],
                                                    compiled["radii"][indices][None, :, None], *obstacle)
            else:
                separation = sphere_box_separation(centers[:, :, None], compiled["radii"][indices][None, :, None], *obstacle)
            hit = mask & (separation <= 0)
            first = item.gather(-1, hit.int().argmax(-1, keepdim=True)).squeeze(-1)
            found = torch.where((found < 0) & hit.any(-1), first, found)
        result[:, indices] = found
    return result, world["centers"]


class EpisodeDiagnostics:
    def __init__(self, task, scene_names, scene_goals, n_scene_slots, speeds):
        self.task = task
        self.n = task.num_envs
        dev = task.device
        self.dev = dev
        self.dt = float(task.dt)
        checker = task.collision
        self.checker = checker
        self.compiled, self.bank = checker.compiled, checker.bank
        self.max_candidates = checker.metadata["static_candidate_count"]
        self.shape_ids = [s["id"] for s in __import__("json").loads(checker.proposal_path.read_text())["shapes"]]
        self.region_of_shape = checker.compiled["region_mask"].int().argmax(0)          # [S]
        kind = torch.as_tensor(classify(self.bank["centers"].cpu().numpy(), self.bank["half_sizes"].cpu().numpy()), device=dev)
        self.box_kind = kind
        self.names = scene_names
        self.goals = scene_goals
        self.NS = n_scene_slots
        self.speeds = speeds
        # per-scene x extents of low and overhead boxes (+/- WINDOW) for crouch and foot-lift evidence
        off, cnt = self.bank["scene_box_offsets"].cpu().numpy(), self.bank["scene_box_counts"].cpu().numpy()
        c, h, k = self.bank["centers"].cpu().numpy(), self.bank["half_sizes"].cpu().numpy(), kind.cpu().numpy()
        S = len(off)
        self.low_x = np.full((S, 2), np.nan, dtype=np.float32)
        self.over_x = np.full((S, 2), np.nan, dtype=np.float32)
        self.box_counts = np.zeros((S, 3), dtype=np.int32)
        for s in range(S):
            a, b = off[s], off[s] + cnt[s]
            if b <= a:
                continue
            for code, store in ((0, self.low_x), (2, self.over_x)):
                m = k[a:b] == code
                if m.any():
                    store[s] = (np.min(c[a:b][m, 0] - h[a:b][m, 0]), np.max(c[a:b][m, 0] + h[a:b][m, 0]))
            for code in range(3):
                self.box_counts[s, code] = int((k[a:b] == code).sum())
        self.low_x = torch.as_tensor(self.low_x, device=dev)
        self.over_x = torch.as_tensor(self.over_x, device=dev)
        z = lambda *shape, dtype=torch.float32: torch.zeros(shape, device=dev, dtype=dtype)
        self.reset_all()
        self.records = []
        self.contact_box = None

    def reset_all(self):
        dev, n = self.dev, self.n
        full = lambda v, dtype=torch.float32: torch.full((n,), v, device=dev, dtype=dtype)
        self.start_xy = torch.zeros((n, 2), device=dev)
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.step_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.best_x = full(-1e9)
        self.last_progress_step = torch.zeros(n, dtype=torch.long, device=dev)
        self.stalled = torch.zeros(n, dtype=torch.long, device=dev)
        self.min_hand = full(1e9); self.min_head = full(1e9); self.min_all = full(1e9)
        self.min_pelvis_z = full(1e9); self.max_foot_z = full(-1e9)
        self.near_low_foot = full(-1e9); self.near_over_pelvis = full(1e9); self.near_low_steps = torch.zeros(n, device=self.dev, dtype=torch.long)
        self.near_over_steps = torch.zeros(n, device=self.dev, dtype=torch.long)
        self.last = dict(xyz=torch.zeros((n, 3), device=dev), v=torch.zeros((n, 2), device=dev), yaw=torch.zeros(n, device=dev))
        # the first contact
        self.hit = torch.zeros(n, dtype=torch.bool, device=dev)
        self.hit_info = {}

    # ---- collision wrapper: runs inside task.step on the post-physics state, before any reset -------------------------
    def wrap_collision(self):
        original = self.task.collision
        diag = self

        def wrapped(scene_ids, data):
            flags = original(scene_ids, data)
            if scene_ids is diag.task.scene_ids and flags.shape[0] == diag.n:
                new = flags.any(-1) & ~diag.hit
                if bool(new.any()):
                    diag._record_contact(new.nonzero().flatten(), flags, data)
            return flags
        self.task.collision = wrapped
        self.original_collision = original

    def _record_contact(self, ids, flags, data):
        task = self.task
        boxes, centers = colliding_boxes(self.compiled, self.bank, task.scene_ids[ids], data.xpos[ids], data.xmat[ids], self.max_candidates)
        shape_hit = boxes >= 0                                           # [b, S]
        region = torch.where(shape_hit, self.region_of_shape[None], torch.full_like(boxes, 99))
        first_region = region.amin(-1)
        # primary shape: the colliding shape of the lowest-numbered region (feet before legs ... hands)
        score = torch.where(shape_hit, self.region_of_shape[None] * 1000 + torch.arange(boxes.shape[1], device=self.dev)[None], 10 ** 6)
        primary = score.argmin(-1)
        r = torch.arange(len(ids), device=self.dev)
        box = boxes[r, primary]
        qpos, qvel = task.data.qpos[ids], task.data.qvel[ids]
        valid = shape_hit.any(-1)
        self.hit[ids] = True
        info = dict(step=self.step_count[ids].clone(), region_mask=flags[ids].clone(), first_region=first_region, primary_shape=primary,
                    box=box, box_kind=torch.where(box >= 0, self.box_kind[box.clamp_min(0)], torch.full_like(box, -1)),
                    box_center=self.bank["centers"][box.clamp_min(0)], box_half=self.bank["half_sizes"][box.clamp_min(0)],
                    shape_center=centers[r, primary], xyz=qpos[:, :3].clone(), quat=qpos[:, 3:7].clone(), v=qvel[:, :3].clone(), valid=valid,
                    all_regions=flags[ids].clone())
        if not self.hit_info:
            self.hit_info = {k: torch.zeros((self.n, *v.shape[1:]), dtype=v.dtype, device=self.dev) for k, v in info.items()}
        for k, v in info.items():
            self.hit_info[k][ids] = v

    # ---- before each control step: running statistics of the live episodes -------------------------------------------
    @torch.no_grad()
    def observe(self, before):
        task = self.task
        alive = before >= 0
        xyz = task.data.qpos[:, :3]
        quat = task.data.qpos[:, 3:7]
        yaw = torch.atan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]), 1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
        vworld = task.data.qvel[:, :2]
        # first observation of an episode fixes its start
        self.start_xy = torch.where(self.fresh[:, None], xyz[:, :2], self.start_xy)
        self.fresh = torch.zeros_like(self.fresh)
        x = xyz[:, 0] - self.start_xy[:, 0]
        self.step_count += alive.long()
        progressed = x > self.best_x + 0.01
        self.last_progress_step = torch.where(progressed & alive, self.step_count, self.last_progress_step)
        self.best_x = torch.maximum(self.best_x, torch.where(alive, x, self.best_x))
        speed = torch.linalg.vector_norm(vworld, dim=-1)
        self.stalled += (alive & (speed < STALL_SPEED) & (self.step_count > 50)).long()
        sdf = task.info["sdf"][:, :, 0]
        self.min_hand = torch.minimum(self.min_hand, torch.where(alive, sdf[:, 5:7].amin(-1), self.min_hand))
        self.min_head = torch.minimum(self.min_head, torch.where(alive, sdf[:, 0:1].amin(-1), self.min_head))
        self.min_all = torch.minimum(self.min_all, torch.where(alive, sdf[:, :11].amin(-1), self.min_all))
        self.min_pelvis_z = torch.minimum(self.min_pelvis_z, torch.where(alive, xyz[:, 2], self.min_pelvis_z))
        feet_z = task.info["positions"][:, 3:5, 2].amax(-1)
        self.max_foot_z = torch.maximum(self.max_foot_z, torch.where(alive, feet_z, self.max_foot_z))
        scene = task.scene_ids
        lx, ox = self.low_x[scene], self.over_x[scene]
        absx = xyz[:, 0]
        near_low = alive & (absx > lx[:, 0] - WINDOW) & (absx < lx[:, 1] + WINDOW)
        near_over = alive & (absx > ox[:, 0] - WINDOW) & (absx < ox[:, 1] + WINDOW)
        self.near_low_foot = torch.maximum(self.near_low_foot, torch.where(near_low, feet_z, self.near_low_foot))
        self.near_over_pelvis = torch.minimum(self.near_over_pelvis, torch.where(near_over, xyz[:, 2], self.near_over_pelvis))
        self.near_low_steps += near_low.long(); self.near_over_steps += near_over.long()
        self.last = dict(xyz=torch.where(alive[:, None], xyz, self.last["xyz"]), v=torch.where(alive[:, None], vworld, self.last["v"]),
                         yaw=torch.where(alive, yaw, self.last["yaw"]))

    # ---- after a step: finalize finished episodes --------------------------------------------------------------------
    @torch.no_grad()
    def finish(self, done_ids, before, m, outcome_flags):
        """done_ids: env indices that just finished; outcome_flags: dict of bool [n] tensors (goal/obstacle/fall/refused)."""
        names = self.names
        dt = self.dt
        for e in done_ids.tolist():
            job = int(before[e])
            scene_slot = job % self.NS
            speed_index = job // self.NS
            xs = float(self.last["xyz"][e, 0] - self.start_xy[e, 0])
            goal_x = float(self.goals[scene_slot])
            rec = dict(scene=names[scene_slot], speed=float(self.speeds[speed_index]), env=e, steps=int(self.step_count[e]),
                       outcome=("clean" if outcome_flags["clean"][e] else "contact" if outcome_flags["contact"][e] else "fall" if outcome_flags["fall"][e]
                                else "refused" if outcome_flags["refused"][e] else "other"),
                       goal=bool(outcome_flags["goal"][e]), progress_m=max(float(self.best_x[e]), xs), route_m=goal_x,
                       stalled_s=float(self.stalled[e]) * dt, since_progress_s=float(self.step_count[e] - self.last_progress_step[e]) * dt,
                       min_hand_clear=float(self.min_hand[e]), min_head_clear=float(self.min_head[e]), min_clear=float(self.min_all[e]),
                       min_pelvis_z=float(self.min_pelvis_z[e]), max_foot_z=float(self.max_foot_z[e]),
                       near_low_foot_z=None if self.near_low_steps[e] == 0 else float(self.near_low_foot[e]),
                       near_over_pelvis_z=None if self.near_over_steps[e] == 0 else float(self.near_over_pelvis[e]),
                       boxes_low_wall_over=[int(x) for x in self.box_counts[scene_slot]],
                       regions_at_end={r: bool(m["episode/body_collision_" + r][e]) for r in REGIONS if ("episode/body_collision_" + r) in m})
            if bool(self.hit[e]):
                h = {k: v[e] for k, v in self.hit_info.items()}
                quat = h["quat"]
                yaw = float(torch.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)))
                vx, vy = float(h["v"][0]), float(h["v"][1])
                fwd = vx * np.cos(yaw) + vy * np.sin(yaw); lat = -vx * np.sin(yaw) + vy * np.cos(yaw)
                x0 = float(h["xyz"][0] - self.start_xy[e, 0])
                rec["contact"] = dict(
                    step=int(h["step"]), time_s=int(h["step"]) * dt,
                    regions=[REGIONS[i] for i in range(len(REGIONS)) if bool(h["region_mask"][i])],
                    first_region=REGIONS[int(h["first_region"])] if int(h["first_region"]) < len(REGIONS) else None,
                    shape=self.shape_ids[int(h["primary_shape"])], box=int(h["box"]),
                    box_type=TYPES[int(h["box_kind"])] if int(h["box_kind"]) >= 0 else None,
                    box_center=[float(v) for v in h["box_center"]], box_half=[float(v) for v in h["box_half"]],
                    shape_z=float(h["shape_center"][2]), pelvis_z=float(h["xyz"][2]), progress_m=x0, progress_frac=x0 / max(goal_x, 1e-6),
                    speed_xy=float(np.hypot(vx, vy)), v_forward=float(fwd), v_lateral=float(lat),
                    heading_rad=yaw, heading_deg=float(np.degrees(yaw)), sideways=bool(abs(np.degrees(yaw)) > 45.))
            self.records.append(rec)
        # clear state of the envs that restarted
        d = done_ids
        full = lambda v: v
        self.fresh[d] = True
        self.step_count[d] = 0; self.best_x[d] = -1e9; self.last_progress_step[d] = 0; self.stalled[d] = 0
        self.min_hand[d] = 1e9; self.min_head[d] = 1e9; self.min_all[d] = 1e9; self.min_pelvis_z[d] = 1e9; self.max_foot_z[d] = -1e9
        self.near_low_foot[d] = -1e9; self.near_over_pelvis[d] = 1e9; self.near_low_steps[d] = 0; self.near_over_steps[d] = 0
        self.hit[d] = False
