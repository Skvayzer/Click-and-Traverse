"""Objects thrown at the live hands of a traversing robot (frozen-policy dynamic-threat benchmark, A1).

The standing reactive benchmark re-aims every approach at the hand where it IS (reactive.StandingObjects.rearm) and
sends the object in along a straight line at an authored speed. This module carries the same idea into traversal
episodes. Each episode has up to three launches, one per speed tier, at scheduled times. When a launch is due the
object is aimed at the current hand centre, led by the measured hand velocity, from the robot's front or outer side.
The path is straight and constant-speed. It reaches the aimed hand centre and flies on for OVERSHOOT metres, so an
unmoved hand is hit. Candidate rays are drawn in advance, and the ray whose approach is clearest of every non-hand
body sphere is chosen live, so the hand is the first body part met, as in the standing bank.

The schedule (times, tiers, side, ray randoms) is fixed per episode row, hence identical for every policy. The
geometry is computed from each policy's own live state, so the benchmark threatens each policy's own hand.
Objects are the plan-blind box (an upright 5.4 x 5.4 x 17 cm bar) with the same physical contact pairs and the
same policy-side field merge as the plan-blind benchmark, so contacts and perception match E3.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .plan_blind import BOX_HALF, HIDDEN, BlindObstacle
from .reactive import BodyEnvelope

SPEEDS = (.4, .8, 1.2)         # m/s, one launch per tier in a random order per episode
SLOTS = len(SPEEDS)
APPROACH = .55                 # m, launch point to the aimed hand centre
OVERSHOOT = .10                # m flown beyond the aimed centre
FIRST = (3., 8.)               # s, window for the first scheduled launch
SPACING = 6.                   # s between scheduled launches
LATE = 3.                      # s a blocked launch may slip before the slot is skipped
PAUSE = 1.                     # s of calm after a throw before the next may start
CANDIDATES = 8
FLOOR = .12                    # m, lowest allowed object centre
SCENE_CLEAR = .03              # m, object centre clearance from static scene geometry along the path
START_CLEAR = .10              # m, object start clearance from static scene geometry
BODY_CLEAR = .01               # m, clearance of the approach from every non-hand body sphere
HAND_ZONE = .22                # m, the path is checked against non-hand spheres only farther than this from the hand
BOX_DIAG = float(np.linalg.norm(BOX_HALF))


def make_schedule(num, seed=20261125):
    """Per-episode throw schedule, drawn once and shared by all policies."""
    rng = np.random.default_rng(seed)
    first = rng.uniform(*FIRST, num)
    return dict(throw_time=(first[:, None] + SPACING * np.arange(SLOTS)[None]).astype(np.float32),
                throw_speed=np.stack([rng.permutation(SPEEDS) for _ in range(num)]).astype(np.float32),
                throw_side=rng.integers(0, 2, (num, SLOTS)).astype(np.int64),
                throw_az=rng.uniform(0., 1., (num, SLOTS, CANDIDATES)).astype(np.float32),
                throw_el=rng.uniform(0., 1., (num, SLOTS, CANDIDATES)).astype(np.float32))


class ThrownObstacle(BlindObstacle):
    """Schedule + live launch state. Drop-in for BlindObstacle in BlindSimulation and task.analytic_objects."""

    def __init__(self, guide, task, device):
        n = task.num_envs
        tensor = lambda key, dtype=None: torch.as_tensor(guide[key], device=device, dtype=dtype)
        self.sched_t = tensor('throw_time', torch.float32)
        self.sched_speed = tensor('throw_speed', torch.float32)
        self.sched_side = tensor('throw_side', torch.long)
        self.sched_az = tensor('throw_az', torch.float32)
        self.sched_el = tensor('throw_el', torch.float32)
        if self.sched_t.shape != (n, SLOTS):
            raise ValueError('Throw schedule does not match the number of simulated episodes')
        self.valid = torch.ones(n, dtype=torch.bool, device=device)
        self.enabled = self.valid.clone()
        self.half = torch.tensor(BOX_HALF, device=device)
        self.hidden = torch.tensor(HIDDEN, device=device)
        self.device = device
        self.ids = torch.arange(n, device=device)
        self.task = task
        self.envelope = BodyEnvelope(task.model, device)
        self.dt = float(task.dt)
        self.pelvis = int(task.pelvis_body)
        # Live state.
        self.next = torch.zeros(n, dtype=torch.long, device=device)
        self.free_at = torch.zeros(n, device=device)
        self.launched = torch.zeros((n, SLOTS), dtype=torch.bool, device=device)
        self.skipped = torch.zeros((n, SLOTS), dtype=torch.bool, device=device)
        self.t0 = torch.full((n, SLOTS), torch.inf, device=device)
        self.dur = torch.zeros((n, SLOTS), device=device)
        self.p0 = torch.zeros((n, SLOTS, 3), device=device)
        self.vel = torch.zeros((n, SLOTS, 3), device=device)
        self.slot_any = torch.zeros((n, SLOTS), dtype=torch.bool, device=device)
        self.slot_hand = torch.zeros((n, SLOTS), dtype=torch.bool, device=device)
        self.prev_hand = None
        self.prev_time = torch.zeros(n, device=device)

    # -- perception and physics both read the object through this single function --------------------------------
    def locate(self, ids, time):
        rel = time[..., None] - self.t0[ids]
        live = (rel >= 0.) & (rel <= self.dur[ids]) & self.launched[ids]
        slot = live.int().argmax(-1)
        rows = torch.arange(len(ids), device=self.device)
        center = self.p0[ids][rows, slot] + self.vel[ids][rows, slot] * rel[rows, slot, None]
        return center, live.any(-1) & self.enabled[ids]

    def active_slot(self, before, after):
        """Slot whose flight window overlaps the control step [before, after]; -1 when none."""
        live = self.launched & (self.t0 <= after[:, None]) & (self.t0 + self.dur >= before[:, None])
        return torch.where(live.any(-1), live.int().argmax(-1), -1)

    @torch.no_grad()
    def update(self, task):
        """Before each control step: track the hands, launch every throw that is due."""
        data = task.data
        time = data.time.to(self.t0.dtype)
        spheres = self.envelope.spheres(data, self.ids)
        hands = spheres[:, self.envelope.hand_mask]
        fresh = time > self.prev_time + 1e-6
        if self.prev_hand is None:
            velocity = torch.zeros_like(hands)
        else:
            velocity = torch.where(fresh[:, None, None], (hands - self.prev_hand) / (time - self.prev_time).clamp_min(1e-3)[:, None, None], 0.)
        self.prev_hand = hands.clone()
        self.prev_time = time.clone()
        k = self.next.clamp_max(SLOTS - 1)
        due_time = self.sched_t[self.ids, k]
        pending = self.enabled & (self.next < SLOTS)
        late = pending & (time > due_time + LATE)
        if bool(late.any()):
            self.skipped[late.nonzero().flatten(), self.next[late]] = True
            self.next[late] += 1
            self.free_at[late] = time[late] + PAUSE
            k = self.next.clamp_max(SLOTS - 1)
            due_time = self.sched_t[self.ids, k]
            pending = self.enabled & (self.next < SLOTS)
        due = pending & (time >= due_time) & (time >= self.free_at)
        ids = due.nonzero().flatten()
        if len(ids):
            self._launch(task, ids, spheres[ids], hands[ids], velocity[ids], time[ids])

    def _launch(self, task, ids, spheres, hands, velocity, time):
        data = task.data
        b = len(ids)
        rows = torch.arange(b, device=self.device)
        k = self.next[ids]
        speed = self.sched_speed[ids, k]
        hand = hands[rows, self.sched_side[ids, k]]
        hand_v = velocity[rows, self.sched_side[ids, k]]
        aim = hand + (hand_v * (APPROACH / speed)[:, None]).clamp(-.25, .25)
        frame = data.xmat[ids, self.pelvis].reshape(b, 3, 3)
        yaw = torch.atan2(frame[:, 1, 0], frame[:, 0, 0])
        local = torch.einsum('bji,bj->bi', frame, hand - data.xpos[ids, self.pelvis])
        outward = torch.where(local[:, 1] >= 0., 1., -1.)           # body-frame lateral sign of the target hand
        # Candidate directions from the hand towards the launch point: front or outer side, never from behind.
        az = outward[:, None] * (-math.radians(20.) + math.radians(180.) * self.sched_az[ids, k])
        el = math.radians(-20.) + math.radians(60.) * self.sched_el[ids, k]
        direction = torch.stack((el.cos() * (yaw[:, None] + az).cos(), el.cos() * (yaw[:, None] + az).sin(), el.sin()), -1)
        start = aim[:, None] + direction * APPROACH                       # [b,C,3]
        end = aim[:, None] - direction * OVERSHOOT
        fractions = torch.linspace(0., 1., 11, device=self.device)
        path = torch.lerp(start[:, :, None], end[:, :, None], fractions[None, None, :, None])   # [b,C,11,3]
        scenes = task.scene_ids[ids]
        sdf = task.bank.sample('sdf', path.reshape(b, -1, 3), scenes).reshape(b, CANDIDATES, len(fractions))
        ok = (path[..., 2].amin(-1) > FLOOR) & (sdf.amin(-1) > SCENE_CLEAR) & (sdf[..., 0] > START_CLEAR)
        # Clearance of the approach from the rest of the body, where the object is still outside the hand zone.
        others = ~self.envelope.hand_mask
        centers = spheres[:, others]
        radii = self.envelope.radii[others]
        gap = torch.linalg.vector_norm(path[:, :, :, None] - centers[:, None, None], dim=-1) - radii - BOX_DIAG
        far = torch.linalg.vector_norm(path - aim[:, None, None], dim=-1) > HAND_ZONE
        body = torch.where(far[..., None], gap, torch.inf).amin((-1, -2))
        ok &= body > BODY_CLEAR
        score = torch.where(ok, body, -torch.inf)
        choice = score.argmax(-1)
        good = ok.any(-1)
        if not bool(good.any()):
            return
        g = good.nonzero().flatten()
        e, c, kk = ids[g], choice[g], k[g]
        total = (APPROACH + OVERSHOOT)
        self.p0[e, kk] = start[g, c]
        self.vel[e, kk] = -direction[g, c] * speed[g, None]
        self.dur[e, kk] = total / speed[g]
        self.t0[e, kk] = time[g]
        self.launched[e, kk] = True
        self.free_at[e] = time[g] + total / speed[g] + PAUSE
        self.next[e] += 1

    def observe(self, hits):
        """After a control step: attribute object contacts (any region, hand region) to the live slot."""
        before = self.prev_time
        slot = self.active_slot(before, before + self.dt)
        have = slot >= 0
        if not bool(have.any()):
            return
        e = have.nonzero().flatten()
        self.slot_any[e, slot[e]] |= hits[e].any(-1)
        self.slot_hand[e, slot[e]] |= hits[e, 5]

    def per_episode(self):
        """Tier-resolved counts for every simulated episode (rows are episodes, columns the speed tiers)."""
        tier = torch.stack([(self.sched_speed - s).abs() < 1e-4 for s in SPEEDS], -1)    # [n, slot, tier]
        grid = lambda x: (x[:, :, None] & tier).any(1)
        return dict(launched=grid(self.launched), skipped=grid(self.skipped), hand=grid(self.slot_hand & self.launched),
                    any=grid(self.slot_any & self.launched))
