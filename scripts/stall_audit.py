"""Windowed terms at the first confirmed route stall in a refused traversal."""
from __future__ import annotations

import torch

TERMS = (
    'stall', 'goal_progress', 'global_power', 'table_power', 'track_pos', 'track_lead',
    'track_upper', 'track_heading', 'track_height', 'handsdf', 'wholebody_hand_clearance',
    'wholebody_arm_clearance', 'headdf', 'feetdf', 'handsgf', 'feetgf', 'headgf',
    'tracking_root_field', 'tracking_orientation', 'upright', 'stand_tall', 'torso_rate',
    'wholebody_upper_clear_posture', 'body_motion', 'body_rotation',
)


class StallAudit:
    def __init__(self, worlds, scenes, device, dt):
        self.before = round(1. / dt)
        self.during = round(2. / dt)
        self.capacity = self.before + self.during
        self.ring = torch.zeros(self.capacity, worlds, len(TERMS), device=device)
        self.age = torch.zeros(worlds, dtype=torch.long, device=device)
        self.streak = torch.zeros_like(self.age)
        self.progress = torch.zeros(worlds, device=device)
        self.seen = torch.zeros(worlds, 2, dtype=torch.bool, device=device)
        self.windows = self.ring.new_zeros((worlds, 2, len(TERMS)))
        self.sum = self.ring.new_zeros((scenes, 2, len(TERMS)))
        self.count = torch.zeros(scenes, 2, device=device)
        self.clock = 0

    def observe(self, metrics, ids, done, refused):
        terms = torch.stack([metrics.get('reward/' + key, torch.zeros_like(self.progress)) for key in TERMS], -1)
        self.ring[self.clock] = terms
        self.age += 1
        moving = metrics['track_route_moving'] & (ids >= 0)
        self.progress = torch.where(moving, self.progress + metrics['track_route_advance_m'], 0.)
        reached = self.progress >= .05
        self.streak = torch.where(moving & ~reached, self.streak + 1, 0.)
        self.progress = torch.where(reached, 0., self.progress)
        fires = (self.streak == self.during) & ~self.seen[:, 1]
        if bool(fires.any()):
            order = (torch.arange(self.capacity, device=ids.device) + self.clock + 1) % self.capacity
            history = self.ring[:, fires][order]
            self.windows[fires, 0] = history[:self.before].mean(0)
            self.windows[fires, 1] = history[self.before:].mean(0)
            self.seen[fires, 0] = self.age[fires] >= self.capacity
            self.seen[fires, 1] = True
        if bool(refused.any()):
            seen = self.seen[refused]
            self.sum.index_add_(0, ids[refused], self.windows[refused] * seen[..., None])
            self.count.index_add_(0, ids[refused], seen.float())
        self.seen[done] = False
        self.age[done] = 0
        self.streak[done] = 0
        self.progress[done] = 0.
        self.clock = (self.clock + 1) % self.capacity

    def summary(self, scene):
        count = self.count[scene].sum(0)
        values = self.sum[scene].sum(0) / count.clamp_min(1.)[:, None]
        return {when: {'episodes': int(count[i]), 'terms': dict(zip(TERMS, values[i].tolist())) if count[i] else {}}
                for i, when in enumerate(('before', 'during'))}
