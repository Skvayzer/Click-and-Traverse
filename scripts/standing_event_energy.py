"""Substep measurements for completed object approaches in the standing benchmark."""

import torch

from cat_mjlab.energy import SHAPE, report_energy

CLASSES = ("failure", "arm_first", "body_first")


def classify(retreat, feet, pelvis, stepped):
    body_first = stepped | (pelvis >= .10)
    arm_first = (retreat >= .10) & (feet < .05) & (pelvis < .05)
    return torch.where(body_first, 2, torch.where(arm_first, 1, 0))


class StandingEventEnergy:
    def __init__(self, task):
        sites = [task.torso_site, *(task.model.site(name).id for name in task.site_names[3:5])]
        self.bodies = torch.as_tensor(task.model.site_bodyid[sites], device=task.device, dtype=torch.long)
        self.offsets = task.data.qpos.new_tensor(task.model.site_pos[sites])
        self.sim_dt = task.sim_dt
        n = task.num_envs
        self.ids = torch.arange(n, device=task.device)
        self.open = torch.zeros(n, dtype=torch.bool, device=task.device)
        self.steps = torch.zeros_like(self.ids)
        self.energy = task.data.qpos.new_zeros((n, *SHAPE))
        self.origin = task.data.qpos.new_zeros((n, 4, 3))
        self.ray = task.data.qpos.new_zeros((n, 3))
        self.peaks = task.data.qpos.new_zeros((n, 3))
        self.foot_xy = task.data.qpos.new_zeros((n, 2))
        self.foot_lift = torch.zeros_like(self.foot_xy)
        self.rows = []
        self.energies = []

    def observe(self, data, ids, clock, target, ray, hands, root):
        # Pre-integration poses at the start of each physics substep. The
        # event starts when the object leaves its authored pause.
        hands = hands[self.ids[:len(ids)], target]
        index = (ids[:, None], self.bodies)
        rotations = data.xmat.reshape(len(self.ids), -1, 3, 3)[index]
        sites = data.xpos[index] + (rotations @ self.offsets[..., None]).squeeze(-1)
        rotation = rotations[:, 0].transpose(-1, -2)
        hand = (rotation @ (hands - sites[:, 0])[..., None]).squeeze(-1)
        positions = torch.cat((hand[:, None], root[:, None], sites[:, 1:]), 1)
        valid = clock >= 0.
        start = valid & ~self.open[ids]
        self.origin[ids] = torch.where(start[:, None, None], positions, self.origin[ids])
        local_ray = (rotation @ ray[..., None]).squeeze(-1)
        self.ray[ids] = torch.where(start[:, None], local_ray, self.ray[ids])
        self.open[ids] = valid
        delta = positions - self.origin[ids]
        foot_delta = delta[:, 2:]
        peaks = torch.stack(((delta[:, 0] * self.ray[ids]).sum(-1),
                             torch.linalg.vector_norm(foot_delta, dim=-1).amax(-1),
                             torch.linalg.vector_norm(delta[:, 1], dim=-1)), 1)
        self.peaks[ids] = torch.where(valid[:, None], torch.maximum(self.peaks[ids], peaks), 0.)
        self.foot_xy[ids] = torch.where(valid[:, None], torch.maximum(
            self.foot_xy[ids], torch.linalg.vector_norm(foot_delta[..., :2], dim=-1)), 0.)
        self.foot_lift[ids] = torch.where(valid[:, None], torch.maximum(self.foot_lift[ids], foot_delta[..., 2]), 0.)

    def record(self, sample_j):
        self.energy.add_(torch.where(self.open[:, None, None, None], sample_j, 0.))
        self.steps.add_(self.open)

    def finish(self, ids, bucket):
        retreat, feet, pelvis = self.peaks[ids].unbind(1)
        # A step requires both foot lift and horizontal travel on the same
        # foot. Displacement alone still counts against the arm-first limit.
        stepped = ((self.foot_xy[ids] >= .05) & (self.foot_lift[ids] >= .02)).any(-1)
        self.rows.append(torch.stack((ids, bucket, classify(retreat, feet, pelvis, stepped), self.steps[ids],
                                      retreat, feet, pelvis, stepped), -1))
        self.energies.append(self.energy[ids])
        self.open[ids] = False
        self.steps[ids] = 0
        self.energy[ids] = 0.
        self.peaks[ids] = 0.
        self.foot_xy[ids] = 0.
        self.foot_lift[ids] = 0.

    def reset(self, done):
        self.open &= ~done
        self.steps.masked_fill_(done, 0)
        self.energy.masked_fill_(done[:, None, None, None], 0.)
        self.peaks.masked_fill_(done[:, None], 0.)
        self.foot_xy.masked_fill_(done[:, None], 0.)
        self.foot_lift.masked_fill_(done[:, None], 0.)

    def summary(self, rows, buckets):
        if not self.rows:
            return {k: {"count": 0} for k in CLASSES}, []
        values = torch.cat(self.rows).cpu()
        energies = torch.cat(self.energies).cpu()
        summary = {}
        for code, name in enumerate(CLASSES):
            selected = values[:, 2] == code
            count = int(selected.sum())
            seconds = float(values[selected, 3].sum()) * self.sim_dt
            summary[name] = {"count": count, "energy": report_energy(energies[selected].sum(0), count, seconds)}
        details = []
        for (world, bucket, code, steps, retreat, feet, pelvis, stepped), energy in zip(values.tolist(), energies):
            seconds = steps * self.sim_dt
            details.append(dict(row=rows[int(world)], bucket=buckets[int(bucket)], response=CLASSES[int(code)],
                                seconds=round(seconds, 4), retreat_m=round(retreat, 4), feet_m=round(feet, 4),
                                pelvis_m=round(pelvis, 4), stepped=bool(stepped), energy=report_energy(energy, 1, seconds)))
        return summary, details
