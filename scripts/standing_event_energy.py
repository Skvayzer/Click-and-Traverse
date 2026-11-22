"""Substep measurements for completed object approaches in the standing benchmark."""

import numpy as np
import torch

from cat_mjlab.energy import GROUPS, MODELS, SHAPE, SOURCES, report_energy

CLASSES = ("failure", "arm_first", "body_first", "still_success")
SIDES = ("left", "right")


def classify(retreat, feet, pelvis, stepped, success):
    body_first = stepped | (pelvis >= .10)
    arm_first = (retreat >= .10) & (feet < .05) & (pelvis < .05)
    return torch.where(success, torch.where(body_first, 2, torch.where(arm_first, 1, 3)), 0)


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

    def abort(self, failed, buckets, hands):
        ids = (self.open & failed).nonzero().flatten()
        if len(ids):
            self.finish(ids, buckets[ids], hands[ids], torch.zeros_like(ids, dtype=torch.bool))

    def finish(self, ids, bucket, hand, success):
        retreat, feet, pelvis = self.peaks[ids].unbind(1)
        # A step requires both foot lift and horizontal travel on the same
        # foot. Displacement alone still counts against the arm-first limit.
        stepped = ((self.foot_xy[ids] >= .05) & (self.foot_lift[ids] >= .02)).any(-1)
        self.rows.append(torch.stack((ids, bucket, classify(retreat, feet, pelvis, stepped, success),
                                      hand, self.steps[ids],
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
            seconds = float(values[selected, 4].sum()) * self.sim_dt
            summary[name] = {"count": count, "energy": report_energy(energies[selected].sum(0), count, seconds)}
        details = []
        for (world, bucket, code, hand, steps, retreat, feet, pelvis, stepped), energy in zip(values.tolist(), energies):
            seconds = steps * self.sim_dt
            details.append(dict(row=rows[int(world)], bucket=buckets[int(bucket)], response=CLASSES[int(code)], hand_side=SIDES[int(hand)],
                                seconds=round(seconds, 4), retreat_m=round(retreat, 4), feet_m=round(feet, 4),
                                pelvis_m=round(pelvis, 4), stepped=bool(stepped), energy=report_energy(energy, 1, seconds)))
        return summary, details



def matched_extra(events, *, draws=2000, seed=0):
    """Match still-success W by bucket and hand, subtract over each event's duration.

    Standardize both response classes to the same bucket/side mix on common support.
    Bootstrap whole evaluation rows to keep repeated approaches together.
    """
    categories = ("arm_first", "body_first", "still_success")
    names = sorted({(e["bucket"], e["hand_side"]) for e in events})
    row_ids = {row: i for i, row in enumerate(sorted({e["row"] for e in events}))}
    strata = {name: i for i, name in enumerate(names)}
    values = np.zeros((len(row_ids), len(strata), 3, 2 + int(np.prod(SHAPE))))
    for event in events:
        response, duration = event["response"], event["seconds"]
        if response not in categories or duration <= 0.:
            continue
        cell = values[row_ids[event["row"]], strata[event["bucket"], event["hand_side"]], categories.index(response)]
        cell[:2] += (1., duration)
        cell[2:] += [event["energy"][s][m][g]["energy_j"] for s in SOURCES for m in MODELS for g in GROUPS]
    counts = values.sum(0)[...,0]
    supported = np.all(counts > 0., axis=1)
    support = [{"bucket": bucket, "hand_side": side,
                **dict(zip(categories, counts[i].astype(int).tolist()))}
               for i, (bucket, side) in enumerate(names) if supported[i]]
    result = dict(k_star=None, ci95=None, valid_bootstrap_fraction=0., common_support=support)
    if not support:
        return result
    data = values[:, supported]
    mix = counts[supported, :2].sum(1)
    mix /= mix.sum()

    def extra(totals):
        count, duration = totals[..., 0], totals[..., 1]
        energy = totals[..., 2:]
        with np.errstate(invalid="ignore", divide="ignore"):
            baseline_w = energy[..., 2, :] / duration[..., 2, None]
            excess = energy[..., :2, :] - duration[..., :2, None] * baseline_w[..., None, :]
            mean_j = excess / count[..., :2, None]
            mean_w = excess / duration[..., :2, None]
        return (np.einsum("s,...scr->...cr", mix, x).reshape(*x.shape[:-3], 2, *SHAPE) for x in (mean_j, mean_w))

    mean_j, mean_w = extra(data.sum(0))
    result["extra"] = {response: {s: {m: {g: dict(energy_j=float(mean_j[c, i, j, k]),
                                                 mean_w=float(mean_w[c, i, j, k]))
                                          for k, g in enumerate(GROUPS)} for j, m in enumerate(MODELS)}
                                  for i, s in enumerate(SOURCES)}
                       for c, response in enumerate(categories[:2])}
    arm, body = mean_j[:,0,1].sum(-1)
    result.update(arm_extra_j=float(arm), body_extra_j=float(body))
    if arm <= 0. or body <= 0.:
        return result
    result["k_star"] = float(body / arm)
    rng = np.random.default_rng(seed)
    n = len(row_ids)
    flat = data.reshape(n, -1)
    ratios = []
    for begin in range(0, draws, 128):
        weights = rng.multinomial(n, np.full(n, 1./n), size=min(128,draws-begin))
        sample, _ = extra((weights @ flat).reshape(-1,*data.shape[1:]))
        a, b = sample[:,:,0,1].sum(-1).T
        valid = np.isfinite(a) & np.isfinite(b) & (a > 0.) & (b > 0.)
        ratios.extend((b[valid] / a[valid]).tolist())
    result["valid_bootstrap_fraction"] = len(ratios) / draws
    if len(ratios) >= .95 * draws:
        result["ci95"] = np.quantile(ratios,[.025,.975]).tolist()
    return result
