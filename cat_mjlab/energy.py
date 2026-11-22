"""G1 motor energy in joules and control-step mean power in watts."""

import torch

GROUPS = {"legs": slice(0, 12), "waist": slice(12, 15), "arms": slice(15, 29)}
SOURCES = ("policy", "applied")
MODELS = ("mech", "heat")
SHAPE = (len(SOURCES), len(MODELS), len(GROUPS))
PRICE_TERMS = ("reactive_power", "table_leg_power", "table_power", "global_power", "route_energy")
DEFAULT_ALPHA = 500.


def _group_sum(joints):
    return torch.stack([joints[..., group].sum(-1) for group in GROUPS.values()], -1)


def _heat(signed, holding):
    # Clamp net mechanical output once. Attribute it to groups with positive
    # net output in proportion to that output, so group telemetry sums to total.
    positive = signed.clamp_min(0.)
    net = signed.sum(-1, keepdim=True).clamp_min(0.)
    total = positive.sum(-1, keepdim=True)
    fraction = net / torch.where(total > 0., total, 1.)
    return holding + positive * fraction



def power_sample(policy_torque, applied_torque, joint_velocity, limits, weights, alpha, model):
    torque = torch.stack((policy_torque, applied_torque), 1)
    work = torch.nan_to_num(torque * joint_velocity[:, None], nan=0., posinf=0., neginf=0.)
    mech = _group_sum(work.abs())
    signed = _group_sum(work)
    holding = alpha * _group_sum(torch.nan_to_num(torque / limits, nan=0., posinf=0., neginf=0.).square())
    heat = _heat(signed, holding)
    telemetry = torch.stack((mech, heat), 2)
    if model is None:
        return telemetry, None
    if model == "heat":
        power = heat[:, 0]
        lower = _heat(signed[:, 0, :2], holding[:, 0, :2])
    elif model == "metab":
        policy = work[:, 0]
        power = _group_sum(policy.clamp_min(0.) / .25 + policy.clamp_max(0.) / -1.2)
        lower = power[:, :2]
    else:
        power = mech[:, 0]
        lower = power[:, :2]
    prices = torch.stack(((power * weights).sum(-1), (lower * weights[:2]).sum(-1)), -1)
    return telemetry, prices


class EnergyMeter:
    def __init__(self, num_envs, torque_limit, settings):
        self.sample = power_sample
        self.limits = torque_limit
        self.alpha = settings.get("alpha", DEFAULT_ALPHA)
        self.model = settings.get("model")
        self.weights = torque_limit.new_tensor([settings.get("leg_mult", 1.), 1., settings.get("arm_mult", 1.)])
        self.step_j = torque_limit.new_zeros((num_envs, *SHAPE))
        self.price_j = torque_limit.new_zeros((num_envs, 2))

    def begin_step(self):
        self.step_j.zero_()
        self.price_j.zero_()

    def record(self, policy_torque, applied_torque, joint_velocity, sim_dt):
        power, prices = self.sample(policy_torque, applied_torque, joint_velocity,
                                    self.limits, self.weights, self.alpha, self.model)
        sample_j = power * sim_dt
        self.step_j.add_(sample_j)
        if prices is not None:
            self.price_j.add_(prices, alpha=sim_dt)
        return sample_j

    def reward_power(self, applied_torque, joint_velocity, ctrl_dt):
        if self.model is not None:
            return (self.price_j / ctrl_dt).unbind(-1)
        # Preserve old variants bit for bit, including the full reduction order
        # and suppression of a nonfinite total.
        work = (applied_torque * joint_velocity).abs()
        return tuple(torch.nan_to_num(x.sum(-1), nan=0., posinf=0., neginf=0.) for x in (work, work[:, :15]))


def report_energy(energy_j, episodes, seconds):
    """JSON-ready group metrics for reductions over episodes or event windows."""
    return {
        source: {
            model: {
                group: {
                    "energy_j": round(float(energy_j[s, m, g]) / episodes, 4) if episodes else None,
                    "mean_w": round(float(energy_j[s, m, g]) / seconds, 4) if seconds else None,
                }
                for g, group in enumerate(GROUPS)
            }
            for m, model in enumerate(MODELS)
        }
        for s, source in enumerate(SOURCES)
    }
