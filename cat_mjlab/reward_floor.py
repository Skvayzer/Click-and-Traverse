"""Opt-in, zone-local monotone replacement for the shaped reward's lower clamp."""
import torch


def hand_reward_floor(pre, context, width=0.):
    """Width is in reward/control-step units, after dt and before collision events.

    Below width, f(x)=width**2/(2*width-x); above it, f(x)=x.
    This C1 floor has an algebraic (not exponentially vanishing) negative tail.
    Blend with the legacy floor over the existing hand-objective phase ramp.
    No lift for flat/retention/narrow, inactive zones, or the default width=0.
    """
    legacy = pre.clamp(0, 10000)
    if width == 0.:
        return legacy, pre < 0, torch.zeros_like(pre), ((pre >= 0) & (pre < 10000)).to(pre.dtype)
    phase = context.get('reward_phase_weight', context['phase_weight'])
    enabled = context.get('reward_enabled', context['enabled'])
    active = context.get('reward_hand_active', context['hand_active']).any(-1)
    valid = context.get('reward_region_valid', context['region_valid']).any(-1)
    role = context['role']
    blend = torch.where(enabled & active & valid & ((role == 1) | (role == 3)), phase, 0.)
    # Clamp the unused denominator too, so autograd never encounters a pole.
    denominator = (2 * width - pre).clamp_min(width)
    soft = torch.where(pre < width, width**2 / denominator, pre).clamp_max(10000)
    result = legacy + blend * (soft - legacy)
    slope = torch.where(pre < width, width**2 / denominator.square(), 1.)
    slope = (1-blend)*(pre >= 0).to(pre.dtype) + blend*slope
    slope = torch.where(pre >= 10000, 0., slope)
    return result, (pre < 0) & (blend == 0), result-legacy, slope
