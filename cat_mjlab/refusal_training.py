"""Progress and reward rules for the optional traversal refusal fine-tune."""
from __future__ import annotations

import torch

CLEARANCE_TERMS = (
    "handsdf", "headdf", "feetdf", "kneesdf", "shldsdf",
    "wholebody_hand_clearance", "wholebody_arm_clearance",
)
STALL_BONUSES = ("upright", "stand_tall", "track_pos", "track_heading", "track_height", "track_upper")
PASSAGE_KINDS = {"side-gap", "table-edge-passage", "contrastive-hand-passage", "width-curriculum-scaffold"}

SETTINGS = dict(progress_m=.05, gate_s=2., timeout_s=8., clearance_reference_speed_mps=.30)


def advance_clock(progress, seconds, moving, advance_m, dt, marker_m):
    """Count consecutive moving-command steps without a net advance marker."""
    progress = torch.where(moving, progress + advance_m, 0.)
    advanced = progress >= marker_m
    return torch.where(advanced, 0., progress), torch.where(moving & ~advanced, seconds + dt, 0.)


def apply_terms(terms, passage, stalled, distance, dt, reference_speed):
    """Integrate negative passage clearance by distance and gate stall bonuses."""
    scale = torch.where(passage, distance / (dt * reference_speed), 1.)
    for name in CLEARANCE_TERMS:
        if name in terms:
            term = terms[name]
            terms[name] = torch.where(term < 0, term * scale, term)
    for name in STALL_BONUSES:
        if name in terms:
            term = terms[name]
            terms[name] = torch.where(stalled & (term > 0), 0., term)


def refusal_failure(moving, clock, timeout, terminated, obstacle, fall, reached, limit_s):
    """A prolonged no-progress or unfinished traversal times out without bootstrapping."""
    return moving & ((clock >= limit_s) | timeout) & ~terminated & ~obstacle & ~fall & ~reached
