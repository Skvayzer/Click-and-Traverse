#!/usr/bin/env python3
"""Batch D (code 9f497ab): held-out E4 runs with --subset-fields for the control, the no-reference mode and the stall-
triggered plan (seeds 0-2), the expert-iteration planner (seed 0), and the 384-scene stall-triggered plan on seed 0."""
from pathlib import Path
S = {0: "/tmp/cat-energy-current-videos/inputs/COMBO_FIX_315808_u150.pt",
     1: "/tmp/cat-overnight/e4-inputs/COMBO_FIX_seed1_u150.pt", 2: "/tmp/cat-overnight/e4-inputs/COMBO_FIX_seed2_u150.pt"}
EI = Path("/tmp/cat-improve/planner_ei1.pt")
STALL = "--tracker-mix plan=1 --tracker-stall-plan 2 4"
lines = [f"stallplan_seed0|{S[0]}|{STALL}"]
for k in (0, 1, 2):
    lines += [f"E4:ctrl_k4_seed{k}|{S[k]}|--tracker-mix plan=1", f"E4:none_seed{k}|{S[k]}|--tracker-mix none=1",
              f"E4:stallplan_seed{k}|{S[k]}|{STALL}"]
if EI.exists():
    lines.append(f"E4:ei1_seed0|{S[0]}|--tracker-mix plan=1 --planner-checkpoint {EI}")
print("\n".join(lines))
