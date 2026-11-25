#!/usr/bin/env python3
"""Batch D: the held-out E4 runs that ran out of GPU memory in batch C (now with --subset-fields), plus the no-reference
mode on E4 for seeds 1-2 (paired with the existing E4 planner seed 1/2 evaluations)."""
from pathlib import Path
S = {0: "/tmp/cat-energy-current-videos/inputs/COMBO_FIX_315808_u150.pt",
     1: "/tmp/cat-overnight/e4-inputs/COMBO_FIX_seed1_u150.pt", 2: "/tmp/cat-overnight/e4-inputs/COMBO_FIX_seed2_u150.pt"}
EI = Path("/tmp/cat-improve/planner_ei1.pt")
lines = [f"E4:ctrl_k4_seed0|{S[0]}|--tracker-mix plan=1"]
lines += [f"E4:none_seed{k}|{S[k]}|--tracker-mix none=1" for k in (0, 1, 2)]
lines += [f"E4:ctrl_k4_seed{k}|{S[k]}|--tracker-mix plan=1" for k in (1, 2)]
if EI.exists():
    lines.append(f"E4:ei1_seed0|{S[0]}|--tracker-mix plan=1 --planner-checkpoint {EI}")
print("\n".join(lines))
