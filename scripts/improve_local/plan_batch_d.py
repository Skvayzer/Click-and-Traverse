#!/usr/bin/env python3
"""Batch D2 (code 9f497ab), trimmed at 08:30: the E4 sweep moved to cluster array 324637; only the I6 stall-triggered plan
on the 384 scenes for seed 0 remains local."""
S0 = "/tmp/cat-energy-current-videos/inputs/COMBO_FIX_315808_u150.pt"
print(f"stallplan_seed0|{S0}|--tracker-mix plan=1 --tracker-stall-plan 2 4")
