#!/usr/bin/env python3
"""Batch D2 (code 9f497ab). Decided when the dep-1 lock is acquired: if cluster array 324637 (arbitration seeds 1-2 + E4
sweep) has started, only the I6 stall-triggered plan on the 384 scenes for seed 0 runs here; otherwise D2 also runs the
arbitration seeds 1-2 and the E4 sweep (and 324637 is then cancelled by hand while still pending)."""
import subprocess
S = {0: "/tmp/cat-energy-current-videos/inputs/COMBO_FIX_315808_u150.pt",
     1: "/tmp/cat-overnight/e4-inputs/COMBO_FIX_seed1_u150.pt", 2: "/tmp/cat-overnight/e4-inputs/COMBO_FIX_seed2_u150.pt"}
STALL = "--tracker-mix plan=1 --tracker-stall-plan 2 4"
ARB = "--tracker-mix plan=1 --plan-select value_safe --plan-arbitrate -3.719"
ARB0 = "--tracker-mix plan=1 --plan-select clear --plan-arbitrate 0"
lines = [f"stallplan_seed0|{S[0]}|{STALL}"]
try:
    st = subprocess.run(["ssh", "-o", "BatchMode=yes", "mbz-hpc", "PATH=/usr/local/bin:/usr/bin:/bin squeue -r -j 324637 -h -o %T"],
                        capture_output=True, text=True, timeout=60).stdout.split()
except Exception:  # noqa: BLE001
    st = ["UNKNOWN"]
if not st or all(x == "PENDING" for x in st):
    # GPU memory (48 GB): 3 full runs (~6.7 GB each) + 10 E4 runs; arb0 seeds 1-2 and E4 for arb0/I6 stay on the cluster
    lines += [f"arbvsafe_seed{k}|{S[k]}|{ARB}" for k in (1, 2)]
    for k in (0, 1, 2):
        lines += [f"E4:ctrl_k4_seed{k}|{S[k]}|--tracker-mix plan=1", f"E4:none_seed{k}|{S[k]}|--tracker-mix none=1",
                  f"E4:arbvsafe_seed{k}|{S[k]}|{ARB}"]
    lines.append(f"E4:ei1_seed0|{S[0]}|--tracker-mix plan=1 --planner-checkpoint /tmp/cat-improve/planner_ei1.pt")
print("\n".join(lines))
