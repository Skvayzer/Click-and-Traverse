#!/usr/bin/env python3
"""E5 / E3 plan-blind ladder table with paired scene-bootstrap intervals (A4 collection)."""
import glob
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_stats as S

CL = Path("/tmp/cat-eval-overnight/data/cluster"); L = Path("/home/konstantinsmirnov/robotics/cat-energy-local-runs")
protos = ["static_0.15", "static_0.3", "moving_0.15", "moving_0.3"]
def g(pattern):
    return sorted(glob.glob(str(pattern)))
fix = [L / "PlanBlind_COMBO_FIX_seed0"] + [Path(p) for p in g(CL / "energy_round5_blind_COMBO_FIX_seed*_319349")]
off = [Path(p) for p in g(CL / "energy_round5_blind_COMBO_NOREFLEX_seed*_319349")]
pol = {"Planner–reflex (COMBO_FIX, 3 seeds)": fix, "Reflex off (COMBO_NOREFLEX, 3 seeds)": off,
       "Plan only, fields masked (3 seeds)": [Path(p) for p in g(CL / "overnight_E5_plan_only_seed*_319436")],
       "Reactive V7r": [Path(p) for p in g(CL / "overnight_E5_reactive_seed0_320668")],
       "Priced COMBO1b u500": [Path(p) for p in g(CL / "overnight_E5_price_reference_seed0_320669")],
       "Price-free COMBO_NOPRICE u50": [Path(p) for p in g(CL / "overnight_E5_no_price_seed0_320670")]}
KEYS = [("clean", "clean"), ("contact", "any contact"), ("hands", "any hand contact"), ("inserted_hand_contact", "inserted-object hand contact"), ("inserted_contact", "inserted-object any contact")]
print("| Protocol | Policy | " + " | ".join(k[1] for k in KEYS) + " |\n| --- | --- | " + " | ".join("---:" for _ in KEYS) + " |")
delta = []
for pr in protos:
    data = {}
    for n, dirs in pol.items():
        fs = [d / f"{pr}.json" for d in dirs if (d / f"{pr}.json").exists()]
        if fs:
            data[n] = [S.load(f) for f in fs]
    if not data:
        continue
    sc = S.common_scenes(*data.values())
    for n, ds in data.items():
        print(f"| {pr} | {n} | " + " | ".join(S.fmt(*S.ci_mean(S.matrix(ds, k, sc))) for k, _ in KEYS) + " |")
    base = "Planner–reflex (COMBO_FIX, 3 seeds)"
    for n, ds in data.items():
        if n != base and base in data:
            delta.append(f"| {pr} | {base} − {n} | " + " | ".join(S.fmt(r["mean"], r["lo"], r["hi"], signed=True) for r in (S.ci_delta(S.matrix(data[base], k, sc), S.matrix(ds, k, sc)) for k, _ in KEYS)) + " |")
print("\nPaired differences, planner–reflex minus the other policy (percentage points; 95% CI over scenes):\n")
print("| Protocol | Comparison | " + " | ".join(k[1] for k in KEYS) + " |\n| --- | --- | " + " | ".join("---:" for _ in KEYS) + " |")
print("\n".join(delta))
