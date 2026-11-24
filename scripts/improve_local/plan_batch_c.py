#!/usr/bin/env python3
"""Decide batch C from batch B's seed-0 results with the pre-registered rules; prints 'tag|ckpt|args' lines.

Pilot gate (docs/improve_overnight.md): mean clean gain vs the same-code control >= 1.5 pp over the two speeds and contact
change <= +0.5 pp at each speed. Arbitration threshold: 10th percentile of V(chosen plan) - V(no reference) in the
value_safe K=4 run (pooled over its process, read from the 0.3 file).
"""
import json
import sys
from pathlib import Path

R = Path("/home/konstantinsmirnov/robotics/cat-energy-local-runs")
SHA = "0772f2b"
S0 = "/tmp/cat-energy-current-videos/inputs/COMBO_FIX_315808_u150.pt"
SEEDS = {1: "/tmp/cat-overnight/e4-inputs/COMBO_FIX_seed1_u150.pt", 2: "/tmp/cat-overnight/e4-inputs/COMBO_FIX_seed2_u150.pt"}
EI = Path("/tmp/cat-improve/planner_ei1.pt")
ARGS = {"vsafe_k4": "--tracker-mix plan=1 --plan-select value_safe",
        "clear_k8": "--tracker-mix plan=1 --plan-samples 8",
        "vsafe_k8": "--tracker-mix plan=1 --plan-select value_safe --plan-samples 8"}


def summary(tag, speed):
    d = json.loads((R / f"improve_{tag}_seed0_{SHA}" / f"speed_{speed}.json").read_text())
    assert d["finished"] == d["target"]
    return d["summary"]


def main():
    out, notes = [], {}
    ctrl = {s: summary("ctrl_k4", s) for s in ("0.15", "0.3")}
    passing = []
    for tag in ARGS:
        try:
            v = {s: summary(tag, s) for s in ("0.15", "0.3")}
        except Exception as e:  # noqa: BLE001
            notes[tag] = f"missing: {e}"
            continue
        dc = [100 * (v[s]["clean"] - ctrl[s]["clean"]) for s in v]
        dk = [100 * (v[s]["contact"] - ctrl[s]["contact"]) for s in v]
        ok = sum(dc) / 2 >= 1.5 and max(dk) <= .5
        notes[tag] = dict(clean_pp=dc, contact_pp=dk, passes=ok)
        if ok:
            passing.append((sum(dc) / 2, tag))
    q = summary("vsafe_k4", "0.3")["plan_value"]["gain_quantiles"]
    delta = q[1]
    notes["arbitrate_delta"] = delta
    out.append(f"arb_vsafe_k4_seed0|{S0}|{ARGS['vsafe_k4']} --plan-arbitrate {delta}")
    out.append(f"arb_clear_k4_seed0|{S0}|--tracker-mix plan=1 --plan-select clear --plan-arbitrate {delta}")
    if passing:
        best = max(passing)[1]
        notes["best"] = best
        for seed, ck in SEEDS.items():
            out.append(f"{best}_seed{seed}|{ck}|{ARGS[best]}")
    # held-out E4 layouts (seed 0): the same-code control and the best variant
    out.append(f"E4:ctrl_k4_seed0|{S0}|--tracker-mix plan=1")
    if passing:
        out.append(f"E4:{best}_seed0|{S0}|{ARGS[best]}")
    if EI.exists():
        for seed, ck in SEEDS.items():
            out.append(f"ei1_seed{seed}|{ck}|--tracker-mix plan=1 --planner-checkpoint {EI}")
    print(json.dumps(notes), file=sys.stderr)
    print("\n".join(out))


if __name__ == "__main__":
    main()
