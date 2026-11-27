#!/usr/bin/env python3
"""B3 fairness comparison (docs/cos_blinkmimic.md): BlinkMimic tracker vs the C4 arms on the COS test-200.
  python scripts/cos_bm_b3_compare.py --b3 B3_DIR --control B3_CONTROL_DIR --c4 /l/.../cos/c4 --out table.json
"""
import argparse, json
from pathlib import Path
import numpy as np


def boot(d, n=4000, seed=0):
    rng = np.random.default_rng(seed); m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--b3", required=True); ap.add_argument("--control", required=True)
    ap.add_argument("--c4", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c4 = Path(a.c4)
    bank = {r["source"]["cos"]["scene"]: r["scene_id"] for r in json.load(open(c4 / "bank_test_packed/addition.json"))["scenes"]}
    B = {r["scene"]: r for r in json.load(open(Path(a.b3) / "per_scene_clearance.json"))}
    Cn = {r["scene"]: r for r in json.load(open(Path(a.control) / "per_scene_clearance.json"))}
    arms = {"plan_off": "fix_none_cos", "c3_oracle": "c3s0_oracle_cos", "v7r": "v7r_cos", "c3_plan": "c3s0_planC2_cos"}
    A = {}
    for k, d in arms.items():
        sp = {}
        for f in sorted((c4 / "evals" / d).glob("speed_*.json")):
            j = json.load(open(f)); sp[str(j["operator_speed"])] = j["per_scene"]
        A[k] = sp
    scenes = sorted(B)
    sub = [s for s in scenes if B[s]["reference_goal_c4"]]
    def arm_vals(k, key, ss, speed=None):
        sps = [speed] if speed else sorted(A[k])
        return np.array([np.mean([A[k][p][bank[s]][key] for p in sps]) for s in ss])
    out = dict(n_all=len(scenes), n_subset=len(sub))
    for name, ss in (("a_subset", sub), ("b_all", scenes)):
        t = np.array([float(B[s]["clean_c4"]) for s in ss])
        res = {"tracker_clean_c4": round(float(t.mean()), 3)}
        for k in arms:
            v = arm_vals(k, "clean", ss)
            res[f"{k}_clean"] = round(float(v.mean()), 3)
            res[f"{k}_clean_by_speed"] = {p: round(float(arm_vals(k, "clean", ss, p).mean()), 3) for p in sorted(A[k])}
            res[f"tracker_minus_{k}"] = dict(mean=round(float((t - v).mean()), 3), ci95=boot(t - v))
        out[name] = res
    c = {"tracker_contact": round(float(np.mean([B[s]["contact"] for s in scenes])), 3),
         "tracker_hand_contact": round(float(np.mean([B[s]["hand_contact"] for s in scenes])), 3),
         "tracker_fall": round(float(np.mean([B[s]["failed"] for s in scenes])), 3)}
    for k in arms:
        c[f"{k}_contact"] = round(float(arm_vals(k, "contact", scenes).mean()), 3)
        c[f"{k}_hand_contact"] = round(float(arm_vals(k, "hands", scenes).mean()), 3)
        c[f"{k}_fall"] = round(float(arm_vals(k, "fall", scenes).mean()), 3)
        c[f"{k}_refused"] = round(float(arm_vals(k, "refused", scenes).mean()), 3)
    out["c_contacts_all"] = c
    def route(R):
        return np.array([float((not R[s]["contact"]) and (not R[s]["failed"]) and R[s]["completed"] and R[s]["end_error_m"] <= 0.30) for s in scenes])
    rt, rc = route(B), route(Cn)
    out["d_route_tracker_only"] = dict(candidate=round(float(rt.mean()), 3), control_no_obstacles=round(float(rc.mean()), 3),
                                       candidate_goal_c4=round(float(np.mean([B[s]["goal_c4"] for s in scenes])), 3),
                                       control_goal_c4=round(float(np.mean([Cn[s]["goal_c4"] for s in scenes])), 3),
                                       control_fall=round(float(np.mean([Cn[s]["failed"] for s in scenes])), 3),
                                       note="not comparable to C4 numbers")
    fams = sorted({B[s]["family"] for s in scenes})
    out["a_subset_by_family"] = {f: dict(n=len([s for s in sub if B[s]["family"] == f]),
                                         tracker=round(float(np.mean([B[s]["clean_c4"] for s in sub if B[s]["family"] == f] or [np.nan])), 3),
                                         plan_off=round(float(arm_vals("plan_off", "clean", [s for s in sub if B[s]["family"] == f]).mean()), 3) if any(B[s]["family"] == f for s in sub) else None)
                                 for f in fams}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
