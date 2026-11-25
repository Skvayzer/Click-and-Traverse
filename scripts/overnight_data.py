#!/usr/bin/env python3
"""Assemble every number behind the overnight figures and the website into one JSON (results.json).

Run again whenever new result files arrive. Inputs are the local mirrors of the per-scene evaluation JSONs
(see PATHS below). Intervals are the paired scene bootstrap of scripts/eval_stats.py.

    python scripts/overnight_data.py --out /path/results.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_stats as S  # noqa: E402

CL = Path(os.environ.get("EVAL_CLUSTER_MIRROR", "/tmp/cat-eval-overnight/data/cluster"))
LOC = Path("/home/konstantinsmirnov/robotics/cat-energy-local-runs")
R3 = Path("/tmp/cat-energy-round3-results")
R4 = Path("/tmp/cat-overnight/results")
EVRUNS = Path("/tmp/cat-eval-overnight/runs")
RATES = ("clean", "contact", "refused", "hands")
GROUPS = {"Side gaps": ("sidegap",), "Cabinets": ("contrast",), "Furniture rooms": ("random-furniture",),
          "Clutter rooms": ("random-generic",), "CAT courses": ("D8", "procedural", "published")}


def exists(*p):
    return all(Path(x).exists() for x in p)


def narrow_paths(speed):
    sp = f"speed_{speed}.json"
    out = {}
    if speed == "0.3":
        out["Released CAT"] = [CL / "speed_sweep/CAT_released/speed_0.3.json"]
    out["Reactive (V7r)"] = [Path("/tmp/cat-improve/baselines/reactive_v7r_314948") / sp]
    out["Planner–reflex"] = [R3 / "energy_round3_COMBO_FIX_315810" / sp, R4 / "energy_round4_COMBO_FIX_seed1_318295" / sp,
                             R4 / "energy_round4_COMBO_FIX_seed2_318295" / sp]
    out["Reflex off"] = [R3 / "energy_round3_COMBO_NOREFLEX_315811" / sp, R4 / "energy_round4_COMBO_NOREFLEX_seed1_318295" / sp,
                         R4 / "energy_round4_COMBO_NOREFLEX_seed2_318295" / sp]
    brace = CL / "overnight_E6_BRACE_384_320664" / sp
    out["BRACE + planner"] = [brace]
    for p in glob.glob(str(CL / "sonic_G2_384_*" / sp)):
        out["SONIC (plain)"] = [Path(p)]
    for p in glob.glob(str(CL / "sonic_latent_eval_u600_384_*" / sp)):
        out["SONIC + adaptor"] = [Path(p)]
    return {k: v for k, v in out.items() if exists(*v)}


def e4_paths(speed):
    sp = f"speed_{speed}.json"
    L = lambda n: LOC / n / sp
    out = {
        "Released CAT": [L("overnight_E4_released_cat_seed0_local_ac58626")],
        "Reactive (V7r)": [L("overnight_E4_reactive_seed0_local_ac58626")],
        "BRACE + planner": [L("overnight_E6_BRACE_E4_local_ac58626")],
        "Planner–reflex": [L("overnight_E4_planner_seed0_local_ac58626"), CL / "overnight_E4_planner_seed1_319447" / sp,
                           L("overnight_E4_planner_seed2_local_ac58626")],
        "Reflex off": [L("overnight_E4_field_no_reflex_seed0_local_ac58626"), CL / "overnight_E4_field_no_reflex_seed1_319448" / sp,
                       CL / "overnight_E4_field_no_reflex_seed2_319448" / sp],
        "Plan only (fields masked)": [L("overnight_E4_plan_only_seed0_local_ac58626"), L("overnight_E4_plan_only_seed1_local_ac58626"),
                                      L("overnight_E4_plan_only_seed2_local_ac58626")],
    }
    off = sorted(glob.glob(str(CL / "overnight_A3_planner_noplan_seed*" / sp)) + glob.glob(str(EVRUNS / "A3_noplan_seed*_local" / sp)))
    if off:
        out["Planner–reflex, plan off"] = [Path(p) for p in off]
    for p in glob.glob(str(CL / "sonic_G2_E4_*" / sp)):
        out["SONIC (plain)"] = [Path(p)]
    for p in glob.glob(str(CL / "sonic_latent_eval_u600_E4_*" / sp)):
        out["SONIC + adaptor"] = [Path(p)]
    return {k: v for k, v in out.items() if exists(*v)}


def group_of(scene):
    for g, prefixes in GROUPS.items():
        if scene.startswith(prefixes):
            return g
    return None


def summarize(paths, scenes, mask=None):
    """metrics for one policy (its seed files) on `scenes`; mask selects a subset of scenes."""
    ds = [S.load(p) for p in paths]
    ids = [s for s in scenes if (mask is None or mask(s))]
    res = {}
    for k in RATES:
        A = S.matrix(ds, k, ids)
        res[k] = list(S.ci_mean(A))
    return res, ds, ids


def build_benchmark(path_fn, speeds, family_of=None, groups=None):
    out = {}
    for speed in speeds:
        P = path_fn(speed)
        if not P:
            continue
        datasets = {k: [S.load(p) for p in v] for k, v in P.items()}
        scenes = S.common_scenes(*datasets.values())
        entry = dict(n_scenes=len(scenes), policies={})
        for name, ds in datasets.items():
            m = {k: list(S.ci_mean(S.matrix(ds, k, scenes))) for k in RATES}
            row = dict(seeds=len(ds), **m, per_seed={k: [float(x) for x in S.matrix(ds, k, scenes).mean(1)] for k in RATES})
            if groups:
                row["groups"] = {}
                for g in list(GROUPS):
                    ids = [s for s in scenes if group_of(s) == g]
                    if ids:
                        row["groups"][g] = {k: list(S.ci_mean(S.matrix(ds, k, ids))) for k in RATES} | {"n": len(ids)}
            if family_of:
                row["families"] = {}
                fam = {s: ds[0]["per_scene"][s]["layout_family"] for s in scenes}
                for f in sorted(set(fam.values())):
                    ids = [s for s in scenes if fam[s] == f]
                    row["families"][f] = {k: list(S.ci_mean(S.matrix(ds, k, ids))) for k in RATES} | {"n": len(ids)}
                    goals = S.matrix(ds, "clean_goals", ids, scale=1.0)
                    tsum = S.matrix(ds, None, ids, scale=1.0, fn=lambda r: (r.get("mean_time_to_clean_goal_s") or 0.0) * r["clean_goals"])
                    row["families"][f]["time_to_goal"] = list(S.ci_ratio(tsum, goals)) if goals.sum() else None
                    row["families"][f]["palm_cm"] = list(S.ci_mean(S.matrix(ds, "mean_episode_min_palm_field_m", ids, scale=100.0)))
                goals = S.matrix(ds, "clean_goals", scenes, scale=1.0)
                tsum = S.matrix(ds, None, scenes, scale=1.0, fn=lambda r: (r.get("mean_time_to_clean_goal_s") or 0.0) * r["clean_goals"])
                row["time_to_goal"] = list(S.ci_ratio(tsum, goals)) if goals.sum() else None
                row["palm_cm"] = list(S.ci_mean(S.matrix(ds, "mean_episode_min_palm_field_m", scenes, scale=100.0)))
            entry["policies"][name] = row
        entry["deltas"] = {}
        for a, b in (("Planner–reflex", "Reflex off"), ("Planner–reflex", "Reactive (V7r)"), ("Reflex off", "Reactive (V7r)"),
                     ("Planner–reflex", "Released CAT"), ("Planner–reflex", "BRACE + planner"),
                     ("Planner–reflex", "Planner–reflex, plan off"), ("Planner–reflex", "Plan only (fields masked)")):
            if a in datasets and b in datasets:
                entry["deltas"][f"{a} − {b}"] = {k: S.ci_delta(S.matrix(datasets[a], k, scenes), S.matrix(datasets[b], k, scenes)) for k in RATES}
        if family_of:
            fam = {s_: next(iter(datasets.values()))[0]["per_scene"][s_]["layout_family"] for s_ in scenes}
            ut = [s_ for s_ in scenes if fam[s_] in ("u-trap", "early-turn")]
            for key, d_ in entry["deltas"].items():
                a_, b_ = key.split(" − ")
                d_["u_turn_clean"] = S.ci_delta(S.matrix(datasets[a_], "clean", ut), S.matrix(datasets[b_], "clean", ut))
        out[speed] = entry
    return out


# ------------------------------------------------------------------------------------------------ A1, A2, A3
def a1():
    """Dynamic threats while walking: per-policy object hand contact (primary), any object contact, clean, refused."""
    pol = {"Planner–reflex": "planner", "Reflex off": "noreflex", "Reactive (V7r)": "reactive"}
    files = {}
    for name, key in pol.items():
        seeds = []
        for p in sorted(glob.glob(str(CL / f"overnight_A1_{key}_seed*_*"))) + sorted(glob.glob(str(EVRUNS / f"A1_{key}_seed*_local"))):
            seeds.append(Path(p))
        files[name] = seeds
    out = {}
    for speed in ("0.15", "0.3"):
        D = {n: [S.load(p / f"throw_{speed}.json") for p in v if (p / f"throw_{speed}.json").exists()] for n, v in files.items()}
        D = {n: v for n, v in D.items() if v}
        if not D:
            continue
        scenes = S.common_scenes(*D.values())
        entry = dict(n_scenes=len(scenes), policies={}, deltas={})
        tiers = ("0.4", "0.8", "1.2")

        def tier_matrix(ds, kind, tier, ids):  # per-scene hits per launched throw within a tier (hand=2, any=3, launched=0)
            ti = tiers.index(tier)
            return S.matrix(ds, None, ids, scale=1.0, fn=lambda r: r["throws"][kind][ti])

        for name, ds in D.items():
            row = dict(seeds=len(ds))
            for k, key in (("clean", "clean"), ("contact", "contact"), ("refused", "refused"), ("hands", "hands"),
                           ("obj_hand", "inserted_hand_contact"), ("obj_any", "inserted_contact")):
                row[k] = list(S.ci_mean(S.matrix(ds, key, scenes)))
            row["tiers"] = {}
            for t in tiers:
                L = tier_matrix(ds, 0, t, scenes); H = tier_matrix(ds, 2, t, scenes); Y = tier_matrix(ds, 3, t, scenes)
                row["tiers"][t] = dict(launched=float(L.sum(1).mean()), hand=list(S.ci_ratio(100 * H, L)), any=list(S.ci_ratio(100 * Y, L)))
            row["per_seed"] = {k: [float(x) for x in S.matrix(ds, k if k != "obj_hand" else "inserted_hand_contact", scenes).mean(1)]
                               for k in ("clean", "contact", "refused")}
            entry["policies"][name] = row
        for a, b in (("Planner–reflex", "Reflex off"), ("Planner–reflex", "Reactive (V7r)"), ("Reflex off", "Reactive (V7r)")):
            if a in D and b in D:
                entry["deltas"][f"{a} − {b}"] = {k: S.ci_delta(S.matrix(D[a], key, scenes), S.matrix(D[b], key, scenes))
                                                 for k, key in (("clean", "clean"), ("contact", "contact"), ("refused", "refused"),
                                                                ("obj_hand", "inserted_hand_contact"), ("obj_any", "inserted_contact"))}
                entry["deltas"][f"{a} − {b}"]["tiers"] = {}
                for t in tiers:
                    la, ha = tier_matrix(D[a], 0, t, scenes), tier_matrix(D[a], 2, t, scenes)
                    lb, hb = tier_matrix(D[b], 0, t, scenes), tier_matrix(D[b], 2, t, scenes)
                    # per-scene hit rate per launched throw (scenes with no launch contribute their own zero rate)
                    ra = 100 * ha / np.maximum(la, 1); rb = 100 * hb / np.maximum(lb, 1)
                    entry["deltas"][f"{a} − {b}"]["tiers"][t] = S.ci_delta(ra, rb)
        out[speed] = entry
    return out


def a2():
    """Robustness: nominal vs robust (R1) vs robust+pushes (R2) for each policy, original 384 scenes."""
    pol = {"Planner–reflex": "planner", "Reflex off": "noreflex", "Reactive (V7r)": "reactive"}
    out = {}
    for speed in ("0.15", "0.3"):
        nominal = narrow_paths(speed)
        cond = {}
        for name, key in pol.items():
            for setting in ("nopush", "push"):
                ps = sorted(glob.glob(str(CL / f"overnight_A2_{key}_seed*_*/{setting}/speed_{speed}.json")) +
                            glob.glob(str(EVRUNS / f"A2_{key}_seed*_local/{setting}/speed_{speed}.json")))
                if ps:
                    cond[(name, setting)] = [Path(p) for p in ps]
            if name in nominal:
                cond[(name, "nominal")] = nominal[name]
        if not any(k[1] != "nominal" for k in cond):
            continue
        datasets = {k: [S.load(p) for p in v] for k, v in cond.items()}
        scenes = S.common_scenes(*datasets.values())
        entry = dict(n_scenes=len(scenes), cells={}, deltas={})
        for (name, setting), ds in datasets.items():
            entry["cells"][f"{name}|{setting}"] = dict(seeds=len(ds), **{k: list(S.ci_mean(S.matrix(ds, k, scenes))) for k in RATES})
        for setting in ("nominal", "nopush", "push"):
            for a, b in (("Planner–reflex", "Reactive (V7r)"), ("Planner–reflex", "Reflex off")):
                if (a, setting) in datasets and (b, setting) in datasets:
                    entry["deltas"][f"{a} − {b} [{setting}]"] = {k: S.ci_delta(S.matrix(datasets[(a, setting)], k, scenes),
                                                                                 S.matrix(datasets[(b, setting)], k, scenes)) for k in RATES}
        for name in pol:
            for setting in ("nopush", "push"):
                if (name, setting) in datasets and (name, "nominal") in datasets:
                    entry["deltas"][f"{name}: {setting} − nominal"] = {
                        k: S.ci_delta(S.matrix(datasets[(name, setting)], k, scenes), S.matrix(datasets[(name, "nominal")], k, scenes))
                        for k in RATES}
        out[speed] = entry
    return out


# ------------------------------------------------------------------------------------------------ standing / energy
def a3_384():
    """Earlier plan-input ablation on the 384 scenes: the same COMBO1b u500 network with its plan input on and off (1 seed)."""
    base = CL / "speed_sweep"
    out = {}
    for sp in ("0.15", "0.3"):
        f_on, f_off = base / "COMBO1b_u500_plan" / f"speed_{sp}.json", base / "COMBO1b_u500_none" / f"speed_{sp}.json"
        if not exists(f_on, f_off):
            continue
        on, off = S.load(f_on), S.load(f_off)
        sc = S.common_scenes([on], [off])
        cells = {n: {k: list(S.ci_mean(S.matrix([d], k, sc))) for k in RATES} for n, d in (("Plan input on", on), ("Plan input off", off))}
        out[sp] = dict(n_scenes=len(sc), cells=cells, delta={k: S.ci_delta(S.matrix([on], k, sc), S.matrix([off], k, sc)) for k in RATES})
    return out


def standing_row(path):
    d = json.loads(Path(path).read_text())
    d = d[0] if isinstance(d, list) else d
    ev = [a for a in d["approaches"] if a["bucket"] in ("danger", "anticipation")]
    n = max(len(ev), 1)
    share = lambda r: 100 * sum(a["response"] == r for a in ev) / n
    feet = np.array([a["feet_m"] for a in ev]) * 100
    return dict(threats=len(ev), arm_first=share("arm_first"), body_first=share("body_first"), failure=share("failure"),
                still=share("still_success"), feet_moved_worlds=100 * d["envs_ever_violating"],
                feet_p50=float(np.percentile(feet, 50)), feet_p90=float(np.percentile(feet, 90)), feet_p95=float(np.percentile(feet, 95)),
                feet_ge5=float(100 * np.mean(feet >= 5)), retreat_p50=float(np.percentile([a["retreat_m"] for a in ev], 50) * 100),
                power_w=d["power_W"]["total"]["standing"], feet_cm=[float(x) for x in feet],
                feet_q=[float(x) for x in np.percentile(feet, np.linspace(0, 100, 101))])


def standing():
    out = {}
    cases = {
        "arm×1 seed 1": R4 / "energy_round4_Flip_arm1_seed1_318296/standing.json",
        "arm×30 seed 1": R4 / "energy_round4_Flip_arm30_seed1_318296/standing.json",
        "arm×1 seed 2": CL / "energy_round5_Flip_arm1_seed2_319350/standing.json",
        "arm×30 seed 2": CL / "energy_round5_Flip_arm30_seed2_319350/standing.json",
        "α250": R4 / "energy_round4_Heat250_seed1_318296/standing.json",
        "α500": R4 / "energy_round4_Flip_arm1_seed1_318296/standing.json",
        "α1000": R4 / "energy_round4_Heat1000_seed1_318296/standing.json",
        "energy only": Path("/tmp/cat-energy-contact-evaluations/energy_contact_ENERGY_ONLY_317956/standing.json"),
        "contact only": Path("/tmp/cat-energy-contact-evaluations/energy_contact_CONTACT_ONLY_317957/standing.json"),
    }
    for k, p in cases.items():
        if Path(p).exists():
            out[k] = standing_row(p)
    ref = CL.parent / "ref/reference.json"
    if ref.exists():
        r = json.loads(ref.read_text())
        s = r["summary"]
        out["_reference"] = dict(events=s["events"], strategies=s["strategies"], k_phys=s["k_phys"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    res = dict(narrow=build_benchmark(narrow_paths, ("0.15", "0.3"), groups=True),
               e4=build_benchmark(e4_paths, ("0.15", "0.3", "0.5"), family_of=True),
               a1=a1(), a2=a2(), a3_384=a3_384(), standing=standing())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, {k: (list(v) if isinstance(v, dict) else v) for k, v in res.items() if k != "standing"})


if __name__ == "__main__":
    main()
