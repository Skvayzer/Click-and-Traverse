#!/usr/bin/env python3
"""C4 comparison of eval_per_scene outputs (docs/cos_planner_reflex.md, C4).

  python scripts/cos_c4_compare.py --arm cos=DIR --arm v7r=DIR --arm fix=DIR [--families BANK_packed/addition.json] \
      --system cos --out table.json

Per arm: clean / contact / hand contact / refused / fall (episode shares), time to the clean goal and min palm field
(--traversal-metrics) per speed and pooled; per COS family when --families is given. Paired scene-bootstrap 95 % CI of
system − arm for clean and contact (scene = unit; both speeds pooled per scene).
"""
import argparse, json
from pathlib import Path
import numpy as np

KEYS = ("clean", "contact", "hands", "refused", "fall")


def load(d):
    """One eval dir, or several comma-separated dirs (seeds): per-scene values are averaged over the dirs."""
    dirs = d.split(",")
    runs = []
    for dd in dirs:
        out = {}
        for f in sorted(Path(dd).glob("speed_*.json")):
            j = json.load(open(f)); out[str(j["operator_speed"])] = j
        runs.append(out)
    if len(runs) == 1:
        return runs[0]
    merged = {}
    for sp in runs[0]:
        ps = {}
        for sc in runs[0][sp]["per_scene"]:
            rows = [r[sp]["per_scene"][sc] for r in runs if sp in r and sc in r[sp]["per_scene"]]
            row = {k: float(np.mean([x[k] for x in rows])) for k in KEYS}
            for k in ("mean_time_to_clean_goal_s", "mean_episode_min_palm_field_m"):
                v = [x[k] for x in rows if x.get(k) is not None]
                row[k] = float(np.mean(v)) if v else None
            ps[sc] = row
        merged[sp] = dict(per_scene=ps)
    return merged


def boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed); m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--families"); ap.add_argument("--system", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = {k: load(v) for k, v in (s.split("=", 1) for s in a.arm)}
    fam = {}
    if a.families:
        for r in json.load(open(a.families))["scenes"]:
            fam[r["scene_id"]] = r["source"]["cos"]["family"]
    res = {}
    for name, sp in arms.items():
        R = {}
        for speed, j in sp.items():
            ps = j["per_scene"]
            R[speed] = {k: round(float(np.mean([v[k] for v in ps.values()])), 4) for k in KEYS}
            t = [v["mean_time_to_clean_goal_s"] for v in ps.values() if v.get("mean_time_to_clean_goal_s") is not None]
            R[speed]["time_to_goal_s"] = round(float(np.mean(t)), 2) if t else None
            p = [v["mean_episode_min_palm_field_m"] for v in ps.values() if v.get("mean_episode_min_palm_field_m") is not None]
            R[speed]["min_palm_field_m"] = round(float(np.mean(p)), 4) if p else None
            if fam:
                R[speed]["by_family"] = {}
                for f in sorted(set(fam.values())):
                    vv = [v for s, v in ps.items() if fam.get(s) == f]
                    if vv:
                        R[speed]["by_family"][f] = dict(n=len(vv), **{k: round(float(np.mean([v[k] for v in vv])), 3) for k in ("clean", "contact", "refused")})
        res[name] = R
    sysd = arms[a.system]; paired = {}
    for name, sp in arms.items():
        if name == a.system:
            continue
        common = None
        for speed in sysd:
            s = set(sysd[speed]["per_scene"]) & set(sp.get(speed, {}).get("per_scene", {}))
            common = s if common is None else common & s
        common = sorted(common or [])
        if not common:
            continue
        P = {}
        for k in ("clean", "contact", "hands", "refused"):
            d = np.array([np.mean([sysd[sp_][ "per_scene"][s][k] - sp[sp_]["per_scene"][s][k] for sp_ in sysd]) for s in common])
            P[k] = dict(mean=round(float(d.mean()), 4), ci95=boot(d))
        if fam:
            P["by_family_clean"] = {}
            for f in sorted(set(fam.values())):
                cs = [s for s in common if fam.get(s) == f]
                if cs:
                    d = np.array([np.mean([sysd[sp_]["per_scene"][s]["clean"] - sp[sp_]["per_scene"][s]["clean"] for sp_ in sysd]) for s in cs])
                    P["by_family_clean"][f] = dict(n=len(cs), mean=round(float(d.mean()), 3), ci95=boot(d))
        paired[f"{a.system} - {name}"] = dict(scenes=len(common), **P)
    out = dict(arms=res, paired=paired)
    json.dump(out, open(a.out, "w"), indent=1)
    for name, R in res.items():
        print(name, {sp: {k: R[sp][k] for k in ("clean", "contact", "hands", "refused", "time_to_goal_s", "min_palm_field_m")} for sp in R})
    for k, v in paired.items():
        print(k, {x: v[x] for x in ("clean", "contact")})


if __name__ == "__main__":
    main()
