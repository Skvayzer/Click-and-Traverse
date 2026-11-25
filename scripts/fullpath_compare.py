#!/usr/bin/env python3
"""Paired comparison of CPU-protocol arms (scripts/fullpath_rollout.py recordings), docs/fullpath_plan.md.

    fullpath_compare.py --arm NAME=DIR[,DIR...] ... --pair A:B ... [--gpu NAME=JSONDIR ...] [--out table.md]

An arm with several directories (one per seed) is pooled. Episodes are keyed (scene, speed, trial); a pair uses the episodes
present in both arms (and the same seeds, by position, when both have several). Differences: paired scene-cluster bootstrap
(10,000 resamples of scenes, the same scene draw for both arms), percentage points. Families: side gap, course (hurdle /
beam / hurdle+beam / walls), table gap, and pooled F144 / F144+TG24.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FAMS = ("side gap", "course: hurdle", "course: beam", "course: hurdle+beam", "course: walls", "course (all)", "table gap",
        "F144", "F144+TG24")


def episodes(d):
    """{(scene, speed, trial): dict(result, family)} from the audit caches (or the npz metas when no cache)."""
    out = {}
    for f in sorted(Path(d).glob("*.npz")):
        a = Path(str(f) + ".audit.json")
        if a.exists():
            r = json.loads(a.read_text())
            if "error" in r:
                continue
            fam = r["family"]
        else:
            m = json.loads(str(np.load(f)["meta"]))
            r = dict(scene=m["scene"], speed=m["speed"], trial=m["trial"], result=m["outcome"]["result"])
            fam = "side gap" if r["scene"].startswith("sidegap") else "table gap" if "table-gap" in r["scene"] else "course (all)"
        out[(r["scene"], float(r["speed"]), int(r["trial"]))] = dict(result=r["result"], family=fam,
                                                                      oracle=(r.get("oracle") or {}).get("ok") if isinstance(r.get("oracle"), dict) else None)
    return out


def in_family(fam, f):
    if fam == "F144":
        return f != "table gap"
    if fam == "F144+TG24":
        return True
    if fam == "course (all)":
        return f.startswith("course")
    return f == fam


def rates(eps, fam, speed=None):
    v = [e for k, e in eps.items() if in_family(fam, e["family"]) and (speed is None or k[1] == speed)]
    if not v:
        return None
    return {m: 100. * np.mean([e["result"] == m for e in v]) for m in ("clean", "contact", "refused")} | dict(n=len(v))


def paired(a_runs, b_runs, fam, metric, speed=None, B=10000, seed=0):
    """Mean difference a - b (pp) and its scene-bootstrap CI over keys present in both, per seed position, pooled."""
    rows = []                                    # (scene, a, b)
    for A, Bd in zip(a_runs, b_runs) if len(a_runs) == len(b_runs) else [(x, b_runs[0]) for x in a_runs]:
        for k, ea in A.items():
            eb = Bd.get(k)
            if eb is None or not in_family(fam, ea["family"]) or (speed is not None and k[1] != speed):
                continue
            rows.append((k[0], float(ea["result"] == metric), float(eb["result"] == metric)))
    if not rows:
        return None
    scenes = sorted({r[0] for r in rows})
    idx = {s: i for i, s in enumerate(scenes)}
    diff = np.zeros(len(scenes)); cnt = np.zeros(len(scenes))
    for s, x, y in rows:
        diff[idx[s]] += x - y; cnt[idx[s]] += 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(scenes), size=(B, len(scenes)))
    boot = diff[draws].sum(1) / cnt[draws].sum(1)
    return dict(delta=100. * diff.sum() / cnt.sum(), lo=100. * np.percentile(boot, 2.5), hi=100. * np.percentile(boot, 97.5),
                n=int(cnt.sum()), scenes=len(scenes))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True, help="NAME=DIR[,DIR...]")
    p.add_argument("--pair", action="append", default=[], help="A:B")
    p.add_argument("--speeds", default="both", help="both | each")
    p.add_argument("--out", type=Path)
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    arms = {}
    for spec in a.arm:
        name, dirs = spec.split("=", 1)
        arms[name] = [episodes(d) for d in dirs.split(",")]
    lines, js = [], {}
    speeds = [None] if a.speeds == "both" else [None, .15, .3]
    for sp in speeds:
        tag = "both speeds" if sp is None else f"{sp:g} m/s"
        lines.append(f"\n### Rates, {tag} (clean / contact / refused, %; n episodes)\n")
        lines.append("| family | " + " | ".join(arms) + " |")
        lines.append("| --- |" + " --- |" * len(arms))
        for fam in FAMS:
            cells = []
            for name, runs in arms.items():
                pooled = {}
                for i, r in enumerate(runs):
                    pooled.update({(k, i): v for k, v in r.items()})
                eps = {k: v for k, v in pooled.items()}
                v = [e for k, e in eps.items() if in_family(fam, e["family"]) and (sp is None or k[0][1] == sp)]
                if not v:
                    cells.append("–"); continue
                cl, co, rf = (100. * np.mean([e["result"] == m for e in v]) for m in ("clean", "contact", "refused"))
                cells.append(f"{cl:.1f} / {co:.1f} / {rf:.1f} ({len(v)})")
                js.setdefault(tag, {}).setdefault(name, {})[fam] = dict(clean=cl, contact=co, refused=rf, n=len(v))
            lines.append(f"| {fam} | " + " | ".join(cells) + " |")
        for pr in a.pair:
            A, Bn = pr.split(":")
            lines.append(f"\n### {A} minus {Bn}, {tag} (pp, paired scene bootstrap 95% CI)\n")
            lines.append("| family | clean | contact | refused | n pairs |")
            lines.append("| --- | --- | --- | --- | --- |")
            for fam in FAMS:
                cells, n = [], None
                for metric in ("clean", "contact", "refused"):
                    r = paired(arms[A], arms[Bn], fam, metric, sp)
                    if r is None:
                        cells.append("–"); continue
                    n = r["n"]
                    cells.append(f"{r['delta']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}]")
                    js.setdefault(tag, {}).setdefault(f"{A}-{Bn}", {}).setdefault(fam, {})[metric] = r
                if n:
                    lines.append(f"| {fam} | " + " | ".join(cells) + f" | {n} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
    if a.json:
        a.json.write_text(json.dumps(js, indent=1))


if __name__ == "__main__":
    main()
