#!/usr/bin/env python3
"""Paired comparison of per-scene evaluation files (eval_per_scene.py outputs) for the improve-overnight log.

Usage: improve_compare.py --speeds 0.15 0.3 --pair NAME "A_dir[,A_dir...]" "B_dir[,B_dir...]" [--pair ...] [--markdown]
Each side is a comma-separated list of run directories, one per seed, in the same seed order on both sides (a side with
one directory is reused for every seed, e.g. the single reactive checkpoint). Reports per-side mean ± s.d. over seeds, the
B − A change with a seed-level paired t interval (if ≥ 2 seeds) and a scene-cluster bootstrap interval (scenes resampled,
seed pairs kept together).
Oracle mode: --oracle NAME "A_dirs" "B_dirs" reports, per scene, the better of A and B (clean first, then fewer contacts).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student

RATES = ("clean", "contact", "refused", "hands")


GROUPS = {"side gaps": ("sidegap",), "cabinets": ("contrast",), "furniture": ("random-furniture",),
          "clutter": ("random-generic",), "CAT courses": ("D8", "procedural", "published")}
FAMILIES = ("u-trap", "early-turn", "table-gap")


def group_of(scene):
    for g, prefixes in GROUPS.items():
        if scene.startswith(prefixes):
            return g
    for f in FAMILIES:
        if f in scene:
            return f
    return "other"


def load(run, speed):
    path = Path(run) / f"speed_{speed:g}.json"
    d = json.loads(path.read_text())
    if d["finished"] != d["target"]:
        raise ValueError(f"incomplete: {path} {d['finished']}/{d['target']}")
    return d["per_scene"]


def matrix(runs, speed, scenes=None):
    rows = [load(r, speed) for r in runs]
    if scenes is None:
        scenes = sorted(rows[0])
    for r in rows:
        if set(r) != set(scenes):
            raise ValueError("scene sets differ")
    # [seeds, scenes, rates], each scene weighted by its episode count (all equal here)
    x = np.array([[[r[s][k] for k in RATES] for s in scenes] for r in rows])
    w = np.array([[r[s]["episodes"] for s in scenes] for r in rows], float)
    return x, w, scenes


def boot(diff, w, n=10000, seed=0):
    """diff [seeds, scenes, rates]; returns 2.5/97.5 percentiles of the episode-weighted mean change (pp)."""
    rng = np.random.default_rng(seed)
    S = diff.shape[1]
    idx = rng.integers(0, S, size=(n, S))
    num = (diff[:, idx] * w[:, idx, None]).sum((0, 2))
    den = w[:, idx].sum((0, 2))[:, None]
    est = num / den
    return np.percentile(est, 2.5, 0) * 100, np.percentile(est, 97.5, 0) * 100


def rate(x, w):
    return (x * w[..., None]).sum(1) / w.sum(1)[:, None] * 100      # [seeds, rates]


def fmt(v):
    return f"{v.mean():.2f}" + (f" ± {v.std(ddof=1):.2f}" if len(v) > 1 else "")


def compare(name, a_runs, b_runs, speeds, label_a="A", label_b="B"):
    n = max(len(a_runs), len(b_runs))
    a_runs = a_runs * n if len(a_runs) == 1 else a_runs
    b_runs = b_runs * n if len(b_runs) == 1 else b_runs
    out = []
    for sp in speeds:
        xa, wa, scenes = matrix(a_runs, sp)
        xb, wb, _ = matrix(b_runs, sp, scenes)
        ra, rb = rate(xa, wa), rate(xb, wb)
        d = rb - ra
        lo, hi = boot(xb - xa, wa)
        row = dict(speed=sp, a={k: fmt(ra[:, i]) for i, k in enumerate(RATES)}, b={k: fmt(rb[:, i]) for i, k in enumerate(RATES)},
                   delta={}, seeds=n)
        for i, k in enumerate(RATES):
            s = f"{d[:, i].mean():+.2f}"
            if n > 1:
                half = student.ppf(.975, n - 1) * d[:, i].std(ddof=1) / np.sqrt(n)
                s += f" t[{d[:, i].mean() - half:+.2f}, {d[:, i].mean() + half:+.2f}]"
            s += f" b[{lo[i]:+.2f}, {hi[i]:+.2f}]"
            row["delta"][k] = s
        row["raw"] = dict(a=ra.mean(0).tolist(), b=rb.mean(0).tolist(), d=d.mean(0).tolist(), lo=lo.tolist(), hi=hi.tolist())
        out.append(row)
    return out


def oracle(a_runs, b_runs, speeds):
    n = max(len(a_runs), len(b_runs))
    a_runs = a_runs * n if len(a_runs) == 1 else a_runs
    b_runs = b_runs * n if len(b_runs) == 1 else b_runs
    out = []
    for sp in speeds:
        xa, wa, scenes = matrix(a_runs, sp)
        xb, _, _ = matrix(b_runs, sp, scenes)
        better_b = (xb[..., 0] > xa[..., 0]) | ((xb[..., 0] == xa[..., 0]) & (xb[..., 1] < xa[..., 1]))
        xo = np.where(better_b[..., None], xb, xa)
        ro = rate(xo, wa)
        out.append(dict(speed=sp, oracle={k: fmt(ro[:, i]) for i, k in enumerate(RATES)},
                        share_b=float(better_b.mean()), a=rate(xa, wa).mean(0).tolist(), b=rate(xb, wa).mean(0).tolist()))
    return out


def by_group(a_runs, b_runs, speeds):
    n = max(len(a_runs), len(b_runs))
    a_runs = a_runs * n if len(a_runs) == 1 else a_runs
    b_runs = b_runs * n if len(b_runs) == 1 else b_runs
    out = []
    for sp in speeds:
        xa, wa, scenes = matrix(a_runs, sp)
        xb, _, _ = matrix(b_runs, sp, scenes)
        groups = np.array([group_of(x) for x in scenes])
        for g in sorted(set(groups)):
            m = groups == g
            ra, rb = rate(xa[:, m], wa[:, m]).mean(0), rate(xb[:, m], wa[:, m]).mean(0)
            lo, hi = boot(xb[:, m] - xa[:, m], wa[:, m], n=4000)
            out.append(dict(speed=sp, group=g, scenes=int(m.sum()), a=ra.tolist(), b=rb.tolist(), lo=lo.tolist(), hi=hi.tolist()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speeds", type=float, nargs="+", default=[.15, .3])
    p.add_argument("--pair", nargs=3, action="append", default=[], metavar=("NAME", "A", "B"))
    p.add_argument("--oracle", nargs=3, action="append", default=[], metavar=("NAME", "A", "B"))
    p.add_argument("--groups", action="store_true", help="also break every --pair down by scene group")
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    report = {}
    for name, a, b in args.pair:
        rows = compare(name, a.split(","), b.split(","), args.speeds)
        report[name] = rows
        print(f"\n| {name} | speed | clean | contact | refused | hands |\n| --- | --- | ---: | ---: | ---: | ---: |")
        for r in rows:
            print(f"| A | {r['speed']:g} | " + " | ".join(r["a"][k] for k in RATES) + " |")
            print(f"| B | {r['speed']:g} | " + " | ".join(r["b"][k] for k in RATES) + " |")
            print(f"| B − A | {r['speed']:g} | " + " | ".join(r["delta"][k] for k in RATES) + " |")
        if args.groups:
            print(f"\n| {name} by group | speed | scenes | clean A→B [Δ CI] | contact A→B [Δ CI] | refused A→B [Δ CI] |\n| --- | --- | ---: | --- | --- | --- |")
            for r in by_group(a.split(","), b.split(","), args.speeds):
                cell = lambda i: f"{r['a'][i]:.1f}→{r['b'][i]:.1f} [{r['lo'][i]:+.1f}, {r['hi'][i]:+.1f}]"
                print(f"| {r['group']} | {r['speed']:g} | {r['scenes']} | {cell(0)} | {cell(1)} | {cell(2)} |")
    for name, a, b in args.oracle:
        rows = oracle(a.split(","), b.split(","), args.speeds)
        report["oracle:" + name] = rows
        print(f"\n| oracle {name} | speed | clean | contact | refused | hands | share from B |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for r in rows:
            print(f"| best of A,B | {r['speed']:g} | " + " | ".join(r["oracle"][k] for k in RATES) + f" | {r['share_b']:.2f} |")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
