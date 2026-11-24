"""Paired scene-bootstrap statistics for the per-scene evaluation JSONs (overnight evaluation program).

Every evaluation JSON holds, per scene, rates over its episodes (2 per scene). Policies are compared on the same scenes.
A policy that has several training seeds is a [seeds, scenes] array. One bootstrap resample draws scene indices with
replacement (the same indices for every policy and every seed), averages over seeds, and takes the statistic; the
2.5/97.5 percentiles over 10,000 resamples are the interval. The seed-level interval treats the per-seed deltas as a
sample (Student t, S-1 d.f.), which with three seeds is the conservative one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_BOOT = 10000
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def load(path):
    d = json.loads(Path(path).read_text())
    return d


def scenes_of(d):
    return sorted(d["per_scene"])


def matrix(paths, key, scenes=None, scale=100.0, fn=None):
    """[seeds, scenes] array of per-scene `key` (or fn(row)) for the given result files."""
    rows = []
    for p in paths:
        d = p if isinstance(p, dict) else load(p)
        ids = scenes or scenes_of(d)
        g = fn if fn else (lambda r: r[key])
        rows.append([scale * g(d["per_scene"][s]) for s in ids])
    return np.asarray(rows, dtype=float)


def common_scenes(*datasets):
    ids = None
    for group in datasets:
        for d in group:
            s = set(d["per_scene"])
            ids = s if ids is None else ids & s
    return sorted(ids)


def boot_indices(n, n_boot=N_BOOT, seed=0):
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


def ci_mean(A, n_boot=N_BOOT, seed=0):
    """Mean over seeds and scenes with a scene-bootstrap interval. A is [S, N]."""
    m = A.mean(0)
    idx = boot_indices(m.shape[0], n_boot, seed)
    b = m[idx].mean(1)
    return float(m.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def ci_delta(A, B, n_boot=N_BOOT, seed=0):
    """Paired difference A - B (both [S, N] over the same scenes). Returns the scene-bootstrap and seed-level intervals."""
    a, b = A.mean(0), B.mean(0)
    idx = boot_indices(a.shape[0], n_boot, seed)
    boot = a[idx].mean(1) - b[idx].mean(1)
    out = dict(mean=float((a - b).mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))
    if A.shape[0] == B.shape[0] and A.shape[0] > 1:
        per_seed = A.mean(1) - B.mean(1)
        s = A.shape[0]
        half = T975.get(s - 1, 2.0) * per_seed.std(ddof=1) / np.sqrt(s)
        out.update(seed_lo=float(per_seed.mean() - half), seed_hi=float(per_seed.mean() + half),
                   seeds_same_sign=bool(np.all(np.sign(per_seed) == np.sign(per_seed.mean())) and per_seed.mean() != 0),
                   per_seed=[float(x) for x in per_seed])
    return out


def ci_ratio(num, den, n_boot=N_BOOT, seed=0):
    """Ratio estimator sum(num)/sum(den) (e.g. time to clean goal over clean arrivals) with a scene bootstrap. [S,N]."""
    n, d = num.sum(0), den.sum(0)
    idx = boot_indices(n.shape[0], n_boot, seed)
    boot = n[idx].sum(1) / np.maximum(d[idx].sum(1), 1e-12)
    ok = np.isfinite(boot) & (d[idx].sum(1) > 0)
    est = n.sum() / d.sum() if d.sum() > 0 else float("nan")
    if not ok.any():
        return float(est), float("nan"), float("nan")
    return float(est), float(np.percentile(boot[ok], 2.5)), float(np.percentile(boot[ok], 97.5))


def fmt(est, lo, hi, d=2, signed=False):
    sign = "+" if signed else ""
    return f"{est:{sign}.{d}f} [{lo:{sign}.{d}f}, {hi:{sign}.{d}f}]"


def delta_row(label, A, B, d=2):
    r = ci_delta(A, B)
    s = fmt(r["mean"], r["lo"], r["hi"], d, True)
    if "seed_lo" in r:
        s += f" | seeds [{r['seed_lo']:+.{d}f}, {r['seed_hi']:+.{d}f}]" + (" same sign" if r["seeds_same_sign"] else "")
    return f"{label}: {s}"
