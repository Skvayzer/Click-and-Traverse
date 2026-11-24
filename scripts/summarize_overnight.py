#!/usr/bin/env python3
"""Audit and summarize complete overnight rollout files copied from Slurm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import t


RATES = ("clean", "contact", "refused", "hands")
GOAL_TIME = "mean_time_to_clean_goal_s"
PALM_CLEARANCE = "mean_episode_min_palm_field_m"
LEGACY_GROUPS = {
    "Side gaps": ("sidegap",),
    "Cabinets": ("contrast",),
    "Furniture": ("random-furniture",),
    "Clutter": ("random-generic",),
    "CAT courses": ("D8", "procedural", "published"),
}
TERMS = {
    "Stall": ("stall",),
    "Tracking": ("track_pos", "track_lead", "track_upper", "track_heading", "track_height"),
    "Clearance": ("handsdf", "wholebody_hand_clearance", "wholebody_arm_clearance", "headdf", "feetdf"),
    "Field": ("handsgf", "feetgf", "headgf", "tracking_root_field", "tracking_orientation"),
    "Posture and motion": ("upright", "stand_tall", "torso_rate", "wholebody_upper_clear_posture", "body_motion", "body_rotation"),
    "Energy": ("global_power", "table_power"),
    "Goal progress": ("goal_progress",),
}


def read_scores(path):
    result = json.loads(path.read_text())
    rows = result["per_scene"]
    if result["finished"] != result["target"] or sum(r["episodes"] for r in rows.values()) != result["target"]:
        raise ValueError(f"Incomplete evaluation in {path}")
    return rows, result


def format_stats(values, *, paired=False):
    center = mean(values)
    sign = "+" if paired else ""
    if len(values) == 1:
        return f"{center:{sign}.2f}"
    return f"{center:{sign}.2f} ± {stdev(values):.2f}"


def confidence_interval(values):
    half = t.ppf(.975, len(values) - 1) * stdev(values) / len(values)**.5
    return mean(values) - half, mean(values) + half


def weighted(rows, metric):
    selected = [v for v in rows.values() if v.get(metric) is not None]
    weight = "clean_goals" if metric == GOAL_TIME else "episodes"
    denominator = sum(v[weight] for v in selected)
    return sum(v[metric] * v[weight] for v in selected) / denominator if denominator else None


def stall_report(paths):
    print("| Speed | Scene group | Window | Episodes across seeds | " + " | ".join(TERMS) + " |")
    print("| --- | --- | --- | ---: |" + " ---: |" * len(TERMS))
    for speed in ("0.15", "0.3"):
        data = [read_scores(folder / f"speed_{speed}.json")[0] for folder in paths]
        if any(rows.keys() != data[0].keys() for rows in data[1:]):
            raise ValueError("Audit scenes are unmatched")
        for group, prefix in LEGACY_GROUPS.items():
            for window in ("before", "during"):
                by_seed = []
                for rows in data:
                    windows = [v["stall_windows"][window] for k, v in rows.items() if k.startswith(prefix)]
                    count = sum(w["episodes"] for w in windows)
                    terms = [sum(sum(w["terms"].get(term, 0.) for term in keys) * w["episodes"] for w in windows)
                             for keys in TERMS.values()]
                    by_seed.append((count, terms))
                total = sum(count for count, _ in by_seed)
                if not total:
                    vals = ["—"] * len(TERMS)
                else:
                    vals = [
                        format_stats([terms[i] / n for n, terms in by_seed if n])
                        for i in range(len(TERMS))
                    ]
                print("| " + " | ".join([speed, group, window, str(total), *vals]) + " |")


def paired_sample(rows, metric):
    weight = "clean_goals" if metric == GOAL_TIME else "episodes"
    scale = 1. if metric == GOAL_TIME else 100.
    values = [rows[k] for k in sorted(rows)]
    counts = np.array([r[weight] if r.get(metric) is not None else 0 for r in values])
    sums = np.array([(r.get(metric) or 0.) * scale for r in values]) * counts
    return sums, counts


def paired_difference(left, right, metric):
    values = []
    pairs = []
    for a, b in zip(left, right, strict=True):
        if a.keys() != b.keys() or any(a[k]["episodes"] != b[k]["episodes"] for k in a):
            raise ValueError("Paired evaluation scenes or episode counts differ")
        sums_a, counts_a = paired_sample(a, metric)
        sums_b, counts_b = paired_sample(b, metric)
        if counts_a.sum() and counts_b.sum():
            pairs.append(((sums_a, counts_a), (sums_b, counts_b)))
            values.append(sums_a.sum() / counts_a.sum() - sums_b.sum() / counts_b.sum())
    if not values:
        return "—"
    if len(values) > 1:
        lo, hi = confidence_interval(values)
    elif len(left) == 1:
        (sums_a, counts_a), (sums_b, counts_b) = pairs[0]
        rng = np.random.default_rng(20261124)
        idx = rng.integers(0, len(sums_a), size=(4000, len(sums_a)))
        count_a = counts_a[idx].sum(1)
        count_b = counts_b[idx].sum(1)
        keep = (count_a > 0) & (count_b > 0)
        boot = sums_a[idx[keep]].sum(1) / count_a[keep] - sums_b[idx[keep]].sum(1) / count_b[keep]
        if not boot.size:
            return format_stats(values, paired=True)
        lo, hi = np.quantile(boot, [.025, .975])
    else:
        return f"{format_stats(values, paired=True)} (one paired seed with both estimates)"
    return f"{format_stats(values, paired=True)} [{lo:+.2f}, {hi:+.2f}]"


def rate_report(conditions, speeds, file_names, *, family=False, include_inserted=False):
    metrics = (*RATES, "inserted_hand_contact", "inserted_revealed") if include_inserted else RATES
    labels = (*metrics, "time to clean goal (s)", "episode min palm (cm)") if family else metrics
    print("| Case | Speed / family | " + " | ".join(labels) + " |")
    print("| --- | --- |" + " ---: |" * len(labels))
    for speed in speeds:
        loaded = {label: [read_scores(folder / file_names(speed)) for folder in folders]
                  for label, folders in conditions}
        datasets = [pairs for pairs in loaded.values()]
        template, meta0 = datasets[0][0]
        if any(x.keys() != template.keys() for pairs in datasets for x, _ in pairs):
            raise ValueError("Evaluation scenes are unmatched")
        if family and any(x["initial_state_sha256"] != meta0["initial_state_sha256"]
                          for pairs in datasets for _, x in pairs):
            raise ValueError("Planning evaluation initial states differ")
        families = [None] + sorted({r["layout_family"] for r in template.values()}) if family else [None]
        baseline = conditions[0][0]
        for layout in families:
            def subset(rows):
                return {k: v for k,v in rows.items() if layout is None or v["layout_family"] == layout}
            mark = f"{speed} / {layout}" if layout else str(speed)
            base = [subset(r) for r, _ in loaded[baseline]]
            for label, _ in conditions:
                rows = [subset(r) for r, _ in loaded[label]]
                values = [format_stats([100 * weighted(r, metric) for r in rows]) for metric in metrics]
                if family:
                    time = [x for r in rows if (x := weighted(r, GOAL_TIME)) is not None]
                    values.append(format_stats(time) if time else "—")
                    values.append(format_stats([100 * weighted(r, PALM_CLEARANCE) for r in rows]))
                print("| " + " | ".join([label, mark, *values]) + " |")
                if label == baseline:
                    continue
                n = max(len(rows), len(base))
                left = rows * n if len(rows) == 1 else rows
                right = base * n if len(base) == 1 else base
                if len(left) != len(right):
                    raise ValueError("Multi-seed paired conditions need the same number of seeds")
                paired_metrics = (*metrics, GOAL_TIME, PALM_CLEARANCE) if family else metrics
                differences = [paired_difference(left, right, metric) for metric in paired_metrics]
                print("| " + " | ".join([f"{label} − {baseline}", mark, *differences]) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["E0", "E1", "E3", "E4"])
    parser.add_argument("--case", action="append", nargs="+", required=True, metavar=("LABEL", "DIR"),
                        help="repeat for each configuration. List seed dirs in matching order, baseline first")
    args = parser.parse_args()
    conditions = [(item[0], [Path(p) for p in item[1:]]) for item in args.case]
    if any(not paths for _, paths in conditions):
        parser.error("Each case needs one or more evaluation directories")
    if args.phase == "E0":
        stall_report(conditions[0][1])
    elif args.phase == "E3":
        protocols = [f"{mode}_{speed}" for mode in ("static","moving") for speed in ("0.15","0.3")]
        rate_report(conditions, protocols, lambda protocol: f"{protocol}.json", include_inserted=True)
    else:
        speeds = ["0.15","0.3","0.5"] if args.phase == "E4" else ["0.15","0.3"]
        rate_report(conditions, speeds, lambda speed: f"speed_{speed}.json", family=args.phase == "E4")


if __name__ == "__main__":
    main()
