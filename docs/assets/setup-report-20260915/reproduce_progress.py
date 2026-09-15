"""Reconstruct validated episode counts and reproduce the progress figure.

Run with the project's environment (NumPy and Matplotlib only):
    python reproduce_progress.py --output-dir /tmp/cat-progress-reproduction

The adjacent CSV is a lossless selection of numeric metrics from the snapshot,
not a new evaluation. No network, GPU, training or checkpoint access is needed.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRANSITIONS_PER_UPDATE = 1_048_576
ROLLOUT_BUFFER_SIZE = 1000
CONTROL_DT = 0.02
PREFIXES = ["rollout", "rollout/pfid/37", "rollout/pfid/38"]
GROUPS = {
    "all": "rollout",
    "original": "original",
    "furniture": "rollout/pfid/37",
    "generic_clutter": "rollout/pfid/38",
}


def load_series(path):
    with path.open(newline="") as handle:
        data = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    assert all(np.isfinite(v) for row in data for v in row.values())
    assert all(row["global_step"] == (i + 1) * TRANSITIONS_PER_UPDATE for i, row in enumerate(data))
    return data


def recover(data, key):
    """Invert a mean of a growing buffer, then a fixed 1000-update buffer."""
    means = [row[key] for row in data]
    raw = []
    for i, value in enumerate(means):
        n = i + 1
        if n <= ROLLOUT_BUFFER_SIZE:
            raw.append(n * value - (n - 1) * (means[i - 1] if i else 0.))
        else:
            raw.append(ROLLOUT_BUFFER_SIZE * (value - means[i - 1]) + raw[i - ROLLOUT_BUFFER_SIZE])
    return np.array(raw)


def reconstruct(data):
    counts = [{"global_step": int(row["global_step"])} for row in data]
    errors = {"episode_count_roundoff": 0., "termination_count_roundoff": 0., "termination_timeout_sum": 0.}
    for prefix in PREFIXES:
        raw_n = recover(data, prefix + "/episodes")
        raw_r = recover(data, prefix + "/termination_rate")
        raw_t = recover(data, prefix + "/timeout_rate")
        ns = np.rint(raw_n).astype(int)
        fs = np.rint(ns * raw_r).astype(int)
        ts = np.rint(ns * raw_t).astype(int)
        errors["episode_count_roundoff"] = max(errors["episode_count_roundoff"], float(np.max(np.abs(raw_n - ns))))
        errors["termination_count_roundoff"] = max(errors["termination_count_roundoff"], float(np.max(np.abs(ns * raw_r - fs))))
        errors["termination_timeout_sum"] = max(errors["termination_timeout_sum"], float(np.max(np.abs(raw_r + raw_t - 1.))))
        # A missing scene callback invalidates indexing means by global updates.
        assert np.all(ns > 0), (prefix, "scene/update contribution missing")
        assert np.all(fs >= 0) and np.all(fs <= ns) and np.array_equal(fs + ts, ns)
        assert errors["episode_count_roundoff"] < 1e-5
        assert errors["termination_count_roundoff"] < 1e-5
        assert errors["termination_timeout_sum"] < 1e-8
        for i in range(len(data)):
            counts[i].update({prefix + "/episodes": int(ns[i]), prefix + "/terminated": int(fs[i]), prefix + "/timeouts": int(ts[i])})

    # Independent global validation: per-transition event rates and cumulative
    # episode counts must agree with the reconstructed per-episode summaries.
    for name, metric in [("episodes", "done_rate"), ("terminated", "termination_step_rate"), ("timeouts", "timeout_step_rate")]:
        derived = recover(data, "rollout/" + metric) * TRANSITIONS_PER_UPDATE
        assert np.max(np.abs(derived - np.rint(derived))) < 1e-5
        assert np.array_equal(np.rint(derived).astype(int), [row["rollout/" + name] for row in counts])
    previous_completed = 0
    for row, count in zip(data, counts):
        for suffix in ["episodes", "terminated", "timeouts"]:
            count["original/" + suffix] = count["rollout/" + suffix] - count["rollout/pfid/37/" + suffix] - count["rollout/pfid/38/" + suffix]
            assert count["original/" + suffix] >= 0
        assert count["original/terminated"] + count["original/timeouts"] == count["original/episodes"]
        assert row["rollout/completed_episodes"] - previous_completed == count["rollout/episodes"]
        previous_completed = row["rollout/completed_episodes"]
    return counts, errors


def window_summary(data, counts, start, end):
    rows, crows = data[start:end], counts[start:end]
    keys = ["training/rollout_reward_mean", "episode/sum_reward", "episode/length", "training/v_loss", "training/policy_loss", "training/entropy_loss", "training/sps"]
    item = {"first_update": start + 1, "last_update": end, "first_step": int(rows[0]["global_step"]), "last_step": int(rows[-1]["global_step"]), "means": {key: statistics.mean(row[key] for row in rows) for key in keys}, "early_termination": {}}
    item["mean_logged_episode_seconds"] = CONTROL_DT * item["means"]["episode/length"]
    for name in ["hand", "arm"]:
        item[name + "_penalty_magnitude_per_step"] = statistics.mean(-row[f"episode/reward/wholebody_{name}_clearance"] / row["episode/length"] for row in rows)
    all_n = sum(row["rollout/episodes"] for row in crows)
    all_f = sum(row["rollout/terminated"] for row in crows)
    for label, prefix in GROUPS.items():
        n = sum(row[prefix + "/episodes"] for row in crows)
        f = sum(row[prefix + "/terminated"] for row in crows)
        item["early_termination"][label] = {"completed_episodes": n, "early_terminations": f, "timeouts": n - f, "rate": f / n, "fraction_of_all_completed_episodes": n / all_n, "fraction_of_all_early_terminations": f / all_f}
    return item


def plot(data, counts, windows, snapshot_utc, output_dir):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.labelcolor": "#334155", "text.color": "#172b4d", "axes.edgecolor": "#94a3b8", "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff"})
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.0), layout="constrained")
    x = np.array([row["global_step"] / 1e9 for row in data])
    blue, teal, orange, purple = "#2563eb", "#087f8c", "#c2410c", "#7c3aed"

    def curve(ax, y, color):
        smoothed = np.array([np.mean(y[max(0, i - 19):i + 1]) for i in range(len(y))])
        ax.plot(x, y, color=color, alpha=.18, lw=.65)
        ax.plot(x, smoothed, color=color, lw=1.8)

    def failure(prefix):
        ns = np.array([row[prefix + "/episodes"] for row in counts])
        fs = np.array([row[prefix + "/terminated"] for row in counts])
        return np.array([100 * fs[max(0, i - 49):i + 1].sum() / ns[max(0, i - 49):i + 1].sum() for i in range(len(ns))])

    curve(axes[0, 0], np.array([row["training/rollout_reward_mean"] for row in data]), blue)
    axes[0, 0].set(title="Reward per transition", ylabel="Mean shaped reward")
    curve(axes[0, 1], np.array([CONTROL_DT * row["episode/length"] for row in data]), teal)
    axes[0, 1].set(title="Completed-episode duration", ylabel="Simulation seconds")
    axes[1, 0].plot(x, failure("rollout"), color=blue, lw=1.8, label="All 39 scenes")
    axes[1, 0].plot(x, failure("original"), color=teal, lw=1.8, label="37 original CAT slots")
    axes[1, 0].set(title="Overall / original early termination", ylabel="Completed episodes (%)", ylim=(0, 100))
    axes[1, 0].legend(frameon=False, loc="upper right")
    axes[1, 1].plot(x, failure("rollout/pfid/37"), color=orange, lw=1.8, label="Tables and chairs")
    axes[1, 1].plot(x, failure("rollout/pfid/38"), color=purple, lw=1.8, label="Generic dense clutter")
    axes[1, 1].set(title="Dense-clutter early termination", ylabel="Completed episodes (%)", ylim=(0, 103))
    axes[1, 1].legend(frameon=False, loc="lower left")
    for ax in axes.flat:
        ax.set_xlabel("Training transitions (billions)")
        ax.grid(alpha=.17)
        for start, end in windows.values():
            ax.axvspan(x[start], x[end - 1], color="#64748b", alpha=.12, zorder=0)
    timestamp = datetime.fromisoformat(snapshot_utc)
    title_date = timestamp.strftime("%-d %B %Y, %H:%M UTC")
    fig.suptitle(f'Current training progress: {data[-1]["global_step"] / 1e9:.3f} billion transitions\n{title_date} · one continuous run · released CAT actor + critic initialization', fontsize=13.5)
    fig.supxlabel("Failure curves: reconstructed, episode-weighted trailing 50 updates. Top curves: 20-update smoothing.\nShading: early / latest comparison windows. Timeouts indicate survival, not successful traversal.", fontsize=9)
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / ("progress-curves." + ext), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, default=Path(__file__).with_name("progress-series.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--snapshot-utc", default="2026-09-15T14:47:33.440566+00:00")
    args = parser.parse_args()
    data = load_series(args.series)
    counts, errors = reconstruct(data)
    windows = {"early": (20, 70), "latest": (len(data) - 50, len(data))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    derived = {"snapshot_utc": args.snapshot_utc, "series_sha256": hashlib.sha256(args.series.read_bytes()).hexdigest(), "updates": len(data), "last_transition": int(data[-1]["global_step"]), "reconstruction_roundoff": errors, "windows": {name: window_summary(data, counts, *bounds) for name, bounds in windows.items()}}
    (args.output_dir / "progress-reproduced-summary.json").write_text(json.dumps(derived, indent=2) + "\n")
    plot(data, counts, windows, args.snapshot_utc, args.output_dir)
    print(json.dumps(derived, indent=2))


if __name__ == "__main__":
    main()
