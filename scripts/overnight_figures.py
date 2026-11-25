#!/usr/bin/env python3
"""Paper-style figures (vector PDF + SVG + PNG) from results.json built by overnight_data.py.

Style: Times-like serif (Liberation Serif / Nimbus Roman), colour-blind-safe palette, plan blue, reflex orange, energy green.
    python scripts/overnight_figures.py --data results.json --out OUTDIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PLAN, REFLEX, ENERGY = "#1f6fb2", "#d0631c", "#2a9d5c"
INK, QUIET, GRID = "#1a1a1a", "#5c5c5c", "#e6e6e6"
SKY, GREY, PURPLE, BROWN, VERMIL = "#8fbbe0", "#7a7a7a", "#7b5ea7", "#8c6d46", "#b8312f"
COLUMN, TEXT = 3.5, 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 300, "axes.unicode_minus": True,
})
STYLE = {  # policy -> colour, marker, filled
    "Released CAT": (GREY, "s", True), "Reactive (V7r)": (INK, "D", True), "Planner–reflex": (PLAN, "o", True),
    "Reflex off": (SKY, "o", False), "BRACE + planner": (PURPLE, "^", True), "Plan only (fields masked)": (BROWN, "v", True),
    "Planner–reflex, plan off": (REFLEX, "P", False), "SONIC (plain)": (VERMIL, "X", True), "SONIC + adaptor": (ENERGY, "X", True),
}


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter):
    ax.text(-.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


def err(est):
    return [[est[0] - est[1]], [est[2] - est[0]]]


# ------------------------------------------------------------------------------------------------ F1 scatter
def fig_scatter(R, out, speeds=("0.3",)):
    for sp in speeds:
        only_e4 = sp not in R["narrow"]
        fig, axes = plt.subplots(1, 1 if only_e4 else 2, figsize=(TEXT / (1.9 if only_e4 else 1), 2.6), squeeze=False)
        axes = axes[0]
        pairs = ((("e4", f"Held-out layouts, 72 layouts, {sp} m/s"),) if only_e4 else
                 (("narrow", f"Narrow scenes, 384 scenes, {sp} m/s"), ("e4", f"Held-out layouts, 72 layouts, {sp} m/s")))
        for ax, (key, title) in zip(axes, pairs):
            e = R[key].get(sp)
            if not e:
                continue
            for name, p in e["policies"].items():
                if name not in STYLE:
                    continue
                c, m, filled = STYLE[name]
                ax.errorbar(p["contact"][0], p["clean"][0], xerr=err(p["contact"]), yerr=err(p["clean"]), fmt=m, ms=5, color=c,
                            mfc=c if filled else "white", mew=1.1, elinewidth=.8, capsize=1.5, zorder=3 if "Planner" in name else 2)
            ax.plot([0, 100], [100, 0], color=GRID, lw=.8, zorder=0)
            ax.set_xlabel("any contact (% of episodes)"); ax.set_ylabel("clean traversal (% of episodes)")
            ax.set_xlim(-3, 103); ax.set_ylim(-3, 103); ax.grid(color=GRID, lw=.4)
            ax.text(.5, 1.02, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        panel(axes[0], "a")
        if not only_e4:
            panel(axes[1], "b")
        handles = [Line2D([], [], marker=STYLE[n][1], color=STYLE[n][0], mfc=STYLE[n][0] if STYLE[n][2] else "white", ls="none", ms=5, mew=1.1, label=n)
                   for n in STYLE if any(n in R[k].get(sp, {}).get("policies", {}) for k in ("narrow", "e4"))]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.2), columnspacing=1.2, handletextpad=.3)
        fig.tight_layout(w_pad=2)
        save(fig, out, f"fig1_clean_vs_contact_{sp.replace('.', '')}")


# ------------------------------------------------------------------------------------------------ F2 per-family
def fig_families(R, out, sp="0.3"):
    order = ["Released CAT", "Reactive (V7r)", "Reflex off", "Planner–reflex"]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT, 4.2), gridspec_kw=dict(width_ratios=[5, 3], hspace=.55, wspace=.18))
    for row, metric, ylabel in ((0, "clean", "clean traversal (%)"), (1, "hands", "hand contact (%)")):
        for col, (key, kind) in enumerate((("narrow", "groups"), ("e4", "families"))):
            ax = axes[row, col]
            e = R[key].get(sp)
            if not e:
                continue
            groups = list(next(iter(e["policies"].values()))[kind])
            names = [n for n in order if n in e["policies"] and kind in e["policies"][n]]
            w = .8 / len(names)
            for i, n in enumerate(names):
                c, m, filled = STYLE[n]
                vals = [e["policies"][n][kind][g][metric] for g in groups]
                x = np.arange(len(groups)) + (i - (len(names) - 1) / 2) * w
                ax.bar(x, [v[0] for v in vals], w * .92, color=c if filled else "white", edgecolor=c, lw=.8,
                       yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], error_kw=dict(lw=.6, capsize=1, capthick=.6, ecolor=INK))
            labels = [g.replace(" rooms", "\nrooms").replace("Side gaps", "Side\ngaps").replace("CAT courses", "CAT\ncourses") if kind == "groups"
                      else g.replace("-", "\n") for g in groups]
            ax.set_xticks(range(len(groups)), labels, fontsize=6.5)
            ax.set_ylabel(ylabel); ax.set_ylim(0, 105 if metric == "clean" else None); ax.grid(axis="y", color=GRID, lw=.4); ax.set_axisbelow(True)
            if row == 0:
                ax.text(.5, 1.03, ("Narrow scenes by group" if key == "narrow" else "Held-out layouts by family") + f", {sp} m/s",
                        transform=ax.transAxes, ha="center", va="bottom")
    panel(axes[0, 0], "a"); panel(axes[0, 1], "b"); panel(axes[1, 0], "c"); panel(axes[1, 1], "d")
    handles = [Line2D([], [], marker="s", color=STYLE[n][0], mfc=STYLE[n][0] if STYLE[n][2] else "white", ls="none", ms=6, label=n) for n in order]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.04), columnspacing=1.5, handletextpad=.2)
    save(fig, out, f"fig2_per_family_{sp.replace('.', '')}")


# ------------------------------------------------------------------------------------------------ F3 refusal causes (E0)
E0 = {  # three-seed temporal audit (docs/overnight_20261123.md, E0): mean reward per control step before / during the first stall
    "0.15": {"before": dict(stall=-.43, goal=3.02, track=2.58, power=-.74), "during": dict(stall=-1.47, goal=1.43, track=2.46, power=-.69), "n": (63, 69)},
    "0.3": {"before": dict(stall=-.16, goal=2.74, track=2.15, power=-.85), "during": dict(stall=-2.13, goal=.70, track=1.78, power=-.75), "n": (87, 92)},
}


def fig_refusal(R, out):
    fig, axes = plt.subplots(1, 3, figsize=(TEXT, 2.3), gridspec_kw=dict(width_ratios=[1.15, 1, 1], wspace=.42))
    ax = axes[0]
    groups = list(R["narrow"]["0.3"]["policies"]["Planner–reflex"]["groups"])
    w = .38
    for i, (sp, c) in enumerate((("0.15", SKY), ("0.3", PLAN))):
        vals = [R["narrow"][sp]["policies"]["Planner–reflex"]["groups"][g]["refused"] for g in groups]
        ax.bar(np.arange(len(groups)) + (i - .5) * w, [v[0] for v in vals], w * .92, color=c, label=f"{sp} m/s",
               yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], error_kw=dict(lw=.6, capsize=1, ecolor=INK))
    ax.set_xticks(range(len(groups)), [g.replace(" rooms", "\nrooms").replace("Side gaps", "Side\ngaps").replace("CAT courses", "CAT\ncourses") for g in groups], fontsize=5.6)
    ax.set_ylabel("refused episodes (%)"); ax.legend(frameon=False, loc="upper right"); ax.grid(axis="y", color=GRID, lw=.4); ax.set_axisbelow(True)
    panel(ax, "a")
    terms = [("goal", "goal\nprogress"), ("stall", "stall\ncost"), ("track", "plan\ntracking"), ("power", "power\ncost")]
    for ax, sp, letter in ((axes[1], "0.3", "b"), (axes[2], "0.15", "c")):
        d = E0[sp]
        x = np.arange(len(terms))
        ax.bar(x - .19, [d["before"][k] for k, _ in terms], .36, color=SKY, label=f"before stall (n={d['n'][0]})")
        ax.bar(x + .19, [d["during"][k] for k, _ in terms], .36, color=REFLEX, label=f"during stall (n={d['n'][1]})")
        ax.axhline(0, color=INK, lw=.6)
        ax.set_xticks(x, [t for _, t in terms], fontsize=6.3); ax.set_ylim(-2.7, 4.6)
        ax.set_ylabel("reward per control step"); ax.set_xlabel(f"{sp} m/s", labelpad=1)
        ax.legend(frameon=False, loc="upper right", fontsize=6.2, handlelength=.9); ax.grid(axis="y", color=GRID, lw=.4); ax.set_axisbelow(True)
        panel(ax, letter)
    save(fig, out, "fig3_refusal_causes")


# ------------------------------------------------------------------------------------------------ F4 energy
def fig_energy(R, out):
    S = R["standing"]
    if "energy only" not in S:
        return
    fig = plt.figure(figsize=(TEXT, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.05, 1.0], wspace=.45)
    ax = fig.add_subplot(gs[0])
    cases = [("arm×1 seed 1", "arm×1"), ("arm×30 seed 1", "arm×30"), ("arm×1 seed 2", "arm×1"), ("arm×30 seed 2", "arm×30"),
             ("α250", "α 250"), ("α500", "α 500"), ("α1000", "α 1000")]
    cases = [c for c in cases if c[0] in S]
    x = np.array([0, 1, 2.6, 3.6, 5.4, 6.4, 7.4][:len(cases)])
    xs = []
    pos = {"arm×1 seed 1": 0, "arm×30 seed 1": 1, "arm×1 seed 2": 2.5, "arm×30 seed 2": 3.5, "α250": 5.4, "α500": 6.4, "α1000": 7.4}
    bot = np.zeros(len(cases)); X = np.array([pos[c[0]] for c in cases])
    for key, color, lab in (("arm_first", ENERGY, "arm-first"), ("body_first", REFLEX, "body-first"), ("failure", GREY, "failure")):
        v = np.array([S[c[0]][key] for c in cases])
        ax.bar(X, v, .8, bottom=bot, color=color, label=lab, lw=0); bot += v
    for xi, c in zip(X, cases):
        ax.text(xi, bot[list(X).index(xi)] + 1.5, f"{S[c[0]]['feet_moved_worlds']:.0f}%", ha="center", fontsize=6, color=INK)
    ax.set_xticks(X, [c[1] for c in cases], fontsize=6, rotation=45, ha="right")
    for xc, lab in ((.5, "price ratio ×30, seed 1"), (3., "ratio ×30, seed 2"), (6.4, "heat α (W), seed 1")):
        ax.text(xc, -.30, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.3, color=QUIET)
    ax.set_ylabel("threatened events (%)"); ax.set_ylim(0, 75)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=6.3, handlelength=.8, columnspacing=.8, bbox_to_anchor=(.5, 1.12))
    ax.text(.99, .90, "numbers: % of worlds with a foot moved", transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=INK)
    panel(ax, "a")
    ax = fig.add_subplot(gs[1])
    for key, color, lab in (("energy only", ENERGY, "energy price only"), ("contact only", PLAN, "contact cost only")):
        f = np.sort(np.array(S[key]["feet_cm"]))
        ax.step(f, np.arange(1, len(f) + 1) / len(f), color=color, lw=1.2, label=f"{lab} (p90 {S[key]['feet_p90']:.1f} cm)")
    ax.axvline(5, color=GRID, lw=.8, ls="--"); ax.text(5.1, .05, "5 cm\nrule", fontsize=6, color=QUIET)
    ax.set_xlim(0, 6); ax.set_xlabel("max foot displacement per threat (cm)"); ax.set_ylabel("fraction of threatened events")
    ax.legend(frameon=False, loc="center right", fontsize=6.2, bbox_to_anchor=(1.04, .45)); ax.grid(color=GRID, lw=.4)
    panel(ax, "b")
    ax = fig.add_subplot(gs[2])
    st = S["_reference"]["strategies"]
    names = [("arm", "arm"), ("body", "lean /\ntwist"), ("step", "side\nstep")]
    for i, (kind, c, off) in enumerate((("mech", PLAN, -.19), ("heat", ENERGY, .19))):
        med = [st[k][kind]["median"] for k, _ in names]
        lo = [st[k][kind]["iqr"][0] for k, _ in names]; hi = [st[k][kind]["iqr"][1] for k, _ in names]
        ax.bar(np.arange(3) + off, med, .36, color=c, label="mechanical" if kind == "mech" else "size-weighted heat",
               yerr=[np.array(med) - lo, np.array(hi) - med], error_kw=dict(lw=.6, capsize=1, ecolor=INK))
    ax.set_yscale("log"); ax.set_xticks(range(3), [n for _, n in names], fontsize=6.5); ax.set_ylabel("minimum energy to clear (J)")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left"); ax.grid(axis="y", color=GRID, lw=.4); ax.set_axisbelow(True)
    k = S["_reference"]["k_phys"]
    ax.set_xlabel(f"paired lean / arm: {k['mech']['median']:.1f}× mech, {k['heat']['median']:.1f}× heat", fontsize=6.5, labelpad=1)
    panel(ax, "c")
    save(fig, out, "fig4_energy")


# ------------------------------------------------------------------------------------------------ F5-F7 (A1-A3)
def fig_a1(R, out):
    if not R.get("a1"):
        return
    fig, axes = plt.subplots(1, 3, figsize=(TEXT, 2.4), gridspec_kw=dict(wspace=.38))
    cols = {"Planner–reflex": PLAN, "Reflex off": SKY, "Reactive (V7r)": INK}
    for ax, sp in zip(axes[:2], ("0.15", "0.3")):
        e = R["a1"].get(sp)
        if not e:
            continue
        tiers = ("0.4", "0.8", "1.2")
        names = [n for n in cols if n in e["policies"]]
        w = .8 / len(names)
        for i, n in enumerate(names):
            vals = [e["policies"][n]["tiers"][t]["hand"] for t in tiers]
            ax.bar(np.arange(3) + (i - (len(names) - 1) / 2) * w, [v[0] for v in vals], w * .92, color=cols[n], label=n,
                   yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], error_kw=dict(lw=.6, capsize=1, ecolor=INK))
        ax.set_xticks(range(3), [f"{t} m/s" for t in tiers]); ax.set_xlabel("object speed"); ax.set_ylabel("hands hit per throw (%)")
        ax.text(.5, 1.03, f"walking at {sp} m/s", transform=ax.transAxes, ha="center"); ax.grid(axis="y", color=GRID, lw=.4); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper left")
    panel(axes[0], "a"); panel(axes[1], "b")
    ax = axes[2]
    rows = []
    for sp in ("0.15", "0.3"):
        e = R["a1"].get(sp, {})
        for dn in ("Planner–reflex − Reflex off", "Planner–reflex − Reactive (V7r)"):
            if dn in e.get("deltas", {}):
                rows.append((f"{sp} m/s\n{dn.split(' − ')[1].replace(' (V7r)', '')}", e["deltas"][dn]["obj_hand"]))
    for i, (lab, d) in enumerate(rows):
        ax.errorbar(d["mean"], i, xerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]], fmt="o", color=PLAN, capsize=2, ms=4)
    ax.axvline(0, color=INK, lw=.6); ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6.2); ax.invert_yaxis()
    ax.set_xlabel("Δ object hand contact (pp)"); panel(ax, "c")
    save(fig, out, "fig5_dynamic_threats_walking")


def fig_a2(R, out):
    if not R.get("a2"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 2.4), gridspec_kw=dict(wspace=.3))
    cols = {"Planner–reflex": PLAN, "Reflex off": SKY, "Reactive (V7r)": INK}
    for ax, sp in zip(axes, ("0.15", "0.3")):
        e = R["a2"].get(sp)
        if not e:
            continue
        settings = [("nominal", "nominal"), ("nopush", "robust"), ("push", "robust + pushes")]
        names = [n for n in cols if any(f"{n}|{s}" in e["cells"] for s, _ in settings)]
        w = .8 / len(names)
        for i, n in enumerate(names):
            vals = [e["cells"].get(f"{n}|{s}", {}).get("clean") for s, _ in settings]
            xs = np.arange(3) + (i - (len(names) - 1) / 2) * w
            for x, v in zip(xs, vals):
                if v:
                    ax.bar(x, v[0], w * .92, color=cols[n], yerr=[[v[0] - v[1]], [v[2] - v[0]]], error_kw=dict(lw=.6, capsize=1, ecolor=INK),
                           label=n if x == xs[0] else None)
        ax.set_xticks(range(3), [l for _, l in settings]); ax.set_ylabel("clean traversal (%)"); ax.set_ylim(0, 100)
        ax.text(.5, 1.03, f"384 scenes, {sp} m/s", transform=ax.transAxes, ha="center"); ax.grid(axis="y", color=GRID, lw=.4); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=6.3, loc="lower left", ncol=1); panel(axes[0], "a"); panel(axes[1], "b")
    save(fig, out, "fig6_robustness")


def fig_a3(R, out):
    e4 = R["e4"]
    if not any("Planner–reflex, plan off" in e4.get(sp, {}).get("policies", {}) for sp in e4):
        return
    fig, axes = plt.subplots(1, 3, figsize=(TEXT, 2.3), gridspec_kw=dict(wspace=.38))
    spd = [sp for sp in ("0.15", "0.3", "0.5") if sp in e4]
    for ax, (key, ylabel, fam) in zip(axes, (("clean", "clean traversal (%)", None), ("time_to_goal", "time to clean goal (s)", None), ("palm_cm", "min palm clearance (cm)", None))):
        for n, c, mk in (("Planner–reflex", PLAN, "o"), ("Planner–reflex, plan off", REFLEX, "P"), ("Reactive (V7r)", INK, "D")):
            pts = [(float(sp), e4[sp]["policies"][n][key]) for sp in spd if n in e4[sp]["policies"] and e4[sp]["policies"][n].get(key)]
            if not pts:
                continue
            xs = [p[0] for p in pts]; v = [p[1] for p in pts]
            ax.errorbar(xs, [a[0] for a in v], yerr=[[a[0] - a[1] for a in v], [a[2] - a[0] for a in v]], marker=mk, color=c, ms=4, lw=1, capsize=1.5, label=n)
        ax.set_xticks([float(s) for s in spd]); ax.set_xlabel("operator speed (m/s)"); ax.set_ylabel(ylabel); ax.grid(color=GRID, lw=.4)
    n_on = e4[spd[0]]["policies"]["Planner–reflex"]["seeds"]; n_off = e4[spd[0]]["policies"]["Planner–reflex, plan off"]["seeds"]
    handles, labels = axes[0].get_legend_handles_labels()
    labels = [f"{l} ({n_on if l == 'Planner–reflex' else n_off if 'plan off' in l else 1} seed{'s' if (n_on if l == 'Planner–reflex' else n_off if 'plan off' in l else 1) > 1 else ''})" for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.1))
    for a, l in zip(axes, "abc"):
        panel(a, l)
    save(fig, out, "fig7_plan_input_speed_sweep")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    R = json.loads(a.data.read_text())
    fig_scatter(R, a.out, ("0.3", "0.15", "0.5"))
    fig_families(R, a.out)
    fig_refusal(R, a.out)
    fig_energy(R, a.out)
    fig_a1(R, a.out); fig_a2(R, a.out); fig_a3(R, a.out)


if __name__ == "__main__":
    main()
