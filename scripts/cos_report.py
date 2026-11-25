#!/usr/bin/env python3
"""Yield and quality report of COS scenes v1 over one or more scene folders (docs/cos_obstacles.md).

  python scripts/cos_report.py --scenes DIR1 DIR2 ... --libs LIB1 LIB2 ... --out REPORT_DIR [--figure]

Writes REPORT_DIR/yield.csv (per source x family: clips, hours, clean events, events with >= 1 fit, scenes, verified,
scenes per mocap hour), REPORT_DIR/quality.csv (per family: penetration / clearance / slack percentiles, realism share,
kinds, tracked share, tracked-trajectory clear share), REPORT_DIR/manifest.json and, with --figure, quality.png.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

FAMS = ["duck", "crawl", "high_step", "sideways", "arms_in", "arms_tucked", "arms_raised", "detour", "multi"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True); ap.add_argument("--libs", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hours, clips, events = collections.Counter(), collections.Counter(), collections.Counter()
    for lib in a.libs:
        for l in open(Path(lib) / "clips.jsonl"):
            c = json.loads(l); hours[c["source"]] += c["seconds"] / 3600; clips[c["source"]] += 1
        for l in open(Path(lib) / "events.jsonl"):
            e = json.loads(l)
            if e["clean"]:
                events[(e["source"], e["family"])] += 1
    S, A, W = [], [], []
    for d in a.scenes:
        S += [json.loads(l) for l in open(Path(d) / "scenes.jsonl")]
        A += [json.loads(l) for l in open(Path(d) / "attempts.jsonl")]
        W += [json.loads(l) for l in open(Path(d) / "windows.jsonl")]
    fitted = collections.Counter((x["source"], x["family"]) for x in A if x["accepted"] > 0)
    tried = collections.Counter((x["source"], x["family"]) for x in A)
    sc = collections.Counter((s["source"], s["family"]) for s in S)
    ver = collections.Counter((s["source"], s["family"]) for s in S if s["verified"])
    sources = sorted(hours)
    rows = []
    for src in sources:
        for f in FAMS:
            ev_f = "sideways" if f == "arms_in" else f
            if not (events[(src, ev_f)] or sc[(src, f)]):
                continue
            rows.append(dict(source=src, family=f, clips=clips[src], hours=round(hours[src], 2),
                             clean_events=events[(src, ev_f)] if f not in ("arms_in", "multi") else "",
                             events_tried=tried[(src, ev_f)] if f not in ("arms_in", "multi") else "",
                             events_with_fit=fitted[(src, ev_f)] if f not in ("arms_in", "multi") else "",
                             scenes=sc[(src, f)], verified=ver[(src, f)],
                             scenes_per_hour=round(ver[(src, f)] / max(hours[src], 1e-6), 1)))
    with open(out / "yield.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    q = []
    for f in FAMS:
        X = [s for s in S if s["family"] == f and s["verified"]]
        if not X:
            continue
        pen = np.array([s["quality"]["d_pen"] for s in X]); clr = np.array([s["quality"]["m_clear"] for s in X])
        slack = np.array([s["quality"]["slack_main"] for s in X])
        tr = [s["tracked"] for s in X if s["tracked"] is not None]
        tc = [s["tracked_clear"] for s in X if s["tracked_clear"] is not None]
        kinds = collections.Counter(k for s in X for k in s["kinds"])
        q.append(dict(family=f, scenes=len(X), windows=len({s["window"] for s in X}), clips=len({s["clip"] for s in X}),
                      pen_p10=round(float(np.percentile(pen, 10)), 3), pen_p50=round(float(np.median(pen)), 3),
                      pen_p90=round(float(np.percentile(pen, 90)), 3), clear_p50=round(float(np.median(clr)), 3),
                      clear_p90=round(float(np.percentile(clr, 90)), 3), slack_p50=round(float(np.median(slack)), 3),
                      realistic=round(float(np.mean([s["quality"]["realistic"] for s in X])), 3),
                      tracked=round(float(np.mean(tr)), 3) if tr else "", tracked_known=len(tr),
                      tracked_clear=round(float(np.mean(np.array(tc) >= 0)), 3) if tc else "",
                      kinds=" ".join(f"{k}:{v}" for k, v in kinds.most_common())))
    with open(out / "quality.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(q[0])); wr.writeheader(); wr.writerows(q)
    manifest = dict(schema="cos-scenes-v1", scene_folders=a.scenes, libraries=a.libs,
                    hours={k: round(v, 2) for k, v in hours.items()}, clips=dict(clips),
                    scenes=len(S), verified=sum(s["verified"] for s in S),
                    windows=len(W), windows_skipped=dict(collections.Counter(w.get("skip") for w in W if w.get("skip"))),
                    by_family={r["family"]: r["scenes"] for r in q},
                    fields=dict(scene="scenes.jsonl: boxes [{center, half, yaw, role, kind}] in the chord frame "
                                      "(origin = window start pelvis, x toward the window end); motions/<window>.npz: M "
                                      "(retargeted G1 qpos, chord frame), D (default motion), tracked (SONIC rollout, "
                                      "if tracked)", tracked="SONIC lite-harness feasibility flag of the window",
                                tracked_clear="min clearance of the tracked rollout to all boxes (m)"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest)[:1500])
    for r in q:
        print(r)
    if a.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fams = [r["family"] for r in q]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for k, (key, title) in enumerate((("d_pen", "explanation strength: default penetration (cm)"),
                                          ("m_clear", "tightness: real-motion clearance (cm)"),
                                          ("slack_main", "clearance to the explaining obstacle (cm)"))):
            data = [100 * np.array([s["quality"][key] for s in S if s["family"] == f and s["verified"]]) for f in fams]
            ax[k].boxplot(data, vert=False, showfliers=False)
            ax[k].set_yticks(range(1, len(fams) + 1)); ax[k].set_yticklabels(fams if k == 0 else [""] * len(fams))
            ax[k].set_title(title, fontsize=10); ax[k].grid(axis="x", alpha=.3)
        fig.tight_layout(); fig.savefig(out / "quality.png", dpi=130)


if __name__ == "__main__":
    main()
