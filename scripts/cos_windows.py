#!/usr/bin/env python3
"""Candidate windows for G1/G2 from the G0 events: clean events, event +- pad s, merged per clip; one id per window."""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--events", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--pad", type=float, default=1.5); ap.add_argument("--min-s", type=float, default=3.0)
ap.add_argument("--max-s", type=float, default=12.0)
a = ap.parse_args()
E = json.loads(Path(a.events).read_text())["events"]
by = {}
for x in E:
    if x["clean"]:
        by.setdefault(x["clip"], []).append(x)
W = []
for c, ev in sorted(by.items()):
    ev.sort(key=lambda x: x["t0"])
    n = int(round(ev[0]["clip_seconds"] / .02))
    groups = []
    for x in ev:
        a0, b0 = max(0, int((x["t0"] - a.pad) / .02)), min(n, int((x["t1"] + a.pad) / .02))
        if groups and a0 <= groups[-1]["f1"] and (b0 - groups[-1]["f0"]) * .02 <= a.max_s:
            g = groups[-1]; g["f1"] = max(g["f1"], b0); g["events"].append([x["family"], x["t0"], x["t1"], x["mag"]])
        else:
            groups.append(dict(f0=a0, f1=b0, events=[[x["family"], x["t0"], x["t1"], x["mag"]]]))
    for k, g in enumerate(groups):
        if (g["f1"] - g["f0"]) * .02 < a.min_s:                   # extend short windows symmetrically
            need = int(a.min_s / .02) - (g["f1"] - g["f0"])
            g["f0"] = max(0, g["f0"] - need // 2); g["f1"] = min(n, g["f1"] + need - need // 2)
        g["f1"] = min(g["f1"], g["f0"] + int(a.max_s / .02))
        W.append(dict(id=f"c{c}_w{k}", clip=c, clip_id=ev[0]["clip_id"], source=ev[0]["source"], frame0=ev[0]["frame0"],
                      families=sorted({e[0] for e in g["events"]}), **g))
Path(a.out).write_text(json.dumps(W))
import collections
print(len(W), "windows,", round(sum((w["f1"] - w["f0"]) * .02 for w in W) / 60, 1), "min;",
      collections.Counter(f for w in W for f in w["families"]))
