"""Report every class's time-matched excess by bucket and target hand."""
from collections import defaultdict
import json
import pathlib
import sys

GROUPS = ("legs", "waist", "arms")
MODELS = ("mech", "heat")
SOURCES = ("policy", "applied")
CLASSES = ("arm_first", "body_first", "still_success", "failure")
CHANNELS = [(s,m,g) for s in SOURCES for m in MODELS for g in GROUPS]

def extra(policy):
    cells = defaultdict(lambda: [0, 0., [0.]*len(CHANNELS)])
    for e in policy["approaches"]:
        if not e["seconds"]:
            continue
        cell = cells[(e["bucket"],e["hand_side"],e["response"])]
        cell[0] += 1
        cell[1] += e["seconds"]
        for i,(s,m,g) in enumerate(CHANNELS):
            cell[2][i] += e["energy"][s][m][g]["energy_j"]
    report = {}
    for cls in CLASSES:
        n, secs, excess = 0, 0., [0.]*len(CHANNELS)
        for (bucket,side,name),(count,t,energy) in tuple(cells.items()):
            base = cells.get((bucket,side,"still_success"))
            if name != cls or not base or not base[1]:
                continue
            n += count
            secs += t
            for i in range(len(excess)):
                excess[i] += energy[i] - t * base[2][i] / base[1]
        report[cls] = dict(matched_count=n, extra={s:{m:{g:{
            "energy_j": round(excess[i]/n,4) if n else None,
            "mean_w": round(excess[i]/secs,4) if secs else None}
            for i,(source,model,g) in enumerate(CHANNELS) if source==s and model==m}
            for m in MODELS} for s in SOURCES})
    return report

def main():
    values = json.loads(pathlib.Path(sys.argv[1]).read_text())
    report = []
    for p in values:
        report.append(dict(checkpoint=p["checkpoint"], counts={k:v["count"] for k,v in p["energy_by_response"].items()},
                           by_class=p["energy_by_response"], matched=p["matched_extra"], extra_by_class=extra(p)))
    print(json.dumps(report, indent=2, allow_nan=False))
if __name__ == "__main__":
    main()
