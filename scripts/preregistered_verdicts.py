#!/usr/bin/env python3
"""Apply the pre-registered pass criteria of docs/eval_overnight.md to results.json."""
import json, sys
R = json.load(open(sys.argv[1]))
out = []
# A1 H1a: planner-reflex minus reflex-off, pooled over speeds: obj hand-contact <= -2 pp with CI upper < 0; clean lower bound >= -2 pp
for sp, e in sorted(R.get("a1", {}).items()):
    d = e.get("deltas", {}).get("Planner–reflex − Reflex off")
    if d:
        h, c = d["obj_hand"], d["clean"]
        ok = h["mean"] <= -2 and h["hi"] < 0 and c["lo"] >= -2
        out.append(f"A1 H1a @{sp}: object hand contact {h['mean']:+.2f} [{h['lo']:+.2f}, {h['hi']:+.2f}] pp, clean {c['mean']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] -> {'PASS' if ok else 'FAIL'}")
    d = e.get("deltas", {}).get("Planner–reflex − Reactive (V7r)")
    if d:
        h = d["obj_hand"]
        out.append(f"A1 H1b @{sp}: object hand contact vs reactive {h['mean']:+.2f} [{h['lo']:+.2f}, {h['hi']:+.2f}] pp -> {'PASS' if h['hi'] < 2 else 'FAIL'} (non-inferiority upper < +2)")
# A2
for sp, e in sorted(R.get("a2", {}).items()):
    c = e["cells"]
    for setting in ("nopush", "push"):
        a, n, r = c.get(f"Planner–reflex|{setting}"), c.get("Planner–reflex|nominal"), c.get("Reactive (V7r)|" + setting)
        if a and n:
            ok1 = a["clean"][0] >= 0.9 * n["clean"][0]
            msg = f"A2 H2 @{sp} {setting}: clean {a['clean'][0]:.1f} vs 0.9*nominal {0.9 * n['clean'][0]:.1f} -> {'ok' if ok1 else 'fail'}"
            if r:
                gap = a["clean"][0] - r["clean"][0]; cgap = a["contact"][0] - r["contact"][0]
                msg += f"; vs reactive clean {gap:+.1f} pp (need >= -3), contact {cgap:+.1f} pp (need <= +3)"
                ok1 = ok1 and gap >= -3 and cgap <= 3
            out.append(msg + f" -> {'PASS' if ok1 else 'FAIL'}")
# A3
e5 = R["e4"].get("0.5", {})
d = e5.get("deltas", {}).get("Planner–reflex − Planner–reflex, plan off")
if d:
    u = d["u_turn_clean"]
    ok = u["mean"] >= 5 and u["lo"] > 0
    out.append(f"A3 H3 @0.5: plan on - off, U + early-turn clean {u['mean']:+.2f} [{u['lo']:+.2f}, {u['hi']:+.2f}] pp -> {'PASS' if ok else 'FAIL'}; overall clean {d['clean']['mean']:+.2f} [{d['clean']['lo']:+.2f}, {d['clean']['hi']:+.2f}], hands {d['hands']['mean']:+.2f} [{d['hands']['lo']:+.2f}, {d['hands']['hi']:+.2f}]")
print("\n".join(out) if out else "no pre-registered result available yet")
