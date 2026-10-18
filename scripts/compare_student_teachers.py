#!/usr/bin/env python3
"""Student vs teachers, per skill (the distillation pass test): clean success on the SAME scenes.

For every skill tag, the student's clean rate over the scenes it was taught is compared with the routed teacher's
rate on those scenes (scripts/eval_per_scene.py results, deterministic). Pass: within --tolerance (default 5 points)
of the teacher on every tag with enough episodes. Dropped scenes (no competent teacher) are reported separately.

Tags (no GPU needed): published types (pub_hurdle, pub_side_crouch, ...), original, procedural by narrowest opening
(outputs/narrow_feasibility_sidegap_v2.json: proc_narrow < 0.45 m, proc_mid < 0.77 m, proc_open), side gaps by
layout and width band, room scene families (furniture / clutter / table edges / passages / flat).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tags_of(scene, narrowest):
    sid, family = scene["scene_id"], scene["family"]
    source = scene.get("source", {}) or {}
    if sid.startswith("published-"):
        return ["pub_" + sid[len("published-"):].rstrip("0123456789").replace("-", "_")]
    if family == "original_cat":
        return ["original"]
    if source.get("kind") == "side-gap":
        info = source.get("side_gap", {})
        w = min(info.get("gap_m", 9.), info.get("gap2_m", 9.))
        return ["gap_" + str(info.get("variant", "v1")), "gapw_lt040" if w < .40 else "gapw_040_050" if w < .50 else "gapw_ge050"]
    if scene.get("task_kind") == "cat":
        n = narrowest.get(sid, 9.)
        return ["proc_narrow" if n < .45 else "proc_mid" if n < .77 else "proc_open"]
    if sid == "flat-balance-walk-v1":
        return ["flat"]
    kind = source.get("kind", "")
    role = (source.get("hand_contrast") or {}).get("role")
    if kind == "table-edge-passage":
        return ["room_table_edges"]
    if role:
        return ["room_passage_" + role]
    return ["room_" + family]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--routing", required=True)
    p.add_argument("--student", action="append", required=True, help="student eval_per_scene.py JSON (repeat: cat, room)")
    p.add_argument("--bank", default="data/furniture/sidegap_v2_packed/manifest.json")
    p.add_argument("--feasibility", default="outputs/narrow_feasibility_sidegap_v2.json")
    p.add_argument("--tolerance", type=float, default=.05)
    p.add_argument("--min-episodes", type=int, default=48)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    routing = json.loads(Path(args.routing).read_text())
    teachers = {name: json.loads(Path(path).read_text())["per_scene"] for name, path in routing["evals"].items()}
    student = {}
    for path in args.student:
        student.update(json.loads(Path(path).read_text())["per_scene"])
    narrowest = {k: v["narrowest_m"] for k, v in json.loads((ROOT / args.feasibility).read_text()).items()}
    scenes = json.loads((ROOT / args.bank).read_text())["scenes"]
    owner, dropped = routing["owner"], set(routing["dropped"])
    acc = collections.defaultdict(lambda: [0., 0., 0., 0., 0.])   # episodes, student clean, teacher clean, student hands, teacher hands
    acc_dropped = collections.defaultdict(lambda: [0., 0.])
    missing = 0
    for s in scenes:
        sid = s["scene_id"]
        if sid not in student:
            missing += sid in owner; continue
        st = student[sid]
        for tag in tags_of(s, narrowest):
            if sid in owner:
                te = teachers[owner[sid]][sid]; e = st["episodes"]
                a = acc[tag]; a[0] += e; a[1] += st["clean"] * e; a[2] += te["clean"] * e
                a[3] += st.get("hands", 0.) * e; a[4] += te.get("hands", 0.) * e
            elif sid in dropped:
                d = acc_dropped[tag]; d[0] += st["episodes"]; d[1] += st["clean"] * st["episodes"]
    rows, fails = [], []
    for tag in sorted(acc):
        e, sc, tc, sh, th = acc[tag]
        if not e:
            continue
        row = dict(tag=tag, episodes=int(e), student=round(sc / e, 3), teacher=round(tc / e, 3), gap=round((sc - tc) / e, 3),
                   student_hands=round(sh / e, 3), teacher_hands=round(th / e, 3))
        row["pass"] = e < args.min_episodes or row["gap"] >= -args.tolerance
        rows.append(row)
        if not row["pass"]:
            fails.append(tag)
    drop_rows = [dict(tag=t, episodes=int(v[0]), student=round(v[1] / v[0], 3)) for t, v in sorted(acc_dropped.items()) if v[0]]
    total = [sum(acc[t][i] for t in acc) for i in range(3)]
    result = dict(pass_all=not fails, failing=fails, overall=dict(student=round(total[1] / max(1, total[0]), 3), teacher=round(total[2] / max(1, total[0]), 3)),
                  per_tag=rows, dropped_scenes=drop_rows, owned_scenes_missing_from_student_eval=missing)
    print(f"{'tag':24s} {'eps':>6s} {'student':>8s} {'teacher':>8s} {'gap':>7s} {'hands s/t':>12s}")
    for r in rows:
        print(f"{r['tag']:24s} {r['episodes']:6d} {r['student']:8.3f} {r['teacher']:8.3f} {r['gap']:+7.3f} {r['student_hands']:5.3f}/{r['teacher_hands']:5.3f} {'' if r['pass'] else 'FAIL'}")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_tag",)}, indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
