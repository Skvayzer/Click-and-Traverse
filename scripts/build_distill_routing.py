#!/usr/bin/env python3
"""Per-scene teacher routing for three-teacher DAgger (traversal / narrow / rooms) from per-scene evaluations.

Inputs: scripts/eval_per_scene.py results (deterministic, K episodes per scene) for each expert, plus the
exclusion lists the CAT experts were trained with. Rules:
  * CAT scenes: the default owner is the expert that TRAINED on the scene (traversal unless the scene is in
    configs/pilots/traversal_excluded_scenes.txt, else narrow). The other expert takes the scene only when its clean
    success is higher by more than --margin (12 episodes per scene: a 1-2 episode difference is noise).
  * room scenes (furniture, clutter, table edges, passages, flat): the rooms expert.
  * a scene whose owner's clean success is below --min-clean is DROPPED from distillation (zero sampling weight):
    imitating a teacher that fails there teaches failure (e.g. published-side-crouch2, narrow expert 0%); the RL
    fine-tune stage learns those.
Output JSON: {"owner": {scene_id: expert}, "dropped": [...], "summary": ...}; plus <out>.dropped.txt for
--zero-weight-scenes @file.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", action="append", required=True, metavar="NAME=FILE", help="expert name = eval_per_scene.py JSON")
    p.add_argument("--bank", default="data/furniture/sidegap_v2_packed/manifest.json")
    p.add_argument("--traversal-excluded", default="configs/pilots/traversal_excluded_scenes.txt")
    p.add_argument("--min-clean", type=float, default=.20)
    p.add_argument("--margin", type=float, default=.15)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    evals = {}
    for item in args.eval:
        name, path = item.split("=", 1)
        evals[name] = json.loads(Path(path).read_text())["per_scene"]
    scenes = json.loads((ROOT / args.bank).read_text())["scenes"]
    excluded = {l.strip() for l in open(ROOT / args.traversal_excluded) if l.strip()}
    owner, dropped, reasons = {}, [], collections.Counter()
    best_clean = {}
    for s in scenes:
        sid = s["scene_id"]
        if s.get("task_kind") == "cat":
            default = "narrow" if sid in excluded else "traversal"
            other = "traversal" if default == "narrow" else "narrow"
            a = evals[default].get(sid, {}).get("clean"); b = evals[other].get(sid, {}).get("clean")
            if a is None or b is None:
                raise ValueError(f"{sid}: missing evaluation ({default}={a}, {other}={b})")
            chosen = other if b > a + args.margin else default
            reasons[f"{default}->{chosen}"] += 1
        else:
            chosen = "rooms"
            if sid not in evals["rooms"]:
                raise ValueError(f"{sid}: missing rooms evaluation")
            reasons["rooms"] += 1
        clean = evals[chosen][sid]["clean"]
        best_clean[sid] = clean
        if clean < args.min_clean:
            dropped.append(sid); reasons[f"dropped_{chosen}"] += 1
        else:
            owner[sid] = chosen
    by_owner = collections.Counter(owner.values())
    mean_clean = {n: round(sum(best_clean[k] for k, v in owner.items() if v == n) / max(1, by_owner[n]), 3) for n in by_owner}
    summary = dict(owners=dict(by_owner), dropped=len(dropped), decisions=dict(reasons), mean_clean_of_owned=mean_clean,
                   min_clean=args.min_clean, margin=args.margin)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(summary=summary, owner=owner, dropped=sorted(dropped),
                                        clean_of_owner=best_clean, evals={i.split('=', 1)[0]: i.split('=', 1)[1] for i in args.eval}), indent=1) + "\n")
    Path(str(args.out) + ".dropped.txt").write_text("\n".join(sorted(dropped)) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
