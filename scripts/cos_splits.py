#!/usr/bin/env python3
"""C1 splits of COS scenes v2 by source take (docs/cos_planner_reflex.md, C1).

  python scripts/cos_splits.py --manifest /l/.../cos/scenes_v2/manifest.json --out /l/.../cos/splits_v2
"""
import argparse, collections, hashlib, json, random, re
from pathlib import Path

FAMS = ("duck", "crawl", "high_step", "sideways", "arms_in", "arms_tucked", "arms_raised", "detour", "multi")


def take_of(clip):
    s = re.sub(r"(__r\d+)$", "", clip)
    return s


def split_of(take):
    h = int(hashlib.sha1(take.encode()).hexdigest(), 16) % 100
    return "test" if h < 10 else ("val" if h < 15 else "train")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    man = json.load(open(a.manifest)); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    takes = {}; cnt = collections.defaultdict(collections.Counter); cand = collections.defaultdict(list)
    for folder in man["scene_folders"]:
        for l in open(Path(folder) / "scenes.jsonl"):
            s = json.loads(l)
            if not s.get("verified"):
                continue
            t = take_of(s["clip"]); sp = split_of(t); takes[t] = sp
            cnt[sp][s["family"]] += 1
            if sp == "test" and s.get("tracked") and s.get("tracked_clear") is not None and s["tracked_clear"] >= 0:
                cand[s["family"]].append(dict(scene=s["scene"], window=s["window"], take=t, family=s["family"],
                                              source=s["source"], folder=folder))
    rng = random.Random(a.seed)
    for f in cand:
        rng.shuffle(cand[f])

    def draw(fam, k, used_w, per_take):
        got = []
        for c in cand[fam]:
            if len(got) >= k:
                break
            if c["window"] in used_w or per_take[c["take"]] >= 2 or c.get("_used"):
                continue
            got.append(c); used_w.add(c["window"]); per_take[c["take"]] += 1; c["_used"] = True
        return got
    used_w, per_take = set(), collections.Counter()
    quota = {f: a.n // len(FAMS) for f in FAMS}
    for f in sorted(FAMS, key=lambda f: -len(cand[f]))[: a.n - sum(quota.values())]:
        quota[f] += 1
    pick = {f: draw(f, quota[f], used_w, per_take) for f in FAMS}
    short = a.n - sum(len(v) for v in pick.values())
    while short > 0:
        progressed = False
        for f in sorted(FAMS, key=lambda f: -len(cand[f])):
            if short <= 0:
                break
            more = draw(f, 1, used_w, per_take)
            if more:
                pick[f] += more; short -= 1; progressed = True
        if not progressed:
            break
    with open(out / "test200.jsonl", "w") as fo:
        for f in FAMS:
            for c in pick[f]:
                c.pop("_used", None); fo.write(json.dumps(c) + "\n")
    json.dump(takes, open(out / "take_split.json", "w"))
    by_split = collections.Counter(takes.values())
    summ = dict(takes=dict(by_split), scenes={k: dict(v) for k, v in cnt.items()},
                test200={f: len(pick[f]) for f in FAMS}, test_candidates={f: len(cand[f]) for f in FAMS},
                overlap_check="takes are keys of one dict: each take has exactly one split")
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
