#!/usr/bin/env python3
"""B1: export the C0-cleaned, levelled COS motion windows as BlinkMimic motions (docs/cos_blinkmimic.md, B1).

  python scripts/cos_bm_export.py --manifest /l/.../cos/scenes_v2/manifest.json --splits /l/.../cos/splits_v2 \
      --out /l/.../cos/tracker/motions_v1

One file per unique window that keeps >= 1 verified scene after C0: OUT/<split>/<window>.npz with qpos [T, 36] (50 Hz MuJoCo
qpos of the training G1: root position, root quaternion wxyz, 29 joints in Unitree SDK order = BlinkMimic's order), fps,
dz (C0 levelling), take, source, families. Split by take (C1). Writes OUT/summary.json (clips, hours per split / family).
"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--splits", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-seconds", type=float, default=2.0)
    a = ap.parse_args()
    from cos_splits import take_of
    man = json.load(open(a.manifest)); takes = json.load(open(Path(a.splits) / "take_split.json"))
    out = Path(a.out)
    win = {}
    for folder in man["scene_folders"]:
        for l in open(Path(folder) / "scenes.jsonl"):
            s = json.loads(l)
            if not s.get("verified"):
                continue
            w = win.setdefault(s["window"], dict(folder=folder, clip=s["clip"], source=s["source"], families=set(),
                                                 split=takes[take_of(s["clip"])], take=take_of(s["clip"])))
            w["families"].add(s["family"])
    S = collections.defaultdict(lambda: dict(clips=0, frames=0)); F = collections.defaultdict(collections.Counter)
    for name, w in win.items():
        d = np.load(Path(w["folder"]) / "motions" / f"{name}.npz")
        q = d["M"].astype(np.float32)
        if len(q) * 0.02 < a.min_seconds:
            continue
        (out / w["split"]).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / w["split"] / f"{name}.npz", qpos=q, fps=np.float32(50.0), dz=np.float32(d["dz"]),
                            take=w["take"], source=w["source"], families=json.dumps(sorted(w["families"])))
        S[w["split"]]["clips"] += 1; S[w["split"]]["frames"] += len(q)
        for f in w["families"]:
            F[w["split"]][f] += 1
    summ = {sp: dict(clips=v["clips"], hours=round(v["frames"] * 0.02 / 3600, 3), takes=len({w["take"] for w in win.values() if w["split"] == sp}),
                     windows_by_family=dict(F[sp])) for sp, v in S.items()}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
