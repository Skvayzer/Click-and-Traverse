#!/usr/bin/env python3
"""B3: exact-geometry clearance of BlinkMimic rollouts in the COS test scenes (35-primitive proxy, synth.Body.sep), plus the
per-family table with Wilson CIs, from ~/blinkmimic/tools/cos_obstacle_eval.py outputs.

  python scripts/cos_bm_b3_clearance.py --run DIR --scenes /l/.../splits_v2/test200.jsonl
"""
import argparse, collections, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--scenes", required=True)
    a = ap.parse_args()
    from cat_mjlab.cos.synth import Body
    from cat_mjlab.exact_geometry import ExactGeometry
    G = ExactGeometry(); hands = G.hand_shapes.tolist()
    run = Path(a.run); per = json.load(open(run / "per_scene.json")); R = np.load(run / "rollouts.npz")
    rows = {json.loads(l)["scene"]: json.loads(l) for l in open(a.scenes) if l.strip()}
    boxes = {}
    for folder in {r["folder"] for r in rows.values()}:
        for l in open(Path(folder) / "scenes.jsonl"):
            s = json.loads(l)
            if s["scene"] in rows:
                boxes[s["scene"]] = s["boxes"]
    q, alive = R["qpos"], R["alive"]
    for k, r in enumerate(per):
        idx = np.flatnonzero(alive[:, k])
        b = Body(G, q[idx, k].astype(np.float64))
        v, _ = b.sep(boxes[r["scene"]]); vh, _ = b.sep(boxes[r["scene"]], shapes=hands)
        r["min_clear_m"] = float(v.min()); r["min_hand_clear_m"] = float(vh.min())
        r["proxy_contact"] = bool(v.min() < 0)
    json.dump(per, open(run / "per_scene_clearance.json", "w"), indent=1)
    keys = ("clean", "contact", "hand_contact", "failed", "goal_reached", "proxy_contact")
    def block(rr):
        n = len(rr); d = dict(n=n, **{k: round(float(np.mean([x[k] for x in rr])), 3) for k in keys})
        d["clean_ci95"] = wilson(sum(x["clean"] for x in rr), n)
        d["median_min_clear_cm"] = round(100 * float(np.median([x["min_clear_m"] for x in rr])), 1)
        d["median_min_hand_clear_cm"] = round(100 * float(np.median([x["min_hand_clear_m"] for x in rr])), 1)
        return d
    out = {"all": block(per)}
    for f in sorted({r["family"] for r in per}):
        out[f] = block([r for r in per if r["family"] == f])
    json.dump(out, open(run / "b3_table.json", "w"), indent=1)
    for k, v in out.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
