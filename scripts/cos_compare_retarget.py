#!/usr/bin/env python3
"""Compare two G1 retargetings of the same AMASS sequences (our GMR vs ember-lab/fleaven), clip by clip.

  python scripts/cos_compare_retarget.py --a LIB_GMR --b LIB_EMBER --out OUT.json [--feas-a F.jsonl --feas-b F.jsonl]

Clips are matched on the normalized AMASS member (subset/subject/sequence without _stageii/_poses suffixes). Per clip:
joint-limit violation share (library check), foot skating (horizontal speed of a sole while it is within 2 cm of its
clip-wide lowest height, mean m/s), floor penetration (share of frames whose lowest proxy point is > 2 cm below the
clip's 5th-percentile sole height), and, if feasibility files are given, the SONIC tracked share of the clip's windows.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def norm_member(m):
    m = m.split(";")[0]
    m = re.sub(r"\.npz$|\.npy$", "", m)
    m = re.sub(r"_stageii$|_poses(_\d+)?(_jpos)?$", "", m)
    parts = m.replace("\\", "/").split("/")
    subset = parts[0] if parts[0] in ("CMU", "KIT", "BMLmovi", "SFU") else parts[-3] if len(parts) >= 3 else parts[0]
    return (subset + "/" + parts[-1]).lower().replace(" ", "").replace("-", "")


def clip_metrics(FF, q):
    F = FF(q)
    sole = F["foot_z"]                                   # [T, 2] lowest sole point
    ref = np.percentile(sole, 5)
    v = np.linalg.norm(np.gradient(F["foot_xy"], 0.02, axis=0), axis=-1)     # [T, 2]
    contact = sole < ref + 0.02
    skate = float((v * contact).sum() / max(contact.sum(), 1))
    lowest = np.minimum(sole.min(1), F["knee_z"] - 0.05)
    pen = float((lowest < ref - 0.02).mean())
    return dict(skate=skate, penetration=pen)


def load(lib):
    C = {}
    for l in open(Path(lib) / "clips.jsonl"):
        c = json.loads(l); C[norm_member(c["member"])] = c
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--feas-a"); ap.add_argument("--feas-b"); ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()
    from cat_mjlab.cos.features import FrameFeatures
    FF = FrameFeatures()
    A, B = load(a.a), load(a.b)
    common = sorted(set(A) & set(B))
    rng = np.random.default_rng(0)
    pick = [common[i] for i in rng.permutation(len(common))[:a.limit]]

    def feas(path, lib):
        if not path:
            return {}
        out = {}
        for l in open(path):
            m = json.loads(l); clip = m["id"].rsplit("__w", 1)[0]
            out.setdefault(clip, []).append(bool(m.get("tracked")))
        return out

    fa, fb = feas(a.feas_a, a.a), feas(a.feas_b, a.b)
    rows = []
    for k in pick:
        r = dict(member=k)
        for tag, lib, C, fe in (("a", a.a, A, fa), ("b", a.b, B, fb)):
            c = C[k]
            q = np.asarray(np.load(Path(lib) / f"qpos_{c['shard']:03d}.npy", mmap_mode="r")[c["start"]:c["start"] + c["frames"]])
            m = clip_metrics(FF, q)
            r[tag] = dict(joint_limit_violation=c["joint_limit_violation"], seconds=c["seconds"], **m,
                          tracked=(float(np.mean(fe[c["clip_id"]])) if c["clip_id"] in fe else None))
        rows.append(r)
    summ = {}
    for tag in ("a", "b"):
        for key in ("joint_limit_violation", "skate", "penetration", "tracked"):
            v = [r[tag][key] for r in rows if r[tag][key] is not None]
            summ[f"{tag}/{key}"] = round(float(np.mean(v)), 4) if v else None
    Path(a.out).write_text(json.dumps(dict(common=len(common), compared=len(rows), summary=summ, a=a.a, b=a.b, rows=rows), indent=1))
    print(json.dumps(dict(common=len(common), compared=len(rows), summary=summ)))


if __name__ == "__main__":
    main()
