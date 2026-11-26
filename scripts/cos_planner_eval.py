#!/usr/bin/env python3
"""C2 kinematic evaluation of planners on held-out COS situations (exact geometry) and benchmark situations (grid).

  python scripts/cos_planner_eval.py --cos-data /l/.../cos/planner_data_v1 --bench-data /l/.../wbp/planner_v1/all \
      --checkpoint NAME=PATH [--checkpoint NAME2=PATH2] --out DIR --device cuda:0

Per situation: K = 4 scene-conditioned samples (goal = the reference chunk end), chosen = best grid score
(planner_eval.best_of). COS: exact 35-primitive proxy vs the scene's boxes (synth.Body.sep) over the 20 keyframes
and their midpoints (plus the start pose) -> collision if clearance < 0. Benchmarks: grid-sphere collision-free (tol 1 cm).
Same situations and generator seed for every checkpoint; paired bootstrap of the differences (situations resampled
by scene).
"""
import argparse, collections, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
FAMS = ("duck", "crawl", "high_step", "sideways", "arms_in", "arms_tucked", "arms_raised", "detour", "multi")


def boxes_local(boxes, origin, yaw):
    c, s = math.cos(yaw), math.sin(yaw); out = []
    for b in boxes:
        p = np.asarray(b["center"], float); dx, dy = p[0] - origin[0], p[1] - origin[1]
        out.append(dict(b, center=[c * dx + s * dy, -s * dx + c * dy, p[2]], yaw=float(b.get("yaw", 0.)) - yaw))
    return out


def boot(diff, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.nonzero(groups == g)[0] for g in ug}
    ms = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug)); ii = np.concatenate([idx[g] for g in pick]); ms.append(diff[ii].mean())
    return [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cos-data", required=True); ap.add_argument("--bench-data")
    ap.add_argument("--checkpoint", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n-cos", type=int, default=2000)
    ap.add_argument("--n-bench", type=int, default=1000); ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--ode-steps", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    import torch, mujoco
    from cat_mjlab.cos.synth import Body
    from cat_mjlab.exact_geometry import ExactGeometry, interpolate_frames
    from cat_mjlab.model import assemble_training_xml
    from cat_mjlab.planner.data import chunk_qpos, load_split, state_qpos
    from cat_mjlab.planner.metrics import ChunkScorer, diversity
    from cat_mjlab.planner.model import FlowPlanner, Normalizer, sample
    from planner_eval import best_of
    from planner_train import held_out_halves
    dev = torch.device(a.device); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    G = ExactGeometry(); scorer = ChunkScorer(mujoco.MjModel.from_xml_string(assemble_training_xml()), dev)
    rng = np.random.default_rng(a.seed)
    # ---- COS situations (stratified by family)
    cos = load_split(a.cos_data, "cos_test")
    fam = cos["meta"][:, 1].astype(int) - 100
    per = a.n_cos // len(FAMS); pick = []
    for f in range(len(FAMS)):
        ii = np.nonzero(fam == f)[0]; pick += rng.choice(ii, min(per, len(ii)), replace=False).tolist()
    rest = np.setdiff1d(np.arange(len(fam)), pick)
    pick += rng.choice(rest, max(0, a.n_cos - len(pick)), replace=False).tolist()
    pick = np.sort(np.array(pick)); cos = {k: v[pick] for k, v in cos.items()}; fam = fam[pick]
    index = json.load(open(Path(a.cos_data) / "scene_index.json"))
    need = collections.defaultdict(set)
    for sid in np.unique(cos["meta"][:, 0]):
        r = index[int(sid) - 1_000_000]; need[r["folder"]].add(r["scene"])
    boxes = {}
    for folder, ids in need.items():
        for l in open(Path(folder) / "scenes.jsonl"):
            s = json.loads(l)
            if s["scene"] in ids:
                boxes[s["scene"]] = s["boxes"]
    scene_of = [index[int(sid) - 1_000_000]["scene"] for sid in cos["meta"][:, 0]]
    sets = {"cos": cos}
    if a.bench_data:
        held = load_split(a.bench_data, "test")
        _, tm = held_out_halves(held["meta"], 0)
        ii = np.nonzero(tm)[0]; ii = np.sort(rng.choice(ii, min(a.n_bench, len(ii)), replace=False))
        sets["bench"] = {k: v[ii] for k, v in held.items()}
    res = {}; per_sit = {}
    hands = G.hand_shapes.tolist()
    for spec in a.checkpoint:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        model = FlowPlanner(**ck["config"]).to(dev).eval(); model.load_state_dict(ck["model"])
        norm = Normalizer(np.zeros((2, 67)), np.zeros((2, 5)), np.zeros((2, 20, 35))).to(dev); norm.load_state_dict(ck["norm"])
        g = torch.Generator(device=dev).manual_seed(a.seed)
        R = {}
        for setname, D in sets.items():
            n = len(D["state"]); recs = collections.defaultdict(list)
            for s0 in range(0, n, a.batch):
                b = {k: torch.as_tensor(np.ascontiguousarray(v[s0:s0 + a.batch]), device=dev) for k, v in D.items()}
                with torch.no_grad():
                    st, gl = norm.norm("state", b["state"]), norm.norm("goal", b["goal"])
                    xs = norm.denorm("chunk", sample(model, st, gl, b["grid"], use_scene=True, samples=a.samples,
                                                     steps=a.ode_steps, generator=g))
                    sc = scorer.score(xs, b["grid"], b["goal"])
                    ch = best_of(sc)
                recs["grid_free_chosen"].append((sc["min_clear"][torch.arange(len(ch)), ch] >= -0.01).cpu().numpy())
                recs["grid_free_single"].append((sc["min_clear"][:, 0] >= -0.01).cpu().numpy())
                recs["goal_err"].append(sc["goal_err"][torch.arange(len(ch)), ch].cpu().numpy())
                recs["diversity"].append(diversity(xs).cpu().numpy())
                if setname == "cos":
                    q = chunk_qpos(xs).cpu().numpy(); q0 = state_qpos(b["state"]).cpu().numpy()
                    chn = ch.cpu().numpy()
                    for j in range(len(q)):
                        k = s0 + j; pose = D["pose"][k]
                        bl = boxes_local(boxes[scene_of[k]], pose[:2], float(pose[2]))
                        clr = []; hcl = []
                        for m in range(q.shape[1]):
                            seq = np.concatenate((q0[j][None], q[j, m]), 0).astype(np.float64)
                            seq = interpolate_frames(seq, np.arange(0, 2.0001, 0.05), dt=0.1)
                            body = Body(G, seq); v, _ = body.sep(bl); clr.append(float(v.min()))
                            if hands:
                                hcl.append(float(body.sep(bl, shapes=hands)[0].min()))
                        recs["exact_clear_chosen"].append(np.array([clr[chn[j]]]))
                        recs["exact_clear_single"].append(np.array([clr[0]]))
                        recs["exact_any_free"].append(np.array([max(clr) >= 0]))
                        if hands:
                            recs["exact_hand_clear_chosen"].append(np.array([hcl[chn[j]]]))
            R[setname] = {k: np.concatenate(v) for k, v in recs.items()}
        per_sit[name] = R
        print(name, "done", flush=True)
    # summaries
    summ = {}
    for name, R in per_sit.items():
        S = {}
        c = R["cos"]
        def block(mask):
            d = dict(n=int(mask.sum()),
                     exact_collision_chosen=round(float((c["exact_clear_chosen"][mask] < 0).mean()), 4),
                     exact_collision_single=round(float((c["exact_clear_single"][mask] < 0).mean()), 4),
                     exact_any_of_k_free=round(float(c["exact_any_free"][mask].mean()), 4),
                     grid_free_chosen=round(float(c["grid_free_chosen"][mask].mean()), 4),
                     goal_err_m=round(float(c["goal_err"][mask].mean()), 4),
                     diversity_m=round(float(c["diversity"][mask].mean()), 4))
            if "exact_hand_clear_chosen" in c:
                d["exact_hand_collision_chosen"] = round(float((c["exact_hand_clear_chosen"][mask] < 0).mean()), 4)
                d["median_hand_clear_m"] = round(float(np.median(c["exact_hand_clear_chosen"][mask])), 4)
            return d
        S["cos_all"] = block(np.ones(len(fam), bool))
        for f in range(len(FAMS)):
            S[f"cos_{FAMS[f]}"] = block(fam == f)
        if "bench" in R:
            bb = R["bench"]
            S["bench"] = dict(n=len(bb["grid_free_chosen"]), grid_free_chosen=round(float(bb["grid_free_chosen"].mean()), 4),
                              grid_free_single=round(float(bb["grid_free_single"].mean()), 4),
                              goal_err_m=round(float(bb["goal_err"].mean()), 4), diversity_m=round(float(bb["diversity"].mean()), 4))
        summ[name] = S
    names = list(per_sit)
    if len(names) >= 2:
        A0, B0 = names[0], names[1]
        groups = cos["meta"][:, 0]
        d = (per_sit[B0]["cos"]["exact_clear_chosen"] < 0).astype(float) - (per_sit[A0]["cos"]["exact_clear_chosen"] < 0).astype(float)
        summ["paired"] = dict(diff=f"{B0} - {A0} exact collision (chosen)", mean=round(float(d.mean()), 4), ci95=boot(d, groups))
        if "bench" in per_sit[A0]:
            gb = sets["bench"]["meta"][:, 0]
            d2 = per_sit[B0]["bench"]["grid_free_chosen"].astype(float) - per_sit[A0]["bench"]["grid_free_chosen"].astype(float)
            summ["paired_bench"] = dict(diff=f"{B0} - {A0} grid free (chosen)", mean=round(float(d2.mean()), 4), ci95=boot(d2, gb))
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    np.savez_compressed(out / "per_situation.npz", fam=fam, scene=cos["meta"][:, 0],
                        **{f"{n}_{s}_{k}": v for n, R in per_sit.items() for s, r in R.items() for k, v in r.items()})
    print(json.dumps({n: {k: summ[n][k] for k in ("cos_all", "bench") if k in summ[n]} for n in names}))
    if "paired" in summ:
        print(json.dumps(summ["paired"]), json.dumps(summ.get("paired_bench")))


if __name__ == "__main__":
    main()
