#!/usr/bin/env python3
"""S1 diagnostics of recorded episodes (scripts/fullpath_rollout.py): lookahead, exact-geometry plan audit, tracking error,
failure split. Pre-registration and definitions: docs/fullpath_plan.md, section S1.

    fullpath_audit.py --runs /tmp/cat-fullpath/runs/fix0_plan --out audit.json [--workers 8]

Per episode it writes one record (JSON lines cache next to each npz: <npz>.audit.json); the summary tables aggregate them by
family (side gaps, courses with low / overhead / both obstacle kinds, table gaps) and speed.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DT, PLAN_DT, F = 0.02, 0.1, 21
_G = None


def geometry():
    global _G
    if _G is None:
        import torch
        torch.set_num_threads(1)
        from cat_mjlab.exact_geometry import ExactGeometry
        _G = ExactGeometry("cpu")
    return _G


def family(scene, kinds):
    if scene.startswith("sidegap"):
        return "side gap"
    if "table-gap" in scene:
        return "table gap"
    low, over = bool((kinds == 0).any()), bool((kinds == 2).any())
    return "course: hurdle+beam" if low and over else "course: hurdle" if low else "course: beam" if over else "course: walls"


def active_event(ev_t, t):
    """Index of the chunk executing at state t (set during an earlier step), or -1."""
    k = int(np.searchsorted(ev_t, t, side="left")) - 1
    return k


def audit_one(path):
    from cat_mjlab.exact_geometry import box_far_end, box_kind, interpolate_frames
    from cat_mjlab.collision import REGIONS
    g = geometry()
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    boxes = {k: d[k] for k in ("box_center", "box_rot", "box_half")}
    kinds = box_kind(boxes)
    start, goal = np.array(meta["start"][:2]), np.array(meta["goal"][:2])
    u = (goal - start) / max(np.linalg.norm(goal - start), 1e-6)
    qpos = d["qpos"]; T = len(qpos) - 1
    res = dict(scene=meta["scene"], speed=meta["speed"], trial=meta["trial"], result=meta["outcome"]["result"],
               hand_contact=meta["outcome"]["hand_contact"], family=family(meta["scene"], kinds), steps=T,
               mix=meta["mix"], oracle=meta.get("oracle"))
    has_plan = "ev_t" in d.files
    prog = qpos[:, :2] @ u
    # ---------------- (b) plan audit
    if has_plan:
        ev_t, chosen = d["ev_t"], d["ev_chosen"]
        n = len(ev_t)
        sep_ch = g.min_separation(chosen.reshape(-1, 36), boxes).reshape(n, F).numpy()
        nxt = np.append(ev_t[1:], T)
        exec_frames = np.clip(np.ceil((nxt - ev_t) * DT / PLAN_DT).astype(int) + 1, 2, F)
        exec_min = np.array([sep_ch[k, :exec_frames[k]].min() for k in range(n)])
        near = d["body_clear"][np.minimum(ev_t, T)] < .15
        rec = dict(replans=int(n), chosen_collide=int((sep_ch.min(1) < 0).sum()), exec_collide=int((exec_min < 0).sum()),
                   exec_min=exec_min.round(4).tolist(), near=near.tolist())
        if "ev_cands" in d.files:
            cands = d["ev_cands"].astype(np.float32)
            k = cands.shape[1]
            sep_c = g.min_separation(cands.reshape(-1, 36), boxes).reshape(n, k, F).amin(-1).numpy()
            rec.update(all_collide=int((sep_c < 0).all(1).sum()), all_collide_near=int(((sep_c < 0).all(1) & near).sum()),
                       best_cand=sep_c.max(1).round(4).tolist())
            if "ev_sclear" in d.files:
                sc = d["ev_sclear"]
                rec.update(sphere_ok_exact_bad=int(((sc >= 0) & (sep_c < 0)).sum()), sphere_bad_exact_ok=int(((sc < 0) & (sep_c >= 0)).sum()),
                           cand_total=int(sc.size))
        res["plan"] = rec
    # ---------------- (c) tracking error at every 5th step with a valid plan
    if has_plan:
        steps = np.arange(1, T, 5)
        k_act = np.array([active_event(ev_t, t) for t in steps])
        ok = (k_act >= 0) & d["valid"][steps]
        steps, k_act = steps[ok], k_act[ok]
        if len(steps):
            plan_q = np.stack([interpolate_frames(chosen[k], [d["plan_t"][t]])[0] for k, t in zip(k_act, steps)])
            pa, ha, fa = g.keypoints(qpos[steps]); pp, hp, fp = g.keypoints(plan_q)
            e_pel = np.linalg.norm((pa - pp)[:, :2].numpy(), axis=-1); e_z = np.abs((pa - pp)[:, 2].numpy())
            e_hand = np.linalg.norm((ha - hp).numpy(), axis=-1); e_foot = np.linalg.norm((fa - fp).numpy(), axis=-1)
            nr = d["body_clear"][steps] < .15
            # path-relative: nearest frame (by pelvis xy) of the same chunk, 0.02 s grid; error of every proxy primitive centre
            pr_q = []
            for k, t in zip(k_act, steps):
                dense = interpolate_frames(chosen[k], np.linspace(0, 2., 101))
                pr_q.append(dense[int(np.linalg.norm(dense[:, :2] - qpos[t, :2], axis=-1).argmin())])
            pr_q = np.stack(pr_q)
            wa, _ = g.primitives(qpos[steps]); wp, _ = g.primitives(pr_q)
            e_prim = np.linalg.norm((wa["centers"] - wp["centers"]).numpy(), axis=-1)        # [n, 35]
            pr_p, pr_h, pr_f = g.keypoints(pr_q)
            res["track_path"] = dict(pelvis=np.linalg.norm((pa - pr_p)[:, :2].numpy(), axis=-1).round(4).tolist(),
                                     hand=np.linalg.norm((ha - pr_h).numpy(), axis=-1).max(1).round(4).tolist(),
                                     foot=np.linalg.norm((fa - pr_f).numpy(), axis=-1).max(1).round(4).tolist(),
                                     body=e_prim.max(1).round(4).tolist(), body_mean=e_prim.mean(1).round(4).tolist())
            res["track"] = dict(pelvis=e_pel.round(4).tolist(), z=e_z.round(4).tolist(), hand=e_hand.max(1).round(4).tolist(),
                                foot=e_foot.max(1).round(4).tolist(), near=nr.tolist())
    # ---------------- (a) + (d) at the first contact / the stall onset
    ev = None
    tc = meta.get("term_cause")
    if res["result"] == "contact" and int(meta["first_contact_step"]) < 0 and tc:
        # ended by the 13-point field (< -4 cm), the elbow probes or the route corridor, not the proxy: the event pose is
        # the terminating state; the obstacle is the box nearest to the robot then
        q_c = d["term_qpos"][None]
        sv, si = g.separation(q_c, boxes)
        box = int(si[0, int(sv[0].argmin())]) if len(boxes["box_center"]) else -1
        point = ("head", "trunk", "trunk", "feet", "feet", "hands", "hands", "legs", "legs", "arms", "arms")[min(int(tc["sdf_point"]), 10)]
        regions = (["route"] if tc["root"] else []) + ([f"field:{point}"] if tc["fields"] else []) + (["field:elbows"] if tc["elbows"] else [])
        ev = dict(kind="contact", step=int(tc["step"]), box=box, box_kind=int(kinds[box]) if box >= 0 else -1,
                  sep_actual=round(float(sv.min()), 4), regions=regions, via="field_or_route")
        pel = q_c[0, :2]; t_state = min(int(tc["step"]), T)
    elif res["result"] == "contact" and int(meta["first_contact_step"]) >= 0:
        c = int(meta["first_contact_step"])
        sb = d["contact_shape_box"]; hit = sb[sb >= 0] - int(d["box_first"])
        hit = hit[(hit >= 0) & (hit < len(boxes["box_center"]))]
        q_c = d["contact_qpos"][None]
        if len(hit):
            box = int(np.bincount(hit).argmax())
        else:                                  # box id outside this scene's range: take the nearest box
            sv_, si_ = g.separation(q_c, boxes)
            box = int(si_[0, int(sv_[0].argmin())]) if len(boxes["box_center"]) else -1
        sep_now = float(g.min_separation(q_c, boxes)[0])
        ev = dict(kind="contact", step=c, box=box, box_kind=int(kinds[box]) if box >= 0 else -1, sep_actual=round(sep_now, 4),
                  regions=meta.get("first_contact_regions", []))
        pel = q_c[0, :2]; t_state = min(c, T)
    elif res["result"] == "refused":
        best = np.maximum.accumulate(prog)
        onset = int(np.argmax(best >= best[-1] - .05))
        sv, si = g.separation(qpos[onset][None], boxes)
        j = int(si[0, int(sv[0].argmin())]) if len(boxes["box_center"]) else -1
        ev = dict(kind="stall", step=onset, box=j, box_kind=int(kinds[j]) if j >= 0 else -1, sep_actual=round(float(sv.min()), 4),
                  progress=round(float(best[-1] - prog[0]), 3), route=round(float((goal - start) @ u), 3))
        pel = qpos[onset, :2]; t_state = onset
    if ev is not None:
        if ev["box"] >= 0:
            ev["need"] = round(box_far_end(boxes, ev["box"], u) - float(pel @ u), 3)
        t = t_state
        if has_plan:
            k = active_event(ev_t, t)
            ev["valid"] = bool(d["valid"][min(t, T)]) and k >= 0
            if k >= 0:
                fr = chosen[k]; pt = float(d["plan_t"][min(t, T)])
                ev["reach"] = round(float(fr[-1, :2] @ u - pel @ u), 3)
                ev["plan_left_s"] = round((F - 1) * PLAN_DT - pt, 2)
                win = interpolate_frames(fr, np.clip(pt + np.linspace(-.2, .2, 9), 0, 2.0))
                ev["plan_sep"] = round(float(g.min_separation(win, boxes).min()), 4)
                ahead = interpolate_frames(fr, [pt, min(pt + 1., 2.)])
                ev["plan_advance_1s"] = round(float((ahead[1, :2] - ahead[0, :2]) @ u), 3)
                plan_now = interpolate_frames(fr, [pt])
                q_now = q_c if ev["kind"] == "contact" else qpos[t][None]
                pa, ha, fa = g.keypoints(q_now); pp, hp, fp = g.keypoints(plan_now)
                ev["err_pelvis"] = round(float(np.linalg.norm((pa - pp)[0, :2].numpy())), 4)
                ev["err_hand"] = round(float(np.linalg.norm((ha - hp)[0].numpy(), axis=-1).max()), 4)
                ev["err_foot"] = round(float(np.linalg.norm((fa - fp)[0].numpy(), axis=-1).max()), 4)
                # deviation of the contacting primitives between the actual and the planned pose
                # path-relative: the plan frame nearest the robot's pelvis (no along-track lag)
                dense = interpolate_frames(fr, np.linspace(0, 2., 101))
                j = int(np.linalg.norm(dense[:, :2] - q_now[0, :2], axis=-1).argmin())
                pr, hr, frr = g.keypoints(dense[j][None])
                ev["perr_pelvis"] = round(float(np.linalg.norm((pa - pr)[0, :2].numpy())), 4)
                ev["perr_hand"] = round(float(np.linalg.norm((ha - hr)[0].numpy(), axis=-1).max()), 4)
                ev["plan_sep_path"] = round(float(g.min_separation(dense[max(0, j - 10):j + 11], boxes).min()), 4)
                if ev["kind"] == "contact" and "contact_shape_box" in d.files and ev.get("via") is None:
                    wa, _ = g.primitives(q_now); wp, _ = g.primitives(plan_now)
                    touched = np.nonzero(d["contact_shape_box"] >= 0)[0]
                    if len(touched):
                        dev = np.linalg.norm((wa["centers"][0, touched] - wp["centers"][0, touched]).numpy(), axis=-1)
                        ev["err_touching"] = round(float(dev.max()), 4)
                        ev["touching_regions"] = sorted({REGIONS[int(g.region_of_shape[s])] for s in touched})
                ev["released"] = bool(d["released"][min(t, T)].any())
                if "ev_cands" in d.files:
                    cands = d["ev_cands"][k].astype(np.float32)
                    ev["cands_all_collide"] = bool((g.min_separation(cands.reshape(-1, 36), boxes).reshape(len(cands), F).amin(-1) < 0).all())
        res["event"] = ev
    return res


def cached(path):
    out = Path(str(path) + ".audit.json")
    if out.exists() and out.stat().st_mtime > Path(path).stat().st_mtime:
        return json.loads(out.read_text())
    try:
        r = audit_one(path)
    except Exception as e:      # a broken file is reported, not fatal
        r = dict(error=repr(e), path=str(path))
    out.write_text(json.dumps(r))
    return r


def pct(x, q):
    x = np.asarray(x, dtype=float)
    return round(float(np.percentile(x, q)), 4) if len(x) else None


def summarize(records, p95=None):
    """Tables of the pre-registered S1 measures; p95 = dict(pelvis=, hand=) thresholds (from clean plan-on episodes)."""
    by = {}
    for r in records:
        if "error" in r:
            continue
        for key in {r["family"], "all F144" if r["family"] != "table gap" else "table gap"}:
            by.setdefault(key, []).append(r)
        if r["family"].startswith("course"):
            by.setdefault("course (all)", []).append(r)
    out = {}
    for fam, rs in sorted(by.items()):
        n = len(rs)
        row = dict(episodes=n, clean=round(np.mean([r["result"] == "clean" for r in rs]), 4),
                   contact=round(np.mean([r["result"] == "contact" for r in rs]), 4),
                   refused=round(np.mean([r["result"] == "refused" for r in rs]), 4))
        pl = [r["plan"] for r in rs if "plan" in r]
        if pl:
            rep = sum(p["replans"] for p in pl)
            em = np.concatenate([p["exec_min"] for p in pl]); nr = np.concatenate([p["near"] for p in pl]).astype(bool)
            row["plan_audit"] = dict(replans=rep, chosen_collide=round(sum(p["chosen_collide"] for p in pl) / rep, 4),
                                     exec_collide=round(sum(p["exec_collide"] for p in pl) / rep, 4),
                                     exec_collide_near=round(float((em[nr] < 0).mean()), 4) if nr.any() else None,
                                     exec_margin_p5=pct(em, 5), exec_margin_p50=pct(em, 50),
                                     exec_margin_near_p50=pct(em[nr], 50) if nr.any() else None)
            if all("all_collide" in p for p in pl):
                row["plan_audit"].update(all4_collide=round(sum(p["all_collide"] for p in pl) / rep, 4),
                                         all4_collide_near=round(sum(p["all_collide_near"] for p in pl) / max(1, int(nr.sum())), 4),
                                         sphere_ok_exact_bad=round(sum(p.get("sphere_ok_exact_bad", 0) for p in pl) /
                                                                   max(1, sum(p.get("cand_total", 0) for p in pl)), 4))
            # executed plans in the 2 s before a contact
            pre = []
            for r in rs:
                e = r.get("event")
                if e and e["kind"] == "contact" and "plan" in r:
                    pre.append(r)
            row["contact_eps_plan_collides_at_contact"] = round(np.mean([r["event"].get("plan_sep", 1) < 0 for r in pre]), 4) if pre else None
        tr = [r["track"] for r in rs if "track" in r]
        if tr:
            cat = lambda k, near=None: np.concatenate([np.asarray(t[k])[np.asarray(t["near"], bool)] if near else np.asarray(t[k]) for t in tr])
            row["tracking_m"] = {k: dict(p50=pct(cat(k), 50), p95=pct(cat(k), 95), near_p50=pct(cat(k, True), 50),
                                         near_p95=pct(cat(k, True), 95)) for k in ("pelvis", "z", "hand", "foot")}
        tp = [r["track_path"] for r in rs if "track_path" in r]
        if tp:
            row["tracking_path_m"] = {k: dict(p50=pct(np.concatenate([t[k] for t in tp]), 50), p95=pct(np.concatenate([t[k] for t in tp]), 95))
                                      for k in ("pelvis", "hand", "foot", "body", "body_mean")}
        evs = [r["event"] for r in rs if r.get("event")]
        for kind in ("contact", "stall"):
            e = [x for x in evs if x["kind"] == kind]
            if not e:
                continue
            la = [x for x in e if "need" in x and "reach" in x]
            blk = dict(events=len(e))
            if la:
                blk.update(need_p50=pct([x["need"] for x in la], 50), reach_p50=pct([x["reach"] for x in la], 50),
                           reach_lt_need=round(np.mean([x["reach"] < x["need"] for x in la]), 4))
            if kind == "contact" and p95:
                split = dict(no_plan=0, bad_plan=0, bad_tracking=0, conflict=0)
                regions = {}
                for x in e:
                    if not x.get("valid", False) or "plan_sep" not in x:
                        c = "no_plan"
                    elif x["plan_sep"] < 0 or x.get("cands_all_collide"):
                        c = "bad_plan"
                    elif x["err_pelvis"] > p95["pelvis"] or x["err_hand"] > p95["hand"]:
                        c = "bad_tracking"
                    else:
                        c = "conflict"
                    split[c] += 1
                    for reg in x.get("touching_regions", x.get("regions", [])):
                        regions.setdefault(c, {}).setdefault(reg, 0)
                        regions[c][reg] += 1
                blk["split"] = {k: round(v / len(e), 4) for k, v in split.items()}
                blk["split_regions"] = regions
                blk["conflict_released"] = round(np.mean([x.get("released", False) for x in e]), 4)
            if kind == "stall":
                sp = dict(no_plan=0, plan_stops=0, robot_stops=0)
                for x in e:
                    if not x.get("valid", False) or "plan_advance_1s" not in x:
                        sp["no_plan"] += 1
                    elif x["plan_advance_1s"] < .05:
                        sp["plan_stops"] += 1
                    else:
                        sp["robot_stops"] += 1
                blk["split"] = {k: round(v / len(e), 4) for k, v in sp.items()}
            row[kind + "_events"] = blk
        out[fam] = row
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--p95-from", type=Path, help="audit json whose clean-episode tracking p95 sets the thresholds")
    a = p.parse_args()
    files = sorted(f for r in a.runs for f in r.glob("*.npz"))
    with ProcessPoolExecutor(a.workers) as ex:
        records = list(ex.map(cached, files, chunksize=4))
    clean_tr = [r["track"] for r in records if "track" in r and r.get("result") == "clean"]
    if a.p95_from:
        p95 = json.loads(a.p95_from.read_text())["p95_thresholds"]
    elif clean_tr:
        p95 = dict(pelvis=pct(np.concatenate([t["pelvis"] for t in clean_tr]), 95),
                   hand=pct(np.concatenate([t["hand"] for t in clean_tr]), 95),
                   foot=pct(np.concatenate([t["foot"] for t in clean_tr]), 95))
    else:
        p95 = None
    summary = dict(files=len(files), errors=[r for r in records if "error" in r][:10], p95_thresholds=p95)
    for sp in sorted({r["speed"] for r in records if "speed" in r}):
        summary[f"speed {sp:g}"] = summarize([r for r in records if r.get("speed") == sp], p95)
    summary["both speeds"] = summarize(records, p95)
    a.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["both speeds"], indent=1)[:6000])


if __name__ == "__main__":
    main()
