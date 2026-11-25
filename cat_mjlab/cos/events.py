"""Deviation-event detectors on G1 motion features (docs/cos_obstacles.md, gate G0).

Every detector returns events (t0, t1 frame indices, half-open, plus a magnitude) for one clip at 50 Hz. Thresholds are the
pre-registered values in docs/cos_obstacles.md; `CAL` holds the upright-walking calibration measured on frames the library
tags as `walk` (scripts/cos_g0_audit.py prints it).
"""
from __future__ import annotations

import numpy as np

DT = 0.02
FAMILIES = ("duck", "crawl", "high_step", "sideways", "arms_raised", "arms_tucked", "detour")

# Upright-walking calibration (library `walk` frames, every 7th frame, 2026-11-26): medians / percentiles.
CAL = dict(top_body_up=1.297, hand_z_walk=0.493, arm_lat_p5=0.241)

TH = dict(
    speed_moving=0.15,          # m/s, pelvis xy speed smoothed over 0.5 s
    duck_drop=0.15,             # m below top_body_up
    duck_min_s=0.3,
    crawl_pelvis=0.50, crawl_contact=0.12, crawl_speed=0.08, crawl_min_s=0.5,
    step_apex_lo=0.20, step_apex_hi=0.60, step_ground=0.05, step_forward=0.25,
    side_lo=np.deg2rad(50), side_hi=np.deg2rad(130), side_speed=0.20, side_min_s=0.5,
    raise_above_walk=0.20, raise_abs=0.72, raise_min_s=0.5,
    tuck_min_s=0.5,
    detour_window_s=3.0, detour_chord=1.2, detour_dev=0.30, detour_turn=np.deg2rad(45), detour_ratio=1.5,
)


def smooth(x, n):
    if n <= 1 or len(x) < n:
        return x
    k = np.ones(n) / n
    pad = np.pad(x, [(n // 2, n - 1 - n // 2)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    if x.ndim == 1:
        return np.convolve(pad, k, mode="valid")
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def runs(mask, min_len=1, merge_gap=10):
    """Half-open runs [a, b) of True in mask; gaps shorter than merge_gap frames are bridged."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(int), 0])
    a, b = np.where(d == 1)[0], np.where(d == -1)[0]
    out = []
    for s, e in zip(a, b):
        if out and s - out[-1][1] < merge_gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if e - s >= min_len]


def kinematics(F):
    """Derived per-frame signals: speed, path direction, moving flag."""
    p = F["pelvis"][:, :2].astype(np.float64)
    v = np.gradient(p, DT, axis=0) if len(p) > 1 else np.zeros_like(p)
    vs = smooth(v, 25)
    speed = np.linalg.norm(vs, axis=1)
    # path direction from displacement over +-0.5 s
    n = len(p); h = 25
    i0, i1 = np.clip(np.arange(n) - h, 0, n - 1), np.clip(np.arange(n) + h, 0, n - 1)
    d = p[i1] - p[i0]
    path_dir = np.arctan2(d[:, 1], d[:, 0])
    return dict(speed=speed, path_dir=path_dir, disp=np.linalg.norm(d, axis=1))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def detect(F, th=TH, cal=CAL):
    """All events of one clip: {family: [(t0, t1, magnitude), ...]}."""
    K = kinematics(F)
    n = len(F["top"])
    moving = K["speed"] > th["speed_moving"]
    ev = {f: [] for f in FAMILIES}
    pz = F["pelvis"][:, 2]
    hand_min = F["hand_z"].min(1)
    # crawl
    crawl = (pz < th["crawl_pelvis"]) & ((hand_min < th["crawl_contact"]) | (F["knee_z"] < th["crawl_contact"])) \
        & (K["speed"] > th["crawl_speed"])
    for a, b in runs(crawl, int(th["crawl_min_s"] / DT)):
        ev["crawl"].append((a, b, float(pz[a:b].min())))
    crawlish = (pz < th["crawl_pelvis"]) | (hand_min < th["crawl_contact"])
    # duck
    duck = (F["top_body"] < cal["top_body_up"] - th["duck_drop"]) & moving & ~crawlish
    for a, b in runs(duck, int(th["duck_min_s"] / DT)):
        ev["duck"].append((a, b, float(cal["top_body_up"] - F["top_body"][a:b].min())))
    # high step: per foot lift segments
    fz = F["foot_z"]
    for s in (0, 1):
        z, other = fz[:, s], fz[:, 1 - s]
        up = z > th["step_ground"]
        for a, b in runs(up, 3, merge_gap=1):
            apex = z[a:b].max()
            if not (th["step_apex_lo"] <= apex <= th["step_apex_hi"]) or a == 0 or b >= n:
                continue
            k = a + int(np.argmax(z[a:b]))
            if other[k] > 0.10:              # stance foot must be on the ground at the apex (no jumps)
                continue
            hd = K["path_dir"][a]
            u = np.array([np.cos(hd), np.sin(hd)])
            fwd = float((F["foot_xy"][b, s] - F["foot_xy"][a, s]) @ u)
            if fwd < th["step_forward"] or not moving[a:b].any():
                continue
            ev["high_step"].append((a, b, float(apex)))
    # sideways: torso yaw vs path direction
    ang = np.abs(wrap(F["torso_yaw"] - K["path_dir"]))
    side = (ang > th["side_lo"]) & (ang < th["side_hi"]) & (K["speed"] > th["side_speed"]) & ~crawlish
    for a, b in runs(side, int(th["side_min_s"] / DT)):
        ev["sideways"].append((a, b, float(np.rad2deg(np.median(ang[a:b])))))
    # arms raised / tucked (while moving, upright)
    hmax = F["hand_z"].max(1)
    up = (hmax > max(th["raise_abs"], cal["hand_z_walk"] + th["raise_above_walk"])) & moving & ~crawlish
    for a, b in runs(up, int(th["raise_min_s"] / DT)):
        ev["arms_raised"].append((a, b, float(hmax[a:b].max())))
    tuck = (F["elbow_lat"] < cal["arm_lat_p5"]) & moving & ~crawlish
    for a, b in runs(tuck, int(th["tuck_min_s"] / DT)):
        ev["arms_tucked"].append((a, b, float(F["elbow_lat"][a:b].min())))
    # detour: bulge of the root path off its chord over 3 s windows
    p = F["pelvis"][:, :2].astype(np.float64)
    w = int(th["detour_window_s"] / DT)
    hits = np.zeros(n, bool); mags = np.zeros(n)
    for a in range(0, max(n - w, 0), 25):
        b = a + w
        c = p[b] - p[a]; L = np.linalg.norm(c)
        if L < th["detour_chord"]:
            continue
        u = c / L
        rel = p[a:b + 1] - p[a]
        perp = rel[:, 0] * u[1] - rel[:, 1] * u[0]
        dev = np.abs(perp).max()
        plen = np.linalg.norm(np.diff(p[a:b + 1], axis=0), axis=1).sum()
        turn = abs(wrap(K["path_dir"][b] - K["path_dir"][a]))
        # one-sided bulge (not an S) that returns to the start heading
        one_sided = (perp.max() < 0.1 * dev) or (perp.min() > -0.1 * dev)
        if dev >= th["detour_dev"] and turn < th["detour_turn"] and plen / L < th["detour_ratio"] and one_sided:
            hits[a:b] = True; mags[a:b] = np.maximum(mags[a:b], dev)
    for a, b in runs(hits, 1, merge_gap=1):
        ev["detour"].append((a, b, float(mags[a:b].max())))
    return ev


# ---- clean tier (amendment 01:15 after the first spot check: ~half of raw events were wrong-reason motions) ----
import re as _re

EXCLUDE = _re.compile(r"fight|combo|kick|punch|box|martial|danc|jump|karate|weapon|sword|push|stumble|throw|drink|money|"
                      r"phone|wave|sit|stool|stand_?up|get_?up|lie|lying|pick|yoga|stretch|squat|kneel_down_with|exercise|"
                      r"golf|tennis|basket|soccer|ball", _re.I)
# CMU subjects whose takes are dance, breakdance, acrobatics or getting up from the floor (CMU mocap subject index)
CMU_EXCLUDE = {5, 49, 55, 60, 61, 85, 87, 88, 89, 90, 93, 94, 131, 140}
CLEAN = dict(run_speed=1.6, travel_m=0.5, crawl_travel_m=0.3, duck_mean_speed=0.25, side_yaw_change=np.deg2rad(45),
             step_pelvis_fwd=0.20, step_overshoot=0.10, step_apex_hi=0.50, pad_s=0.5)


def clean(fam, a, b, F, name, source):
    """True if an event (frames [a, b) of one clip's features F) is a plausible obstacle-avoidance deviation."""
    if source == "omniretarget" or EXCLUDE.search(name):
        return False
    m = _re.match(r"(\d+)_\d+__", name)
    if source == "amass_CMU" and m and int(m.group(1)) in CMU_EXCLUDE:
        return False
    n = len(F["top"]); pad = int(CLEAN["pad_s"] / DT)
    a0, b0 = max(0, a - pad), min(n - 1, b + pad)
    p = F["pelvis"][:, :2]
    travel = float(np.linalg.norm(p[b0] - p[a0]))
    if fam == "crawl":
        return travel >= CLEAN["crawl_travel_m"]
    if fam == "high_step":
        b1 = min(b, n - 1)
        hd = np.arctan2(*(p[b1] - p[a])[::-1]) if np.linalg.norm(p[b1] - p[a]) > 1e-3 else 0.
        u = np.array([np.cos(hd), np.sin(hd)])
        if float((p[b1] - p[a]) @ u) < CLEAN["step_pelvis_fwd"]:
            return False
        fz = F["foot_z"][a:b1]; s = int(np.argmax(fz.max(0)))
        if fz[:, s].max() > CLEAN["step_apex_hi"]:
            return False
        if float(np.linalg.norm(p[b1] - p[a])) / max((b1 - a) * DT, DT) > CLEAN["run_speed"]:
            return False                                       # running stride, heel kicked up behind
        k = a + int(np.argmax(fz[:, s]))                       # apex: the foot must be under or ahead of the pelvis
        if float((F["foot_xy"][k, s] - p[k]) @ u) < -0.05:
            return False
        fx = (F["foot_xy"][a:b1 + 1, s] - F["foot_xy"][a, s]) @ u
        return float(fx.max() - fx[-1]) <= CLEAN["step_overshoot"]
    if travel < CLEAN["travel_m"]:
        return False
    if fam == "duck":
        return float(np.linalg.norm(p[min(b, n - 1)] - p[a])) / max((b - a) * DT, DT) >= CLEAN["duck_mean_speed"]
    if fam == "sideways":
        y = np.unwrap(F["torso_yaw"][a:b])
        return float(y.max() - y.min()) < CLEAN["side_yaw_change"]
    return True
