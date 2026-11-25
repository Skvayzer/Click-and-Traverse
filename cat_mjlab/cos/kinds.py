"""Several obstacle kinds per deviation family and diverse multi-fit search (docs/cos_obstacles.md, scenes v1).

Each sampler returns a list of boxes in the chord frame (role main / support) or None; acceptance is the same exact
check as synth.obstacles_for_event (M clear >= margin, main part within margin + slack of M, D penetrates the main part
by > MIN_PEN). fits_for_event cycles through the family's kinds and rejects near-duplicates of fits it already kept.
"""
from __future__ import annotations

import math

import numpy as np

from .synth import (MARGIN, MIN_PEN, SLACK, check, lateral_in, posts_for_beam, sample_arms_raised, sample_crawl,
                    sample_detour, sample_duck, sample_high_step, top_in, bottom_in, _wall_pair)


def duck_shelf(ev, M, D, st, rng):
    """Slab cantilevered from a side wall over the path (low shelf / overhang)."""
    a, b = ev["fa"], ev["fb"]
    k = a + int(np.argmin(st["top_body"][a:b]))
    xc = float(M.pel[k, 0]) + rng.uniform(-0.15, 0.15)
    ell = rng.uniform(0.3, 1.0); d = ell
    mt = top_in(M, xc - d / 2, xc + d / 2, pad=0.05); dt = top_in(D, xc - d / 2, xc + d / 2, pad=0.05)
    lo_, hi_ = mt + MARGIN, min(mt + MARGIN + SLACK, dt - 0.02)
    if hi_ <= lo_:
        return None
    zb = rng.uniform(lo_, hi_); th = rng.uniform(0.03, 0.15)
    side = rng.choice([-1., 1.])
    ml, mr = lateral_in(M, xc - d / 2, xc + d / 2, pad=0.05)
    wall_face = (ml if side > 0 else -mr) + MARGIN + rng.uniform(0.1, 0.5)
    y_end = -side * rng.uniform(0.0, 0.5)
    y_wall = side * wall_face
    half_y = abs(y_wall - y_end) / 2; yc = (y_wall + y_end) / 2
    H = rng.uniform(1.8, 2.4); tw = rng.uniform(0.05, 0.15)
    return [dict(center=[xc, yc, zb + th / 2], half=[d / 2, half_y, th / 2], yaw=0., role="main", kind="shelf"),
            dict(center=[xc, y_wall + side * tw, H / 2], half=[max(d / 2, 0.4), tw, H / 2], yaw=0., role="support")]


def duck_low_ceiling(ev, M, D, st, rng):
    """Low ceiling starting at the duck (entrance of a low passage), on 4 posts."""
    a, b = ev["fa"], ev["fb"]
    k = a + int(np.argmin(st["top_body"][a:b]))
    x0 = float(M.pel[k, 0]) + rng.uniform(-0.3, 0.05); ell = rng.uniform(0.3, 1.2); x1 = x0 + ell
    mt = top_in(M, x0, x1, pad=0.05); dt = top_in(D, x0, x1, pad=0.05)
    lo_, hi_ = mt + MARGIN, min(mt + MARGIN + SLACK, dt - 0.02)
    if hi_ <= lo_:
        return None
    zb = rng.uniform(lo_, hi_); th = rng.uniform(0.05, 0.2)
    hw = rng.uniform(0.6, 1.4); yc = rng.uniform(-0.2, 0.2); r = 0.05
    out = [dict(center=[(x0 + x1) / 2, yc, zb + th / 2], half=[ell / 2, hw, th / 2], yaw=0., role="main", kind="low_ceiling")]
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append(dict(center=[(x0 + x1) / 2 + sx * (ell / 2 - r), yc + sy * (hw - r), zb / 2], half=[r, r, zb / 2],
                            yaw=0., role="support"))
    return out


def crawl_table(ev, M, D, st, rng):
    """Crawl under a table: top above the crawling body, 4 legs outside it."""
    xa, xb = float(M.pel[ev["fa"], 0]), float(M.pel[ev["fb"] - 1, 0])
    if xb - xa < 0.2:
        return None
    ell = rng.uniform(0.6, 1.6); x0 = rng.uniform(xa - 0.3, max(xa - 0.3, xb - ell)); x1 = x0 + ell
    mt = top_in(M, x0, x1, pad=0.05)
    h = rng.uniform(max(0.6, mt + MARGIN), max(0.6, mt + MARGIN) + 0.2)
    if h > 0.9:
        return None
    top = 0.04
    yl, yr = lateral_in(M, x0, x1, zmax=h, pad=0.05)
    hw = max(abs(yl), abs(yr)) + MARGIN + rng.uniform(0.05, 0.3)
    out = [dict(center=[(x0 + x1) / 2, 0., h - top / 2], half=[ell / 2, hw, top / 2], yaw=0., role="main", kind="table")]
    r, zl = 0.025, (h - top) / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append(dict(center=[(x0 + x1) / 2 + sx * (ell / 2 - r), sy * (hw - r), zl], half=[r, r, zl], yaw=0.,
                            role="support"))
    return out


def high_step_low_box(ev, M, D, st, rng):
    """A deeper, lower box to step over (same clearance-profile placement as the hurdle)."""
    out = sample_high_step(ev, M, D, st, rng)
    if out is None:
        return None
    b = out[0]
    d = rng.uniform(0.15, 0.45)
    b["half"][0] = d / 2
    b["kind"] = "low_box"
    return out


def _gap(ev, M, D, st, rng, kind):
    w = _wall_pair(ev, M, D, rng, kind)
    if w is None:
        return None
    for b in w:
        b["kind"] = kind
    return w


def gap_doorway(ev, M, D, st, rng, kind="doorway"):
    w = _wall_pair(ev, M, D, rng, kind)
    if w is None:
        return None
    for b in w:
        b["kind"] = kind
    yl = w[0]["center"][1] - w[0]["half"][1]; yr = w[1]["center"][1] + w[1]["half"][1]
    H = min(b["half"][2] * 2 for b in w); zl = rng.uniform(1.85, 2.1)
    for b in w:
        b["half"][2] = max(b["half"][2], (zl + 0.2) / 2); b["center"][2] = b["half"][2]
    x = w[0]["center"][0]; hx = w[0]["half"][0]
    w.append(dict(center=[x, (yl + yr) / 2, zl + 0.1], half=[hx, (yl - yr) / 2 + 0.05, 0.1], yaw=0., role="support"))
    return w


def gap_cabinet(ev, M, D, st, rng, kind="cabinet_gap"):
    """One side of the gap is a cabinet (0.8-1.2 m high, 0.4-0.6 m deep), the other a wall."""
    w = _wall_pair(ev, M, D, rng, kind)
    if w is None:
        return None
    j = int(rng.integers(0, 2)); b = w[j]
    s = 1. if j == 0 else -1.
    inner = b["center"][1] - s * b["half"][1]
    dep = rng.uniform(0.2, 0.3); h = rng.uniform(0.8, 1.2)
    b.update(center=[b["center"][0], inner + s * dep, h / 2], half=[max(b["half"][0], 0.3), dep, h / 2])
    for x in w:
        x["kind"] = kind
    return w


def arms_raised_pair(ev, M, D, st, rng):
    """Counters on both sides (a kitchen-like corridor) below the raised hands."""
    out = sample_arms_raised(ev, M, D, st, rng)
    if out is None or out[0]["kind"] != "counter":
        return None
    c = dict(out[0]); c["center"] = list(c["center"]); c["half"] = list(c["half"])
    x0, x1 = c["center"][0] - c["half"][0], c["center"][0] + c["half"][0]
    h = c["half"][2] * 2
    ml, mr = lateral_in(M, x0, x1, zmax=h + MARGIN, pad=0.03)
    side = -math.copysign(1., c["center"][1])
    inner = (ml if side > 0 else -mr) + MARGIN + rng.uniform(0., SLACK)
    c["center"][1] = side * (inner + c["half"][1])
    for b in (out[0], c):
        b["kind"] = "counter_pair"
    return [out[0], c]


def detour_wall_stub(ev, M, D, st, rng):
    """A wall reaching from the side across the straight path (walk around its end)."""
    out = sample_detour(ev, M, D, st, rng)
    if out is None:
        return None
    b = out[0]
    face = b["center"][1] + (b["half"][1] if b["center"][1] < 0 else -b["half"][1])   # face toward M
    side = -1. if b["center"][1] < face else 1.                                         # obstacle lies on this side of its face
    far = face + side * rng.uniform(1.0, 2.0)
    t = rng.uniform(0.05, 0.12)
    H = rng.uniform(1.2, 2.2)
    b.update(center=[b["center"][0], (face + far) / 2, H / 2], half=[t, abs(far - face) / 2, H / 2], yaw=0., kind="wall_stub")
    return out


def detour_table(ev, M, D, st, rng):
    out = sample_detour(ev, M, D, st, rng)
    if out is None:
        return None
    b = out[0]
    h = rng.uniform(0.7, 0.8); top = 0.04
    hx, hy = max(b["half"][0], 0.3), max(b["half"][1], 0.3)
    cx, cy = b["center"][0], b["center"][1]
    res = [dict(center=[cx, cy, h - top / 2], half=[hx, hy, top / 2], yaw=b["yaw"], role="main", kind="table")]
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"]); r = 0.025; zl = (h - top) / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = sx * (hx - r), sy * (hy - r)
            res.append(dict(center=[cx + c * lx - s * ly, cy + s * lx + c * ly, zl], half=[r, r, zl], yaw=b["yaw"],
                            role="main", kind="table"))
    return res


KINDS = {
    "duck": [sample_duck, duck_shelf, duck_low_ceiling],
    "crawl": [sample_crawl, crawl_table],
    "high_step": [sample_high_step, high_step_low_box],
    "sideways": [lambda *x: _gap(*x, kind="gap"), gap_doorway, gap_cabinet],
    "arms_tucked": [lambda *x: _gap(*x, kind="narrow_gap"), lambda *x: gap_doorway(*x, kind="narrow_doorway")],
    "arms_raised": [sample_arms_raised, arms_raised_pair],
    "detour": [sample_detour, detour_wall_stub, detour_table],
}


def descriptor(boxes):
    m = [b for b in boxes if b["role"] == "main"]
    c = np.mean([b["center"] for b in m], 0); h = np.max([b["half"] for b in m], 0)
    return m[0].get("kind", "?"), np.r_[c, h]


def near_duplicate(desc, kept, tol=(0.10, 0.10, 0.05, 0.08, 0.08, 0.05)):
    k, v = desc
    return any(k == k2 and np.all(np.abs(v - v2) < np.asarray(tol)) for k2, v2 in kept)


def fits_for_event(ev, M, D, st, rng, *, K=6, tries=150):
    """Up to K accepted, mutually non-duplicate fits cycling through the family's kinds; plus attempt statistics."""
    frames = np.arange(0, len(M.pel), 2)
    samplers = KINDS[ev["family"]]
    acc, kept, n, dup = [], [], 0, 0
    for i in range(tries):
        cand = samplers[i % len(samplers)](ev, M, D, st, rng)
        n += 1
        if cand is None:
            continue
        cm, cmain, dmin, region = check(M, D, cand, frames)
        if not (cm >= MARGIN and cmain <= MARGIN + SLACK and dmin < -MIN_PEN):
            continue
        desc = descriptor(cand)
        if near_duplicate(desc, kept):
            dup += 1
            continue
        kept.append(desc)
        acc.append(dict(boxes=cand, m_clear=cm, m_clear_main=cmain, d_pen=-dmin, d_region=region, kind=desc[0]))
        if len(acc) >= K:
            break
    return acc, dict(tried=n, duplicates=dup)
