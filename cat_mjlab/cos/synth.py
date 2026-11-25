"""Counterfactual obstacle synthesis (docs/cos_obstacles.md, G2).

For one window M (qpos [T, 36], expressed in its chord frame: start pelvis at the origin, goal on +x) and one deviation
event in it, sample obstacles of the event's family so that
  (1) M clears every box of the obstacle by >= margin (exact proxy geometry),
  (2) the default motion D(M) touches the obstacle's main part (separation < 0),
  (3) the main part's clearance to M is <= margin + slack (it really constrains M),
then add distractors (clear of M and D by >= margin, so they explain nothing) and room walls. Boxes are
{center [3], half [3], yaw} in the chord frame; roles: main, support, distractor, wall.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..collision import box_box_separation, capsule_box_separation, sphere_box_separation
from .features import _extent

MARGIN, SLACK, MIN_PEN = 0.05, 0.20, 0.03


def rotz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def boxes_arrays(boxes):
    if not boxes:
        return dict(box_center=np.zeros((0, 3)), box_rot=np.zeros((0, 3, 3)), box_half=np.zeros((0, 3)))
    return dict(box_center=np.array([b["center"] for b in boxes], float),
                box_rot=np.array([rotz(b.get("yaw", 0.)) for b in boxes]),
                box_half=np.array([b["half"] for b in boxes], float))


class Body:
    """Proxy primitives of a motion, computed once; separations to many boxes at once."""

    def __init__(self, geom, qpos):
        self.g = geom
        self.pel = np.asarray(qpos)[:, :3]
        q = torch.as_tensor(np.asarray(qpos), dtype=torch.float32)
        self.world, _ = geom.primitives(q)
        lo, hi = _extent(self.world, geom.compiled)
        self.lo, self.hi = lo.numpy(), hi.numpy()                     # [N, S, 3]
        self.region = geom.region_of_shape.numpy()

    def sep(self, boxes, frames=None, shapes=None):
        """[N, B] min separation over primitives (or the subset `shapes`), and [N, B] argmin primitive."""
        A = boxes_arrays(boxes)
        center, rot, half = (torch.as_tensor(A[k], dtype=torch.float32) for k in ("box_center", "box_rot", "box_half"))
        w = self.world if frames is None else {k: v[frames] for k, v in self.world.items()}
        n, m = len(w["centers"]), len(center)
        c = self.g.compiled
        sep = torch.empty((n, self.g.n_shapes, m))
        ob = (center[None, None], rot[None, None], half[None, None])
        for kind, idx in c["indices"].items():
            if not len(idx):
                continue
            if kind == "box":
                s = box_box_separation(w["centers"][:, idx, None], w["rotations"][:, idx, None],
                                       c["half_sizes"][idx][None, :, None], *ob)
            elif kind == "capsule":
                e = w["endpoints"][:, idx]
                s = capsule_box_separation(e[:, :, None, 0], e[:, :, None, 1], c["radii"][idx][None, :, None], *ob)
            else:
                s = sphere_box_separation(w["centers"][:, idx, None], c["radii"][idx][None, :, None], *ob)
            sep[:, idx] = s
        if shapes is not None:
            sub = torch.as_tensor(shapes)
            v, i = sep[:, sub].min(1)
            return v.numpy(), sub[i].numpy()
        v, i = sep.min(1)
        return v.numpy(), i.numpy()


# ------------------------------------------------------------------------------------------------------------------
# statistics of a motion in an x-range (chord frame), from primitive AABBs
def _xmask(B, x0, x1, frames=None, pad=0.):
    lo, hi = (B.lo, B.hi) if frames is None else (B.lo[frames], B.hi[frames])
    return (hi[..., 0] >= x0 - pad) & (lo[..., 0] <= x1 + pad), lo, hi


def top_in(B, x0, x1, pad=0.):
    m, lo, hi = _xmask(B, x0, x1, pad=pad)
    return float(hi[..., 2][m].max()) if m.any() else -np.inf


def bottom_in(B, x0, x1, pad=0.):
    m, lo, hi = _xmask(B, x0, x1, pad=pad)
    return float(lo[..., 2][m].min()) if m.any() else np.inf


def lateral_in(B, x0, x1, zmax=10., pad=0.):
    """(max y, min y) of the body parts overlapping the x-range and reaching below zmax."""
    m, lo, hi = _xmask(B, x0, x1, pad=pad)
    m = m & (lo[..., 2] < zmax)
    if not m.any():
        return -np.inf, np.inf
    return float(hi[..., 1][m].max()), float(lo[..., 1][m].min())


# ------------------------------------------------------------------------------------------------------------------
def posts_for_beam(xc, yc, hl, zb, th, yaw, rng):
    """Two posts at the beam ends (floor to beam top)."""
    r = rng.uniform(0.04, 0.08)
    ztop = zb + th
    R = rotz(yaw)
    out = []
    for s in (-1, 1):
        p = np.array([xc, yc, 0.]) + R @ np.array([0., s * (hl + r), 0.])
        out.append(dict(center=[p[0], p[1], ztop / 2], half=[r, r, ztop / 2], yaw=yaw, role="support"))
    return out


def sample_duck(ev, M, D, st, rng):
    """Beam or bar across the path above the crouched body, below the upright head."""
    a, b = ev["fa"], ev["fb"]
    k = a + int(np.argmin(st["top_body"][a:b]))
    xc = float(M.pel[k, 0]) + rng.uniform(-0.15, 0.15)
    bar = rng.random() < 0.3
    d = rng.uniform(0.03, 0.06) if bar else rng.uniform(0.08, 0.30)
    th = d if bar else rng.uniform(0.05, 0.25)
    mt = top_in(M, xc - d / 2, xc + d / 2, pad=0.1)
    dt = top_in(D, xc - d / 2, xc + d / 2, pad=0.1)
    lo_, hi_ = mt + MARGIN, min(mt + MARGIN + SLACK, dt - 0.02)
    if hi_ <= lo_:
        return None
    zb = rng.uniform(lo_, hi_)
    hl = rng.uniform(0.6, 1.5); yc = rng.uniform(-0.25, 0.25); yaw = rng.uniform(-0.25, 0.25)
    main = dict(center=[xc, yc, zb + th / 2], half=[d / 2, hl, th / 2], yaw=yaw, role="main", kind="bar" if bar else "beam")
    return [main] + posts_for_beam(xc, yc, hl, zb, th, yaw, rng)


def sample_crawl(ev, M, D, st, rng):
    """Low ceiling (on posts) or tunnel (side walls) over part of the crawl."""
    xa, xb = float(M.pel[ev["fa"], 0]), float(M.pel[ev["fb"] - 1, 0])
    if xb - xa < 0.2:
        return None
    ell = rng.uniform(0.3, max(0.31, min(2.0, xb - xa)))
    x0 = rng.uniform(xa, max(xa, xb - ell)); x1 = x0 + ell
    mt = top_in(M, x0, x1, pad=0.05)
    zb = rng.uniform(mt + MARGIN, mt + MARGIN + SLACK)
    if zb > 1.0:
        return None
    th = rng.uniform(0.05, 0.25)
    yl, yr = lateral_in(M, x0, x1, zmax=zb, pad=0.05)
    half_w = max(abs(yl), abs(yr)) + MARGIN + rng.uniform(0.05, 0.5)
    yc = 0.0
    main = dict(center=[(x0 + x1) / 2, yc, zb + th / 2], half=[ell / 2, half_w + 0.1, th / 2], yaw=0., role="main",
                kind="ceiling")
    out = [main]
    if rng.random() < 0.5:                       # tunnel: side walls
        t = rng.uniform(0.05, 0.12)
        for s in (-1, 1):
            out.append(dict(center=[(x0 + x1) / 2, s * (half_w + t), zb / 2], half=[ell / 2, t, zb / 2], yaw=0.,
                            role="support"))
        main["kind"] = "tunnel"
    else:
        r = 0.05
        for sx in (-1, 1):
            for sy in (-1, 1):
                out.append(dict(center=[(x0 + x1) / 2 + sx * (ell / 2 - r), sy * (half_w + 0.1 - r), zb / 2],
                                half=[r, r, zb / 2], yaw=0., role="support"))
    return out


def sample_high_step(ev, M, D, st, rng):
    """Hurdle where the floor clearance under the motion is highest near the swing-foot apex: clearance profile
    c(x) = lowest body point over all frames among parts covering x (2 cm bins); height <= c - margin."""
    a, b = ev["fa"], ev["fb"]
    fz = st["foot_z"][a:b]
    s = int(np.argmax(fz.max(0))); k = a + int(np.argmax(fz[:, s]))
    xf = float(st["foot_xy"][k, s, 0])
    xs = np.arange(xf - 0.35, xf + 0.35, 0.02)
    lo, hi = M.lo, M.hi
    clr = np.array([lo[..., 2][(hi[..., 0] >= x) & (lo[..., 0] <= x + 0.02)].min(initial=10.) for x in xs])
    good = np.where(clr - MARGIN >= 0.10)[0]
    if not len(good):
        return None
    i = int(rng.choice(good))
    # widest run of good bins around i bounds the hurdle depth
    j0 = i
    while j0 - 1 >= 0 and clr[j0 - 1] - MARGIN >= 0.10:
        j0 -= 1
    j1 = i
    while j1 + 1 < len(xs) and clr[j1 + 1] - MARGIN >= 0.10:
        j1 += 1
    span = (j1 - j0 + 1) * 0.02
    d = rng.uniform(0.03, max(0.031, min(0.25, span - 0.01)))
    xc = rng.uniform(xs[j0] + d / 2, max(xs[j0] + d / 2, xs[j1] + 0.02 - d / 2))
    mb = bottom_in(M, xc - d / 2, xc + d / 2)
    hi_ = mb - MARGIN; lo_ = max(0.10, hi_ - SLACK)
    if hi_ <= lo_:
        return None
    h = rng.uniform(lo_, hi_)
    hl = rng.uniform(0.4, 1.2); yc = float(st["foot_xy"][k, s, 1]) * rng.uniform(0., 1.)
    return [dict(center=[xc, yc, h / 2], half=[d / 2, hl, h / 2], yaw=0., role="main", kind="hurdle")]


def _wall_pair(ev, M, D, rng, kind):
    xa, xb = float(M.pel[ev["fa"], 0]), float(M.pel[ev["fb"] - 1, 0])
    ell = rng.uniform(0.15, max(0.16, min(1.2, xb - xa)))
    if xb - xa < ell:
        x0 = (xa + xb) / 2 - ell / 2
    else:
        x0 = rng.uniform(xa, xb - ell)
    x1 = x0 + ell
    H = rng.uniform(1.4, 2.2)
    ml, mr = lateral_in(M, x0, x1, zmax=H, pad=0.03)
    if not np.isfinite(ml):
        return None
    yl = ml + MARGIN + rng.uniform(0., SLACK)
    yr = mr - MARGIN - rng.uniform(0., SLACK)
    t = rng.uniform(0.05, 0.20)
    out = []
    for s, y in ((1, yl), (-1, yr)):
        out.append(dict(center=[(x0 + x1) / 2, y + s * t, H / 2], half=[ell / 2, t, H / 2], yaw=0., role="main", kind=kind))
    return out


def sample_sideways(ev, M, D, st, rng):
    return _wall_pair(ev, M, D, rng, "gap")


def sample_arms_tucked(ev, M, D, st, rng):
    return _wall_pair(ev, M, D, rng, "narrow_gap")


def sample_arms_raised(ev, M, D, st, rng):
    """Counter or table beside the path on the raised hand's side, top below the raised hand."""
    a, b = ev["fa"], ev["fb"]
    hz = st["hand_z"][a:b]                      # [n, 2] bottom of the hand spheres
    s = int(np.argmax(hz.max(0)))
    side = 1. if float(np.median(st["hand_xyz"][a:b, s, 1] - M.pel[a:b, 1])) >= 0 else -1.
    xa, xb = float(M.pel[a, 0]), float(M.pel[b - 1, 0])
    ell = rng.uniform(0.4, max(0.41, min(1.5, xb - xa)))
    x0 = rng.uniform(xa, max(xa, xb - ell)); x1 = x0 + ell
    hmax = float(np.percentile(hz[:, s], 20)) - MARGIN
    if hmax < 0.6:
        return None
    h = rng.uniform(0.6, min(hmax, 1.0))
    ml, mr = lateral_in(M, x0, x1, zmax=h + MARGIN, pad=0.03)
    inner = (ml if side > 0 else -mr) + MARGIN + rng.uniform(0., SLACK)
    depth = rng.uniform(0.4, 0.8)
    yc = side * (inner + depth / 2)
    if rng.random() < 0.5:
        return [dict(center=[(x0 + x1) / 2, yc, h / 2], half=[ell / 2, depth / 2, h / 2], yaw=0., role="main", kind="counter")]
    top = 0.04
    out = [dict(center=[(x0 + x1) / 2, yc, h - top / 2], half=[ell / 2, depth / 2, top / 2], yaw=0., role="main", kind="table")]
    r = 0.025; zl = (h - top) / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append(dict(center=[(x0 + x1) / 2 + sx * (ell / 2 - r - 0.03), yc + sy * (depth / 2 - r - 0.03), zl],
                            half=[r, r, zl], yaw=0., role="support"))
    return out


def sample_detour(ev, M, D, st, rng):
    """Box or column on the straight chord, on the far side of the bulge."""
    a, b = ev["fa"], ev["fb"]
    y = M.pel[a:b, 1]
    k = a + int(np.argmax(np.abs(y)))
    side = math.copysign(1., float(M.pel[k, 1]))
    if abs(M.pel[k, 1]) < 0.2:
        return None
    hx, hy = rng.uniform(0.1, 0.4), rng.uniform(0.1, 0.4)
    xc = float(M.pel[k, 0]) + rng.uniform(-0.2, 0.2)
    ml, mr = lateral_in(M, xc - hx, xc + hx, pad=0.05)
    face = (mr - MARGIN - rng.uniform(0., SLACK)) if side > 0 else (ml + MARGIN + rng.uniform(0., SLACK))
    yc = face - side * hy
    H = rng.uniform(0.5, 2.0)
    kind = "column" if H > 1.5 and max(hx, hy) < 0.25 else "box"
    return [dict(center=[xc, yc, H / 2], half=[hx, hy, H / 2], yaw=rng.uniform(-0.5, 0.5), role="main", kind=kind)]


SAMPLERS = dict(duck=sample_duck, crawl=sample_crawl, high_step=sample_high_step, sideways=sample_sideways,
                arms_tucked=sample_arms_tucked, arms_raised=sample_arms_raised, detour=sample_detour)


# ------------------------------------------------------------------------------------------------------------------
def check(M, D, boxes, frames):
    """(M clearance over all boxes, M clearance over main boxes, D min separation over main boxes, D argmin region)."""
    vm, _ = M.sep(boxes, frames)
    main = [b for b in boxes if b["role"] == "main"]
    vd, idd = D.sep(main, frames)
    mi = [i for i, b in enumerate(boxes) if b["role"] == "main"]
    j = np.unravel_index(int(np.argmin(vd)), vd.shape)
    return float(vm.min()), float(vm[:, mi].min()), float(vd.min()), int(D.region[idd[j]])


def obstacles_for_event(ev, M, D, st, rng, *, tries=60, want=3):
    """Up to `want` accepted obstacle sets for one event, plus attempt counts."""
    frames = np.arange(0, len(M.pel), 2)
    acc, n = [], 0
    for _ in range(tries):
        cand = SAMPLERS[ev["family"]](ev, M, D, st, rng)
        n += 1
        if cand is None:
            continue
        cm, cmain, dmin, region = check(M, D, cand, frames)
        if cm >= MARGIN and cmain <= MARGIN + SLACK and dmin < -MIN_PEN:
            acc.append(dict(boxes=cand, m_clear=cm, m_clear_main=cmain, d_pen=-dmin, d_region=region))
            if len(acc) >= want:
                break
    return acc, n


def distractors(M, D, main_boxes, rng, *, k=None, x_range=(-1., 4.), tries=40):
    """k boxes clear of M and D by >= margin and of the main obstacle by >= 0.1 m (AABB test)."""
    k = int(rng.integers(2, 6)) if k is None else k
    frames = np.arange(0, len(M.pel), 2)
    out = []
    mains = boxes_arrays(main_boxes)
    ext = np.abs(mains["box_rot"]) @ mains["box_half"][..., None] if len(main_boxes) else np.zeros((0, 3, 1))
    mlo = mains["box_center"] - ext[..., 0]; mhi = mains["box_center"] + ext[..., 0]
    for _ in range(tries):
        if len(out) >= k:
            break
        t = rng.random()
        if t < 0.4:
            half = [rng.uniform(0.15, 0.4), rng.uniform(0.15, 0.4), rng.uniform(0.2, 0.6)]; kind = "box"
        elif t < 0.7:
            r = rng.uniform(0.08, 0.2); half = [r, r, rng.uniform(0.8, 1.1)]; kind = "column"
        else:
            half = [rng.uniform(0.4, 1.2), rng.uniform(0.05, 0.12), rng.uniform(0.5, 1.1)]; kind = "wall"
        c = [rng.uniform(*x_range), rng.uniform(-2.0, 2.0), half[2]]
        b = dict(center=c, half=half, yaw=rng.uniform(-math.pi, math.pi), role="distractor", kind=kind)
        r_ = max(half[0], half[1]) * 1.42
        if len(main_boxes) and np.any((c[0] + r_ > mlo[:, 0] - .1) & (c[0] - r_ < mhi[:, 0] + .1) &
                                      (c[1] + r_ > mlo[:, 1] - .1) & (c[1] - r_ < mhi[:, 1] + .1)):
            continue
        if M.sep([b], frames)[0].min() >= MARGIN and D.sep([b], frames)[0].min() >= MARGIN:
            out.append(b)
    return out


def room_walls(M, D, rng, x_range):
    """Two long side walls beyond both motions (>= margin + 0.3 m), open at start and goal."""
    yl = max(M.hi[..., 1].max(), D.hi[..., 1].max()) + MARGIN + rng.uniform(0.3, 1.5)
    yr = min(M.lo[..., 1].min(), D.lo[..., 1].min()) - MARGIN - rng.uniform(0.3, 1.5)
    L = x_range[1] - x_range[0]; xc = (x_range[0] + x_range[1]) / 2; H = rng.uniform(1.8, 2.6)
    return [dict(center=[xc, yl + 0.1, H / 2], half=[L / 2, 0.1, H / 2], yaw=0., role="wall", kind="room_wall"),
            dict(center=[xc, yr - 0.1, H / 2], half=[L / 2, 0.1, H / 2], yaw=0., role="wall", kind="room_wall")]


def upsample(q, factor=2):
    """Linear (quaternion nlerp) upsampling of qpos [T, 36] by an integer factor (adds in-between frames)."""
    from ..exact_geometry import interpolate_frames
    t = np.arange((len(q) - 1) * factor + 1) * (0.02 / factor)
    return interpolate_frames(q, t, dt=0.02)


def verify(geom, Mq, Dq, boxes):
    """Independent full-rate check (50 Hz frames plus midpoints), all boxes: (M clearance >= margin, D touches main)."""
    Mu, Du = Body(geom, upsample(Mq)), Body(geom, upsample(Dq))
    vm, _ = Mu.sep(boxes)
    main = [b for b in boxes if b["role"] == "main"]
    vd, _ = Du.sep(main)
    body = [i for i, n in enumerate(geom.shape_names) if int(geom.region_of_shape[i]) in (2, 3)
            or "shoulder_pitch" in n or "shoulder_roll" in n]            # trunk, head, shoulders (no swinging arms)
    vb, _ = Du.sep(main, shapes=body)
    nonmain = [b for b in boxes if b["role"] in ("distractor", "wall")]
    vdn = Du.sep(nonmain)[0].min() if nonmain else 10.
    return dict(m_clear=float(vm.min()), d_main=float(vd.min()), d_main_body=float(vb.min()), d_other=float(vdn),
                ok=bool(vm.min() >= MARGIN and vd.min() < 0), ok_pen=bool(vm.min() >= MARGIN and vd.min() < -MIN_PEN))
