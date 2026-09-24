#!/usr/bin/env python3
"""Generate many diverse approaching-object standing scenes.

The approval bank was a 10x3 grid: ten fixed directions crossed with three
distance buckets. A grid of thirty configurations is exactly what a policy
memorises, so every axis here is sampled continuously instead:

    direction   azimuth uniform on the full circle, elevation in [-35, +65] deg
    distance    uniform inside a stratified bucket, not a bucket midpoint
    shape       sphere / box / cylinder / thin rod, random size and yaw
    speed       uniform 0.05 - 0.50 m/s
    start pose  arms randomised inside the real +/-0.8 action reach

Certification is unchanged and reused from build_reactive_approval: every
accepted scene carries a continuous full-path lower bound proving the object
never reaches the robot, and the approach always targets the NEAR hand so an
object from the right never crosses the torso to reach the left hand.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

from build_reactive_approval import geometry_arrays, point_sdf, sampled_path  # noqa: E402
from cat_mjlab.reactive import BodyEnvelope, digest  # noqa: E402
from cat_mjlab.collision import PROPOSAL  # noqa: E402
from reactive_tuck_geometry import Geometry  # noqa: E402

BUCKETS = {"danger": (.03, .08), "anticipation": (.09, .20), "negative": (.20, .35)}
SHAPES = {"sphere": (0, (.03, .07)), "box": (1, (.02, .09)), "cylinder": (2, (.025, .08)),
          "rod": (2, (.008, .012))}


def sample_direction(rng):
    """Continuous approach ray. Elevation is bounded so 'below' still rises from the floor."""
    azimuth = rng.uniform(0., 2. * np.pi)
    elevation = np.radians(rng.uniform(-35., 65.))
    horizontal = np.cos(elevation)
    return np.array([horizontal * np.cos(azimuth), horizontal * np.sin(azimuth), np.sin(elevation)])


def sample_shape(rng):
    name = list(SHAPES)[rng.integers(len(SHAPES))]
    kind, (lo, hi) = SHAPES[name]
    if name == "rod":
        size = np.array([rng.uniform(lo, hi), rng.uniform(.08, .22), rng.uniform(lo, hi)])
    elif name == "sphere":
        radius = rng.uniform(lo, hi)
        size = np.array([radius, radius, radius])
    else:
        size = np.array([rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi)])
    yaw = rng.uniform(0., 2. * np.pi)
    cos, sin = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cos, -sin, 0.], [sin, cos, 0.], [0., 0., 1.]])
    if rng.random() < .35:  # tip long shapes over, so rods are not always vertical
        rotation = rotation @ np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    return name, kind, size, rotation


def author_scene(geo, env, nominal, rng, bucket, device, attempts=120):
    from scipy.optimize import brentq
    lo, hi = BUCKETS[bucket]
    gap = float(rng.uniform(lo + .003, hi - .003))
    name, kind, size, rotation = sample_shape(rng)
    support = float(size[0] if kind == 0 else np.abs(rotation[2]) @ size)
    ray = sample_direction(rng)
    hand = 0 if ray[1] > 0 else 1                      # always the near hand
    for attempt in range(attempts):
        q = nominal.copy()
        # Root pose was identical in all 2000 scenes of the previous bank: one position,
        # one orientation. Only the arms were randomised, so the policy met every object
        # from a single standing pose and world heading. Yaw matters most -- it changes the
        # relationship between the world-frame approach and the robot's own frame -- and it
        # is free here because the object path below is derived from the resulting hand
        # position, so the scene stays self-consistent.
        yaw = rng.uniform(0., 2. * np.pi)
        q[3], q[6] = np.cos(yaw / 2.), np.sin(yaw / 2.)
        q[0] += rng.uniform(-.35, .35)
        q[1] += rng.uniform(-.35, .35)
        # Randomise on every attempt, including the first: gating this behind
        # attempt>0 made the first try always succeed from the nominal pose, so
        # every scene shipped with an identical posture.
        scale = .65 if attempt < 60 else .35
        for sign, offset in ((1, 22), (-1, 29)):
            change = rng.uniform(-scale, scale, 7)
            change[1] = sign * rng.uniform(.05, scale)
            q[offset:offset + 7] = np.clip(q[offset:offset + 7] + change,
                                           geo.lo[offset - 19:offset - 12], geo.hi[offset - 19:offset - 12])
        geo.q = q
        geo.set(q[19:])
        self_gap = float(geo.self_distances().min())
        if self_gap < 1e-6:
            continue
        centers, radii, hands = geometry_arrays(geo, env, device)

        def miss(distance):
            probe = hands[hand] + ray * distance
            return float(point_sdf(hands[hand:hand + 1], probe, rotation, size, kind, device)[0] - .10344617 - gap)

        try:
            distance = brentq(miss, .001, 2., xtol=1e-7)
        except ValueError:
            continue
        end = hands[hand] + ray * distance
        start = end + ray * rng.uniform(.22, .45)
        if ray[2] < 0.:                                 # rising approaches start at floor level
            start[2] = max(start[2], support + .003)
        if min(start[2], end[2]) - support < .002 or np.linalg.norm(end - start) < .025:
            continue
        check = sampled_path(start, end, centers, radii, rotation, size, kind, device, n=81)
        if check["continuous_body_lower_bound_m"] < .004:
            continue
        return dict(q=q, hand=hand, start=start, end=end, gap=gap, shape=name, kind=kind,
                    size=size, rotation=rotation, self_gap=self_gap, attempt=attempt,
                    centers=centers, radii=radii, hands=hands, support=support, ray=ray)
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--base-bank", type=Path,
                   default=ROOT / "data/furniture/cat_flat_hand_balance_v2_20260921/manifest.json")
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output bank already exists; refusing to overwrite")

    rng = np.random.default_rng(args.seed)
    geo = Geometry()
    env = BodyEnvelope(geo.model, args.device)
    nominal = geo.q.copy()
    foot = [i for i in range(geo.model.ngeom)
            if "foot" in (geo.model.geom(i).name or "") and geo.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX]
    nominal[2] -= min(geo.data.geom_xpos[i, 2]
                      - np.sum(np.abs(geo.data.geom_xmat[i].reshape(3, 3)[2]) * geo.model.geom_size[i]) for i in foot)

    names = list(BUCKETS)
    rows, certs, rejected = [], [], 0
    while len(rows) < args.count:
        bucket = names[len(rows) % len(names)]           # equal stratified mass per bucket
        found = author_scene(geo, env, nominal, rng, bucket, args.device)
        if found is None:
            rejected += 1
            continue
        speed = float(rng.uniform(.05, .50))
        hold = float(rng.uniform(.4, 1.2))
        travel = float(np.linalg.norm(found["end"] - found["start"]))
        certificate = sampled_path(found["start"], found["end"], found["centers"], found["radii"],
                                   found["rotation"], found["size"], found["kind"], args.device, n=1001)
        if certificate["continuous_body_lower_bound_m"] <= 0.:
            rejected += 1
            continue
        index = len(rows)
        rows.append(dict(
            id=f"reactive-{index:05d}-{bucket}", direction="sampled", bucket=bucket,
            target="left" if found["hand"] == 0 else "right", target_hands=[found["hand"]],
            shape=found["shape"], qpos=found["q"].tolist(),
            start=[found["start"].tolist(), [10., 10., 10.]],
            end=[found["end"].tolist(), [10., 10., 10.]],
            rotations=[found["rotation"].tolist()] * 2, sizes=[found["size"].tolist()] * 2,
            kinds=[found["kind"]] * 2, valid=[True, False], speed=[speed] * 2, hold=[hold] * 2,
            clearance=[found["gap"]] * 2, duration_s=2 * travel / speed + hold + .25,
            arrival_s=travel / speed, sampling_weight=1.,
            flag="", path_note="continuously sampled approach ray",
            approach_ray=found["ray"].tolist()))
        certs.append(dict(id=rows[-1]["id"], target_surface_m=found["gap"],
                          object_paths=[certificate],
                          initial_self_proxy_gap_m=found["self_gap"],
                          authoring_attempts=found["attempt"],
                          passed=certificate["continuous_body_lower_bound_m"] > 0.))
        if len(rows) % 100 == 0:
            print(f"{len(rows)}/{args.count} certified, {rejected} rejected", flush=True)

    args.output.mkdir(parents=True)
    base = args.base_bank.resolve()
    manifest = dict(
        schema="cat-reactive-standing-v1", reactive_mass=.25, retained_mass=.75,
        base_bank=dict(path=str(base), sha256=digest(base)),
        collision_bank=dict(path=str(base.parent.parent / (base.parent.name + "_collision") / "manifest.json"),
                            sha256=digest(base.parent.parent / (base.parent.name + "_collision") / "manifest.json")),
        reset_bank=dict(path=str(base.parent.parent / (base.parent.name + "_resets") / "manifest.json"),
                        sha256=digest(base.parent.parent / (base.parent.name + "_resets") / "manifest.json")),
        proxy=dict(path=str(PROPOSAL), sha256=digest(PROPOSAL)),
        background_scene_index=2350, episode_steps=4000, scenes=rows,
        retained_scene_count=2375,
        sampling="75% unchanged base sampler; 25% continuously sampled reactive scenes, equal bucket mass")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (args.output / "certification.json").write_text(
        json.dumps(dict(scenes=certs, all_passed=all(c["passed"] for c in certs)), indent=1) + "\n")
    print(json.dumps(dict(generated=len(rows), rejected=rejected,
                          all_certified=all(c["passed"] for c in certs)), indent=1))


if __name__ == "__main__":
    main()
