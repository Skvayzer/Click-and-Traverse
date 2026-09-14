"""Strict episode accounting and paired comparisons for furniture traversal.

These metrics do not infer success from an environment's ``done`` flag or an
x-coordinate. Ordinary foot-floor support is excluded by the environment's
contact classifier; every robot-furniture contact fails strict success.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Iterable

import numpy as np


def route_projection(position, route):
    """Return nearest polyline arclength, lateral distance, and total length."""
    points = np.asarray(route, dtype=float)
    position = np.asarray(position, dtype=float)[:2]
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise ValueError("route must contain at least two xy points")
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(points)) or not np.any(lengths > 0):
        raise ValueError("route must be finite and have positive length")
    fractions = np.clip(np.sum((position-points[:-1])*vectors, axis=1)
                        / np.maximum(lengths**2, 1e-12), 0, 1)
    projections = points[:-1] + fractions[:, None]*vectors
    distances = np.linalg.norm(projections-position, axis=1)
    index = int(np.argmin(distances))
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    return float(arc[index]+fractions[index]*lengths[index]), float(distances[index]), float(arc[-1])


@dataclass
class EpisodeRecord:
    scene_id: str
    case_id: str
    training_seed: int
    episode_seed: int
    controller: str
    time_budget: float
    elapsed: float
    reached_goal: bool
    strict_success: bool
    furniture_contact: bool
    hand_contact: bool
    fall: bool
    self_collision: bool
    nonfoot_floor_contact: bool
    timeout: bool
    first_contact_part: str | None
    first_contact_time: float | None
    minimum_clearance: float | None
    path_length: float
    route_length: float
    route_progress: float
    progress_before_first_contact: float
    bottlenecks_cleared: int
    bottlenecks_before_first_contact: int
    bottlenecks_total: int
    termination_reason: str
    numerical_failure: bool = False
    provenance: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class EpisodeTracker:
    """Unbatched accounting, evaluated before any training autoreset wrapper."""

    def __init__(self, scene, *, case_id="0", training_seed=0, episode_seed=0,
                 controller="policy", goal_tolerance=0.25, provenance=None):
        self.scene = scene
        self.case_id, self.training_seed, self.episode_seed = str(case_id), int(training_seed), int(episode_seed)
        self.controller, self.goal_tolerance = controller, float(goal_tolerance)
        if not math.isfinite(self.goal_tolerance) or self.goal_tolerance <= 0:
            raise ValueError("goal_tolerance must be finite and positive")
        self.time_budget = float(scene["time_budget"])
        if self.time_budget <= 0 or not math.isfinite(self.time_budget):
            raise ValueError("time_budget must be finite and positive")
        self.route = np.asarray(scene["route"], dtype=float)
        self.last_position = np.asarray(scene["start"][:2], dtype=float)
        _, _, self.route_length = route_projection(self.last_position, self.route)
        self.bottlenecks = list(scene.get("bottlenecks", []))
        self.bottlenecks_cleared = 0
        self.elapsed = self.path_length = self.progress = 0.0
        self.progress_before_contact = 0.0
        self.bottlenecks_before_contact = 0
        self.furniture_contact = self.hand_contact = self.fall = self.self_collision = False
        self.nonfoot_floor_contact = False
        self.numerical_failure = False
        self.reached_goal = False
        self.first_contact_part = self.first_contact_time = None
        self.minimum_clearance = math.inf
        self.provenance = provenance or {}

    def update(self, *, position, elapsed, furniture_contact=False, hand_contact=False,
               fall=False, self_collision=False, minimum_clearance=None,
               contact_part=None, nonfoot_floor_contact=False, goal_reached=None,
               numerical_failure=False, first_contact_time=None):
        position = np.asarray(position, dtype=float)[:2]
        elapsed = float(elapsed)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("position must be a finite xy pair")
        if not math.isfinite(elapsed) or elapsed < self.elapsed:
            raise ValueError("elapsed time must be finite and monotonic")
        if first_contact_time is not None:
            first_contact_time = float(first_contact_time)
            if (not math.isfinite(first_contact_time) or first_contact_time < 0
                    or first_contact_time > elapsed + 1e-6):
                raise ValueError("first_contact_time must be finite and within elapsed episode time")
            first_contact_time = min(first_contact_time, elapsed)
        self.path_length += float(np.linalg.norm(position-self.last_position))
        self.last_position, self.elapsed = position.copy(), elapsed
        progress, _, _ = route_projection(position, self.route)
        self.progress = max(self.progress, progress)
        # Count successive gates actually visited, not gates skipped by nearest-
        # segment projection when a winding route runs close to another aisle.
        if self.bottlenecks_cleared < len(self.bottlenecks):
            gate = self.bottlenecks[self.bottlenecks_cleared]
            center = gate.get("center", gate.get("position", gate.get("xy")))
            if center is None:
                raise ValueError("bottleneck needs center/position/xy coordinates")
            radius = float(gate.get("capture_radius", 0.45))
            if np.linalg.norm(position-np.asarray(center)[:2]) <= radius:
                self.bottlenecks_cleared += 1
        if self.first_contact_time is None:
            if furniture_contact or hand_contact or self_collision or nonfoot_floor_contact:
                self.first_contact_time = elapsed if first_contact_time is None else first_contact_time
                self.first_contact_part = str(contact_part or ("hand" if hand_contact else "unclassified"))
            else:
                # These remain the last observed clean control-frame values;
                # a substep contact timestamp does not imply substep positions.
                self.progress_before_contact = self.progress
                self.bottlenecks_before_contact = self.bottlenecks_cleared
        self.hand_contact |= bool(hand_contact)
        self.furniture_contact |= bool(furniture_contact)
        self.self_collision |= bool(self_collision)
        self.nonfoot_floor_contact |= bool(nonfoot_floor_contact)
        self.fall |= bool(fall)
        self.numerical_failure |= bool(numerical_failure)
        if minimum_clearance is not None and math.isfinite(float(minimum_clearance)):
            self.minimum_clearance = min(self.minimum_clearance, float(minimum_clearance))
        reached = np.linalg.norm(position-np.asarray(self.scene["goal"])[:2]) <= self.goal_tolerance
        if goal_reached is not None:
            reached = reached and bool(goal_reached)
        self.reached_goal |= bool(reached and elapsed <= self.time_budget and not numerical_failure)

    def finish(self, *, termination_reason=None):
        failed = (self.furniture_contact or self.hand_contact or self.fall or self.self_collision
                  or self.nonfoot_floor_contact or self.numerical_failure)
        strict = self.reached_goal and not failed
        timeout = self.elapsed >= self.time_budget and not self.reached_goal
        reason = termination_reason or (
            "numerical_failure" if self.numerical_failure else
            "furniture_contact" if self.furniture_contact else
            "self_collision" if self.self_collision else
            "nonfoot_floor_contact" if self.nonfoot_floor_contact else
            "hand_contact" if self.hand_contact else
            "fall" if self.fall else "goal" if self.reached_goal else
            "timeout" if timeout else "incomplete")
        return EpisodeRecord(
            scene_id=str(self.scene["scene_id"]), case_id=self.case_id,
            training_seed=self.training_seed, episode_seed=self.episode_seed,
            controller=self.controller, time_budget=self.time_budget,
            elapsed=self.elapsed, reached_goal=self.reached_goal, strict_success=strict,
            furniture_contact=self.furniture_contact, hand_contact=self.hand_contact,
            fall=self.fall, self_collision=self.self_collision, nonfoot_floor_contact=self.nonfoot_floor_contact, timeout=timeout,
            first_contact_part=self.first_contact_part, first_contact_time=self.first_contact_time,
            minimum_clearance=self.minimum_clearance if math.isfinite(self.minimum_clearance) else None,
            path_length=self.path_length, route_length=self.route_length,
            route_progress=min(self.progress/self.route_length, 1.0),
            progress_before_first_contact=min(self.progress_before_contact/self.route_length, 1.0),
            bottlenecks_cleared=self.bottlenecks_cleared,
            bottlenecks_before_first_contact=self.bottlenecks_before_contact,
            bottlenecks_total=len(self.bottlenecks), termination_reason=reason,
            numerical_failure=self.numerical_failure,
            provenance=self.provenance)


def summarize(records: Iterable[EpisodeRecord | dict]):
    rows = [r.to_dict() if isinstance(r, EpisodeRecord) else r for r in records]
    if not rows:
        raise ValueError("cannot summarize zero episodes")
    rate_keys = ["strict_success", "reached_goal", "hand_contact", "furniture_contact", "fall", "self_collision", "nonfoot_floor_contact", "timeout"]
    solved_times = [r["elapsed"] for r in rows if r["strict_success"]]
    return {
        "episodes": len(rows),
        **{f"{k}_rate": float(np.mean([r[k] for r in rows])) for k in rate_keys},
        "numerical_failure_rate": float(np.mean([r.get("numerical_failure", False) for r in rows])),
        "median_strict_success_time": float(np.median(solved_times)) if solved_times else None,
        "mean_capped_completion_time": float(np.mean([
            r["elapsed"] if r["strict_success"] else r["time_budget"] for r in rows])),
        "mean_route_progress_before_first_contact": float(np.mean([r["progress_before_first_contact"] for r in rows])),
        "mean_bottlenecks_before_first_contact": float(np.mean([r["bottlenecks_before_first_contact"] for r in rows])),
    }


def paired_comparison(reference: Iterable[dict], candidate: Iterable[dict], *, seed=0, resamples=2000):
    """Paired effects with whole-scene bootstrap clusters (all seeds stay together)."""
    def index(rows):
        result = {}
        for row in rows:
            key = tuple(row[k] for k in ("scene_id", "case_id", "training_seed", "episode_seed"))
            if key in result:
                raise ValueError(f"duplicate episode identity: {key}")
            result[key] = row
        return result
    before, after = index(reference), index(candidate)
    if not before or before.keys() != after.keys():
        raise ValueError("paired controllers must have identical nonempty episode identities")
    groups = {}
    paired_times = []
    for key, ref in before.items():
        new = after[key]
        if not math.isclose(ref["time_budget"], new["time_budget"]):
            raise ValueError("paired time budgets differ")
        delta = [float(new["strict_success"])-float(ref["strict_success"]),
                 float(new["hand_contact"])-float(ref["hand_contact"])]
        groups.setdefault(key[0], []).append(delta)
        if ref["strict_success"] and new["strict_success"]:
            paired_times.append(new["elapsed"]-ref["elapsed"])
    clusters = [np.asarray(group) for group in groups.values()]
    effect = np.concatenate(clusters).mean(axis=0)
    interval = None
    if len(clusters) >= 2 and resamples > 0:
        rng = np.random.default_rng(seed)
        bootstrap = np.asarray([np.concatenate([clusters[j] for j in rng.integers(len(clusters), size=len(clusters))]).mean(axis=0)
                                for _ in range(resamples)])
        interval = np.quantile(bootstrap, [0.025, 0.975], axis=0).T.tolist()
    return {"episodes": len(before), "scene_clusters": len(clusters),
            "strict_success_rate_difference": float(effect[0]),
            "hand_contact_rate_difference": float(effect[1]),
            "scene_cluster_bootstrap_95_percent_intervals": interval,
            "interval_order": ["strict_success_rate_difference", "hand_contact_rate_difference"],
            "jointly_solved_episodes": len(paired_times),
            "median_paired_time_difference": float(np.median(paired_times)) if paired_times else None}
