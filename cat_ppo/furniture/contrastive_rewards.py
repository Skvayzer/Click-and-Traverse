"""Geometry-conditioned passage objectives without additional observations.

Zones depend only on ordered route progress. Hand targets use route-relative
XY coordinates and absolute world height, so yaw or crouching cannot move the
target with the robot. The two alternatives are paired left/right regions.
"""
from __future__ import annotations

import math

import jax.numpy as jp
import numpy as np

SCHEMA = "hand-contrast-v1"
MAX_ZONES = 6
MAX_REGIONS = 2
ROLES = ("open", "forward_protected", "narrow", "transition")


def pack_hand_contrast(scenes):
    """Validate and pack optional scene metadata as small immutable arrays."""
    scenes = list(scenes)
    if not scenes:
        raise ValueError("At least one scene slot is required")
    count = len(scenes)
    arrays = {
        "enabled": np.zeros(count, dtype=bool),
        "role": np.zeros(count, dtype=np.int32),
        "zone_valid": np.zeros((count, MAX_ZONES), dtype=bool),
        "start_m": np.zeros((count, MAX_ZONES), dtype=np.float32),
        "end_m": np.zeros((count, MAX_ZONES), dtype=np.float32),
        "fade_m": np.zeros((count, MAX_ZONES), dtype=np.float32),
        "forward_weight": np.zeros((count, MAX_ZONES), dtype=np.float32),
        "hand_active": np.zeros((count, MAX_ZONES, 2), dtype=bool),
        "hand_regions_min": np.zeros((count, MAX_ZONES, MAX_REGIONS, 2, 3), dtype=np.float32),
        "hand_regions_max": np.zeros((count, MAX_ZONES, MAX_REGIONS, 2, 3), dtype=np.float32),
        "region_valid": np.zeros((count, MAX_ZONES, MAX_REGIONS), dtype=bool),
    }
    for scene_index, scene in enumerate(scenes):
        metadata = None if scene is None else scene.get("hand_contrast")
        if metadata is None:
            continue
        if (metadata.get("schema") != SCHEMA or metadata.get("role") not in ROLES
                or not isinstance(metadata.get("group_id"), (str, int))):
            raise ValueError("Invalid hand contrast schema, role or group identity")
        radius = metadata.get("navigation_radius_m", math.nan)
        if not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 0 < radius <= .23:
            raise ValueError("Hand contrast navigation radius must be in (0, .23]")
        zones = metadata.get("zones")
        if not isinstance(zones, list) or not 1 <= len(zones) <= MAX_ZONES:
            raise ValueError(f"Hand contrast requires 1..{MAX_ZONES} zones")
        arrays["enabled"][scene_index] = True
        arrays["role"][scene_index] = ROLES.index(metadata["role"])
        previous_end = -math.inf
        for index, zone in enumerate(zones):
            scalars = np.asarray([zone.get(key, math.nan) for key in
                                  ("start_m", "end_m", "fade_m", "forward_weight")], dtype=float)
            if not np.isfinite(scalars).all():
                raise ValueError("Hand contrast zone parameters must be finite")
            start, end, fade, weight = scalars
            if (start < 0 or end <= start or start < previous_end or fade < 0
                    or fade > (end - start) / 2 or not 0 <= weight <= 1):
                raise ValueError("Hand contrast zones must be ordered, nonoverlapping, with valid fades/weights")
            previous_end = end
            active = np.asarray(zone.get("hand_active"))
            valid = np.asarray(zone.get("region_valid"))
            lower = np.asarray(zone.get("hand_regions_min"), dtype=float)
            upper = np.asarray(zone.get("hand_regions_max"), dtype=float)
            if (active.shape != (2,) or valid.shape != (MAX_REGIONS,)
                    or not np.isin(active, [False, True]).all()
                    or not np.isin(valid, [False, True]).all()
                    or lower.shape != (MAX_REGIONS, 2, 3) or upper.shape != lower.shape
                    or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                    or np.any(lower > upper)
                    or (active.any() and not valid.any())):
                raise ValueError("Invalid paired hand regions or arm activation")
            for key, value in zip(("start_m", "end_m", "fade_m", "forward_weight"), scalars):
                arrays[key][scene_index, index] = value
            arrays["zone_valid"][scene_index, index] = True
            arrays["hand_active"][scene_index, index] = active
            arrays["region_valid"][scene_index, index] = valid
            arrays["hand_regions_min"][scene_index, index] = lower
            arrays["hand_regions_max"][scene_index, index] = upper
    return arrays


def hand_contrast_context(metadata, progress, tangent):
    """Evaluate one scene's arrays; never blend targets across separate zones."""
    inside = (metadata["zone_valid"] & (progress >= metadata["start_m"])
              & (progress < metadata["end_m"]))
    enabled = metadata["enabled"] & jp.any(inside)
    index = jp.argmax(inside.astype(jp.int32))
    start, end, fade = (metadata[key][index] for key in ("start_m", "end_m", "fade_m"))
    ramp = jp.clip(jp.minimum(progress - start, end - progress) / jp.maximum(fade, 1e-8), 0., 1.)
    phase_weight = jp.where(enabled, jp.where(fade > 0., ramp, 1.), 0.)
    tangent = jp.asarray(tangent)
    tangent = tangent / jp.maximum(jp.linalg.norm(tangent), 1e-8)
    return {
        "enabled": enabled, "role": metadata["role"], "zone_index": index,
        "phase_weight": phase_weight,
        # Metrics use only the fully active plateau, and require visiting each
        # relevant zone. These small bookkeeping masks are not observations.
        "core_active": enabled & (phase_weight >= 1. - 1e-6),
        "required_forward_zones": (metadata["enabled"] & metadata["zone_valid"]
                                   & (metadata["forward_weight"] > 0.)),
        "required_hand_zones": (metadata["enabled"] & metadata["zone_valid"]
                                & jp.any(metadata["hand_active"], axis=-1)),
        "forward_weight": metadata["forward_weight"][index] * phase_weight,
        # Arm-neutral relaxation includes the entire lead-in/out, even where
        # the reward ramp itself is zero at the exact first boundary.
        "hand_active": metadata["hand_active"][index] & enabled,
        "hand_regions_min": metadata["hand_regions_min"][index],
        "hand_regions_max": metadata["hand_regions_max"][index],
        "region_valid": metadata["region_valid"][index] & enabled,
        "route_tangent": tangent,
    }


def eval_context(arrays, sceneidx, progress, tangent):
    """Select a scene from packed arrays and evaluate its current zone."""
    return hand_contrast_context({key: jp.asarray(value)[sceneidx] for key, value in arrays.items()},
                                 progress, tangent)


def reward_terms(context, hands_world, root_xy, pelvis_forward, torso_forward, *, region_scale=.15,
                 hand_good_distance=.05):
    """Return bounded costs and posture diagnostics using existing simulator state.

    Heading averages 1-cos(error) for pelvis and torso. Hand error is the mean
    smooth bounded distance d²/(d²+scale²) over active arms, minimized over valid paired
    regions. Reaching any allowed pair makes the cost zero with no extra bonus.
    The separate posture metric tolerates 5 cm outside the region for natural
    walking motion; successful episodes must still pass collision checks.
    """
    if not math.isfinite(region_scale) or region_scale <= 0:
        raise ValueError("Hand contrast region scale must be positive and finite")
    if not math.isfinite(hand_good_distance) or hand_good_distance < 0:
        raise ValueError("Hand posture metric tolerance must be nonnegative and finite")
    tangent = context["route_tangent"]
    headings = jp.stack([jp.asarray(pelvis_forward)[:2], jp.asarray(torso_forward)[:2]])
    cosine = jp.clip(jp.sum(headings * tangent, axis=-1)
                     / jp.maximum(jp.linalg.norm(headings, axis=-1), 1e-8), -1., 1.)
    heading_cost = jp.mean(1. - cosine) * context["forward_weight"]
    hands = jp.asarray(hands_world)
    delta = hands[:, :2] - jp.asarray(root_xy)
    local = jp.stack([delta @ tangent, delta @ jp.array([-tangent[1], tangent[0]]), hands[:, 2]], axis=-1)
    distance = jp.maximum(context["hand_regions_min"] - local[None], 0.) + jp.maximum(
        local[None] - context["hand_regions_max"], 0.)
    squared_distance = jp.sum(distance ** 2, axis=-1)
    active = context["hand_active"].astype(jp.float32)
    # A hard cap at one would make the new shaping flat outside 15 cm, where
    # nominal hands often start. This bounded form still grades progress from
    # every finite distance, without rewarding motion farther inside a region.
    bounded_distance = squared_distance / (squared_distance + region_scale ** 2)
    per_region = jp.sum(bounded_distance * active, axis=-1)
    per_region /= jp.maximum(jp.sum(active), 1.)
    # Valid costs are in [0, 1]; a finite sentinel also keeps derivatives and
    # inactive-zone arithmetic finite when neither alternative is enabled.
    paired_cost = jp.min(jp.where(context["region_valid"], per_region, 2.))
    any_active = jp.any(context["hand_active"]) & jp.any(context["region_valid"]) & context["enabled"]
    hand_cost = jp.where(any_active, paired_cost * context["phase_weight"], 0.)
    inside_pair = jp.all((squared_distance <= hand_good_distance ** 2 + 1e-10)
                         | ~context["hand_active"][None], axis=-1)
    hand_good = ~any_active | jp.any(inside_pair & context["region_valid"])
    heading_good = jp.all(cosine >= math.cos(math.radians(15.)))
    return {
        "wholebody_hand_contrast_heading": heading_cost,
        "wholebody_hand_contrast_region": hand_cost,
    }, {
        "hand_contrast_heading_cost": heading_cost,
        "hand_contrast_region_cost": hand_cost,
        "hand_contrast_heading_good": heading_good.astype(jp.float32),
        "hand_contrast_hand_good": hand_good.astype(jp.float32),
        "hand_contrast_forward_weight": context["forward_weight"],
        "hand_contrast_hand_active": any_active.astype(jp.float32),
    }
