"""Explicit CAT joint/feature contracts and bounded whole-body target generation."""
from itertools import product
import numpy as np
from cat_ppo.furniture.grippers import hand_envelope, geometry_contract

LEG_NAMES = [f"{side}_{joint}_joint" for side in ("left", "right")
             for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
WAIST_NAMES = [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
ARM_NAMES = [f"{side}_{joint}_joint" for side in ("left", "right")
             for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
JOINT_NAMES = LEG_NAMES + WAIST_NAMES + [f"{side}_{joint}_joint" for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
LEGACY_OBS_NAMES = LEG_NAMES + WAIST_NAMES + ARM_NAMES
PF_GROUPS = (("head", 1), ("pelv", 1), ("tors", 1), ("feet", 2), ("hands", 2), ("knees", 2), ("shlds", 2))
# Legacy physical-furniture experiment only. Native CAT whole-body training
# uses wholebody_observation_contract() and reuses its existing hand fields.
PROBE_SPECS = []
for side in ("left", "right"):
    envelope = hand_envelope(side)
    center, half_size = envelope["center"], envelope["half_size"]
    PROBE_SPECS.append((f"{side}_palm", f"{side}_wrist_yaw_link", tuple(center), 0.03))
    for i, signs in enumerate(product((-1, 1), repeat=3)):
        point = tuple(c + s * h for c, s, h in zip(center, signs, half_size))
        PROBE_SPECS.append((f"{side}_hand_corner_{i}", f"{side}_wrist_yaw_link", point, 0.0))
    PROBE_SPECS.append((f"{side}_forearm", f"{side}_elbow_link", (0.07, 0.0, -0.005), 0.05))
    PROBE_SPECS.append((f"{side}_elbow", f"{side}_elbow_link", (0.0, 0.0, 0.0), 0.05))
PROBE_CHANNELS = ("clearance", "clearance_0.2s", "clearance_0.4s", "boundary_x", "boundary_y", "boundary_z", "age_seconds", "unknown", "position_uncertainty_m")


def joint_names(action_dofs=29):
    if action_dofs not in (12, 23, 29):
        raise ValueError("action_dofs must be 12, 23 or 29")
    actions = {12: LEG_NAMES, 23: LEGACY_OBS_NAMES, 29: JOINT_NAMES}[action_dofs]
    observed = JOINT_NAMES if action_dofs == 29 else LEGACY_OBS_NAMES
    return list(actions), list(observed)


def motor_targets(action, previous, nominal, lower, upper, action_indices,
                  *, leg_scale=0.5, upper_scale=0.8, upper_rate=2.0, dt=0.02, xp=np):
    """Original leg increments; upper-body nominal offsets with rad/s slew limits.

    Inactive joints return to nominal. Actions and final targets are bounded;
    nonfinite actions are handled by the environment's numerical-failure gate.
    """
    action = xp.clip(action, -1.0, 1.0)
    ids = xp.asarray(action_indices)
    requested = nominal[ids] + action * upper_scale
    leg = ids < 12
    requested = xp.where(leg, previous[ids] + action * leg_scale, requested)
    requested = xp.where(leg, requested, xp.clip(requested, previous[ids] - upper_rate * dt, previous[ids] + upper_rate * dt))
    requested = xp.clip(requested, lower[ids], upper[ids])
    if hasattr(nominal, "at"):
        return nominal.at[ids].set(requested)
    result = np.array(nominal, copy=True)
    result[np.asarray(ids)] = requested
    return result


def posture_actions(action, action_names, mode="learned", *, probe_features=None,
                    upper_scale=0.8, xp=np):
    """Explicit comparison heuristics; bounded targets are applied afterward.

    These presets are not validated dynamic controllers. Contextual mode picks
    a per-arm preset from observed hand clearance and upward obstruction.
    """
    modes = ("learned", "nominal", "raised", "tucked", "contextual")
    if mode not in modes:
        raise ValueError(f"upper_body_mode must be one of {modes}")
    if mode == "learned":
        return action
    if mode == "contextual" and probe_features is None:
        raise ValueError("contextual posture requires observed probe features")
    offsets = []
    for name in action_names:
        side = "left" if name.startswith("left_") else "right"
        sign = 1.0 if side == "left" else -1.0
        # G1 elbow's positive axis lowers this forearm geometry. FK-verified
        # raised preset decreases elbow flexion, with arms kept near the body.
        raised = -0.8 if "shoulder_pitch" in name else (-0.15 * sign if "shoulder_roll" in name else (-0.8 if "elbow" in name else 0.0))
        tucked = -0.2 if "shoulder_pitch" in name else (-0.15 * sign if "shoulder_roll" in name else (0.8 if "elbow" in name else 0.0))
        offset = {"nominal": 0.0, "raised": raised, "tucked": tucked}.get(mode, 0.0)
        if mode == "contextual":
            start = 0 if side == "left" else len(PROBE_SPECS) // 2
            hand = probe_features[start:start + 9]
            danger = xp.min(hand[:, :3]) < 0.3
            overhead = xp.any((hand[:, 5] < -0.3) & (hand[:, 0] < 0.4))
            offset = xp.where(danger, xp.where(overhead, tucked, raised), 0.0)
        offsets.append(offset)
    fixed = xp.clip(xp.asarray(offsets) / max(float(upper_scale), 1e-6), -1., 1.)
    is_upper = xp.asarray([name not in LEG_NAMES for name in action_names])
    return xp.where(is_upper, fixed, action)


def _base_features(actions, observed):
    common = ([f"gyro.{a}" for a in "xyz"] + [f"gravity.{a}" for a in "xyz"]
        + [f"joint_position.{n}" for n in observed] + [f"joint_velocity.{n}" for n in observed]
        + [f"last_action.{n}" for n in actions] + [f"motor_target.{n}" for n in actions]
        + ["command.move", "command.x", "command.y", "command.yaw", "foot_height"]
        + [f"phase.{fn}.{side}" for fn in ("cos", "sin") for side in ("left", "right")])
    pf = []
    for group, count in PF_GROUPS:
        for field, axes in (("gf", "xyz"), ("bf", "xyz"), ("df", ("distance",))):
            pf.extend(f"pf.{group}.{field}.{i}.{a}" for i in range(count) for a in axes)
    hints = []
    for name, count in (("head_pos", 1), ("head_vel", 1), ("pelv_pos", 1), ("tors_pos", 1),
                        ("feet_pos", 2), ("feet_vel", 2), ("hands_pos", 2), ("hands_vel", 2),
                        ("knees_pos", 2), ("shlds_pos", 2)):
        hints.extend(f"hint.{name}.{i}.{a}" for i in range(count) for a in "xyz")
    hints += ["hint.torso_roll", "hint.torso_pitch", "hint.gait.left", "hint.gait.right",
              "hint.contact.left", "hint.contact.right", "hint.kp_scale", "hint.kd_scale"]
    hints += [f"hint.rfi.{n}" for n in JOINT_NAMES]
    return common + pf, common + [f"hint.linear_velocity.{a}" for a in "xyz"] + pf + hints


def legacy_observation_contract():
    actor, critic = _base_features(LEG_NAMES, LEGACY_OBS_NAMES)
    assert len(actor) == 162 and len(critic) == 250
    return {"schema": "cat-observation-contract-v1", "action_names": list(LEG_NAMES),
            "actor_features": actor, "critic_features": critic}


def observation_contract(action_dofs=29):
    """Legacy physical-furniture v1 contract; not the native CAT training input."""
    actions, observed = joint_names(action_dofs)
    actor, critic = _base_features(actions, observed)
    extras = [f"probe.{name}.{channel}" for name, _, _, _ in PROBE_SPECS for channel in PROBE_CHANNELS]
    return {"schema": "cat-furniture-observation-v1", "action_names": actions,
            "observed_joint_names": observed, "actor_features": actor + extras,
            "critic_features": critic + extras, "baseline_actor_size": len(actor),
            "baseline_critic_size": len(critic), "prediction_horizons_seconds": [0.2, 0.4],
            "hand_geometry": geometry_contract()}


ELBOW_SITES = ("left_elbow_probe", "right_elbow_probe")
ELBOW_RADIUS_M = 0.05


SDF_RATE_FEATURES = [f"pf.hands.dfrate.{i}" for i in range(2)] + [f"pf.elbows.dfrate.{i}" for i in range(2)]


def wholebody_observation_contract(sdf_rate=False):
    """Shared native/JAX contract: hand sphere fields plus one sample/elbow.
    """
    actor, critic = _base_features(JOINT_NAMES, JOINT_NAMES)
    # Match CAT's per-group layout: both GF vectors, both BF vectors, distances.
    elbows = [f"pf.elbows.{field}.{i}.{axis}"
              for field, axes in (("gf", "xyz"), ("bf", "xyz"), ("df", ("distance",)))
              for i in range(2) for axis in axes]
    return {"schema": "cat-wholebody-spheres-v2", "action_names": list(JOINT_NAMES),
            "observed_joint_names": list(JOINT_NAMES),
            # sdf_rate: optional rate of change of the hand/elbow distance samples (actor only), so
            # the actor can tell an object closing in from one that sits still; the critic already
            # sees keypoint velocities. Added as a function-preserving expansion (zero input weights).
            "actor_features": actor + elbows + (list(SDF_RATE_FEATURES) if sdf_rate else []), "critic_features": critic + elbows,
            "sdf_rate_features": bool(sdf_rate),
            "baseline_actor_size": len(actor), "baseline_critic_size": len(critic),
            "prediction_horizons_seconds": [], "field_sample_count": 13,
            "additional_field_samples": ["left_elbow", "right_elbow"],
            "hand_field_semantics": "existing hand slots: sphere-center GF/BF and SDF(center)-fixed radius",
            "elbow_radius_m": ELBOW_RADIUS_M,
            "hand_geometry": geometry_contract(include_spheres=True)}


def route_coordinate(position, route, xp=np):
    """Nearest polyline projection: cumulative distance and lateral error."""
    starts, delta = route[:-1], route[1:] - route[:-1]
    lengths = xp.linalg.norm(delta, axis=-1)
    t = xp.clip(xp.sum((position - starts) * delta, axis=-1) / xp.maximum(lengths**2, 1e-12), 0, 1)
    projections = starts + t[:, None] * delta
    distance = xp.linalg.norm(position - projections, axis=-1)
    segment = xp.argmin(distance)
    offsets = xp.concatenate([xp.zeros(1), xp.cumsum(lengths)[:-1]])
    return offsets[segment] + t[segment] * lengths[segment], distance[segment]


def mjlab_observation_contract(sdf_rate=False):
    """Native geometry/sensor contract: actor 222 (226 with sdf_rate) / critic 310."""
    return wholebody_observation_contract(sdf_rate)
