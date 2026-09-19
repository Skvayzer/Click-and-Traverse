"""Physical upper-target regularization and shared CAT outcome diagnostics.

Costs use the filtered targets actually sent to the PD controller, not raw
actions. All costs are dimensionless; native CAT applies reward scales and dt.
The candidate weights belong to wholebody_config and are opt-in.
"""
import jax.numpy as jp
from mujoco_playground._src import collision


def native_fault_flags(env, data, info):
    """Exact compact CAT terminal causes, preserving simultaneous causes.

Obstacle fields represent clearance violations, not rigid obstacle contacts.
Unlike fields/self-contact, falls and numerical faults have no 50-step grace.
"""
    grace = info["step"] >= 50
    threshold = -env._config.term_collision_threshold
    flags = {
        "fall_inverted": env.get_gravity(data, "pelvis")[2] < 0.,
        "fall_head_low": info["head_pos"][2] < .7,
        "self_contact": grace & (
            collision.geoms_colliding(data, env._right_foot_geom_id, env._left_foot_geom_id)
            | collision.geoms_colliding(data, env._left_foot_geom_id, env._right_shin_geom_id)
            | collision.geoms_colliding(data, env._right_foot_geom_id, env._left_shin_geom_id)),
        "numerical": jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any(),
    }
    for label, key in (("head", "headdf"), ("pelvis", "pelvdf"), ("torso", "torsdf"),
                       ("feet", "feetdf"), ("hands", "handsdf"), ("knees", "kneesdf"),
                       ("shoulders", "shldsdf")):
        flags[label + "_field"] = grace & jp.any(info[key] < threshold)
    flags["elbows_field"] = (grace & jp.any(info["wholebody_elbow_clearance"] < threshold)
                             & env._config.terminate_on_elbow_collision)
    navigation = info.get("room_navigation", {})
    flags["room_root_field"] = navigation.get("enabled", jp.array(False)) & navigation.get("violation", jp.array(False))
    flags["body_collision"] = info.get("wholebody_body_collision", jp.array(False))
    flags["fall"] = flags["fall_inverted"] | flags["fall_head_low"]
    flags["obstacle"] = (jp.any(jp.stack([value for key, value in flags.items() if key.endswith("_field")]))
                         | flags["body_collision"])
    flags["any"] = flags["fall"] | flags["self_contact"] | flags["obstacle"] | flags["numerical"]
    return flags


def goal_status(env, data, info, prior_outside):
    """Geometric endpoint and path-bound diagnostics; does not change termination.

Callers must additionally exclude native terminal transitions from clean goal
credit. CAT allows approaches before minimum x but excludes lateral bypasses;
rooms require the root to remain inside the field in both horizontal axes.
"""
    if hasattr(env, "_pf_origins"):
        scene = env._field_pf_id
        origin, shape, dx = env._pf_origins[scene], env._pf_shapes[scene], env._pf_dxs[scene]
        plane = (env._pf_crossed_is_x_plane[scene] if getattr(env, "_pf_expanded", False)
                 else env._pf_scene_original[scene])
    else:
        origin = jp.asarray(env._config.pf_config.origin)
        shape = jp.asarray(env.sdf.shape[:3])
        dx = env._config.pf_config.dx
        plane = jp.array(True)
    extent = origin + (shape - 1) * dx
    root = data.qpos[:3]
    points = jp.concatenate([root[None], info["feet_pos"]], axis=0)
    lateral_exit = jp.any((points[:, 1] < origin[1]) | (points[:, 1] > extent[1]))
    xy_exit = jp.any((root[:2] < origin[:2]) | (root[:2] > extent[:2]))
    outside = prior_outside | jp.where(plane, lateral_exit, xy_exit)
    raw = env._crossed_goal(root) & jp.all(env._crossed_goal(info["feet_pos"])) & ~xy_exit
    return {"raw_goal": raw, "goal_reached": raw & ~outside, "outside_bounds": outside}


def hand_clearance_pressure(handsdf, *, target_clearance=.04,
                            anticipation_distance=.20, near_weight=.8):
    """Bounded hand-surface pressure, independent of posture or room family.

The near term protects a small attainable margin; the lighter early term gives
the policy a gradient before the hand is in immediate danger. Both terms are
zero outside their stated distances and saturate at contact. This is a cost,
not a height reward: raising, tucking and other clearance-improving poses are
judged by the same two inherited CAT hand distances.
"""
    if not (0. < target_clearance <= anticipation_distance and 0. <= near_weight <= 1.):
        raise ValueError("Require 0 < hand target <= anticipation distance and near weight in [0,1]")
    distance = jp.asarray(handsdf).reshape(2)
    near = jp.clip((target_clearance - distance) / target_clearance, 0., 1.)
    early = jp.clip((anticipation_distance - distance) / anticipation_distance, 0., 1.)
    pressure = near_weight * near ** 2 + (1. - near_weight) * early ** 2
    return jp.mean(pressure), {
        "hand_clearance_pressure": jp.mean(pressure),
        "left_hand_clearance_pressure": pressure[0],
        "right_hand_clearance_pressure": pressure[1],
        "hand_protection_active_fraction": jp.mean((distance < anticipation_distance).astype(jp.float32)),
    }


def upper_stability_terms(target, previous, previous_previous, nominal, handsdf,
                          elbow_clearance, *, dt, velocity_scale=2.,
                          acceleration_scale=20., posture_scale=.8,
                          hand_margin=.12, elbow_margin=.08, clearance_taper=.12,
                          waist_weight=4., hand_protection=False,
                          arm_velocity_weight=.5, arm_acceleration_weight=.1,
                          protection_target_clearance=.04,
                          protection_anticipation_distance=.20,
                          protection_near_weight=.8, protection_posture_taper=.10,
                          contrast_arm_active=None):
    """Dimensionless weighted physical-motion costs and interpretable telemetry.

First three of the17 upper joints are waist joints. Their weight is4; each arm
joint has weight1. Constant-velocity target motion has zero acceleration cost.
The nominal-posture cost is zero when any hand/elbow is within its protection
margin, then smoothly becomes active over12cm of additional clearance.

The opt-in hand-protection profile releases each arm's posture cost earlier
and independently. Waist stabilization retains its original motion weights
and normalization; waist posture regularization remains active near obstacles.
Arm acceleration/velocity contributions are reduced without renormalizing
the weights, so unchanged waist motion has exactly the same cost as before.
"""
    weights = jp.ones_like(target).at[:3].set(waist_weight)
    weighted_mean = lambda values: jp.sum(weights * values) / jp.sum(weights)
    velocity = (target - previous) / dt
    acceleration = (target - 2. * previous + previous_previous) / (dt * dt)
    offset = target - nominal
    comfortable = jp.minimum(jp.min((handsdf - hand_margin) / clearance_taper),
                             jp.min((elbow_clearance - elbow_margin) / clearance_taper))
    blend = jp.clip(comfortable, 0., 1.)
    clear_gate = blend * blend * (3. - 2. * blend)
    costs = {
        "wholebody_upper_target_velocity": weighted_mean((velocity / velocity_scale) ** 2),
        "wholebody_upper_target_acceleration": weighted_mean((acceleration / acceleration_scale) ** 2),
        "wholebody_upper_clear_posture": clear_gate * weighted_mean((offset / posture_scale) ** 2),
    }
    if hand_protection:
        # Upper-joint order is three waist joints, seven left-arm joints, then
        # seven right-arm joints. All inputs are already available in CAT.
        comfortable = jp.minimum(
            (jp.asarray(handsdf).reshape(2) - protection_anticipation_distance) / protection_posture_taper,
            (jp.asarray(elbow_clearance).reshape(2) - elbow_margin) / protection_posture_taper)
        blend = jp.clip(comfortable, 0., 1.)
        arm_gates = blend * blend * (3. - 2. * blend)
        if contrast_arm_active is not None:
            # Route zones retain the release after the hands reach clearance;
            # otherwise an effective protective pose would turn its own
            # nominal-posture penalty back on before the passage exit.
            arm_gates *= ~jp.asarray(contrast_arm_active, dtype=bool)
        posture_gates = jp.concatenate([jp.ones(3), jp.repeat(arm_gates, 7)])
        velocity_weights = weights.at[3:].multiply(arm_velocity_weight)
        acceleration_weights = weights.at[3:].multiply(arm_acceleration_weight)
        denominator = jp.sum(weights)
        costs = {
            "wholebody_upper_target_velocity": jp.sum(velocity_weights * (velocity / velocity_scale) ** 2) / denominator,
            "wholebody_upper_target_acceleration": jp.sum(acceleration_weights * (acceleration / acceleration_scale) ** 2) / denominator,
            "wholebody_upper_clear_posture": weighted_mean(posture_gates * (offset / posture_scale) ** 2),
        }
        clear_gate = jp.mean(arm_gates)
    elif contrast_arm_active is not None:
        # The optional zone release also works with the original stability
        # profile, preserving its existing waist gate and motion costs.
        arm_gates = clear_gate * ~jp.asarray(contrast_arm_active, dtype=bool)
        posture_gates = jp.concatenate([jp.full(3, clear_gate), jp.repeat(arm_gates, 7)])
        costs["wholebody_upper_clear_posture"] = jp.where(
            jp.any(contrast_arm_active), weighted_mean(posture_gates * (offset / posture_scale) ** 2),
            costs["wholebody_upper_clear_posture"])
    telemetry = {
        "upper_target_velocity_rms": jp.sqrt(jp.mean(velocity ** 2)),
        "upper_target_acceleration_rms": jp.sqrt(jp.mean(acceleration ** 2)),
        "upper_target_offset_rms": jp.sqrt(jp.mean(offset ** 2)),
        "upper_target_distance_normalized": jp.sqrt(jp.mean((offset / posture_scale) ** 2)),
        "waist_target_offset_rms": jp.sqrt(jp.mean(offset[:3] ** 2)),
        "clear_posture_gate": clear_gate,
        "minimum_hand_clearance": jp.min(handsdf),
        "minimum_elbow_clearance": jp.min(elbow_clearance),
        "hand_penetration": jp.any(handsdf < 0.).astype(jp.float32),
        "elbow_penetration": jp.any(elbow_clearance < 0.).astype(jp.float32),
    }
    if hand_protection:
        _, pressure = hand_clearance_pressure(
            handsdf, target_clearance=protection_target_clearance,
            anticipation_distance=protection_anticipation_distance,
            near_weight=protection_near_weight)
        telemetry.update(pressure)
        telemetry.update({
            "left_arm_posture_gate": arm_gates[0],
            "right_arm_posture_gate": arm_gates[1],
            "arm_target_offset_rms": jp.sqrt(jp.mean(offset[3:] ** 2)),
            "left_arm_target_offset_rms": jp.sqrt(jp.mean(offset[3:10] ** 2)),
            "right_arm_target_offset_rms": jp.sqrt(jp.mean(offset[10:] ** 2)),
            "arm_target_velocity_rms": jp.sqrt(jp.mean(velocity[3:] ** 2)),
            "arm_target_acceleration_rms": jp.sqrt(jp.mean(acceleration[3:] ** 2)),
            "left_shoulder_pitch_target_offset": offset[3],
            "right_shoulder_pitch_target_offset": offset[10],
            "left_elbow_target_offset": offset[6],
            "right_elbow_target_offset": offset[13],
        })
    return costs, telemetry


EPISODE_KEYS = ("goal_reached", "raw_goal", "fall", "obstacle", "self_contact", "numerical",
                "outside_bounds", "hand_violation", "elbow_violation", "body_collision",
                "body_collision_feet", "body_collision_legs", "body_collision_trunk",
                "body_collision_head", "body_collision_arms", "body_collision_hands", "reset_replaced")
