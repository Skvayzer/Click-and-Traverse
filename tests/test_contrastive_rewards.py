"""Contextual posture choices with native collision/observation semantics."""
import jax
import jax.numpy as jp
import numpy as np
import pytest

from cat_ppo.furniture.contrastive_rewards import (
    MAX_ZONES, eval_context, pack_hand_contrast, reward_terms,
)
from cat_ppo.furniture.wholebody_stability import upper_stability_terms


def zone(start=1., end=3., *, active=(True, True), weight=1., height=1.):
    lower = [[[.2, .05, height], [.2, -.2, height]],
             [[-.15, .05, .6], [-.15, -.2, .6]]]
    upper = [[[.5, .2, height + .15], [.5, -.05, height + .15]],
             [[.05, .2, .75], [.05, -.05, .75]]]
    return dict(start_m=start, end_m=end, fade_m=.2, forward_weight=weight,
                hand_active=list(active), hand_regions_min=lower,
                hand_regions_max=upper, region_valid=[True, True])


def scene(*zones, role="forward_protected"):
    return {"hand_contrast": dict(schema="hand-contrast-v1", group_id="matched-0", role=role,
                                  navigation_radius_m=.15, zones=list(zones or [zone()]))}


def context(*zones, progress=2., role="forward_protected", tangent=(1., 0.)):
    return eval_context(pack_hand_contrast([scene(*zones, role=role)]), 0, progress, jp.array(tangent))


RAISED = jp.array([[.35, .125, 1.075], [.35, -.125, 1.075]])
TUCKED = jp.array([[-.05, .125, .675], [-.05, -.125, .675]])


def terms(ctx, hands=RAISED, *, root=(0., 0.), pelvis=(1., 0.), torso=(1., 0.)):
    return reward_terms(ctx, hands, jp.array(root), jp.array(pelvis), jp.array(torso))


def test_paired_regions_accept_either_complete_pose_without_additional_height_bonus():
    ctx = context()
    for pose in (RAISED, TUCKED, RAISED.at[:, 2].add(.03)):
        costs, metrics = terms(ctx, pose)
        assert costs["wholebody_hand_contrast_region"] == 0
        assert metrics["hand_contrast_hand_good"] == 1
    mixed = RAISED.at[1].set(TUCKED[1])
    costs, metrics = terms(ctx, mixed)
    assert costs["wholebody_hand_contrast_region"] > 0
    assert metrics["hand_contrast_hand_good"] == 0


def test_region_distance_masks_inactive_arm_and_is_bounded():
    ctx = context(zone(active=(True, False)))
    costs, metrics = terms(ctx, RAISED.at[1].set(jp.array([50., 50., 50.])))
    assert costs["wholebody_hand_contrast_region"] == 0
    assert metrics["hand_contrast_hand_good"] == 1
    costs, _ = terms(ctx, jp.full((2, 3), 50.))
    assert .99 < costs["wholebody_hand_contrast_region"] < 1


def test_region_shaping_still_grades_progress_farther_than_its_normalization_scale():
    ctx = context()
    def cost(height):
        return terms(ctx, RAISED.at[:, 2].add(height))[0]["wholebody_hand_contrast_region"]
    assert 0 < cost(.5) < cost(1.) < 1
    assert jax.grad(cost)(jp.array(1.)) > 0


def test_heading_checks_both_pelvis_and_torso_and_only_requested_forward_zones():
    ctx = context()
    forward, _ = terms(ctx)
    sideways, metrics = terms(ctx, pelvis=(0., 1.), torso=(0., 1.))
    twisted, _ = terms(ctx, pelvis=(0., 1.))
    backwards, _ = terms(ctx, pelvis=(-1., 0.), torso=(-1., 0.))
    assert forward["wholebody_hand_contrast_heading"] == 0
    assert sideways["wholebody_hand_contrast_heading"] == 1
    assert twisted["wholebody_hand_contrast_heading"] == .5
    assert backwards["wholebody_hand_contrast_heading"] == 2
    assert metrics["hand_contrast_heading_good"] == 0
    narrow, _ = terms(context(zone(weight=0., active=(False, False)), role="narrow"),
                      pelvis=(0., 1.), torso=(0., 1.))
    assert all(float(value) == 0. for value in narrow.values())


def test_route_frame_rotates_xy_but_absolute_height_prevents_crouch_shortcut():
    ctx = context(tangent=(0., 1.))
    root = jp.array([2., 3.])
    rotated = RAISED.at[:, :2].set(jp.stack([-RAISED[:, 1], RAISED[:, 0]], axis=-1) + root)
    costs, _ = terms(ctx, rotated, root=root, pelvis=(0., 1.), torso=(0., 1.))
    assert all(float(value) == 0. for value in costs.values())
    lowered, metrics = terms(ctx, rotated.at[:, 2].add(-.3), root=root,
                             pelvis=(0., 1.), torso=(0., 1.))
    assert lowered["wholebody_hand_contrast_region"] > 0
    assert metrics["hand_contrast_hand_active"] == 1
    # Turning cannot disable the phase or carry the hand regions with the body.
    turned, _ = terms(ctx, rotated, root=root)
    assert turned["wholebody_hand_contrast_heading"] == 1
    assert ctx["enabled"] and np.all(ctx["hand_active"])


def test_phase_has_clear_leadin_exit_and_never_blends_targets_across_gap():
    arrays = pack_hand_contrast([None, scene(zone(), zone(4., 6., height=1.4))])
    at = lambda progress: eval_context(arrays, 1, progress, jp.array([1., 0.]))
    assert arrays["zone_valid"].shape == (2, MAX_ZONES)
    assert not eval_context(arrays, 0, 2., jp.array([1., 0.]))["enabled"]
    assert not at(.9)["enabled"]
    assert at(1.)["enabled"] and np.all(at(1.)["hand_active"])
    assert at(1.)["phase_weight"] == 0
    assert not at(1.)["core_active"]
    assert float(at(1.1)["phase_weight"]) == pytest.approx(.5, abs=1e-6)
    assert not at(1.1)["core_active"]
    assert at(1.2)["core_active"]
    np.testing.assert_array_equal(at(1.2)["required_forward_zones"], [1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(at(1.2)["required_hand_zones"], [1, 1, 0, 0, 0, 0])
    for progress in (3., 3.5, 6.):
        gap = at(progress)
        assert not gap["enabled"] and not np.any(gap["hand_active"])
        assert not gap["core_active"]
        assert all(float(v) == 0. for v in terms(gap)[0].values())
    assert at(4.5)["hand_regions_min"][0, 0, 2] == pytest.approx(1.4)


def test_region_invalid_alternative_cannot_win():
    z = zone()
    z["region_valid"] = [True, False]
    assert terms(context(z), TUCKED)[0]["wholebody_hand_contrast_region"] > 0


def test_posture_metric_tolerates_small_walking_motion_without_zeroing_reward_cost():
    ctx = context()
    # Upper target height is 1.15 m; this hand is 4 cm above its region.
    near = RAISED.at[0, 2].set(1.19)
    costs, metrics = terms(ctx, near)
    assert costs["wholebody_hand_contrast_region"] > 0
    assert metrics["hand_contrast_hand_good"] == 1
    far = near.at[0, 2].set(1.21)
    assert terms(ctx, far)[1]["hand_contrast_hand_good"] == 0


@pytest.mark.parametrize("profile", [False, True])
def test_arm_neutral_release_persists_after_clearance_and_preserves_waist_motion(profile):
    target = jp.full(17, .2)
    args = (target, target, target, jp.zeros(17), jp.full((2, 1), .8), jp.full(2, .8))
    original, _ = upper_stability_terms(*args, dt=.02, hand_protection=profile)
    released, _ = upper_stability_terms(*args, dt=.02, hand_protection=profile,
                                        contrast_arm_active=jp.array([True, False]))
    assert released["wholebody_upper_clear_posture"] < original["wholebody_upper_clear_posture"]
    for key in ("wholebody_upper_target_velocity", "wholebody_upper_target_acceleration"):
        np.testing.assert_array_equal(released[key], original[key])
    waist_only = jp.zeros(17).at[:3].set(.2)
    args = (waist_only, waist_only, waist_only, jp.zeros(17), jp.full((2, 1), .8), jp.full(2, .8))
    old, _ = upper_stability_terms(*args, dt=.02, hand_protection=profile)
    new, _ = upper_stability_terms(*args, dt=.02, hand_protection=profile,
                                   contrast_arm_active=jp.array([True, True]))
    np.testing.assert_array_equal(old["wholebody_upper_clear_posture"], new["wholebody_upper_clear_posture"])
    restored, _ = upper_stability_terms(*args, dt=.02, hand_protection=profile,
                                        contrast_arm_active=jp.array([False, False]))
    for key in old:
        np.testing.assert_array_equal(restored[key], old[key])


def test_native_sdf_budget_rejects_small_positive_clearance_as_reward_certificate():
    from cat_ppo.envs.g1.env_cat import G1CatEnv
    native = lambda distance: float(G1CatEnv._re_sdf(None, jp.full((2, 1), distance)))
    assert native(.034) < -23.
    assert -.61 < native(.12) < -.59
    assert -.019 < native(.19) < -.018
    # With the proposed -1 heading scale, a comfortably protected forward pose
    # outranks sideways; mere positive SDF at 34 mm does not. Keep native SDF.
    forward_costs, _ = terms(context())
    sideways_costs, _ = terms(context(), pelvis=(0., 1.), torso=(0., 1.))
    score = lambda distance, costs: native(distance) - sum(float(v) for v in costs.values())
    assert score(.12, forward_costs) > score(.19, sideways_costs)
    assert score(.034, forward_costs) < score(.19, sideways_costs)
    # This does not impose a universal 12 cm hard margin on narrow passages.
    narrow = context(zone(weight=0., active=(False, False)), role="narrow")
    assert sum(float(v) for v in terms(narrow, pelvis=(0., 1.), torso=(0., 1.))[0].values()) == 0


def test_generated_protected_geometry_prefers_raised_forward_in_static_field_posture_budget():
    """Include native SDF and added posture costs, not just the new heading term.

    This is a static shaping comparison at equal motion, not a prediction of
    full dynamic return. Collision validity and dynamic policy learning remain
    separate checks.
    """
    import math
    import mujoco
    from cat_ppo.envs.g1 import constants
    from cat_ppo.envs.g1.env_cat import G1CatEnv
    from cat_ppo.furniture.contrastive_passages import generate_contrastive_group, _poses
    from cat_ppo.furniture.generalist_fields import sample_ragged_field
    from cat_ppo.furniture.grippers import hand_sphere
    from cat_ppo.furniture.hand_passages import _robot, _sdf, arm_pose
    from cat_ppo.furniture.wholebody_stability import hand_clearance_pressure

    value = generate_contrastive_group(20260919, certify=False)[1]
    sdf, origin = _sdf(value)
    model, _ = _robot()
    data = mujoco.MjData(model)
    packed = pack_hand_contrast([value])
    nominal = jp.asarray(arm_pose()[19:])
    yaw = value["generator"]["passage_yaw_rad"]
    tangent = jp.array([math.cos(yaw), math.sin(yaw)])
    radii = jp.array([hand_sphere(side)["radius"] for side in ("left", "right")])

    def sample(names):
        positions = np.array([data.site(name).xpos for name in names])
        return sample_ragged_field(sdf.reshape(-1, 1), positions, origin=origin,
                                   dx=.04, shape=np.array(sdf.shape), offset=0)

    def score(module, mode, turn):
        qpos = arm_pose(mode, .95)
        qpos[:2] = _poses(value, [module["center_local_x_m"]])[0, :2]
        heading = yaw + turn
        qpos[3:7] = [math.cos(heading / 2), 0., 0., math.sin(heading / 2)]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        hand_distance = sample(constants.HAND_SITES).reshape(2) - radii
        elbow_distance = sample(["left_elbow_probe", "right_elbow_probe"]).reshape(2)
        native = sum(float(G1CatEnv._re_sdf(None, sample(names))) for names in
                     (["head"], constants.FEET_SITES, constants.KNEE_SITES, constants.SHOULDER_SITES))
        native += float(G1CatEnv._re_sdf(None, hand_distance))
        ctx = eval_context(packed, 0, module["center_local_x_m"] + 3.5, tangent)
        hand_positions = jp.array([data.site(name).xpos for name in constants.HAND_SITES])
        heading_xy = jp.array([math.cos(heading), math.sin(heading)])
        contrast, _ = reward_terms(ctx, hand_positions, jp.asarray(qpos[:2]), heading_xy, heading_xy)
        target = jp.asarray(qpos[19:])
        costs, _ = upper_stability_terms(target, target, target, nominal, hand_distance,
            elbow_distance, dt=.02, hand_protection=True, contrast_arm_active=ctx["hand_active"])
        pressure, _ = hand_clearance_pressure(hand_distance)
        elbow_cost = jp.mean(jp.maximum(.08 - elbow_distance, 0.) ** 2)
        total = (native - sum(float(v) for v in contrast.values())
                 - .5 * float(pressure) - 2. * float(elbow_cost)
                 - .05 * float(costs["wholebody_upper_clear_posture"]))
        return total, hand_distance

    for module in value["hand_contrast"]["modules"]:
        raised_score, clearance = score(module, "raised", 0.)
        sideways_score, _ = score(module, "nominal", math.pi / 2)
        assert jp.min(clearance) > .10
        assert raised_score > sideways_score + .5


def test_context_and_rewards_compile_batched_with_finite_costs_including_inactive_zones():
    packed = pack_hand_contrast([scene()])
    evaluate = jax.jit(jax.vmap(lambda progress: terms(eval_context(packed, 0, progress, jp.array([1., 0.])))))
    costs, metrics = evaluate(jp.array([0., 1., 1.1, 2., 3., 4.]))
    assert all(np.isfinite(value).all() for value in (*costs.values(), *metrics.values()))


@pytest.mark.parametrize("mutation", ["overlap", "reversed", "nan", "no_target", "too_many", "radius"])
def test_invalid_metadata_fails_closed(mutation):
    value = scene()
    metadata = value["hand_contrast"]
    if mutation == "overlap":
        metadata["zones"].append(zone(2., 4.))
    elif mutation == "reversed":
        metadata["zones"][0]["hand_regions_min"][0][0][0] = 10.
    elif mutation == "nan":
        metadata["zones"][0]["forward_weight"] = float("nan")
    elif mutation == "no_target":
        metadata["zones"][0]["region_valid"] = [False, False]
    elif mutation == "too_many":
        metadata["zones"] = [zone(3. * i, 3. * i + 2.) for i in range(MAX_ZONES + 1)]
    else:
        metadata["navigation_radius_m"] = .3
    with pytest.raises(ValueError):
        pack_hand_contrast([value])


def test_config_is_opt_in_and_preserves_observation_action_and_native_reward_contracts():
    from cat_ppo.envs.g1.env_cat import g1_loco_task_config
    from cat_ppo.envs.g1.env_cat_wholebody import wholebody_config
    base = g1_loco_task_config().env_config
    old = wholebody_config(base, stabilization=True, hand_protection=True)
    new = wholebody_config(base, stabilization=True, hand_protection=True, hand_contrast=True)
    assert not old.wholebody_hand_contrast and new.wholebody_hand_contrast
    assert (new.num_obs, new.num_pri, new.num_act) == (222, 310, 29)
    assert new.hand_contrast_region_scale == .15
    assert new.reward_config.scales.wholebody_hand_contrast_heading == -1.
    assert new.reward_config.scales.wholebody_hand_contrast_region == -1.
    for key, value in old.reward_config.scales.items():
        assert new.reward_config.scales[key] == value
    assert new.term_collision_threshold == old.term_collision_threshold
    with pytest.raises(ValueError):
        wholebody_config(base, hand_contrast=True)
