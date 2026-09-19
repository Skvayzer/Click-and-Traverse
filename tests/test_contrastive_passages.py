"""Actual proxy/field checks for contrastive arm protection, not image tests."""
import copy
import math

import numpy as np
import pytest

from cat_ppo.furniture.contrastive_passages import (
    ROLES, _floor_referenced_pose, certify_contrastive_scene, generate_contrastive_group, module_free_intervals,
)
from cat_ppo.furniture.hand_passages import _pose_geometry, _robot, arm_pose
from cat_ppo.furniture.scenes import _digest, validate_scene


@pytest.fixture(scope="module")
def group():
    return generate_contrastive_group(8101)


def test_matched_group_changes_clearance_not_arrangement():
    scenes = generate_contrastive_group(8101, certify=False)
    assert [scene["hand_contrast"]["role"] for scene in scenes] == list(ROLES)
    assert scenes == generate_contrastive_group(8101, certify=False)
    assert scenes[0]["geometry_hash"] != generate_contrastive_group(8102, certify=False)[0]["geometry_hash"]
    assert len({s["hand_contrast"]["group_id"] for s in scenes}) == 1
    assert len({s["geometry_hash"] for s in scenes}) == 4
    for scene in scenes:
        validate_scene(scene)
        assert scene["route"] == scenes[0]["route"]
        assert scene["generator"] == scenes[0]["generator"]
        assert scene["source"]["hand_contrast"] == scene["hand_contrast"]
        assert scene["feasibility"]["root_route_validated"]
        assert "certificate" not in scene["hand_contrast"]
    transition = scenes[-1]["hand_contrast"]["modules"]
    assert [module["role"] for module in transition] == ["forward_protected", "narrow", "forward_protected"]


def test_actual_certificates_have_intended_contrast_and_safe_transitions(group):
    for scene in group:
        contrast = scene["hand_contrast"]
        certificate = contrast["certificate"]
        assert certificate["primitive_count"] == 35
        assert certificate["geometry_hash"] == scene["geometry_hash"]
        assert certificate["route_hash"] == _digest(scene["route"])
        assert certificate["route_transition_validated"]
        assert certificate["body_min_separation_m"] > .008
        assert certificate["hand_field_min_clearance_m"] > (.30 if contrast["role"] == "open" else .025)
        assert certificate["transition_sample_count"] >= 2
        assert certificate["stance_sample_count"] >= 564
        assert certificate["sampled_kinematic_stances_validated"]
        assert not certificate["dynamic_feasibility_validated"]
        assert not certificate["complete_se2_search"]
        assert not certificate["target_region_full_arm_ik_certified"]
        for module in certificate["module_audit"]:
            if module["role"] == "open":
                assert module["nominal_forward_body_min_m"] > .06
            elif module["role"] == "forward_protected":
                assert module["nominal_forward_hand_min_m"] < -.002
                assert module["raised_forward_body_min_m"] > .03
            else:
                assert module["nominal_forward_body_min_m"] < -.005
                assert module["raised_forward_body_min_m"] < -.005
                assert module["tucked_forward_body_min_m"] < -.005
                assert module["nominal_sideways_body_min_m"] > .03


def test_target_boxes_are_swept_field_certified_and_zones_leave_turn_bays(group):
    for scene in group:
        contrast = scene["hand_contrast"]
        zones = contrast["zones"]
        for zone in zones:
            lo, hi = np.asarray(zone["hand_regions_min"]), np.asarray(zone["hand_regions_max"])
            assert lo.shape == hi.shape == (2, 2, 3)
            assert (lo < hi).all()
            assert zone["start_m"] >= 0
            assert zone["end_m"] <= scene["route_length_m"]
            assert zone["start_m"] + zone["fade_m"] < zone["hazard_start_m"]
            assert zone["hazard_end_m"] < zone["end_m"] - zone["fade_m"]
        for a, b in zip(zones, zones[1:]):
            assert a["end_m"] < b["start_m"]
        for alternatives in contrast["certificate"]["target_region_audit"]:
            assert alternatives[0]["valid"]
            assert alternatives[0]["analytic_swept_hand_min_clearance_m"] > .005
            assert alternatives[0]["field_swept_hand_min_clearance_m"] > .025


def test_crouches_bend_legs_and_keep_feet_on_reference_floor():
    _, compiled = _robot()
    feet = [compiled["shape_names"].index(f"{side}_upper_foot") for side in ("left", "right")]
    baseline = arm_pose("raised", .95)
    floor_z = _pose_geometry(baseline)["centers"][feet, 2].min()
    for variant in ("crouch_shallow", "crouch_deep", "step_left", "step_right"):
        q = _floor_referenced_pose("raised", variant)
        assert np.any(abs(q[7:19] - baseline[7:19]) > .01)
        np.testing.assert_allclose(_pose_geometry(q)["centers"][feet, 2].min(), floor_z, atol=1e-6)
        if variant.startswith("crouch"):
            assert q[2] < baseline[2]


def test_bad_hand_target_is_rejected(group):
    scene = copy.deepcopy(group[1])
    # Move the supposedly safe raised target straight into the left cabinet.
    for zone in scene["hand_contrast"]["zones"]:
        zone["hand_regions_min"][0][0][1] += .4
        zone["hand_regions_max"][0][0][1] += .4
    with pytest.raises(ValueError, match="Raised target region lacks"):
        certify_contrastive_scene(scene)


@pytest.mark.parametrize("seed,split", [(True, "train"), (-1, "train"), (1, "unknown")])
def test_invalid_arguments(seed, split):
    with pytest.raises(ValueError):
        generate_contrastive_group(seed, split)


def test_no_exterior_bypass_and_missing_baffle_is_rejected(group):
    for scene in group:
        assert scene["hand_contrast"]["certificate"]["exterior_lanes_sealed"]
        for module in scene["hand_contrast"]["modules"]:
            np.testing.assert_allclose(module_free_intervals(scene, module),
                                       [[-module["width_m"] / 2, module["width_m"] / 2]], atol=1e-7)
    broken = copy.deepcopy(group[1])
    broken["boxes"] = [box for box in broken["boxes"] if box["name"] != "baffle_0_+1"]
    broken["geometry_hash"] = _digest(dict(boxes=broken["boxes"], room_dimensions=broken["room_dimensions"]))
    with pytest.raises(ValueError, match="Exterior bypass"):
        certify_contrastive_scene(broken)
