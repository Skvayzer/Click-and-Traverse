"""Geometric tests for the targeted arm-protection scene curriculum."""
import math

import numpy as np
import pytest

from cat_ppo.furniture.hand_passages import (
    DIFFICULTIES, KINDS, arm_pose, certify_hand_passage,
    generate_hand_passage_scene,
)
from cat_ppo.furniture.scenes import _digest, validate_scene


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_scene_geometry_route_and_metadata(kind, difficulty):
    scene = generate_hand_passage_scene(6001, kind=kind, difficulty=difficulty, certify=False)
    validate_scene(scene)
    assert scene["feasibility"]["root_route_validated"]
    assert scene["hand_protection"]["level"] == DIFFICULTIES.index(difficulty)
    assert scene["source"]["hand_protection"] == scene["hand_protection"]
    assert scene["family"] == ("furniture" if kind == KINDS[0] else "generic_clutter")
    assert scene == generate_hand_passage_scene(6001, kind=kind, difficulty=difficulty, certify=False)
    assert scene["geometry_hash"] != generate_hand_passage_scene(6002, kind=kind, difficulty=difficulty, certify=False)["geometry_hash"]


@pytest.mark.parametrize("kind, difficulty, seed", [(KINDS[0], "easy", 6001), (KINDS[1], "hard", 7009)])
def test_nominal_hands_blocked_protected_path_clears_real_training_checks(kind, difficulty, seed):
    scene = generate_hand_passage_scene(seed, kind=kind, difficulty=difficulty)
    certificate = scene["hand_protection"]["certificate"]
    assert certificate["primitive_count"] == 35
    assert certificate["nominal_forward_hand_collision_fraction"] > .12
    assert certificate["nominal_forward_hand_min_separation_m"] < -.002
    assert certificate["raised_route_body_min_separation_m"] > .008
    assert certificate["raised_route_hand_field_min_clearance_m"] > .025
    assert certificate["raised_90percent_route_body_min_separation_m"] > .008
    assert certificate["raised_90percent_route_hand_field_min_clearance_m"] > .025
    assert certificate["raised_transition_body_min_separation_m"] > .008
    assert certificate["raised_transition_hand_field_min_clearance_m"] > .025
    assert not certificate["dynamic_feasibility_validated"]
    assert not certificate["complete_se2_search"]
    assert certificate["nominal_lateral_yaw_local_audit"]["configuration_count"] == 312
    # Straight nominal sideways travel really is an alternative in some
    # scenes. The certificate must expose it, not falsely claim forced lifting.
    audit = certificate["nominal_constant_yaw_path_audit"]
    assert len(audit) == 24
    assert any(abs(abs(row["yaw_offset_rad"]) - math.pi / 2) < 1e-8 for row in audit)
    assert certificate["nominal_constant_yaw_bypass_found"] == any(row["collision_free"] for row in audit)


def test_arm_pose_changes_only_upper_joints_and_respects_existing_action_range():
    nominal = arm_pose()
    for mode in ("raised", "tucked"):
        qpos = arm_pose(mode)
        np.testing.assert_array_equal(qpos[:19], nominal[:19])
        assert np.max(np.abs(qpos - nominal)) <= .8 + 1e-7
        np.testing.assert_allclose(arm_pose(mode, .5), .5 * (nominal + qpos))


def test_unconstrained_scene_is_rejected_as_hand_protection_challenge():
    scene = generate_hand_passage_scene(6001, certify=False)
    scene["boxes"] = [box for box in scene["boxes"] if box["category"] == "wall"]
    scene["geometry_hash"] = _digest(dict(boxes=scene["boxes"], room_dimensions=scene["room_dimensions"]))
    with pytest.raises(ValueError, match="not sufficiently challenged"):
        certify_hand_passage(scene)


def test_invalid_generation_arguments_rejected():
    for kwargs in ({"seed": True}, {"seed": -1}, {"seed": 1, "kind": "arbitrary"},
                   {"seed": 1, "difficulty": "unvalidated"}):
        with pytest.raises(ValueError):
            generate_hand_passage_scene(**kwargs)
