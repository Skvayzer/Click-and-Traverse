"""Tabletop/leg geometry and actual approved-volume route certificates."""
import copy

import numpy as np
import pytest

from cat_ppo.furniture.contrastive_passages import generate_contrastive_group
from cat_ppo.furniture.scenes import _digest, validate_scene
from cat_ppo.furniture.table_edge_passages import (
    ROLES, certify_table_edge_scene, generate_table_edge_pair, module_cross_section_openings,
)


@pytest.fixture(scope="module")
def pair():
    return generate_table_edge_pair(20260919)


def test_matching_table_pair_changes_gap_and_preserves_real_geometry():
    scenes = generate_table_edge_pair(20260919, certify=False)
    assert scenes == generate_table_edge_pair(20260919, certify=False)
    assert [s["hand_contrast"]["role"] for s in scenes] == list(ROLES)
    assert scenes[0]["route"] == scenes[1]["route"]
    assert scenes[0]["generator"] == scenes[1]["generator"]
    assert scenes[0]["hand_contrast"]["group_id"] == scenes[1]["hand_contrast"]["group_id"]
    assert scenes[0]["geometry_hash"] != scenes[1]["geometry_hash"]
    for scene in scenes:
        validate_scene(scene)
        assert scene["hand_contrast"]["geometry_family"] == "table_edges"
        assert scene["source"]["hand_contrast"] == scene["hand_contrast"]
        assert scene["counts"]["tables"] == 6
        assert sum(b["category"] == "table_leg" for b in scene["boxes"]) == 24
        assert sum(b["category"] == "table_top" for b in scene["boxes"]) == 6
        assert not any(b["category"] == "cabinet" for b in scene["boxes"])
        assert .70 <= scene["hand_contrast"]["tabletop_top_z_m"] <= .72


def test_actual_raised_hand_route_is_clear_and_nominal_hands_collide(pair):
    for scene in pair:
        contrast, certificate = scene["hand_contrast"], scene["hand_contrast"]["certificate"]
        assert certificate["primitive_count"] == 35
        assert certificate["geometry_hash"] == scene["geometry_hash"]
        assert certificate["route_hash"] == _digest(scene["route"])
        assert certificate["route_transition_validated"]
        assert certificate["body_min_separation_m"] > .008
        assert certificate["hand_field_min_clearance_m"] > (.30 if contrast["role"] == "open" else .025)
        assert certificate["stance_sample_count"] >= 564
        assert certificate["tabletop_geometry_validated"]
        assert certificate["raised_hand_above_table_min_m"] > .08
        assert not certificate["dynamic_feasibility_validated"]
        assert not certificate["complete_se2_search"]
    protected = pair[1]["hand_contrast"]
    certificate = protected["certificate"]
    assert certificate["raised_hand_sphere_lateral_overhang_min_m"] > .004
    assert all(min(row["above_top_clearance_m"]) > .05 for row in certificate["above_table_stance_audit"])
    for row in certificate["module_audit"]:
        assert row["nominal_forward_hand_min_m"] < -.002
        assert row["tucked_forward_hand_min_m"] < -.002
        assert row["raised_forward_body_min_m"] > .03
    for zone, audits in zip(protected["zones"], certificate["target_region_audit"]):
        assert zone["region_valid"] == [True, False]
        assert audits[0]["valid"] and not audits[1]["valid"]
        assert min(audits[0]["above_top_clearance_m"]) > .06
        assert audits[0]["field_swept_hand_min_clearance_m"] > .025


def test_baffles_seal_tabletop_band_without_claiming_solid_floor(pair):
    for scene in pair:
        certificate = scene["hand_contrast"]["certificate"]
        assert certificate["exterior_lanes_sealed"]
        assert "not floor-level" in certificate["exterior_lane_seal_scope"]
        for module in scene["hand_contrast"]["modules"]:
            np.testing.assert_allclose(module_cross_section_openings(scene, module, certificate["exterior_lane_seal_height_m"]),
                                       [[-module["width_m"] / 2, module["width_m"] / 2]], atol=1e-7)
            # Underneath the real tabletop, the free center span is wider.
            below = module_cross_section_openings(scene, module, .20)
            assert max(b - a for a, b in below) > module["width_m"] + .5
    broken = copy.deepcopy(pair[1])
    broken["boxes"] = [b for b in broken["boxes"] if b["name"] != "baffle_0_+1"]
    broken["geometry_hash"] = _digest(dict(boxes=broken["boxes"], room_dimensions=broken["room_dimensions"]))
    with pytest.raises(ValueError, match="Unsealed tabletop-level"):
        certify_table_edge_scene(broken)


def test_tall_control_is_unchanged():
    group = generate_contrastive_group(20260919, certify=False)
    assert len(group) == 4
    assert all(s["counts"]["tables"] == 0 for s in group)
    assert all(s["hand_contrast"]["cabinet_height_m"] > 1.2 for s in group)
    assert "geometry_family" not in group[0]["hand_contrast"]


@pytest.mark.parametrize("seed,split", [(True, "train"), (-1, "train"), (1, "unknown")])
def test_invalid_arguments(seed, split):
    with pytest.raises(ValueError):
        generate_table_edge_pair(seed, split)


def test_missing_leg_invalidates_table_geometry(pair):
    broken = copy.deepcopy(pair[1])
    broken["boxes"] = [b for b in broken["boxes"] if b["name"] != "table_0_+1_leg_+1_+1"]
    broken["geometry_hash"] = _digest(dict(boxes=broken["boxes"], room_dimensions=broken["room_dimensions"]))
    with pytest.raises(ValueError, match="six equally high tabletops with four legs each"):
        certify_table_edge_scene(broken)


def test_lowered_hand_target_cannot_be_certified_as_above_table(pair):
    broken = copy.deepcopy(pair[1])
    for zone in broken["hand_contrast"]["zones"]:
        for key in ("hand_regions_min", "hand_regions_max"):
            for hand in zone[key][0]:
                hand[2] -= .15
    with pytest.raises(ValueError, match="Raised hand-target region does not clear tabletop"):
        certify_table_edge_scene(broken)
