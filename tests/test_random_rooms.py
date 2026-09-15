"""Irregular-room geometry and continuous root-route admission regressions."""

import math

import numpy as np
import pytest

from cat_ppo.furniture.random_rooms import (
    RESET_CENTER_CLEARANCE_M, RESET_FIELD_MARGIN_M, RESET_HAND_ENVELOPE_RADIUS_M,
    RESET_XY_JITTER_M, generate_random_room, transform_object,
)
from cat_ppo.furniture.scenes import (
    ROOT_CLEARANCE_RADIUS, _horizontal_distance, _root_obstacles, _table,
    validate_scene,
)


@pytest.fixture(scope="module")
def furniture():
    # Exercise the first actual training-bank room after endpoint admission.
    return generate_random_room(4001)


def test_rigid_rotation_moves_table_legs_with_tabletop():
    source = _table("table", 0, 0, 1.8, .8, .78, "fixture")
    transformed = transform_object(source, 4., 3., math.pi / 2)
    for before, after in zip(source, transformed):
        assert after["center"] == pytest.approx(
            [4. - before["center"][1], 3. + before["center"][0], before["center"][2]])
        assert after["yaw"] == pytest.approx(math.pi / 2)
        assert after["half_size"] == before["half_size"]
    for i in range(len(source)):
        for j in range(i):
            assert math.dist(source[i]["center"], source[j]["center"]) == pytest.approx(
                math.dist(transformed[i]["center"], transformed[j]["center"]))
    assert source[0]["center"][:2] == [0, 0]


def test_dense_furniture_is_not_a_grid_and_has_continuous_independent_yaws(furniture):
    validate_scene(furniture)
    assert 8 <= furniture["counts"]["tables"] <= 12
    assert 25 <= furniture["counts"]["chairs"] <= 45
    profiles = furniture["generator"]["shape_profiles"]
    tables = [p for p in profiles if p["kind"] == "table"]
    assert len({round(p["center_xy_m"][0], 2) for p in tables}) >= len(tables) - 1
    assert len({round(p["center_xy_m"][1], 2) for p in tables}) >= len(tables) - 1
    assert len({round(p["yaw_rad"], 2) for p in profiles}) > .8 * len(profiles)
    assert sum(abs(math.sin(2 * p["yaw_rad"])) > .2 for p in profiles) > .7 * len(profiles)
    assert furniture["layout_metrics"]["root_height_occupied_area_fraction"] > .20
    assert furniture["layout_metrics"]["root_inflated_blocked_area_fraction"] > .55


@pytest.mark.parametrize("kind", ["furniture", "generic_clutter"])
def test_seed_and_split_are_reproducible_and_disjoint(kind):
    a = generate_random_room(53, kind=kind)
    assert a == generate_random_room(53, kind=kind)
    assert a["geometry_hash"] != generate_random_room(54, kind=kind)["geometry_hash"]
    assert a["geometry_hash"] != generate_random_room(53, kind=kind, split="validation")["geometry_hash"]
    if kind == "generic_clutter":
        assert a["counts"]["generic_objects"] >= 33
        assert a["counts"]["tables"] == a["counts"]["chairs"] == 0
        assert {p["kind"] for p in a["generator"]["shape_profiles"]} == {
            "crate", "low_block", "partition", "shelf"}


@pytest.mark.parametrize("kind,seed", [("furniture", 0), ("furniture", 7),
                                      ("generic_clutter", 0), ("generic_clutter", 7)])
def test_every_case_has_continuous_root_route_and_jitter_safe_endpoints(kind, seed):
    scene = generate_random_room(seed, kind=kind)
    root_boxes = _root_obstacles(scene["boxes"])
    assert len(scene["start_goals"]) >= 2
    for case in scene["start_goals"]:
        assert math.dist(case["start"][:2], case["goal"]) > .5 * min(scene["room_dimensions"][:2])
        # Independent dense 1 cm sampling, substantially finer than planner grid
        # or the generator's 4 cm conservative analytic certificate.
        for first, last in zip(case["route"], case["route"][1:]):
            for fraction in np.linspace(0, 1, math.ceil(math.dist(first, last) / .01) + 1):
                point = np.asarray(first) * (1 - fraction) + np.asarray(last) * fraction
                assert min(_horizontal_distance(point, box) for box in root_boxes) >= ROOT_CLEARANCE_RADIUS
        for endpoint in (case["start"][:2], case["goal"]):
            assert min(_horizontal_distance(endpoint, box) for box in root_boxes) >= RESET_CENTER_CLEARANCE_M
            for dx in (-RESET_XY_JITTER_M, RESET_XY_JITTER_M):
                for dy in (-RESET_XY_JITTER_M, RESET_XY_JITTER_M):
                    point = [endpoint[0] + dx, endpoint[1] + dy]
                    assert min(_horizontal_distance(point, box) for box in root_boxes) > ROOT_CLEARANCE_RADIUS + .035
                    assert min(_horizontal_distance(point, box) for box in root_boxes) >= RESET_HAND_ENVELOPE_RADIUS_M + RESET_FIELD_MARGIN_M
    assert all(case["root_route_validated"] for case in scene["case_feasibility"])
    assert scene["feasibility"]["full_body_validated"] is False
    assert scene["feasibility"]["dynamic_feasibility_validated"] is False


def test_objects_are_inside_walls_and_do_not_interpenetrate(furniture):
    # Separating axes reconstructed directly from transformed primitive corners,
    # independently of the generator's complete-object packing footprint test.
    parts = [p for p in furniture["boxes"] if "furniture_id" in p]
    for part in parts:
        c, s = math.cos(part["yaw"]), math.sin(part["yaw"])
        hx, hy = part["half_size"][:2]
        ex, ey = abs(c) * hx + abs(s) * hy, abs(s) * hx + abs(c) * hy
        for axis, extent in ((0, ex), (1, ey)):
            assert part["center"][axis] - extent > .06
            assert part["center"][axis] + extent < furniture["room_dimensions"][axis] - .06
    main_surfaces = [p for p in parts if p["category"] in ("tabletop", "chair_seat")]
    for i, first in enumerate(main_surfaces):
        for second in main_surfaces[:i]:
            separated = False
            displacement = np.asarray(first["center"][:2]) - second["center"][:2]
            for angle in (first["yaw"], first["yaw"] + math.pi / 2,
                          second["yaw"], second["yaw"] + math.pi / 2):
                axis = np.array([math.cos(angle), math.sin(angle)])
                radii = []
                for part in (first, second):
                    ca, sa = math.cos(part["yaw"]), math.sin(part["yaw"])
                    radii.append(abs(np.dot(axis, [ca, sa])) * part["half_size"][0]
                                 + abs(np.dot(axis, [-sa, ca])) * part["half_size"][1])
                separated |= abs(np.dot(displacement, axis)) >= sum(radii)
            assert separated


@pytest.mark.parametrize("kwargs", [dict(seed=-1), dict(seed=True), dict(seed=1.2),
                                    dict(seed=1, split="unknown"), dict(seed=1, kind="chairs"),
                                    dict(seed=1, difficulty="unknown")])
def test_invalid_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_random_room(**kwargs)
