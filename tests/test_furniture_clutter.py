import pytest
import numpy as np

from cat_ppo.furniture.clutter import generate_clutter_scene
from cat_ppo.furniture.scenes import generate_scene, signed_distance, validate_scene


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_generic_clutter_changes_geometry_keeps_admitted_dense_route(split):
    scene = generate_clutter_scene(seed=0, split=split)
    original = generate_scene(seed=0, split=split)
    validate_scene(scene)
    assert scene["geometry_hash"] != original["geometry_hash"]
    assert scene["counts"]["generic_objects"] == 45
    assert scene["counts"]["tables"] == scene["counts"]["chairs"] == 0
    assert scene["counts"]["bottlenecks"] == 6
    assert scene["feasibility"]["root_route_validated"]
    assert scene["route"] == original["route"]
    assert not scene["feasibility"]["full_body_validated"]
    assert scene == generate_clutter_scene(seed=0, split=split)


def test_generic_crate_collision_replaces_table_hole_with_real_solid_geometry():
    scene = generate_clutter_scene(seed=2)
    original = generate_scene(seed=2)
    # Sorted object 36 is the first table, replaced by a crate. The policy field
    # must describe the new solid volume, not the old under-table passage.
    crate = next(box for box in scene["boxes"] if box["name"] == "generic_36_crate")
    point = np.array([crate["center"][0], crate["center"][1], .20])
    assert signed_distance(point, original["boxes"]) > 0
    assert signed_distance(point, scene["boxes"]) < 0
    assert "shelves" in scene["counts"] and "shelfs" not in scene["counts"]
