import pytest

from cat_ppo.furniture.clutter import generate_clutter_scene
from cat_ppo.furniture.scenes import generate_scene, validate_scene


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
