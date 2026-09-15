import pytest

from cat_ppo.furniture.capacity import estimated_memory_cap


def scene(boxes):
    return {"boxes": [{} for _ in range(boxes)]}


@pytest.mark.parametrize("boxes,configured,expected", [
    (5, 8192, 8192), (8, 8192, 8192), (17, 4096, 4096),
    (42, 2048, 2048), (53, 2048, 1024), (79, 2048, 1024),
    (91, 2048, 1024), (91, 1024, 1024), (298, 1024, 256),
])
def test_caps_growing_scene_geometry_without_reducing_observed_safe_batches(boxes, configured, expected):
    result = estimated_memory_cap(scene(boxes), configured, 25)
    assert result["num_envs"] == expected
    assert result["estimated_peak_bytes"] <= 25 * 2**30
    # Halving keeps the existing PPO/stage arithmetic exact.
    assert 8388608 % (result["num_envs"] * 16 * 64) == 0
    assert result["guarantee"] is False


def test_disabled_cap_preserves_configuration_and_impossible_scene_fails_before_launch():
    assert estimated_memory_cap(scene(298), 1024, None)["num_envs"] == 1024
    with pytest.raises(ValueError, match="batch floor"):
        estimated_memory_cap(scene(10000), 1024, 25)
    with pytest.raises(ValueError, match="finite and positive"):
        estimated_memory_cap(scene(5), 8192, float("nan"))
