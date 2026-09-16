"""Ordered navigation and exact swept-cylinder geometry regressions."""
import math

import jax
import jax.numpy as jp
import numpy as np
import pytest

from cat_ppo.furniture.room_geometry import (
    root_cylinder_clearance, root_cylinder_segment_clearance,
)
from cat_ppo.furniture.room_navigation import (
    pack_room_scenes, root_clearance, route_context, swept_root_clearance,
)


def box(center=(0., 0., .7), half=(.1, .8, .05), yaw=0., category="tabletop"):
    return dict(center=list(center), half_size=list(half), yaw=yaw, category=category)


def pack(route, boxes=()):
    arrays = pack_room_scenes([dict(route=route, boxes=list(boxes))])
    return {key: jp.asarray(value[0]) for key, value in arrays.items()}


def context(root, metadata, *, previous=None, segment=0, violation=False, **kwargs):
    return route_context(jp.asarray(root), jp.asarray(root if previous is None else previous),
                         jp.asarray(segment), jp.asarray(violation), metadata["route"],
                         metadata["route_count"], metadata["obstacles"], metadata["obstacle_count"], **kwargs)


def test_scene_packing_masks_native_cat_slots_and_preserves_root_band_geometry():
    scene = dict(route=[[0., 0.], [1., 0.]], start=[0., 0., 0.], goal=[1., 0.],
                 boxes=[box(), box(center=(0., 0., 1.8)), box(category="floor")])
    arrays = pack_room_scenes([None, scene])
    np.testing.assert_array_equal(arrays["enabled"], [False, True])
    np.testing.assert_array_equal(arrays["route_count"], [0, 2])
    np.testing.assert_array_equal(arrays["obstacle_count"], [0, 1])
    assert arrays["obstacles"].dtype == np.float32
    assert np.isposinf(swept_root_clearance(jp.zeros(2), jp.ones(2),
                                          arrays["obstacles"][0], arrays["obstacle_count"][0]))


def test_geometry_matches_independent_numpy_reference_on_random_rotated_boxes():
    rng = np.random.default_rng(12)
    boxes = [box(center=(*rng.uniform(-2, 2, 2), .65),
                 half=(*rng.uniform(.003, .8, 2), .12), yaw=rng.uniform(-math.pi, math.pi))
             for _ in range(15)]
    meta = pack([[0., 0.], [1., 1.]], boxes)
    first = rng.uniform(-3, 3, (300, 2)).astype(np.float32)
    last = rng.uniform(-3, 3, (300, 2)).astype(np.float32)
    last[:30] = first[:30]
    expected = root_cylinder_segment_clearance(first, last, boxes)
    actual = jax.jit(swept_root_clearance)(first, last, meta["obstacles"], meta["obstacle_count"])
    np.testing.assert_allclose(actual, expected, atol=1.5e-6, rtol=1e-5)
    np.testing.assert_allclose(root_clearance(first, meta["obstacles"], meta["obstacle_count"]),
                               root_cylinder_clearance(first, boxes), atol=1.5e-6, rtol=1e-5)


def test_swept_collision_catches_thin_tabletop_between_clear_endpoints_and_sticks():
    meta = pack([[-1., 0.], [-1., 1.5], [1., 1.5], [1., 0.]],
                [box(half=(.003, .5, .01))])
    assert root_clearance(jp.array([-1., 0.]), meta["obstacles"], meta["obstacle_count"]) > 0
    assert root_clearance(jp.array([1., 0.]), meta["obstacles"], meta["obstacle_count"]) > 0
    crossed = context([1., 0.], meta, previous=[-1., 0.], segment=2)
    assert crossed["swept_violation"] and crossed["violation"]
    assert not crossed["route_complete"]
    returned = context([1., 0.], meta, segment=2, violation=crossed["violation"])
    assert not returned["swept_violation"] and returned["violation"]
    assert not returned["route_complete"]
    np.testing.assert_array_equal(returned["guidance"], [0., 0., 0.])


def test_hairpin_near_future_segment_or_goal_does_not_jump_progress():
    route = [[0., 0.], [0., 3.], [.8, 3.], [.8, 0.], [.1, 0.]]
    meta = pack(route)
    initial = context([.09, 0.], meta)
    assert initial["segment"] == 0 and not initial["route_complete"]
    assert initial["direction"][1] > .9
    corner = context([0., 3.], meta)
    assert corner["segment"] == 1
    displaced = context([.8, .05], meta)
    assert displaced["segment"] == 0 and not displaced["route_complete"]


def test_corner_cannot_advance_when_carrot_connection_cuts_inflated_obstacle():
    # Both centerline segments clear this box, but a diagonal cut from 15 cm
    # before the bend into the next segment intersects the 23 cm root cylinder.
    meta = pack([[-1., 0.], [0., 0.], [0., 1.]],
                [box(center=(-.45, .45, .7), half=(.2, .2, .05))])
    premature = context([-.145, 0.], meta)
    assert premature["segment"] == 0
    np.testing.assert_allclose(premature["target"], [0., 0.], atol=1e-7)
    reached = context([0., 0.], meta)
    assert reached["segment"] == 1
    assert reached["direction"][1] > .99


def test_hidden_recovery_target_stops_instead_of_navigating_through_wall():
    meta = pack([[0., 0.], [0., 3.]], [box(center=(.5, 1.5, .7), half=(.05, 2., .1))])
    result = context([1., 1.], meta)
    assert result["blocked"] and not result["violation"]
    np.testing.assert_array_equal(result["guidance"], [0., 0., 0.])


@pytest.mark.parametrize("seed,kind", [(4001, "furniture"), (4002, "furniture"),
                                      (5001, "generic_clutter"), (5002, "generic_clutter")])
def test_recorded_rooms_point_controller_reaches_goal_without_swept_collision(seed, kind):
    from cat_ppo.furniture.random_rooms import generate_random_room
    scene = generate_random_room(seed=seed, kind=kind)
    meta = pack(scene["route"], scene["boxes"])
    root = jp.array(scene["start"][:2], dtype=jp.float32)
    first = context(root, meta)
    initial_direction = np.asarray(scene["route"][1]) - np.asarray(scene["route"][0])
    assert float(jp.dot(first["direction"], initial_direction / np.linalg.norm(initial_direction))) > .999

    @jax.jit
    def integrate(root):
        def advance(carry, _):
            position, previous, segment, violation, finished, blocked, minimum = carry
            status = route_context(position, previous, segment, violation, meta["route"],
                                   meta["route_count"], meta["obstacles"], meta["obstacle_count"])
            finished = finished | status["route_complete"]
            next_position = position + jp.where(finished, jp.zeros(2), status["guidance"][:2] * .02)
            return (next_position, position, status["segment"], status["violation"], finished,
                    blocked | status["blocked"], jp.minimum(minimum, status["swept_clearance"])), None
        initial = (root, root, jp.int32(0), jp.array(False), jp.array(False), jp.array(False), jp.array(jp.inf))
        return jax.lax.scan(advance, initial, None, length=2500)[0]

    final, _, segment, violation, finished, blocked, minimum = integrate(root)
    assert finished and not violation and not blocked
    assert minimum > 0
    assert segment == meta["route_count"] - 2
    assert np.linalg.norm(np.asarray(final) - np.asarray(scene["goal"])) <= .201


def test_batched_jit_context_is_finite_and_progress_remains_per_episode():
    meta = pack([[0., 0.], [0., 1.], [1., 1.]])
    roots = jp.array([[0., 0.], [0., 1.], [1., 1.]])
    result = jax.jit(jax.vmap(lambda root, segment: context(root, meta, segment=segment)))(roots, jp.array([0, 0, 1]))
    np.testing.assert_array_equal(result["segment"], [0, 1, 1])
    np.testing.assert_array_equal(result["route_complete"], [False, False, True])
    assert np.isfinite(result["guidance"]).all()


def test_all_48_training_rooms_track_the_ordered_route_from_reset_jitter():
    from cat_ppo.furniture.expanded_fields import DEFAULT_GENERIC_SEEDS, DEFAULT_ROOM_SEEDS
    from cat_ppo.furniture.random_rooms import generate_random_room
    scenes = [generate_random_room(seed=seed, kind=kind)
              for kind, seeds in (("furniture", DEFAULT_ROOM_SEEDS),
                                  ("generic_clutter", DEFAULT_GENERIC_SEEDS)) for seed in seeds]
    # Nominal plus opposing corners of the actual uniform +/-8 cm reset box.
    # This catches a controller that works only exactly on its stored polyline.
    jitters = np.array([[0., 0.], [-.08, -.08], [.08, .08]], dtype=np.float32)
    expanded = [scene for scene in scenes for _ in jitters]
    metadata = {key: jp.asarray(value) for key, value in pack_room_scenes(expanded).items()}
    roots = jp.asarray(np.array([scene["start"][:2] for scene in scenes], dtype=np.float32)[:, None, :]
                       + jitters[None]).reshape(-1, 2)

    @jax.jit
    def integrate(roots):
        n = len(roots)
        def advance(carry, _):
            position, previous, segment, violation, finished, blocked, minimum = carry
            status = jax.vmap(route_context)(position, previous, segment, violation,
                metadata["route"], metadata["route_count"], metadata["obstacles"], metadata["obstacle_count"])
            finished |= status["route_complete"]
            next_position = position + jp.where(finished[:, None], 0., status["guidance"][:, :2] * .02)
            return (next_position, position, status["segment"], status["violation"], finished,
                    blocked | status["blocked"], jp.minimum(minimum, status["swept_clearance"])), None
        initial = (roots, roots, jp.zeros(n, dtype=jp.int32), jp.zeros(n, dtype=bool),
                   jp.zeros(n, dtype=bool), jp.zeros(n, dtype=bool), jp.full(n, jp.inf))
        return jax.lax.scan(advance, initial, None, length=2500)[0]

    _, _, _, violation, finished, blocked, minimum = jax.device_get(integrate(roots))
    failed = np.flatnonzero(~finished | violation | blocked | (minimum <= 0))
    assert len(failed) == 0, [(expanded[index]["seed"], jitters[index % len(jitters)].tolist(),
                             bool(finished[index]), bool(violation[index]), bool(blocked[index]),
                             float(minimum[index])) for index in failed]
