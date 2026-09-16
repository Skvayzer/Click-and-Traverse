"""Independent volume/contact regressions for production robot primitives."""
import copy
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from scipy.optimize import linprog, minimize_scalar
from scipy.spatial.transform import Rotation

from cat_ppo.furniture.body_collision_geometry import (
    BOX, CAPSULE, SPHERE, box_aabb, box_box_separation, capsule_aabb,
    capsule_box_separation, compile_proposal, reduce_candidates,
    segment_box_squared_distance_local, sphere_aabb, sphere_box_separation,
    transform_primitives,
)


IDENTITY = np.eye(3, dtype=np.float32)


def _yaw(angle):
    return Rotation.from_euler("z", angle).as_matrix().astype(np.float32)


def test_sphere_catches_volume_intersection_without_center_inside():
    result = sphere_box_separation(
        np.array([[.15, 0., 0.], [.21, 0., 0.], [0., 0., 0.]]), .1,
        np.zeros(3), IDENTITY, np.array([.1, .1, .1]))
    np.testing.assert_allclose(result, [-.05, .01, -.2], atol=2e-7)


def test_capsule_shin_hits_thin_leg_between_clear_endpoint_spheres():
    start, end = np.array([0., 0., -.3]), np.array([0., 0., .3])
    center, half = np.zeros(3), np.array([.004, .008, .01])
    endpoint = sphere_box_separation(np.stack([start, end]), .035, center, _yaw(.72), half)
    assert np.all(endpoint > .25)
    assert capsule_box_separation(start, end, .035, center, _yaw(.72), half) == pytest.approx(-.035)


def test_capsule_exact_edge_distance_not_just_endpoint_or_corner_distances():
    # Axis passes diagonally by a box edge. The closest segment point lies
    # strictly between endpoints, and the closest box point is inside its edge.
    start, end = np.array([0., .5, .04]), np.array([.5, 0., .04])
    distance = segment_box_squared_distance_local(start, end, [.1, .1, .1])
    assert float(distance) == pytest.approx((.3 / np.sqrt(2)) ** 2, abs=2e-8)
    assert float(capsule_box_separation(start, end, .1, [0., 0., 0.], IDENTITY,
                                        [.1, .1, .1])) == pytest.approx(.3 / np.sqrt(2) - .1, abs=2e-7)


@pytest.mark.parametrize("start,end,expected", [
    ([.4, .3, .2], [.4, .3, .2], .3 ** 2 + .2 ** 2 + .1 ** 2),
    ([-1., .4, 0.], [1., .4, 0.], .3 ** 2),
    ([-1., 0., 0.], [1., 0., 0.], 0.),
    ([0., 0., 0.], [0., 0., 0.], 0.),
    ([-1., .1, .1], [1., .1, .1], 0.),
])
def test_segment_parallel_zero_length_crossing_and_tangent(start, end, expected):
    value = jax.jit(segment_box_squared_distance_local)(jp.array(start), jp.array(end), jp.array([.1] * 3))
    assert np.isfinite(value)
    assert float(value) == pytest.approx(expected, abs=3e-8)


def test_random_segment_distance_matches_independent_scalar_optimizer():
    rng = np.random.default_rng(61824)
    starts, ends = rng.uniform(-1., 1., (2, 160, 3))
    half = rng.uniform(.01, .25, (160, 3))
    actual = jax.jit(segment_box_squared_distance_local)(jp.array(starts), jp.array(ends), jp.array(half))
    expected = []
    for first, last, size in zip(starts, ends, half):
        def squared(fraction):
            offset = np.maximum(np.abs(first + fraction * (last - first)) - size, 0.)
            return float(np.dot(offset, offset))
        optimum = minimize_scalar(squared, bounds=(0., 1.), method="bounded",
                                  options={"xatol": 1e-12})
        expected.append(min(squared(0.), squared(1.), optimum.fun))
    np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=3e-6)


def test_flat_rotated_toe_and_edge_hit_are_detected():
    # A thin long foot reaches a tiny chair leg despite its center being clear.
    rotation = _yaw(.63)
    half = np.array([.107, .041, .017])
    toe = rotation @ np.array([.104, .038, 0.])
    leg_half = np.array([.003, .003, .15])
    overlap = box_box_separation(np.zeros(3), rotation, half, toe, IDENTITY, leg_half)
    clear = box_box_separation(np.zeros(3), rotation, half,
                               rotation @ np.array([.125, .06, 0.]), IDENTITY, leg_half)
    assert float(overlap) < 0.
    assert float(clear) > 0.
    # The approved thin sole can pass under a ledge that a big foot sphere
    # would reject. This guards against accidentally replacing the OBB by one.
    above = box_box_separation(np.zeros(3), rotation, half,
                               [0., 0., .039], IDENTITY, [.2, .2, .01])
    assert float(above) == pytest.approx(.012, abs=2e-7)


def test_box_containment_contact_and_nearly_parallel_axes_are_finite():
    centers = jp.array([[0., 0., 0.], [.2, 0., 0.], [.2001, 0., 0.]])
    result = box_box_separation(jp.zeros(3), IDENTITY, [.1] * 3, centers,
                                jp.stack([IDENTITY, IDENTITY, _yaw(1e-9)]), [.1] * 3)
    np.testing.assert_allclose(result, [-.2, 0., .0001], atol=2e-7)
    assert np.isfinite(result).all()


def test_random_obb_collisions_match_independent_linear_feasibility_solver():
    rng = np.random.default_rng(484921)
    count = 100
    centers_a, centers_b = rng.uniform(-.6, .6, (2, count, 3))
    rotations_a = Rotation.random(count, random_state=rng).as_matrix()
    rotations_b = Rotation.random(count, random_state=rng).as_matrix()
    half_a, half_b = rng.uniform(.04, .32, (2, count, 3))
    actual = jax.jit(box_box_separation)(jp.array(centers_a), jp.array(rotations_a), jp.array(half_a),
                                        jp.array(centers_b), jp.array(rotations_b), jp.array(half_b))
    expected = []
    cross_axis_only_separations = 0
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    for ca, ra, ha, cb, rb, hb in zip(centers_a, rotations_a, half_a, centers_b, rotations_b, half_b):
        # Intersection exists iff there are local coordinates inside both boxes
        # that map to the same world point. This does not implement SAT.
        result = linprog(np.zeros(6), A_eq=np.concatenate([ra, -rb], axis=1), b_eq=cb - ca,
                         bounds=list(zip(-ha, ha)) + list(zip(-hb, hb)), method="highs")
        expected.append(result.success)
        # Ensure this fixed regression set includes disjoint boxes for which
        # all six face projections overlap: omitting the nine cross axes must
        # fail this test, even if ordinary axis-aligned examples still pass.
        axes = np.concatenate([ra.T, rb.T])
        first_projection = (signs * ha @ ra.T + ca) @ axes.T
        second_projection = (signs * hb @ rb.T + cb) @ axes.T
        face_separates = np.any((first_projection.min(axis=0) > second_projection.max(axis=0))
                               | (second_projection.min(axis=0) > first_projection.max(axis=0)))
        cross_axis_only_separations += int(not face_separates and not result.success)
    np.testing.assert_array_equal(np.asarray(actual) <= 0., expected)
    assert cross_axis_only_separations > 0


def test_contact_geometry_is_invariant_under_shared_rigid_transform():
    rotation = Rotation.from_euler("xyz", [.6, -.4, .2]).as_matrix().astype(np.float32)
    translation = np.array([2., -3., .7], dtype=np.float32)
    first, last = np.array([-.3, .3, .15]), np.array([.4, -.2, .1])
    obstacle = np.array([.1, .1, .08])
    base = capsule_box_separation(first, last, .04, obstacle, _yaw(.35), [.07, .03, .08])
    moved = capsule_box_separation(rotation @ first + translation, rotation @ last + translation, .04,
                                   rotation @ obstacle + translation, rotation @ _yaw(.35), [.07, .03, .08])
    assert float(moved) == pytest.approx(float(base), abs=4e-7)


def test_jit_broadcast_candidate_pairs_and_masked_padding():
    centers = jp.array([[[0., 0., 0.]], [[1., 0., 0.]]])
    obstacles = jp.array([[[0., 0., 0.], [1., 0., 0.]], [[0., 0., 0.], [1., 0., 0.]]])
    mask = jp.array([[True, False], [False, False]])

    @jax.jit
    def query(centers, obstacles, mask):
        gaps = box_box_separation(centers, IDENTITY, jp.array([.1] * 3),
                                  obstacles, IDENTITY, jp.array([.1] * 3))
        return reduce_candidates(gaps, mask)

    collision, distance = query(centers, obstacles, mask)
    np.testing.assert_array_equal(collision, [True, False])
    assert float(distance[0]) == pytest.approx(-.2)
    assert np.isposinf(distance[1])
    batched = jax.jit(jax.vmap(query))(jp.stack([centers, centers]),
                                      jp.stack([obstacles, obstacles]), jp.stack([mask, mask]))
    np.testing.assert_array_equal(batched[0], [[True, False], [True, False]])


def test_primitive_world_transforms_and_aabbs_enclose_rotated_shapes():
    body_rotations = np.stack([IDENTITY, _yaw(np.pi / 2)])
    body_positions = np.array([[0., 0., 0.], [2., 3., 1.]])
    local_center = np.array([[.2, 0., .1]])
    local_endpoints = np.array([[[-.1, 0., 0.], [.1, 0., 0.]]])
    result = jax.jit(transform_primitives)(body_positions, body_rotations, jp.array([1]),
                                          local_center, np.array([IDENTITY]), local_endpoints)
    np.testing.assert_allclose(result["centers"], [[2., 3.2, 1.1]], atol=2e-7)
    np.testing.assert_allclose(result["endpoints"], [[[2., 2.9, 1.], [2., 3.1, 1.]]], atol=2e-7)
    low, high = box_aabb(result["centers"], result["rotations"], jp.array([[.1, .03, .02]]))
    np.testing.assert_allclose(high - low, [[.06, .2, .04]], atol=3e-7)
    low, high = capsule_aabb(result["endpoints"][..., 0, :], result["endpoints"][..., 1, :], jp.array([.02]))
    np.testing.assert_allclose(high - low, [[.04, .24, .04]], atol=3e-7)
    low, high = sphere_aabb(result["centers"], jp.array([.04]))
    np.testing.assert_allclose(high - low, [[.08] * 3], atol=3e-7)
    batch = transform_primitives(jp.stack([body_positions, body_positions]),
                                 jp.stack([body_rotations, body_rotations]), jp.array([1]),
                                 local_center, np.array([IDENTITY]), local_endpoints)
    assert batch["endpoints"].shape == (2, 1, 2, 3)


def test_compile_approved_proposal_preserves_shapes_and_resolves_body_names():
    from cat_ppo.envs.g1.env_cat_wholebody import assemble_training_xml

    path = Path(__file__).resolve().parents[1] / "docs/assets/collision-proxy-proposal-20260916/proposal.json"
    proposal = json.loads(path.read_text())
    model = mujoco.MjModel.from_xml_string(assemble_training_xml())
    # Saved numeric IDs can become stale when a scene adds unrelated bodies.
    changed = copy.deepcopy(proposal)
    for shape in changed["shapes"]:
        shape["body_id"] = 99999
    constants = compile_proposal(changed, model)
    assert len(constants["shape_names"]) == 35
    assert {name: len(value) for name, value in constants["indices"].items()} == {
        "box": 11, "capsule": 22, "sphere": 2}
    for index, shape in enumerate(proposal["shapes"]):
        assert model.body(int(constants["body_ids"][index])).name == shape["body_name"]
        np.testing.assert_allclose(constants["local_centers"][index], shape["center"], atol=2e-8)
        if "sole" in shape["id"]:
            assert constants["kind_ids"][index] == BOX
            np.testing.assert_allclose(constants["half_sizes"][index], shape["half_size"], atol=1e-8)
    assert len(constants["indices"]["capsule"]) == np.sum(constants["kind_ids"] == CAPSULE)
    assert len(constants["indices"]["sphere"]) == np.sum(constants["kind_ids"] == SPHERE)
    changed["shapes"][0]["body_name"] = "missing_robot_link"
    with pytest.raises(ValueError, match="missing"):
        compile_proposal(changed, model)


@pytest.mark.parametrize("candidate_count", [9, 17])
@pytest.mark.parametrize("mode", ["last_candidate", "masked_tail", "mixed"])
def test_training_chunked_candidates_match_all_pairs_including_last_and_padding(candidate_count, mode):
    """Exercise the actual training reducer across two/three 8-wide chunks."""
    from cat_ppo.envs.g1.body_collision import body_collisions

    count = candidate_count
    positions = np.array([[.5, .5, .5], [1.5, .5, .5], [2.5, .5, .5]], dtype=np.float32)
    rotations = np.broadcast_to(IDENTITY, (3, 3, 3)).copy()
    local_rotations = rotations.copy()
    local_rotations[0] = _yaw(.35)
    ends = np.zeros((3, 2, 3), dtype=np.float32)
    ends[1, :, 2] = [-.25, .25]
    half_sizes = np.array([[.1, .04, .02], [0., 0., 0.], [0., 0., 0.]], dtype=np.float32)
    radii = np.array([0., .025, .07], dtype=np.float32)
    compiled = dict(body_ids=np.arange(3, dtype=np.int32), local_centers=np.zeros((3, 3), np.float32),
                    local_rotations=local_rotations, local_endpoints=ends,
                    half_sizes=half_sizes, radii=radii,
                    indices={name: np.array([index], np.int32)
                             for index, name in enumerate(("box", "capsule", "sphere"))})
    # Three real CSR cells, each with a distinct candidate list. Only the very
    # last candidate can overlap its querying shape; every earlier one is far.
    obstacle_centers = np.full((1 + 3 * count, 3), 50., dtype=np.float32)
    obstacle_halves = np.full_like(obstacle_centers, .004)
    obstacle_rotations = np.broadcast_to(IDENTITY, (len(obstacle_centers), 3, 3)).copy()
    candidate_ids = np.arange(1, 1 + 3 * count, dtype=np.int32).reshape(3, count)
    hits = positions.copy()
    hits[0] += local_rotations[0] @ np.array([.098, 0., 0.], np.float32)  # toe edge
    hits[2, 0] += .072
    for index, ids in enumerate(candidate_ids):
        obstacle_centers[ids[-1]] = hits[index]
    # Padded IDs are zero. Deliberately make that sentinel collide with all
    # shapes, so forgetting either CSR masking or chunk padding masking fails.
    obstacle_centers[0] = 0.
    obstacle_halves[0] = 100.
    valid_counts = np.full(3, count, dtype=np.int32)
    if mode == "masked_tail":
        valid_counts[:] = count - 1
    elif mode == "mixed":
        valid_counts[1] = count - 1
    bank = {key: jp.asarray(value) for key, value in dict(
        centers=obstacle_centers, half_sizes=obstacle_halves, rotations=obstacle_rotations,
        scene_grid_origins=np.zeros((1, 3), np.float32), scene_grid_shapes=np.array([[3, 1, 1]], np.int32),
        scene_grid_offsets=np.array([0], np.int32), cell_size=np.float32(1.),
        cell_starts=np.array([1, 1 + count, 1 + 2 * count], np.int32), cell_counts=valid_counts,
        candidate_ids=np.r_[np.int32(0), candidate_ids.ravel()]).items()}

    def brute_force_all_candidates():
        centers = bank["centers"][candidate_ids]
        rotation = bank["rotations"][candidate_ids]
        halves = bank["half_sizes"][candidate_ids]
        gaps = jp.stack([
            box_box_separation(positions[0], local_rotations[0], half_sizes[0], centers[0], rotation[0], halves[0]),
            capsule_box_separation(positions[1] + ends[1, 0], positions[1] + ends[1, 1], radii[1],
                                   centers[1], rotation[1], halves[1]),
            sphere_box_separation(positions[2], radii[2], centers[2], rotation[2], halves[2]),
        ])
        return jp.any((jp.arange(count)[None] < valid_counts[:, None]) & (gaps <= 0.), axis=-1)

    expected = jax.jit(brute_force_all_candidates)()
    actual = jax.jit(lambda poses: body_collisions(
        compiled, bank, jp.int32(0), poses, rotations, max_candidates=count))(jp.asarray(positions))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, valid_counts == count)
