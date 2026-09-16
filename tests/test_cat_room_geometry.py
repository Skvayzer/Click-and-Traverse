"""Analytic room geometry regressions independent of voxel-center sampling."""

import math

import numpy as np
import pytest

from cat_ppo.furniture.room_geometry import (
    conservative_rasterize_boxes, root_cylinder_clearance,
    root_cylinder_segment_clearance,
)


def box(center=(0., 0., .7), half=(.1, .1, .1), yaw=0., category="obstacle"):
    return dict(center=list(center), half_size=list(half), yaw=yaw, category=category)


def test_thin_leg_between_sample_centers_is_covered():
    leg = box(center=(.02, .02, .05), half=(.003, .003, .05), yaw=.31)
    occupancy = conservative_rasterize_boxes([leg], (3, 3, 5), (0., 0., 0.), .04)
    # Four XY cells touch this tiny leg, although none of their centers is inside.
    assert occupancy[:2, :2, :4].all()  # z=.12 cell touches the leg's z=.10 top.
    assert not occupancy[2].any()
    assert not occupancy[:, 2].any()
    assert not occupancy[:, :, 4].any()


def test_closed_voxel_boundary_contact_is_occupied():
    point = box(center=(.5, .5, .5), half=(0., 0., 0.))
    assert conservative_rasterize_boxes([point], (2, 2, 2), (0., 0., 0.), 1.).all()
    separated = box(center=(.50001, .5, .5), half=(0., 0., 0.))
    result = conservative_rasterize_boxes([separated], (2, 2, 2), (0., 0., 0.), 1.)
    assert not result[0].any()
    assert result[1].all()


def test_rotated_long_rectangle_does_not_fill_its_axis_aligned_bounds():
    diagonal = box(center=(0., 0., 0.), half=(.8, .025, .02), yaw=math.pi / 4)
    occupancy = conservative_rasterize_boxes([diagonal], (9, 9, 1), (-.8, -.8, 0.), .2)
    assert occupancy[2, 2, 0] and occupancy[6, 6, 0]
    assert not occupancy[2, 6, 0] and not occupancy[6, 2, 0]
    assert occupancy.sum() < 30


def test_translated_negative_origin_is_equivalent_and_clips_outside_boxes():
    source = box(center=(.03, .05, .08), half=(.022, .014, .03), yaw=-.63)
    original = conservative_rasterize_boxes([source], (6, 6, 6), (0., 0., 0.), .04)
    offset = np.array([-3., -7., -1.])
    shifted = dict(source, center=(np.asarray(source["center"]) + offset).tolist())
    actual = conservative_rasterize_boxes([shifted], (6, 6, 6), offset, .04)
    np.testing.assert_array_equal(actual, original)
    outside = box(center=(-10., -10., -10.))
    assert not conservative_rasterize_boxes([outside], (6, 6, 6), (0., 0., 0.), .04).any()


def test_floor_is_excluded_and_non_floor_zero_height_geometry_is_retained():
    floor = box(center=(0., 0., 0.), half=(10., 10., .1), category="floor")
    point = box(center=(0., 0., 0.), half=(0., 0., 0.))
    assert not conservative_rasterize_boxes([floor], (2, 2, 2), (0., 0., 0.), .04).any()
    assert conservative_rasterize_boxes([point], (2, 2, 2), (0., 0., 0.), .04).sum() == 1


@pytest.mark.parametrize("leg", [
    box((1.62125872, 2.10165938, .211), (.023, .023, .211), -2.79396528),
    box((4.299328558, 10.377343365, .211), (.023, .023, .211), .767624642),
    box((.618448567, 3.535788689, .211), (.023, .023, .211), .961052524),
])
def test_previously_omitted_real_room_4001_legs_are_present(leg):
    # Actual diagnosed primitive coordinates from recorded room 4001. Translate
    # a local window by an integer number of cells to retain the active grid phase.
    origin = np.array([-.08, -.08, .02])
    begin = np.floor((np.asarray(leg["center"]) - origin) / .04).astype(int) - [3, 3, 5]
    local_origin = origin + begin * .04
    result = conservative_rasterize_boxes([leg], (7, 7, 12), local_origin, .04)
    assert result.any()
    assert result[:, :, 1:-1].any(axis=(0, 1)).all()


def test_root_point_clearance_handles_rotation_height_and_radius():
    obstacle = box(center=(0., 0., .7), half=(.4, .1, .1), yaw=math.pi / 2)
    points = [[.4, 0.], [0., .5], [0., 0.]]
    np.testing.assert_allclose(root_cylinder_clearance(points, [obstacle], radius=.2), [.1, -.1, -.3], atol=1e-14)
    assert np.isposinf(root_cylinder_clearance(points, [obstacle], z_interval=(1.1, 1.2))).all()
    # Closed height intervals include obstacles touching the cylinder's top.
    assert np.isfinite(root_cylinder_clearance(points, [box(center=(0., 0., 1.15))])).all()


def test_continuous_segment_finds_collision_between_clear_endpoints():
    obstacle = box(half=(.005, .005, .1))
    assert (root_cylinder_clearance([[-1., 0.], [1., 0.]], [obstacle]) > .7).all()
    assert root_cylinder_segment_clearance([-1., 0.], [1., 0.], [obstacle]) == pytest.approx(-.23)


def test_continuous_segment_corner_distance_and_tangency():
    obstacle = box(half=(.1, .1, .1))
    # The segment passes the NE corner; its interior projection, not an endpoint,
    # determines clearance, exercising a case endpoint sampling misses.
    result = root_cylinder_segment_clearance([0., .5], [.5, 0.], [obstacle], radius=.1)
    assert result == pytest.approx(.3 / math.sqrt(2) - .1)
    assert root_cylinder_segment_clearance([-1., .33], [1., .33], [obstacle]) == pytest.approx(0., abs=1e-14)
    assert root_cylinder_segment_clearance([-1., .3301], [1., .3301], [obstacle]) > 0.


def test_segment_broadcast_degenerate_parallel_and_rotated_equivalence():
    obstacle = box(half=(.4, .1, .1))
    starts = np.array([[-1., .5], [0., .5], [.7, .4], [.7, .4]])
    ends = np.array([[1., .5], [0., .5], [.7, -.4], [.7, .4]])
    expected = root_cylinder_segment_clearance(starts, ends, [obstacle], radius=.2)
    np.testing.assert_allclose(expected, [.2, .2, .1, math.hypot(.3, .3) - .2])
    angle = .71
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    rotated = box(half=(.4, .1, .1), yaw=angle)
    actual = root_cylinder_segment_clearance(np.einsum('ij,nj->ni', rotation, starts),
                                            np.einsum('ij,nj->ni', rotation, ends), [rotated], radius=.2)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert root_cylinder_segment_clearance([[-1., .5], [-1., -.5]], [1., .5], [obstacle]).shape == (2,)


@pytest.mark.parametrize("kwargs", [dict(dx=0.), dict(dx=np.nan), dict(shape=(1, 2.2, 3)),
                                    dict(shape=(1, 0, 3)), dict(sample_origin=(0., np.inf, 0.))])
def test_invalid_grid_is_rejected(kwargs):
    arguments = dict(shape=(3, 3, 3), sample_origin=(0., 0., 0.), dx=.04)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        conservative_rasterize_boxes([], **arguments)
