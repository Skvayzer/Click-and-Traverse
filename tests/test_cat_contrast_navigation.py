"""Certified per-scene navigation radius and ordered reward-phase regressions."""
from copy import deepcopy
import json
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest

from cat_ppo.envs.g1.room_navigation import RoomNavigationMixin
from cat_ppo.furniture.room_geometry import root_cylinder_segment_clearance
from cat_ppo.furniture.room_navigation import (
    pack_room_scenes, route_context, scene_navigation_radius,
)
from cat_ppo.furniture.scenes import _digest


def _narrow_scene():
    scene = dict(
        room_dimensions=[3., 3., 2.], route=[[0., 0.], [2., 0.]],
        start=[0., 0., .8], goal=[2., 0., .8],
        boxes=[dict(center=[1., side * .24, .7], half_size=[.7, .05, .7], yaw=0.)
               for side in (-1, 1)],
    )
    # Synthetic certificates exercise the navigation contract, not the robot
    # certifier (which has separate full-geometry tests).
    certificate = dict(
        schema="hand-contrast-certificate-v1", navigation_radius_m=.15,
        route_transition_validated=True, primitive_count=35,
        body_proxy_sha256="a" * 64, route_sample_count=81, transition_sample_count=17,
        body_min_separation_m=.02, hand_field_min_clearance_m=.02,
        geometry_hash=_digest(dict(boxes=scene["boxes"], room_dimensions=scene["room_dimensions"])),
        route_hash=_digest(scene["route"]),
    )
    zone = dict(start_m=.2, end_m=1.8, fade_m=.1, forward_weight=0.,
                hand_active=[False, False], region_valid=[False, False],
                hand_regions_min=np.zeros((2, 2, 3)).tolist(),
                hand_regions_max=np.zeros((2, 2, 3)).tolist())
    scene["hand_contrast"] = dict(schema="hand-contrast-v1", navigation_radius_m=.15,
                                   certificate=certificate, group_id="test", role="narrow", zones=[zone])
    return scene


def _context(meta, root=(.6, 0.)):
    root = jp.asarray(root)
    return route_context(root, root, jp.int32(0), jp.array(False),
                         meta["route"], meta["route_count"], meta["obstacles"],
                         meta["obstacle_count"], radius=meta["navigation_radius"])


def test_narrow_radius_changes_only_certified_scene_and_allows_checked_route():
    narrow = _narrow_scene()
    legacy = deepcopy(narrow)
    legacy.pop("hand_contrast")
    packed = pack_room_scenes([None, legacy, narrow])
    np.testing.assert_allclose(packed["navigation_radius"], [.23, .23, .15])
    assert root_cylinder_segment_clearance([[0., 0.]], [[2., 0.]], narrow["boxes"])[0] < 0
    assert root_cylinder_segment_clearance([[0., 0.]], [[2., 0.]], narrow["boxes"],
                                           radius=scene_navigation_radius(narrow))[0] > 0
    states = [_context({key: jp.asarray(value[index]) for key, value in packed.items()})
              for index in (1, 2)]
    assert states[0]["violation"] and states[0]["blocked"]
    assert not states[1]["violation"] and not states[1]["blocked"]
    assert states[1]["guidance"][0] > .5


@pytest.mark.parametrize("mutation", [
    lambda s: s["hand_contrast"].update(navigation_radius_m=.1),
    lambda s: s["hand_contrast"].update(schema="unknown"),
    lambda s: s["hand_contrast"].pop("certificate"),
    lambda s: s["hand_contrast"]["certificate"].update(route_transition_validated=False),
    lambda s: s["hand_contrast"]["certificate"].update(transition_sample_count=0),
    lambda s: s["hand_contrast"]["certificate"].update(body_min_separation_m=-.01),
    lambda s: s["hand_contrast"]["certificate"].update(hand_field_min_clearance_m=0.),
    lambda s: s["hand_contrast"]["certificate"].update(primitive_count=34),
    lambda s: s["hand_contrast"]["certificate"].update(body_proxy_sha256="bad"),
    lambda s: s["route"][1].__setitem__(0, 2.1),
    lambda s: s["boxes"][0]["center"].__setitem__(1, -.20),
])
def test_navigation_rejects_stale_or_unsupported_certificates(mutation):
    scene = _narrow_scene()
    mutation(scene)
    with pytest.raises(ValueError):
        pack_room_scenes([scene])


def test_ordered_progress_and_tangent_ignore_robot_heading_and_nearby_future_route():
    scene = dict(route=[[0., 0.], [0., 2.], [.5, 2.], [.5, 0.]], boxes=[])
    meta = {key: jp.asarray(value[0]) for key, value in pack_room_scenes([scene]).items()}
    initial = _context(meta, root=(.45, .1))
    assert initial["segment"] == 0
    np.testing.assert_allclose(initial["progress_m"], .1)
    np.testing.assert_array_equal(initial["tangent"], [0., 1.])
    root = jp.array([.2, 2.])
    later = route_context(root, root, jp.int32(1), jp.array(False),
                          meta["route"], meta["route_count"], meta["obstacles"], meta["obstacle_count"])
    np.testing.assert_allclose(later["progress_m"], 2.2)
    np.testing.assert_array_equal(later["tangent"], [1., 0.])


class _NativeHooks:
    def _sample_body_fields(self, positions, *, root_xy=None):
        return jp.zeros_like(positions), jp.zeros_like(positions), jp.ones((len(positions), 1))

    def compute_cmd_from_rtf(self, rtf, cgf, cbf):
        return jp.zeros(4)


class _Hooks(RoomNavigationMixin, _NativeHooks):
    pass


def test_jit_true_and_delayed_queries_use_the_same_certified_narrow_radius():
    env = object.__new__(_Hooks)
    env._room_arrays = {key: jp.asarray(value) for key, value in pack_room_scenes([_narrow_scene()]).items()}
    env._room_scene_index = jp.array([0])
    env._field_pf_id = jp.int32(0)

    def query(root):
        with env._navigation_scope():
            info = env._update_navigation(SimpleNamespace(qpos=root))
            gf, bf, _ = env._sample_body_fields(jp.zeros((11, 3)), root_xy=root - jp.array([.1, 0.]))
            command = env.compute_cmd_from_rtf(gf[1], gf, bf)
            return info, command

    info, command = jax.jit(query)(jp.asarray([.6, 0.]))
    assert not info["room_navigation"]["violation"]
    assert command[1] > .5
    np.testing.assert_allclose(command[2:], 0.)


@pytest.mark.parametrize("case", ["disabled", "missing_bank", "wrong_proxy"])
def test_contrastive_loader_requires_actual_matching_body_collision_bank(case):
    env = object.__new__(_Hooks)
    env.body_collision_enabled = case != "disabled"
    if case != "missing_bank":
        env._body_collision_bank = {}
    env.body_collision_contract = dict(schema="cat-body-collision-v1", proxy_sha256="a" * 64)
    if case == "wrong_proxy":
        env.body_collision_contract["proxy_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        env._validate_contrast_collision_contract(_narrow_scene())


def test_contrastive_loader_accepts_actual_matching_body_collision_bank():
    env = object.__new__(_Hooks)
    env.body_collision_enabled = True
    env._body_collision_bank = {}
    env.body_collision_contract = dict(schema="cat-body-collision-v1", proxy_sha256="a" * 64)
    env._validate_contrast_collision_contract(_narrow_scene())


class _LoaderBase(_NativeHooks):
    def __init__(self, directory, scene, *, collision=True, contrast=True):
        self.compatibility_mode = False
        self._config = SimpleNamespace(pf_config=SimpleNamespace(bank_manifest=directory / "manifest.json"),
                                       wholebody_hand_contrast=contrast)
        self.field_bank_manifest = dict(scenes=[
            dict(family="original_cat", scene_id="native", path="unused"),
            dict(family="furniture", scene_id="narrow", path="room", source={
                "occupancy": "conservative-voxel-cell-OBB-intersection-v1",
                "room_navigation": "ordered-certified-route-v1",
            }),
        ])
        (directory / "room").mkdir()
        (directory / "room" / "scene.json").write_text(json.dumps(scene))
        self.body_collision_enabled = collision
        self._body_collision_bank = {}
        self.body_collision_contract = dict(schema="cat-body-collision-v1", proxy_sha256="a" * 64)

    def observation_contract(self):
        return {"actor": 222, "critic": 310}


class _LoadedNavigation(RoomNavigationMixin, _LoaderBase):
    pass


def test_loader_packs_contrast_metadata_and_runtime_operands_keep_legacy_scene_disabled(tmp_path):
    from cat_ppo.learning.policy.ppo.field_arguments import FieldArguments
    env = _LoadedNavigation(tmp_path, _narrow_scene())
    np.testing.assert_allclose(env._room_arrays["navigation_radius"], [.23, .15])
    assert env.observation_contract()["hand_contrast_navigation"]["observations_added"] == 0
    # This small navigation-only fixture has no ragged SDF/body simulator.
    del env.field_bank_manifest, env._body_collision_bank
    fields = FieldArguments(env)

    def query(index, arrays):
        with fields.bind(arrays), env._navigation_scope():
            env._field_pf_id = index
            return env._update_navigation(SimpleNamespace(qpos=jp.array([.6, 0., .8])))

    execute = jax.jit(query)
    native, narrow = (execute(jp.int32(index), fields.values) for index in (0, 1))
    assert not native["hand_contrast"]["enabled"]
    assert narrow["hand_contrast"]["enabled"]
    assert not narrow["room_navigation"]["violation"]
    assert narrow["hand_contrast"]["forward_weight"] == 0.
    np.testing.assert_allclose(narrow["room_navigation"]["progress_m"], .6)
    np.testing.assert_array_equal(narrow["hand_contrast"]["route_tangent"], [1., 0.])
    changed = dict(fields.values[0])
    changed["hand_contrast_start_m"] = changed["hand_contrast_start_m"].at[1, 0].set(1.)
    changed_info = execute(jp.int32(1), (changed, fields.values[1]))
    assert not changed_info["hand_contrast"]["enabled"]


def test_loader_rejects_contrast_scenes_without_enabled_full_body_collision(tmp_path):
    with pytest.raises(ValueError, match="validated full-body collision bank"):
        _LoadedNavigation(tmp_path, _narrow_scene(), collision=False)


def test_loader_rejects_contrast_bank_without_its_reward_profile(tmp_path):
    with pytest.raises(ValueError, match="enabled together"):
        _LoadedNavigation(tmp_path, _narrow_scene(), contrast=False)
