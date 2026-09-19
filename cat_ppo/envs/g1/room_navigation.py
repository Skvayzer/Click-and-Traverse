"""Room-only route integration; released CAT scenes retain their native fields."""
import json
from contextlib import contextmanager
from pathlib import Path

import jax.numpy as jp
import numpy as np


class RoomNavigationMixin:
    """One route state per episode, shared by true and delayed CAT observations.

    Route state is simulator bookkeeping, not additional policy observations.
    Immutable room metadata is passed as runtime operands by FieldArguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.compatibility_mode:
            return
        from cat_ppo.furniture.room_navigation import pack_room_scenes, scene_navigation_radius
        from cat_ppo.furniture.room_geometry import root_cylinder_segment_clearance
        manifest_path = Path(self._config.pf_config.bank_manifest).resolve()
        rooms, mapping = [None], []
        for record in self.field_bank_manifest["scenes"]:
            is_room = record.get("task_kind", "cat" if record["family"] == "original_cat" else "room") == "room"
            if not is_room:
                mapping.append(0)
                continue
            source = record.get("source", {})
            if (source.get("occupancy") != "conservative-voxel-cell-OBB-intersection-v1"
                    or source.get("room_navigation") != "ordered-certified-route-v1"):
                raise ValueError("Room bank predates conservative geometry/ordered navigation; rebuild into a NEW bank")
            geometry = json.loads((manifest_path.parent / record["path"] / "scene.json").read_text())
            route = np.asarray(geometry["route"])
            radius = scene_navigation_radius(geometry)
            if geometry.get("hand_contrast") is not None:
                self._validate_contrast_collision_contract(geometry)
            if np.any(root_cylinder_segment_clearance(
                    route[:-1], route[1:], geometry["boxes"], radius=radius) <= 0.):
                raise ValueError(f"Room route fails continuous root-cylinder clearance: {record['scene_id']}")
            mapping.append(len(rooms))
            rooms.append(geometry)
        self._room_arrays = {key: jp.asarray(value) for key, value in pack_room_scenes(rooms).items()}
        self._has_hand_contrast = any(scene is not None and scene.get("hand_contrast") is not None
                                      for scene in rooms)
        if self._has_hand_contrast != bool(getattr(self._config, "wholebody_hand_contrast", False)):
            raise ValueError("Contrastive passage metadata and wholebody_hand_contrast must be enabled together")
        if self._has_hand_contrast:
            from cat_ppo.furniture.contrastive_rewards import pack_hand_contrast
            self._room_arrays.update({"hand_contrast_" + key: jp.asarray(value)
                                      for key, value in pack_hand_contrast(rooms).items()})
        self._room_scene_index = jp.asarray(mapping, dtype=jp.int32)

    def _validate_contrast_collision_contract(self, geometry):
        # BodyCollisionMixin initializes first through super().__init__ and
        # validates both its field geometry and certified reset-bank hashes.
        # A scene label or the enabled config flag alone is insufficient.
        contract = getattr(self, "body_collision_contract", {})
        if (not getattr(self, "body_collision_enabled", False)
                or not hasattr(self, "_body_collision_bank")
                or contract.get("schema") != "cat-body-collision-v1"):
            raise ValueError("Contrastive passages require the validated full-body collision bank")
        expected = geometry["hand_contrast"]["certificate"]["body_proxy_sha256"]
        if contract.get("proxy_sha256") != expected:
            raise ValueError("Contrastive passage certificate uses different body collision primitives")

    def observation_contract(self):
        contract = super().observation_contract()
        if hasattr(self, "_room_arrays"):
            contract["room_navigation"] = "ordered-certified-route-v1"
        if getattr(self, "_has_hand_contrast", False):
            contract["hand_contrast_navigation"] = {
                "schema": "hand-contrast-v1", "radius": "certified per scene",
                "body_collision_required": True, "observations_added": 0,
            }
        return contract

    @contextmanager
    def _navigation_scope(self):
        # Attributes below are only temporary bindings while JAX traces a
        # public reset/step. Episode state lives in info; never retain tracers
        # on the Python environment after the call returns.
        missing = object()
        names = ("_room_context", "_room_query_command", "_field_pf_id")
        previous = {name: getattr(self, name, missing) for name in names}
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is missing:
                    self.__dict__.pop(name, None)
                else:
                    setattr(self, name, value)

    def reset(self, rng):
        with self._navigation_scope():
            return super().reset(rng)

    def reset_with_pf_id(self, rng, pf_id):
        with self._navigation_scope():
            return super().reset_with_pf_id(rng, pf_id)

    def step(self, state, action):
        with self._navigation_scope():
            return super().step(state, action)

    def _room_metadata(self):
        index = self._room_scene_index[self._field_pf_id]
        return {key: value[index] for key, value in self._room_arrays.items()}

    def _update_navigation(self, data, info=None):
        if not hasattr(self, "_room_arrays"):
            return super()._update_navigation(data, info)
        from cat_ppo.furniture.room_navigation import route_context
        meta = self._room_metadata()
        root = data.qpos[:2]
        previous = None if info is None else info["room_navigation"]
        context = route_context(
            root, root if previous is None else previous["root_xy"],
            jp.int32(0) if previous is None else previous["segment"],
            jp.array(False) if previous is None else previous["violation"],
            meta["route"], meta["route_count"], meta["obstacles"], meta["obstacle_count"],
            radius=meta["navigation_radius"])
        context["root_xy"] = root
        context["enabled"] = meta["enabled"]
        context["violation"] &= meta["enabled"]
        for key in ("root_clearance", "swept_clearance"):
            context[key] = jp.where(meta["enabled"], context[key], 0.)
        self._room_context = context
        result = {"room_navigation": context}
        if getattr(self, "_has_hand_contrast", False):
            from cat_ppo.furniture.contrastive_rewards import eval_context
            arrays = {key.removeprefix("hand_contrast_"): value
                      for key, value in self._room_arrays.items()
                      if key.startswith("hand_contrast_")}
            result["hand_contrast"] = eval_context(
                arrays, self._room_scene_index[self._field_pf_id],
                context["progress_m"], context["tangent"])
        return result

    def _room_query_guidance(self, guidance, boundary, clearance, root_xy):
        from cat_ppo.furniture.room_navigation import swept_root_clearance
        context, meta = self._room_context, self._room_metadata()
        # Held odometry uses the SAME current waypoint, without advancing state.
        # A stale pose with no clear connection gets a stop command.
        delta = context["target"] - root_xy
        length = jp.linalg.norm(delta)
        direction = delta / jp.maximum(length, 1e-6)
        visible = swept_root_clearance(root_xy, context["target"], meta["obstacles"],
                                       meta["obstacle_count"], radius=meta["navigation_radius"]) > 0.
        active = visible & ~context["blocked"] & ~context["violation"]
        speed = jp.linalg.norm(context["guidance"][:2])
        velocity = direction * speed * active
        self._room_query_command = jp.concatenate([jp.where(jp.linalg.norm(velocity) > .01, .75, 0.)[None], velocity, jp.zeros(1)])
        # Preserve CAT's vertical body shaping, BF/SDF inputs and all protection
        # rewards. Horizontal attraction comes ONLY from the ordered route.
        replacement = guidance.at[:, :2].set(jp.broadcast_to(direction * .6, guidance[:, :2].shape))
        unit_boundary = boundary / (jp.linalg.norm(boundary, axis=-1, keepdims=True) + 1e-9)
        inward = jp.minimum(jp.sum(replacement * unit_boundary, axis=-1, keepdims=True), 0.)
        replacement -= inward * unit_boundary * (clearance < .5)
        replacement *= active
        # Pelvis is the route reference; body projections must not select a
        # different corridor or alter its command.
        replacement = replacement.at[1].set(jp.concatenate([velocity, jp.zeros(1)]))
        return jp.where(context["enabled"], replacement, guidance)

    def _sample_body_fields(self, positions, *, root_xy=None):
        gf, bf, sdf = super()._sample_body_fields(positions, root_xy=root_xy)
        if hasattr(self, "_room_context"):
            root_xy = self._room_context["root_xy"] if root_xy is None else root_xy
            gf = self._room_query_guidance(gf, bf, sdf, root_xy)
        return gf, bf, sdf

    def compute_cmd_from_rtf(self, rtf, cgf, cbf):
        original = super().compute_cmd_from_rtf(rtf, cgf, cbf)
        if not hasattr(self, "_room_context"):
            return original
        return jp.where(self._room_context["enabled"], self._room_query_command, original)

    def _update_phase(self, state):
        navigation = state.info.get("room_navigation")
        if navigation is not None:
            # Native CAT's stop latch is permanent. A room may temporarily stop
            # because no visible recovery target exists, then recover after a
            # push/motion makes its current route segment visible again.
            resume = navigation["enabled"] & (state.info["command"][0] > .5)
            state.info["stop_timestep"] = jp.where(resume, 100, state.info["stop_timestep"])
        return super()._update_phase(state)

    def _room_elbow_guidance(self, guidance, boundary, clearance, info, *, actor):
        if "room_navigation" not in info:
            return guidance
        command = info["command_delay" if actor else "command"]
        xy = command[1:3]
        xy = .6 * xy / jp.maximum(jp.linalg.norm(xy), 1e-6)
        replacement = guidance.at[:, :2].set(jp.broadcast_to(xy, guidance[:, :2].shape))
        normal = boundary / (jp.linalg.norm(boundary, axis=-1, keepdims=True) + 1e-9)
        inward = jp.minimum(jp.sum(replacement * normal, axis=-1, keepdims=True), 0.)
        replacement -= inward * normal * (clearance < .5)
        replacement *= command[0] > .5
        return jp.where(info["room_navigation"]["enabled"], replacement, guidance)

    def _crossed_goal(self, positions):
        original = super()._crossed_goal(positions)
        if not hasattr(self, "_room_context"):
            return original
        context = self._room_context
        return original & (~context["enabled"] | (context["route_complete"] & ~context["violation"]))
