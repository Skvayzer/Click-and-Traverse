"""Whole-body fine-tuning as an extension of the released CAT generalist task.

The inherited task owns the reward, disturbances, gait, observations, field
latency and termination.  This module changes only the requested body action
interface, fixed Dex3 hand spheres, and two elbow field samples. The optional
BodyCollisionMixin adds simulator-internal primitive-volume obstacle failures
and an explicit terminal penalty, without enlarging policy observations or
adding obstacle contact impulses to CAT's native floor/self-contact physics.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict

from cat_ppo.envs.g1 import constants as consts
from cat_ppo.envs.g1.env_cat import G1CatEnv, delay_body_pos, world_to_navi_vel, EPS
from cat_ppo.envs.g1.env_loco import G1LocoEnv
from cat_ppo.furniture import control
from cat_ppo.furniture.grippers import (
    install_fixed_hands, hand_envelope, hand_sphere, validate_hand_spheres,
)


def wholebody_config(base_config, *, bank_manifest=None, compatibility_mode=False,
                     stabilization=False):
    """Extend the supplied *released* env config without replacing its settings.

    Compatibility mode uses the original robot and 12-action observation
    contract, without additive rewards or probes.  It is intended for numerical
    reference comparisons, not the new whole-body experiment.
    """
    config = (base_config.copy_and_resolve_references()
              if hasattr(base_config, "copy_and_resolve_references")
              else config_dict.ConfigDict(deepcopy(base_config)))
    config.unlock()
    config.compatibility_mode = bool(compatibility_mode)
    if compatibility_mode and stabilization:
        raise ValueError("Stabilization adds whole-body rewards and cannot be used in exact CAT compatibility mode")
    config.wholebody_stabilization = bool(stabilization)
    from cat_ppo.envs.g1.body_collision import PROPOSAL
    config.wholebody = config_dict.create(body_collision=config_dict.create(
        enabled=False, bank_manifest="", reset_manifest="", proposal=str(PROPOSAL),
        event_penalty=1.0))
    if bank_manifest is not None:
        from cat_ppo.furniture.generalist_fields import bank_config
        config.pf_config.update(bank_config(bank_manifest))
    config.upper_action_scale = 0.8
    config.upper_target_rate = 2.0
    config.hand_protection_enabled = not compatibility_mode
    config.terminate_on_elbow_collision = not compatibility_mode
    config.hand_clearance_margin = 0.12
    config.arm_clearance_margin = 0.08
    config.clutter_episode_length = 4000
    # Candidate physical normalization and small additive cost weights. Native
    # CAT still multiplies the combined reward by its .02s control timestep.
    config.upper_velocity_cost_scale = 2.0  # rad/s; existing target slew limit
    config.upper_acceleration_cost_scale = 20.0  # rad/s^2
    config.upper_posture_clearance_taper = .12  # m beyond protection margins
    config.upper_cost_waist_weight = 4.0
    contract = (control.legacy_observation_contract() if compatibility_mode
                else control.wholebody_observation_contract())
    config.num_act = len(contract["action_names"])
    config.num_obs = len(contract["actor_features"])
    config.num_pri = len(contract["critic_features"])
    if not compatibility_mode:
        config.reward_config.scales.wholebody_hand_clearance = -5.0
        config.reward_config.scales.wholebody_arm_clearance = -2.0
        if stabilization:
            config.reward_config.scales.wholebody_upper_target_velocity = -.05
            config.reward_config.scales.wholebody_upper_target_acceleration = -.02
            config.reward_config.scales.wholebody_upper_clear_posture = -.05
    return config


def _numbers(values):
    return " ".join(format(float(value), ".10g") for value in values)


def assemble_training_xml(asset_root=None):
    """Original physics plus one field-query sphere/hand and one site/elbow.

    No obstacle geoms or extra self-contact pairs are introduced.  Original
    explicit hand/thigh pairs retain their Dex3 box solely for those physics
    contacts. There are no box-corner query sites. Sphere geoms are noncontact
    visualizations with zero added mass; their centers reuse CAT's palm sites.
    """
    asset_root = Path(asset_root or consts.ROOT_PATH).resolve()
    root = ET.parse(asset_root / "g1_mjx_feetonly_torque.xml").getroot()
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", str(asset_root / mesh.get("file")))
    # Merge the scene wrapper's floor, sensor and rendering settings.  Keeping
    # its original floor attributes also preserves the explicit foot pairs.
    scene = ET.parse(asset_root / "scene_mjx_feetonly_flat_terrain.xml").getroot()
    for child in scene:
        if child.tag == "include":
            continue
        existing = root.find(child.tag)
        if existing is not None and child.tag in ("asset", "worldbody", "sensor"):
            existing.extend(deepcopy(list(child)))
        else:
            root.append(deepcopy(child))
    install_fixed_hands(root)
    bodies = {body.get("name"): body for body in root.findall(".//body")}
    for side in ("left", "right"):
        envelope = hand_envelope(side)
        sphere = hand_sphere(side)
        ET.SubElement(
            bodies[f"{side}_wrist_yaw_link"], "geom",
            name=f"furniture_{side}_hand_envelope", type="box", **{
                "class": "collision", "pos": _numbers(envelope["center"]),
                "size": _numbers(envelope["half_size"]), "density": "0",
                "contype": "0", "conaffinity": "0", "group": "3",
                "rgba": "0.25 0.6 0.85 0.3",
            },
        )
        wrist = bodies[f"{side}_wrist_yaw_link"]
        wrist.find(f"site[@name='{side}_palm']").set("pos", _numbers(sphere["center"]))
        ET.SubElement(wrist, "geom", name=f"furniture_{side}_hand_sphere", type="sphere",
                      pos=_numbers(sphere["center"]), size=str(sphere["radius"]),
                      density="0", contype="0", conaffinity="0", group="4", rgba="0.25 0.6 0.85 0.2")
        ET.SubElement(bodies[f"{side}_elbow_link"], "site", name=f"{side}_elbow_probe",
                      pos="0 0 0", size="0.005", group="5")
    return ET.tostring(root, encoding="unicode")


class _WholeBodyTask(G1CatEnv):
    """Small extension hooks; released CAT reset/step remain the implementation."""

    def __init__(self, task_type="flat_terrain", config=None, config_overrides=None,
                 compatibility_mode=None):
        if config is None:
            raise ValueError("Pass the pinned released generalist env configuration")
        configured_mode = bool(getattr(config, "compatibility_mode", False))
        self.compatibility_mode = (configured_mode if compatibility_mode is None
                                   else bool(compatibility_mode))
        self._contract = (control.legacy_observation_contract() if self.compatibility_mode
                          else control.wholebody_observation_contract())
        self._xml_directory = None
        super().__init__(task_type=task_type, config=config,
                         config_overrides=config_overrides)

    def _task_xml_path(self, task_type):
        if self.compatibility_mode:
            return super()._task_xml_path(task_type)
        if task_type != "flat_terrain":
            raise ValueError("CAT whole-body training uses flat_terrain field-only physics")
        self._assembled_xml = assemble_training_xml()
        self._xml_directory = tempfile.TemporaryDirectory(prefix="cat-wholebody-")
        path = Path(self._xml_directory.name) / "scene.xml"
        path.write_text(self._assembled_xml)
        return str(path)

    def _post_init(self):
        if self.compatibility_mode:
            return super()._post_init()
        G1LocoEnv._post_init(self, hand_geom_names=(
            "furniture_left_hand_envelope", "furniture_right_hand_envelope"))
        self.action_joint_names, self.obs_joint_names = control.joint_names(29)
        for index, name in enumerate(control.JOINT_NAMES):
            joint = self.mj_model.joint(name)
            if (self.mj_model.actuator(name).id != index or
                    int(joint.qposadr[0]) != 7 + index or
                    int(joint.dofadr[0]) != 6 + index):
                raise ValueError(f"CAT joint coordinate ordering changed at {name}")
        self.action_joint_ids = jp.arange(29)
        self.obs_joint_ids = jp.arange(29)
        self._hand_radii = jp.asarray([hand_sphere(side)["radius"] for side in ("left", "right")])
        self._elbow_site_ids = np.asarray([self.mj_model.site(name).id for name in control.ELBOW_SITES])
        data = mujoco.MjData(self.mj_model)
        data.qpos[:] = np.asarray(self._init_q)
        mujoco.mj_forward(self.mj_model, data)
        self.hand_sphere_validation = validate_hand_spheres(self.mj_model, data)

    def _sample_body_fields(self, positions, *, root_xy=None):
        gf, bf, clearance = super()._sample_body_fields(positions, root_xy=root_xy)
        if not self.compatibility_mode:
            # CAT orders its eleven samples as head/pelvis/torso/feet/hands/
            # knees/shoulders. Adjust fresh true OR delayed queries exactly once.
            clearance = clearance.at[5:7, 0].add(-self._hand_radii)
        return gf, bf, clearance

    def reset(self, rng):
        state = super().reset(rng)
        if self.compatibility_mode:
            return state
        from cat_ppo.furniture.wholebody_stability import (
            EPISODE_KEYS, goal_status, native_fault_flags,
        )
        upper = state.info["motor_targets"][12:]
        state.info["wholebody_previous_upper_target"] = upper
        state.info["wholebody_previous_previous_upper_target"] = upper
        state.info["wholebody_applied_upper_target"] = upper
        flags = native_fault_flags(self, state.data, state.info)
        state.info["wholebody_faults"] = {key: jp.array(False) for key in flags}
        episode = {key: jp.array(False) for key in EPISODE_KEYS}
        episode["reset_replaced"] = state.info.get("wholebody_reset_replaced", jp.array(False))
        episode["outside_bounds"] = goal_status(self, state.data, state.info, jp.array(False))["outside_bounds"]
        state.info["wholebody_episode"] = episode
        _, telemetry = self._upper_stability_terms(state.info)
        telemetry["upper_joint_velocity_rms"] = jp.sqrt(jp.mean(state.data.qvel[18:35] ** 2))
        telemetry["upper_action_distance_normalized"] = jp.array(0.)
        state.info["wholebody_telemetry"] = telemetry
        return state

    def step(self, state, action):
        if self.compatibility_mode:
            return super().step(state, action)
        # Own two-target history: native step changes motor_targets and resets
        # them at termination, so neither value may be read after that reset.
        previous = state.info["wholebody_previous_upper_target"]
        result = super().step(state, action)
        result.info["wholebody_previous_previous_upper_target"] = previous
        result.info["wholebody_previous_upper_target"] = result.info["wholebody_applied_upper_target"]
        return result

    def _upper_stability_terms(self, info):
        from cat_ppo.furniture.wholebody_stability import upper_stability_terms
        return upper_stability_terms(
            info["motor_targets"][12:], info["wholebody_previous_upper_target"],
            info["wholebody_previous_previous_upper_target"], self._default_qpos[12:],
            info["handsdf"], info["wholebody_elbow_clearance"], dt=self.dt,
            velocity_scale=self._config.upper_velocity_cost_scale,
            acceleration_scale=self._config.upper_acceleration_cost_scale,
            posture_scale=self._config.upper_action_scale,
            hand_margin=self._config.hand_clearance_margin,
            elbow_margin=self._config.arm_clearance_margin,
            clearance_taper=self._config.upper_posture_clearance_taper,
            waist_weight=self._config.upper_cost_waist_weight)

    def _reset_root_pose(self, qpos):
        # A scene bank keeps released scenes' reset pose bit-for-bit unchanged;
        # added room scenes relocate the same randomized pose to a clear start.
        adjust = getattr(self, "adjust_reset_pose", None)
        return qpos if adjust is None else adjust(qpos)

    def _motor_targets(self, action, previous):
        if self.compatibility_mode:
            return super()._motor_targets(action, previous)
        # Do not clip leg actions here: the original task does not.  The policy
        # distribution already bounds actions; retaining exact increment math
        # also makes inherited CAT behavior testable on arbitrary test inputs.
        lower = jp.clip(previous[:12] + action[:12] * self._config.action_scale,
                        self._soft_lowers[:12], self._soft_uppers[:12])
        upper = self._default_qpos[12:] + action[12:] * self._config.upper_action_scale
        max_step = self._config.upper_target_rate * self.dt
        upper = jp.clip(upper, previous[12:] - max_step, previous[12:] + max_step)
        upper = jp.clip(upper, self._soft_lowers[12:], self._soft_uppers[12:])
        return jp.concatenate([lower, upper])

    def _episode_step_limit(self, info):
        original = super()._episode_step_limit(info)
        if not hasattr(self, "_pf_scene_original"):
            return original
        scene = info.get("pf_id", self._field_pf_id)
        if getattr(self, "_pf_expanded", False):
            return self._pf_scene_episode_lengths[scene]
        return jp.where(self._pf_scene_original[scene], original,
                        self._config.clutter_episode_length)

    def observation_contract(self):
        return dict(self._contract, baseline="released_CAT_generalist",
                    perception_mode="CAT_odometry_held_5_control_steps",
                    probe_sampler="released_CAT_corner_weight_order_and_clamping",
                    obstacle_physics="potential_fields_only",
                    compatibility_mode=self.compatibility_mode)

    def _elbow_fields(self, positions, info, *, actor):
        guidance = self.sample_field(self.gf, positions)
        boundary = self.sample_field(self.bf, positions)
        clearance = self.sample_field(self.sdf, positions) - control.ELBOW_RADIUS_M
        if hasattr(self, "_room_elbow_guidance"):
            guidance = self._room_elbow_guidance(guidance, boundary, clearance, info, actor=actor)
        # Match CAT: reset normalizes vectors; later GF follows the move flag.
        move = (jp.asarray(info["step"]) == 0) | (info["command"][0] > 0.5)
        guidance = guidance * move / (jp.linalg.norm(guidance, axis=-1, keepdims=True) + EPS)
        boundary = boundary / (jp.linalg.norm(boundary, axis=-1, keepdims=True) + EPS)
        if actor:
            guidance = world_to_navi_vel(info["navi2world_pose"], guidance)
            boundary = world_to_navi_vel(info["navi2world_pose"], boundary) * (clearance < 0.5)
            clearance = jp.clip(clearance, -1.0, 0.5)
        # Field-major ordering is the same as existing paired hand/foot slots.
        return jp.concatenate([guidance.reshape(-1), boundary.reshape(-1), clearance.reshape(-1)])

    def _get_obs(self, data, info, feet_contact):
        baseline = super()._get_obs(data, info, feet_contact)
        if self.compatibility_mode:
            return baseline
        positions = data.site_xpos[self._elbow_site_ids]
        true_features = self._elbow_fields(positions, info, actor=False)
        odom = info["odom_delay"]
        delayed_positions = delay_body_pos(
            data.qpos[:3], data.qpos[3:7], odom[:3], odom[3:7], positions)
        actor_features = self._elbow_fields(delayed_positions, info, actor=True)
        info["wholebody_elbow_clearance"] = true_features[-2:]
        info["wholebody_clearances"] = jp.concatenate([info["handsdf"].reshape(-1), true_features[-2:]])
        return {"state": jp.concatenate([baseline["state"], jp.nan_to_num(actor_features)]),
                "privileged_state": jp.concatenate([
                    baseline["privileged_state"], jp.nan_to_num(true_features)])}

    def _get_reward(self, data, action, info, done, feet_contact):
        rewards = super()._get_reward(data, action, info, done, feet_contact)
        if self.compatibility_mode:
            return rewards
        enabled = self._config.hand_protection_enabled
        hand_deficit = jp.maximum(self._config.hand_clearance_margin - info["handsdf"], 0.0)
        rewards["wholebody_hand_clearance"] = jp.where(enabled, jp.mean(hand_deficit ** 2), 0.0)
        deficit = jp.maximum(self._config.arm_clearance_margin
                             - info["wholebody_elbow_clearance"], 0.0)
        rewards["wholebody_arm_clearance"] = jp.where(enabled, jp.mean(deficit ** 2), 0.0)
        costs, telemetry = self._upper_stability_terms(info)
        telemetry["upper_joint_velocity_rms"] = jp.sqrt(jp.mean(data.qvel[18:35] ** 2))
        telemetry["upper_action_distance_normalized"] = jp.sqrt(jp.mean(action[12:] ** 2))
        info["wholebody_telemetry"] = telemetry
        info["wholebody_applied_upper_target"] = info["motor_targets"][12:]
        if self._config.wholebody_stabilization:
            rewards.update(costs)
        return rewards

    def _crossed_goal(self, positions):
        original = super()._crossed_goal(positions)
        if not hasattr(self, "_pf_scene_goals"):
            return original
        scene = self._field_pf_id
        near_goal = jp.linalg.norm(positions[..., :2] - self._pf_scene_goals[scene, :2], axis=-1) <= 0.5
        use_plane = (self._pf_crossed_is_x_plane[scene] if getattr(self, "_pf_expanded", False)
                     else self._pf_scene_original[scene])
        return jp.where(use_plane, original, near_goal)

    def _get_termination(self, data, info):
        original_done = super()._get_termination(data, info)
        if self.compatibility_mode:
            return original_done
        # Hand sphere penetration already uses CAT's inherited handsdf rule.
        elbow_collision = jp.any(info["wholebody_elbow_clearance"]
                                 < -self._config.term_collision_threshold)
        done = (original_done | (self._config.terminate_on_elbow_collision
                                & (info["step"] >= 50) & elbow_collision))
        from cat_ppo.furniture.wholebody_stability import goal_status, native_fault_flags
        flags = native_fault_flags(self, data, info)
        done = done | flags["room_root_field"] | flags["body_collision"]
        info["wholebody_faults"] = flags
        episode = info["wholebody_episode"]
        goal = goal_status(self, data, info, episode["outside_bounds"])
        episode = dict(episode)
        episode["outside_bounds"] = goal["outside_bounds"]
        for key in ("goal_reached", "raw_goal"):
            episode[key] = episode[key] | (goal[key] & ~done)
        for key in ("fall", "obstacle", "self_contact", "numerical"):
            episode[key] = episode[key] | flags[key]
        episode["body_collision"] = episode["body_collision"] | flags["body_collision"]
        from cat_ppo.envs.g1.body_collision import REGIONS
        regions = info.get("wholebody_collision_regions", jp.zeros(len(REGIONS), dtype=bool))
        for index, region in enumerate(REGIONS):
            key = "body_collision_" + region
            episode[key] = episode[key] | regions[index]
        if getattr(self, "body_collision_enabled", False):
            episode["goal_reached"] &= ~flags["any"]
        episode["hand_violation"] = episode["hand_violation"] | flags["hands_field"]
        episode["elbow_violation"] = episode["elbow_violation"] | flags["elbows_field"]
        info["wholebody_episode"] = episode
        return done


# The scene mixin owns per-environment scene IDs and CAT's adaptive reset
# metadata.  A ragged bank retains released fields without padding all 37 small
# scenes to the dimensions of a furnished room.
from cat_ppo.furniture.generalist_fields import RaggedSceneMixin  # noqa: E402
from cat_ppo.envs.g1.room_navigation import RoomNavigationMixin  # noqa: E402
from cat_ppo.envs.g1.body_collision import BodyCollisionMixin  # noqa: E402


class G1CatWholeBodyEnv(RoomNavigationMixin, BodyCollisionMixin, RaggedSceneMixin, _WholeBodyTask):
    """One CAT generalist task containing original scenes and added room fields."""
