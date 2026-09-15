"""Whole-body fine-tuning as an extension of the released CAT generalist task.

The inherited task owns the reward, disturbances, gait, observations, field
latency and termination.  This module changes only the requested body action
interface, fixed Dex3 hands, and explicit hand/arm field probes.  Furniture
remains a potential field during training, as in released CAT.  The separate
physical furniture environment is useful for contact evaluation, not training.
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
from cat_ppo.envs.g1.env_cat import G1CatEnv, delay_body_pos, world_to_navi_vel
from cat_ppo.envs.g1.env_loco import G1LocoEnv
from cat_ppo.furniture import control
from cat_ppo.furniture.grippers import install_fixed_hands, hand_envelope, validate_hand_envelopes
from cat_ppo.furniture.perception import hand_protection_cost


def wholebody_config(base_config, *, bank_manifest=None, compatibility_mode=False):
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
    if bank_manifest is not None:
        from cat_ppo.furniture.generalist_fields import bank_config
        config.pf_config.update(bank_config(bank_manifest))
    config.upper_action_scale = 0.8
    config.upper_target_rate = 2.0
    config.hand_protection_enabled = not compatibility_mode
    config.terminate_on_probe_collision = not compatibility_mode
    config.hand_clearance_margin = 0.12
    config.arm_clearance_margin = 0.08
    config.hand_urgency_gain = 1.0
    config.clutter_episode_length = 4000
    contract = (control.legacy_observation_contract() if compatibility_mode
                else control.observation_contract(29))
    config.num_act = len(contract["action_names"])
    config.num_obs = len(contract["actor_features"])
    config.num_pri = len(contract["critic_features"])
    if not compatibility_mode:
        config.reward_config.scales.wholebody_hand_clearance = -5.0
        config.reward_config.scales.wholebody_arm_clearance = -2.0
    return config


def _numbers(values):
    return " ".join(format(float(value), ".10g") for value in values)


def assemble_training_xml(asset_root=None):
    """Original feet-only flat physics, with welded Dex3 geometry and probes.

    No obstacle geoms or extra self-contact pairs are introduced.  Original
    explicit hand/thigh pairs refer to the corrected Dex3 bounding box; meshes
    and boxes have zero general collision masks and the boxes have no mass.
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
        ET.SubElement(
            bodies[f"{side}_wrist_yaw_link"], "geom",
            name=f"furniture_{side}_hand_envelope", type="box", **{
                "class": "collision", "pos": _numbers(envelope["center"]),
                "size": _numbers(envelope["half_size"]), "density": "0",
                "contype": "0", "conaffinity": "0", "group": "3",
                "rgba": "0.25 0.6 0.85 0.3",
            },
        )
    for name, body, point, _ in control.PROBE_SPECS:
        ET.SubElement(bodies[body], "site", name=f"furniture_probe_{name}",
                      pos=_numbers(point), size="0.005", group="5")
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
                          else control.observation_contract(29))
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
        self._probe_site_ids = np.asarray([
            self.mj_model.site(f"furniture_probe_{spec[0]}").id
            for spec in control.PROBE_SPECS])
        self._probe_radii = jp.asarray([spec[3] for spec in control.PROBE_SPECS])
        self._hand_probe_ids = jp.asarray([
            i for i, spec in enumerate(control.PROBE_SPECS)
            if "hand_corner" in spec[0] or "palm" in spec[0]])
        self._arm_probe_ids = jp.asarray([
            i for i, spec in enumerate(control.PROBE_SPECS)
            if "forearm" in spec[0] or "elbow" in spec[0]])
        data = mujoco.MjData(self.mj_model)
        data.qpos[:] = np.asarray(self._init_q)
        mujoco.mj_forward(self.mj_model, data)
        self.hand_envelope_validation = validate_hand_envelopes(self.mj_model, data)

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
        return jp.where(self._pf_scene_original[scene], original,
                        self._config.clutter_episode_length)

    def observation_contract(self):
        return dict(self._contract, baseline="released_CAT_generalist",
                    perception_mode="CAT_odometry_held_5_control_steps",
                    probe_sampler="released_CAT_corner_weight_order_and_clamping",
                    obstacle_physics="potential_fields_only",
                    compatibility_mode=self.compatibility_mode)

    def _probe_features(self, positions, velocity, *, age):
        # Use the same sampler as CAT, including its released interpolation
        # convention.  Silently fixing that convention changes learned inputs.
        distances = [self.sample_field(self.sdf, positions + horizon * velocity)[:, 0]
                     - self._probe_radii for horizon in (0.0, 0.2, 0.4)]
        boundary = self.sample_field(self.bf, positions)
        boundary /= jp.maximum(jp.linalg.norm(boundary, axis=-1, keepdims=True), 1e-6)
        return jp.concatenate([
            jp.clip(jp.stack(distances, axis=-1), -1.0, 2.0), boundary,
            jp.full((len(control.PROBE_SPECS), 1), age),
            # Known static fields, with the same boundary clamping as CAT.
            jp.zeros((len(control.PROBE_SPECS), 2)),
        ], axis=-1)

    def _get_obs(self, data, info, feet_contact):
        baseline = super()._get_obs(data, info, feet_contact)
        if self.compatibility_mode:
            return baseline
        positions = data.site_xpos[self._probe_site_ids]
        previous = info.get("wholebody_probe_positions", positions)
        velocity = (positions - previous) / self.dt
        true_features = self._probe_features(positions, velocity, age=0.0)
        odom = info["odom_delay"]
        delayed_positions = delay_body_pos(
            data.qpos[:3], data.qpos[3:7], odom[:3], odom[3:7], positions)
        age = (jp.maximum(jp.asarray(info["step"]) - 1, 0) % 5) * self.dt
        actor_features = self._probe_features(delayed_positions, velocity, age=age)
        actor_features = actor_features.at[:, 3:6].set(
            world_to_navi_vel(info["navi2world_pose"], actor_features[:, 3:6]))
        info["wholebody_probe_positions"] = positions
        info["wholebody_probe_velocity"] = velocity
        info["wholebody_probe_features"] = true_features
        return {"state": jp.concatenate([baseline["state"], actor_features.reshape(-1)]),
                "privileged_state": jp.concatenate([
                    baseline["privileged_state"], true_features.reshape(-1)])}

    def _get_reward(self, data, action, info, done, feet_contact):
        rewards = super()._get_reward(data, action, info, done, feet_contact)
        if self.compatibility_mode:
            return rewards
        features = info["wholebody_probe_features"]
        enabled = self._config.hand_protection_enabled
        rewards["wholebody_hand_clearance"] = jp.where(enabled, hand_protection_cost(
            features[self._hand_probe_ids], info["wholebody_probe_velocity"][self._hand_probe_ids],
            clearance_margin=self._config.hand_clearance_margin,
            urgency_gain=self._config.hand_urgency_gain), 0.0)
        deficit = jp.maximum(self._config.arm_clearance_margin
                             - features[self._arm_probe_ids, 0], 0.0)
        rewards["wholebody_arm_clearance"] = jp.where(enabled, jp.mean(deficit ** 2), 0.0)
        return rewards

    def _crossed_goal(self, positions):
        original = super()._crossed_goal(positions)
        if not hasattr(self, "_pf_scene_goals"):
            return original
        scene = self._field_pf_id
        near_goal = jp.linalg.norm(positions[..., :2] - self._pf_scene_goals[scene, :2], axis=-1) <= 0.5
        return jp.where(self._pf_scene_original[scene], original, near_goal)

    def _get_termination(self, data, info):
        original_done = super()._get_termination(data, info)
        if self.compatibility_mode:
            return original_done
        probe_collision = jp.any(info["wholebody_probe_features"][:, 0]
                                 < -self._config.term_collision_threshold)
        return (original_done | (self._config.terminate_on_probe_collision
                                 & (info["step"] >= 50) & probe_collision))


# The scene mixin owns per-environment scene IDs and CAT's adaptive reset
# metadata.  A ragged bank retains released fields without padding all 37 small
# scenes to the dimensions of a furnished room.
from cat_ppo.furniture.generalist_fields import RaggedSceneMixin  # noqa: E402


class G1CatWholeBodyEnv(RaggedSceneMixin, _WholeBodyTask):
    """One CAT generalist task containing original scenes and added room fields."""
