"""CAT-native furniture traversal with physical collisions and 29-joint control.

The original G1Cat task is unchanged. Perception modes here are controlled
simulator map experiments; they are not a deployed RGB-D/LiDAR reconstruction.
"""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jp
import jaxlie
import mujoco
from mujoco import mjx
import numpy as np
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from cat_ppo.envs.g1 import base, constants as consts
from cat_ppo.envs.g1.env_cat import G1CatEnv, base2navi_transform, g1_loco_task_config
from cat_ppo.envs.g1.env_loco import G1LocoEnv
from cat_ppo.furniture import control
from cat_ppo.furniture.perception import sample_grid, probe_features, apply_uncertainty_margin, hand_protection_cost
from cat_ppo.furniture.scenes import load_scene


def default_config():
    config = g1_loco_task_config().env_config.copy_and_resolve_references()
    config.action_dofs = 29
    config.upper_action_scale = 0.8
    config.upper_target_rate = 2.0
    config.upper_body_mode = "learned"
    config.probe_features_enabled = True
    config.perception_mode = "oracle"
    config.map_latency_steps = 0
    config.map_update_interval_steps = 1
    config.unknown_probability = 0.0
    config.map_position_noise_std = 0.0
    config.prediction_enabled = True
    config.uncertainty_margin_enabled = True
    config.fixed_uncertainty_margin = 0.01
    config.uncertainty_std_multiplier = 2.0
    config.relative_motion_bound = 1.0
    config.goal_radius = 0.3
    config.clearance_margin = 0.08
    config.progress_reward = 5.0
    config.clearance_cost = 2.0
    config.hand_protection_enabled = True
    config.hand_clearance_margin = 0.12
    config.hand_clearance_cost = 5.0
    config.hand_urgency_gain = 1.0
    config.contact_cost = 10.0
    config.success_reward = 10.0
    config.time_cost = 0.05
    config.push_config.enable = False
    config.dm_rand_config.enable_pd = False
    config.dm_rand_config.enable_rfi = False
    return config


def _numbers(values):
    return " ".join(format(float(v), ".10g") for v in values)


def assemble_scene_xml(scene, asset_root=None):
    """Derive physics from canonical scene boxes, independently of its voxel field.

    Only collision primitives get enabled. Visual meshes remain noncolliding.
    Added hand/finger envelopes change neither robot joints nor inertial masses.
    """
    asset_root = Path(asset_root or consts.ROOT_PATH).resolve()
    root = ET.parse(asset_root / "g1_mjx_feetonly_torque.xml").getroot()
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", str(asset_root / mesh.get("file")))
    world = root.find("worldbody")
    for geom in world.iter("geom"):
        if geom.get("class") in ("collision", "foot"):
            geom.set("contype", "1")
            geom.set("conaffinity", "6")  # furniture(2), floor(4); explicit original self pairs remain
            geom.set("condim", "3")
            geom.set("friction", "0.8 0.005 0.0001")
        else:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
    bodies = {b.get("name"): b for b in world.iter("body")}

    def collision(body, name, kind, size, **attributes):
        ET.SubElement(bodies[body], "geom", name=name, type=kind, size=_numbers(size),
            contype="1", conaffinity="6", condim="3", density="0", group="3",
            rgba="0.25 0.6 0.85 0.35", friction="0.8 0.005 0.0001", **attributes)

    collision("pelvis", "furniture_pelvis", "box", (0.105, 0.115, 0.085), pos="0 0 -0.025")
    collision("torso_link", "furniture_torso", "capsule", (0.115,), fromto="0 0 0.04 0 0 0.25")
    collision("torso_link", "furniture_head", "sphere", (0.10,), pos="0 0 0.4")
    for side in ("left", "right"):
        collision(f"{side}_shoulder_pitch_link", f"furniture_{side}_shoulder", "sphere", (0.055,))
        collision(f"{side}_shoulder_roll_link", f"furniture_{side}_upper_arm", "capsule", (0.05,), fromto="0 0 -0.02 0 0 -0.17")
        collision(f"{side}_elbow_link", f"furniture_{side}_forearm", "capsule", (0.05,), fromto="0 0 0 0.14 0 -0.01")
        collision(f"{side}_wrist_yaw_link", f"furniture_{side}_hand_envelope", "box", control.HAND_HALF_SIZE, pos=_numbers(control.HAND_CENTER))
    for name, body, point, _ in control.PROBE_SPECS:
        ET.SubElement(bodies[body], "site", name=f"furniture_probe_{name}", pos=_numbers(point), size="0.005", group="5")
    # Explicit nonadjacent self pairs; never enable overlapping visual meshes or
    # adjacent links. New envelopes participate independently of field probes.
    contacts = root.find("contact")
    for side in ("left", "right"):
        other = "right" if side == "left" else "left"
        for moving in (f"furniture_{side}_hand_envelope", f"furniture_{side}_forearm"):
            for fixed in ("furniture_torso", "furniture_pelvis", f"{other}_thigh"):
                ET.SubElement(contacts, "pair", geom1=moving, geom2=fixed, condim="3")
    ET.SubElement(contacts, "pair", geom1="furniture_left_hand_envelope", geom2="furniture_right_hand_envelope", condim="3")
    ET.SubElement(contacts, "pair", geom1="furniture_left_forearm", geom2="furniture_right_forearm", condim="3")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.01", contype="4", conaffinity="1", condim="3", friction="1 0.005 0.0001", rgba="0.75 0.75 0.75 1")
    for i, box in enumerate(scene["boxes"]):
        ET.SubElement(world, "geom", name=f"furniture_object_{i}", type="box",
            pos=_numbers(box["center"]), size=_numbers(box["half_size"]),
            euler=_numbers((0, 0, box.get("yaw", 0))), contype="2", conaffinity="1",
            condim="3", friction="0.8 0.005 0.0001", rgba="0.5 0.35 0.2 1")
    return ET.tostring(root, encoding="unicode")


class G1FurnitureEnv(G1CatEnv):
    """Brax/MJX environment; one compiled physical scene per instance."""

    def __init__(self, scene_dir, config=None):
        self.scene_dir = Path(scene_dir).resolve()
        self.scene = load_scene(self.scene_dir)
        config = default_config() if config is None else config.copy_and_resolve_references()
        if config.perception_mode not in ("oracle", "corrupted"):
            raise ValueError("perception_mode must be oracle or corrupted")
        if int(config.map_latency_steps) < 0 or int(config.map_update_interval_steps) < 1:
            raise ValueError("map latency must be nonnegative and update interval positive")
        if not 0 <= config.unknown_probability <= 1 or config.map_position_noise_std < 0:
            raise ValueError("invalid perception corruption parameters")
        if min(config.fixed_uncertainty_margin, config.uncertainty_std_multiplier, config.relative_motion_bound) < 0:
            raise ValueError("uncertainty margins and proposed motion bounds must be nonnegative")
        self._contract = control.observation_contract(config.action_dofs)
        if config.upper_body_mode not in ("learned", "nominal", "raised", "tucked", "contextual"):
            raise ValueError("invalid upper_body_mode")
        config.num_obs = len(self._contract["actor_features"])
        config.num_pri = len(self._contract["critic_features"])
        config.num_act = len(self._contract["action_names"])
        config.episode_length = int(np.ceil(float(self.scene["time_budget"]) / config.ctrl_dt))
        self._assembled_xml = assemble_scene_xml(self.scene)
        self._xml_directory = tempfile.TemporaryDirectory(prefix="cat-furniture-")
        xml_path = Path(self._xml_directory.name) / "scene.xml"
        xml_path.write_text(self._assembled_xml)
        base.G1Env.__init__(self, str(xml_path), config)
        G1LocoEnv._post_init(self)
        self.action_joint_names, self.obs_joint_names = control.joint_names(config.action_dofs)
        # Released CAT indexes qpos/qvel with actuator IDs. Assert that contract.
        for i, name in enumerate(control.JOINT_NAMES):
            joint = self.mj_model.joint(name)
            if self.mj_model.actuator(name).id != i or int(joint.qposadr[0]) != 7 + i or int(joint.dofadr[0]) != 6 + i:
                raise ValueError(f"G1 joint/actuator ordering mismatch at {name}")
        self.action_joint_ids = jp.asarray([control.JOINT_NAMES.index(n) for n in self.action_joint_names])
        self.obs_joint_ids = jp.asarray([control.JOINT_NAMES.index(n) for n in self.obs_joint_names])
        self._head_site_id = self.mj_model.site("head").id
        self._knees_site_id = np.asarray([self.mj_model.site(n).id for n in consts.KNEE_SITES])
        self._shlds_site_id = np.asarray([self.mj_model.site(n).id for n in consts.SHOULDER_SITES])
        self._probe_site_ids = np.asarray([self.mj_model.site(f"furniture_probe_{p[0]}").id for p in control.PROBE_SPECS])
        self._probe_radii = jp.asarray([p[3] for p in control.PROBE_SPECS])
        self._hand_probe_ids = jp.asarray([i for i, spec in enumerate(control.PROBE_SPECS) if "hand_corner" in spec[0] or "palm" in spec[0]])
        fields = {}
        for name in ("sdf", "bf", "gf"):
            path = self.scene_dir / f"{name}.npy"
            # load_scene has already verified checksums, shape and finiteness.
            array = np.load(path, allow_pickle=False)
            fields[name] = jp.asarray(array)
        self.sdf, self.bf, self.gf = fields["sdf"][..., None], fields["bf"], fields["gf"]
        grid = self.scene["grid"]
        self.dx = float(grid["voxel_size"])
        self._resolution_error_bound = float(grid["resolution_error_bound_m"])
        if grid["axis_order"] != "xyz" or tuple(grid["shape"]) != self.sdf.shape[:3] or not np.isfinite(self.dx) or self.dx <= 0:
            raise ValueError("invalid scene grid geometry")
        self.pf_origin = jp.asarray(grid["sample_origin"])
        self.Nx, self.Ny, self.Nz = self.sdf.shape[:3]
        if min(self.sdf.shape[:3]) < 2 or self.bf.shape != self.sdf.shape[:3] + (3,) or self.gf.shape != self.bf.shape:
            raise ValueError("scene field shapes must be matching xyz arrays of at least two voxels per axis")
        self._route = jp.asarray(self.scene["route"], dtype=jp.float32)
        self._route_length = float(np.sum(np.linalg.norm(np.diff(np.asarray(self.scene["route"]), axis=0), axis=-1)))
        self._goal = jp.asarray(self.scene["goal"])
        self._latency = int(config.map_latency_steps) if config.perception_mode == "corrupted" else 0
        self._interval = int(config.map_update_interval_steps) if config.perception_mode == "corrupted" else 1
        self._packet_size = 77 + len(control.PROBE_SPECS) * 9 + 4
        self._furniture_geom_ids = np.asarray([self.mj_model.geom(f"furniture_object_{i}").id for i in range(len(self.scene["boxes"]))])
        names = [self.mj_model.geom(i).name or "" for i in range(self.mj_model.ngeom)]
        self._hand_geom_mask = jp.asarray(["hand" in n for n in names])
        self._arm_geom_mask = jp.asarray(["forearm" in n or "upper_arm" in n or "shoulder" in n for n in names])
        self._furniture_geom_mask = jp.asarray([n.startswith("furniture_object_") for n in names])
        self._robot_geom_mask = jp.asarray(self.mj_model.geom_bodyid > 0)
        self._init_q = self._init_q.at[:2].set(jp.asarray(self.scene["start"][:2]))
        yaw = float(self.scene["start"][2])
        self._init_q = self._init_q.at[3:7].set(jp.asarray([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        self._validate_reset()

    def observation_contract(self):
        return dict(self._contract, perception_mode=self._config.perception_mode,
                    prediction_model="constant_velocity_at_capture_time", scene_schema=self.scene["schema"],
                    prediction_enabled=self._config.prediction_enabled,
                    probe_features_enabled=self._config.probe_features_enabled,
                    upper_body_mode=self._config.upper_body_mode,
                    uncertainty_margin_enabled=self._config.uncertainty_margin_enabled,
                    uncertainty_bound_status="proposed_not_calibrated",
                    grid_resolution_error_bound_m=self._resolution_error_bound,
                    first_contact_part_codes={"0": "none", "1": "other_body", "2": "hand", "3": "arm"})

    def _validate_reset(self):
        """Host FK/collision validation once; every reset uses this exact safe pose."""
        data = mujoco.MjData(self.mj_model)
        data.qpos[:] = np.asarray(self._init_q)
        mujoco.mj_forward(self.mj_model, data)
        feet = set(int(i) for i in self._feet_geom_id)
        for contact in data.contact:
            a, b = (int(i) for i in contact.geom)
            allowed = (a == self._floor_geom_id and b in feet) or (b == self._floor_geom_id and a in feet)
            if contact.dist <= 0 and not allowed:
                raise ValueError(f"scene reset intersects forbidden geometry: {self.mj_model.geom(a).name}, {self.mj_model.geom(b).name}")
        self.reset_validation = {"nominal_full_body_checked": True, "randomized_pose": False,
                                 "method": "MuJoCo FK and physical primitive collision at fixed start"}

    def sample_field(self, field, pos):
        value, known = sample_grid(field, pos, self.pf_origin, self.dx)
        return jp.where(known[:, None], value, -1.0 if field.shape[-1] == 1 else 0.0)

    def _contacts(self, data):
        pairs = jp.asarray(data.contact.geom)
        a, b = pairs[:, 0], pairs[:, 1]
        foot_a = jp.any(a[:, None] == self._feet_geom_id[None, :], axis=-1)
        foot_b = jp.any(b[:, None] == self._feet_geom_id[None, :], axis=-1)
        allowed = ((a == self._floor_geom_id) & foot_b) | ((b == self._floor_geom_id) & foot_a)
        active = (data.contact.dist <= 0) & (a >= 0) & (b >= 0) & ~allowed
        hand = self._hand_geom_mask[jp.maximum(a, 0)] | self._hand_geom_mask[jp.maximum(b, 0)]
        arm = self._arm_geom_mask[jp.maximum(a, 0)] | self._arm_geom_mask[jp.maximum(b, 0)]
        furniture = self._furniture_geom_mask[jp.maximum(a, 0)] | self._furniture_geom_mask[jp.maximum(b, 0)]
        self_contact = self._robot_geom_mask[jp.maximum(a, 0)] & self._robot_geom_mask[jp.maximum(b, 0)]
        floor_contact = (a == self._floor_geom_id) | (b == self._floor_geom_id)
        return jp.asarray([jp.any(active), jp.any(active & hand), jp.any(active & arm),
                           jp.any(active & furniture), jp.any(active & self_contact),
                           jp.any(active & floor_contact)], dtype=jp.float32)

    def _physics_step(self, data, targets, info):
        def advance(carry, index):
            data, rng, contacts, first, first_part = carry
            rng, key = jax.random.split(rng)
            torque = info["kp_scale"] * self._kps * (targets - data.qpos[7:]) - info["kd_scale"] * self._kds * data.qvel[6:]
            torque += info["rfi_lim_scale"] * jax.random.uniform(key, torque.shape, minval=-1, maxval=1)
            data = mjx.step(self.mjx_model, data.replace(ctrl=jp.clip(torque, -self.torque_limit, self.torque_limit)))
            current = self._contacts(data)
            part = jp.where(current[1] > 0, 2, jp.where(current[2] > 0, 3, 1))
            first_part = jp.where((first < 0) & (current[0] > 0), part, first_part)
            # mjx.step reports contacts from the forward pass before its Euler
            # integration, so this contact belongs to the substep's start.
            first = jp.where((first < 0) & (current[0] > 0), index, first)
            return (data, rng, jp.maximum(contacts, current), first, first_part), None
        initial = self._contacts(data)
        initial_part = jp.where(initial[1] > 0, 2, jp.where(initial[2] > 0, 3, 1))
        carry = (data, info["rng"], initial, jp.where(initial[0] > 0, 0, -1), jp.where(initial[0] > 0, initial_part, 0))
        data, rng, contacts, first, first_part = jax.lax.scan(advance, carry, jp.arange(self.n_substeps))[0]
        # Refresh the final integrated pose before goal/clearance checks. Without
        # this pass, a last-substep contact could be incorrectly called success.
        data = mjx.forward(self.mjx_model, data)
        final_contacts = self._contacts(data)
        new_contact = (first < 0) & (final_contacts[0] > 0)
        final_part = jp.where(final_contacts[1] > 0, 2, jp.where(final_contacts[2] > 0, 3, 1))
        first_part = jp.where(new_contact, final_part, first_part)
        first = jp.where(new_contact, self.n_substeps, first)
        return data, rng, jp.maximum(contacts, final_contacts), first, first_part

    def _positions(self, data):
        return {"head": data.site_xpos[self._head_site_id][None, :],
                "pelv": data.site_xpos[self._pelvis_imu_site_id][None, :],
                "tors": data.site_xpos[self._torso_imu_site_id][None, :],
                "feet": data.site_xpos[self._feet_site_id], "hands": data.site_xpos[self._hands_site_id],
                "knees": data.site_xpos[self._knees_site_id], "shlds": data.site_xpos[self._shlds_site_id]}

    def _update_features(self, data, info, *, reset=False):
        positions = self._positions(data)
        all_positions = jp.concatenate(list(positions.values()))
        pelvis_rotation = data.site_xmat[self._pelvis_imu_site_id]
        nav_rotation = base2navi_transform(pelvis_rotation)
        pose = jp.eye(4).at[:3, :3].set(nav_rotation).at[:2, 3].set(data.qpos[:2])
        pose = pose.at[2, 3].set(self._config.reward_config.base_height_target)
        info["navi2world_pose"] = pose
        info["navi2world_rot"] = nav_rotation
        info["navi_torso_rpy"] = jp.asarray(jaxlie.SO3.from_matrix(nav_rotation.T @ data.site_xmat[self._torso_imu_site_id]).as_rpy_radians())
        info["navi_pelvis_rpy"] = jp.asarray(jaxlie.SO3.from_matrix(nav_rotation.T @ pelvis_rotation).as_rpy_radians())
        info["global_lin_vel"] = self.get_global_linvel(data, "pelvis")
        info["global_ang_vel"] = self.get_global_angvel(data, "pelvis")
        info["navi_torso_ang_vel"] = nav_rotation.T @ self.get_global_angvel(data, "torso")
        for name, pos in positions.items():
            shape = pos[0] if len(pos) == 1 else pos
            if name in ("head", "feet", "hands"):
                previous = shape if reset else info[f"{name}_pos"]
                info[f"{name}_vel"] = (shape - previous) / self.dt
            info[f"{name}_pos"] = shape
        probe_positions = data.site_xpos[self._probe_site_ids]
        probe_velocity = jp.zeros_like(probe_positions) if reset else (probe_positions - info["probe_positions"]) / self.dt
        info["probe_positions"] = probe_positions
        info["probe_velocities"] = probe_velocity
        gf, known = sample_grid(self.gf, all_positions, self.pf_origin, self.dx)
        bf, _ = sample_grid(self.bf, all_positions, self.pf_origin, self.dx)
        df, _ = sample_grid(self.sdf, all_positions, self.pf_origin, self.dx)
        gf /= jp.maximum(jp.linalg.norm(gf, axis=-1, keepdims=True), 1e-6)
        bf /= jp.maximum(jp.linalg.norm(bf, axis=-1, keepdims=True), 1e-6)
        field = jp.concatenate([gf, bf, df], axis=-1)
        field = jp.where(known[:, None], field, jp.asarray([0, 0, 0, 0, 0, 0, -1]))
        command = self.compute_cmd_from_rtf(gf[1], gf[jp.asarray([0, 3, 4, 5, 6])], bf[jp.asarray([0, 3, 4, 5, 6])])
        command = jp.where(jp.linalg.norm(data.qpos[:2] - self._goal) < self._config.goal_radius, jp.zeros(4), command)
        info["command"] = command
        true_probes = probe_features(self.sdf, self.bf, probe_positions, probe_velocity, self._probe_radii, self.pf_origin, self.dx,
                                     prediction_enabled=self._config.prediction_enabled)
        info["true_probe_features"] = true_probes
        info["rng"], noise_key, unknown_key = jax.random.split(info["rng"], 3)
        corrupt = self._config.perception_mode == "corrupted"
        std = self._config.map_position_noise_std if corrupt else 0.0
        offset = std * jax.random.normal(noise_key, (3,))
        unknown = jax.random.bernoulli(unknown_key, self._config.unknown_probability if corrupt else 0.0, (len(all_positions) + len(probe_positions),))
        observed_gf, observed_known = sample_grid(self.gf, all_positions + offset, self.pf_origin, self.dx)
        observed_bf, _ = sample_grid(self.bf, all_positions + offset, self.pf_origin, self.dx)
        observed_df, _ = sample_grid(self.sdf, all_positions + offset, self.pf_origin, self.dx)
        observed_gf /= jp.maximum(jp.linalg.norm(observed_gf, axis=-1, keepdims=True), 1e-6)
        observed_bf /= jp.maximum(jp.linalg.norm(observed_bf, axis=-1, keepdims=True), 1e-6)
        observed_known &= ~unknown[:len(all_positions)]
        observed = jp.concatenate([observed_gf, observed_bf, observed_df], axis=-1)
        observed = jp.where(observed_known[:, None], observed, jp.asarray([0, 0, 0, 0, 0, 0, -1]))
        observed_command = self.compute_cmd_from_rtf(observed[1, :3], observed[jp.asarray([0, 3, 4, 5, 6]), :3], observed[jp.asarray([0, 3, 4, 5, 6]), 3:6])
        probes = probe_features(self.sdf, self.bf, probe_positions, probe_velocity, self._probe_radii,
            self.pf_origin, self.dx, position_offset=offset, unknown=unknown[len(all_positions):], uncertainty=std,
            prediction_enabled=self._config.prediction_enabled)
        packet = jp.concatenate([observed.reshape(-1), probes.reshape(-1), observed_command])
        if reset:
            info["map_history"] = jp.tile(packet, (self._latency + 1, 1))
            info["map_capture_history"] = jp.zeros((self._latency + 1,), dtype=jp.int32)
        else:
            update = info["step"] % self._interval == 0
            packet = jp.where(update, packet, info["map_history"][0])
            capture = jp.where(update, info["step"], info["map_capture_history"][0])
            info["map_history"] = jp.concatenate([packet[None], info["map_history"][:-1]], axis=0)
            info["map_capture_history"] = jp.concatenate([capture[None], info["map_capture_history"][:-1]])
        received = info["map_history"][-1]
        stale = received[:77].reshape(11, 7)
        info["actor_probe_features"] = received[77:-4].reshape(len(control.PROBE_SPECS), 9).at[:, 6].set((info["step"] - info["map_capture_history"][-1]) * self.dt)
        info["actor_probe_features"] = apply_uncertainty_margin(info["actor_probe_features"],
            enabled=self._config.uncertainty_margin_enabled, fixed_margin=self._config.fixed_uncertainty_margin,
            std_multiplier=self._config.uncertainty_std_multiplier, relative_motion_bound=self._config.relative_motion_bound,
            resolution_error_bound=self._resolution_error_bound)
        info["command_delay"] = received[-4:]
        offset_index = 0
        for name, count in control.PF_GROUPS:
            for suffix, values in (("", field), ("_delay", stale)):
                rows = values[offset_index:offset_index + count]
                info[f"{name}gf{suffix}"] = rows[:, :3]
                info[f"{name}bf{suffix}"] = rows[:, 3:6]
                info[f"{name}df{suffix}"] = rows[:, 6:7]
            offset_index += count
        return info

    def _get_obs(self, data, info, feet_contact):
        obs = G1CatEnv._get_obs(self, data, info, feet_contact)
        actor_probes = info["actor_probe_features"]
        actor_probes = actor_probes.at[:, 3:6].set(actor_probes[:, 3:6] @ info["navi2world_rot"])
        if not self._config.probe_features_enabled:
            actor_probes = jp.zeros_like(actor_probes)
        obs["state"] = jp.concatenate([obs["state"], actor_probes.reshape(-1)])
        obs["privileged_state"] = jp.concatenate([obs["privileged_state"], info["true_probe_features"].reshape(-1)])
        return obs

    def reset(self, rng):
        data = mjx_env.init(self.mjx_model, qpos=self._init_q, qvel=jp.zeros(self.mjx_model.nv), ctrl=jp.zeros(29))
        rng, phase_key = jax.random.split(rng)
        phase = jp.where(jax.random.bernoulli(phase_key), self._init_phase_l, self._init_phase_r)
        info = {"rng": rng, "step": jp.asarray(0, dtype=jp.int32),
                "last_act": jp.zeros(self.action_size), "last_last_act": jp.zeros(self.action_size),
                "motor_targets": self._default_qpos.copy(), "last_joint_vel": jp.zeros(29),
                "foot_height": jp.asarray(0.07), "phase": phase, "phase_dt": jp.asarray(2 * jp.pi * self.dt * 1.4),
                "gait_mask": jp.zeros(2), "kp_scale": jp.asarray(1.0), "kd_scale": jp.asarray(1.0),
                "rfi_lim_scale": jp.zeros(29), "forbidden_contact": jp.asarray(False),
                "hand_contact": jp.asarray(False), "arm_contact": jp.asarray(False),
                "furniture_contact": jp.asarray(False), "self_collision": jp.asarray(False),
                "nonfoot_floor_contact": jp.asarray(False),
                "first_contact_part": jp.asarray(0, dtype=jp.int32),
                "first_contact_time": jp.asarray(-1.0), "fell": jp.asarray(False),
                "success": jp.asarray(False), "numerical_failure": jp.asarray(False)}
        info = self._update_features(data, info, reset=True)
        coordinate, _ = control.route_coordinate(data.qpos[:2], self._route, xp=jp)
        info["route_coordinate"] = coordinate
        info["max_route_coordinate"] = coordinate
        info["min_hand_clearance"] = jp.min(info["true_probe_features"][self._hand_probe_ids, 0])
        info["goal_reached"] = jp.asarray(False)
        info["fall"] = info["fell"]
        info["route_progress"] = coordinate
        info["minimum_clearance"] = jp.min(info["true_probe_features"][:, 0])
        feet = jp.asarray([geoms_colliding(data, int(i), self._floor_geom_id) for i in self._feet_geom_id])
        metrics = {k: jp.asarray(0.0) for k in self._metric_names()}
        obs = self._get_obs(data, info, feet)
        return mjx_env.State(data=data, obs=obs, reward=jp.asarray(0.0), done=jp.asarray(0.0), metrics=metrics, info=info)

    @staticmethod
    def _metric_names():
        return ("reward/progress", "reward/clearance", "reward/hand_protection", "reward/control", "reward/locomotion", "reward/terminal", "success", "successful_completion_time", "completion_time", "forbidden_contact", "hand_contact", "arm_contact", "fall", "numerical_failure", "timeout", "route_progress_m", "route_fraction", "cross_track_m", "min_hand_clearance_m", "map_age_seconds", "unknown_fraction")

    def step(self, state, action):
        info = dict(state.info)
        numerical = ~jp.all(jp.isfinite(action))
        action = jp.clip(jp.nan_to_num(action), -1, 1)
        action = control.posture_actions(action, self.action_joint_names, self._config.upper_body_mode,
            probe_features=info["actor_probe_features"], upper_scale=self._config.upper_action_scale, xp=jp)
        targets = control.motor_targets(action, info["motor_targets"], self._default_qpos,
            self._soft_lowers, self._soft_uppers, self.action_joint_ids,
            leg_scale=self._config.action_scale, upper_scale=self._config.upper_action_scale,
            upper_rate=self._config.upper_target_rate, dt=self.dt, xp=jp)
        data, rng, contacts, first_substep, first_part = self._physics_step(state.data, targets, info)
        numerical |= ~jp.all(jp.isfinite(data.qpos)) | ~jp.all(jp.isfinite(data.qvel))
        info["rng"] = rng
        info["step"] += 1
        info["motor_targets"] = targets
        info["phase"] = (info["phase"] + info["phase_dt"] + jp.pi) % (2 * jp.pi) - jp.pi
        cycle = jp.cos(info["phase"])
        info["gait_mask"] = jp.where(cycle > self._gait_bound, 1.0, jp.where(cycle < -self._gait_bound, -1.0, 0.0))
        info = self._update_features(data, info)
        coordinate, cross_track = control.route_coordinate(data.qpos[:2], self._route, xp=jp)
        progress = coordinate - info["route_coordinate"]
        info["route_coordinate"] = coordinate
        info["max_route_coordinate"] = jp.maximum(info["max_route_coordinate"], coordinate)
        info["forbidden_contact"] |= contacts[0] > 0
        info["hand_contact"] |= contacts[1] > 0
        info["arm_contact"] |= contacts[2] > 0
        info["furniture_contact"] |= contacts[3] > 0
        info["self_collision"] |= contacts[4] > 0
        info["nonfoot_floor_contact"] |= contacts[5] > 0
        info["first_contact_part"] = jp.where((info["first_contact_time"] < 0) & (contacts[0] > 0), first_part, info["first_contact_part"])
        contact_time = (info["step"] - 1) * self.dt + jp.maximum(first_substep, 0) * self.sim_dt
        info["first_contact_time"] = jp.where((info["first_contact_time"] < 0) & (contacts[0] > 0), contact_time, info["first_contact_time"])
        info["fell"] |= (self.get_gravity(data, "pelvis")[2] < 0) | (info["head_pos"][2] < 0.7)
        info["numerical_failure"] |= numerical
        elapsed = info["step"] * self.dt
        timeout = elapsed >= float(self.scene["time_budget"])
        reached = (jp.linalg.norm(data.qpos[:2] - self._goal) <= self._config.goal_radius) & (coordinate >= self._route_length - self._config.goal_radius)
        clean = ~info["forbidden_contact"] & ~info["fell"] & ~info["numerical_failure"]
        info["success"] = reached & clean & (elapsed <= float(self.scene["time_budget"]))
        info["goal_reached"] = reached
        info["fall"] = info["fell"]
        info["route_progress"] = coordinate
        done = info["success"] | info["forbidden_contact"] | info["fell"] | numerical | timeout
        clearances = info["true_probe_features"][:, :3]
        min_clearance = jp.min(clearances[self._hand_probe_ids, 0])
        info["min_hand_clearance"] = jp.minimum(info["min_hand_clearance"], min_clearance)
        info["minimum_clearance"] = jp.minimum(info["minimum_clearance"], jp.min(clearances[:, 0]))
        progress_reward = self._config.progress_reward * progress
        clearance_reward = -self._config.clearance_cost * jp.mean(jp.maximum(self._config.clearance_margin - clearances, 0)**2) * self.dt
        hand_reward = -self._config.hand_clearance_cost * hand_protection_cost(
            info["true_probe_features"][self._hand_probe_ids], info["probe_velocities"][self._hand_probe_ids],
            clearance_margin=self._config.hand_clearance_margin, urgency_gain=self._config.hand_urgency_gain) * self.dt
        hand_reward = jp.where(self._config.hand_protection_enabled, hand_reward, 0.0)
        control_reward = -(0.001 * jp.mean((action - info["last_act"])**2) + 1e-5 * jp.mean(data.ctrl**2) + self._config.time_cost) * self.dt
        scales = self._config.reward_config.scales
        locomotion_reward = self.dt * (
            scales.foot_slip * self._cost_foot_slip(data, info["gait_mask"])
            + scales.foot_clearance * self._cost_foot_clearance(data, info["foot_height"], info["gait_mask"], info["command"][0])
            + scales.foot_balance * self._cost_foot_balance(data, info["navi2world_pose"], info["command"][0]))
        terminal_reward = self._config.success_reward * info["success"] - self._config.contact_cost * (info["forbidden_contact"] | info["fell"] | numerical)
        reward = jp.nan_to_num(progress_reward + clearance_reward + hand_reward + control_reward + locomotion_reward + terminal_reward, nan=-self._config.contact_cost, posinf=-self._config.contact_cost, neginf=-self._config.contact_cost)
        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        info["last_joint_vel"] = data.qvel[6:]
        feet = jp.asarray([geoms_colliding(data, int(i), self._floor_geom_id) for i in self._feet_geom_id])
        obs = self._get_obs(data, info, feet)
        metrics = dict(zip(self._metric_names(), (progress_reward, clearance_reward, hand_reward, control_reward, locomotion_reward, terminal_reward,
            info["success"], elapsed * info["success"], jp.where(done, jp.where(info["success"], elapsed, float(self.scene["time_budget"])), 0.0),
            info["forbidden_contact"], info["hand_contact"], info["arm_contact"], info["fell"], numerical, timeout,
            coordinate, coordinate / max(self._route_length, 1e-6), cross_track, info["min_hand_clearance"],
            jp.max(info["actor_probe_features"][:, 6]), jp.mean(info["actor_probe_features"][:, 7]))))
        metrics = {k: jp.asarray(v, dtype=jp.float32) for k, v in metrics.items()}
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(jp.float32), metrics=metrics, info=info)
