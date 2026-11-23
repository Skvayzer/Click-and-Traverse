"""Inference adapter for the published BRACE G1 student and a shared CAT plan."""
from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch

from .planner.fk import TorchFK

MODEL_SHA256 = "6d6e8bfff1f47c3f549a7ff0de24e475a112bebd32f0e7cd1dcf5d20036e29b5"
CONFIG_SHA256 = "8e0aefdcb2f79b3b3f5e9a0d9bc1e6cc86dacba8525beef93cae9cbc6aca2683"

CURRENT_FIELDS = ("dof_pos", "dof_vel", "anchor_rot", "root_local_ang_vel")


def quaternion_xyzw(rotation):
    """Convert a rotation matrix to an xyzw quaternion via its best-conditioned component."""
    xx, yy, zz = rotation.diagonal(dim1=-2, dim2=-1).unbind(-1)
    xy, xz, yx, yz, zx, zy = (rotation[..., i, j] for i, j in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)))
    squared = torch.stack((1 + xx + yy + zz, 1 + xx - yy - zz, 1 - xx + yy - zz, 1 - xx - yy + zz), -1)
    candidates = torch.stack((
        torch.stack((squared[..., 0], zy - yz, xz - zx, yx - xy), -1),
        torch.stack((zy - yz, squared[..., 1], xy + yx, xz + zx), -1),
        torch.stack((xz - zx, xy + yx, squared[..., 2], yz + zy), -1),
        torch.stack((yx - xy, zx + xz, zy + yz, squared[..., 3]), -1),
    ), -2)
    pivot = squared.argmax(-1)
    chosen = candidates.gather(-2, pivot[..., None, None].expand(*pivot.shape, 1, 4)).squeeze(-2)
    q = torch.nn.functional.normalize(chosen, dim=-1)
    q = torch.where(q[..., :1] < 0., -q, q)
    return q[..., [1, 2, 3, 0]]


def reference_model(path):
    """Load the author's G1 kinematics without the simulator-specific floor pairs."""
    import mujoco

    path = Path(path).resolve()
    root = ElementTree.parse(path).getroot()
    root.remove(root.find("contact"))
    compiler = root.find("compiler")
    compiler.set("meshdir", str((path.parent / compiler.get("meshdir")).resolve()))
    return mujoco.MjModel.from_xml_string(ElementTree.tostring(root, encoding="unicode"))


def plan_pose(tracker, ids, times):
    xy, yaw, z, upper = tracker.plan_at(ids, times)
    roll = tracker._interp(tracker.plan_roll, ids, times)
    pitch = tracker._interp(tracker.plan_pitch, ids, times)
    legs = tracker._interp(tracker.plan_legs, ids, times)
    cy, sy = (yaw / 2).cos(), (yaw / 2).sin()
    cp, sp = (pitch / 2).cos(), (pitch / 2).sin()
    cr, sr = (roll / 2).cos(), (roll / 2).sin()
    quat = torch.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1)
    return torch.cat((xy, z[:, None], quat, legs, upper), -1)


class BraceTracker:
    def __init__(self, task, model_path, config_path, robot_path, *, seed, provider="cuda"):
        import onnxruntime as ort
        import yaml

        for path, expected in ((model_path, MODEL_SHA256), (config_path, CONFIG_SHA256)):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"BRACE file differs from the published release: {path}")
        meta = yaml.safe_load(Path(config_path).read_text())
        robot, control = meta["robot"], meta["control"]
        if abs(task.dt - float(meta["timing"]["control_dt"])) > 1.e-9:
            raise ValueError("BRACE control period differs from the evaluation task")
        names = robot["joint_names"]
        source_order = [task.model.joint(i).name for i in range(1, 30)]
        if set(source_order) != set(names) or task.tracker is None:
            raise ValueError("BRACE requires the common 29-DOF G1 and a full plan reference")
        dev = task.device
        self.task, self.device, self.worlds = task, dev, task.num_envs
        self.obs_order = torch.tensor([source_order.index(joint) for joint in names], device=dev)
        self.action_order = self.obs_order.argsort()
        def control_values(key):
            return torch.tensor(control[key], dtype=torch.float32, device=dev)[self.action_order]
        if not torch.allclose(control_values("effort_limits"), task.torque_limit):
            raise ValueError("BRACE effort limits differ from the shared G1 torque limits")
        task.kps = control_values("stiffness")
        task.kds = control_values("damping")
        task.upper_gravity = None
        task.upper_average = False
        ref = reference_model(robot_path)
        self.fk = TorchFK(ref, dev)
        self.reference_order = torch.tensor(
            list(range(7)) + [source_order.index(ref.joint(i).name) + 7 for i in range(1, 30)], device=dev)
        self.body_order = torch.tensor([ref.body(name).id for name in robot["body_names"]], device=dev)
        self.anchor = robot["anchor_body_index"]
        self.anchor_live = task.model.body(robot["anchor_body_name"]).id
        self.seconds = torch.as_tensor([0., *meta["motion"]["future_dt_seconds"]], device=dev)
        self.history = {}
        self.action_history = torch.zeros(self.worlds, 25, 29, device=dev)
        self.last_target = torch.zeros_like(self.action_history[:, 0])
        generator = torch.Generator(device=dev).manual_seed(seed)
        self.noise = torch.randn(self.worlds, 29, generator=generator, device=dev)
        self.provider = provider
        opts = ort.SessionOptions()
        if provider == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("BRACE requires an ONNX Runtime CUDA provider for the full benchmark")
            stream = torch.cuda.current_stream(dev)
            providers = [("CUDAExecutionProvider", {"device_id": dev.index or 0, "user_compute_stream": str(stream.cuda_stream)})]
        else:
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        if provider == "cuda" and self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("BRACE CUDA provider failed to load")
        self.expected = {node.name for node in self.session.get_inputs()}

    def _reference(self):
        task, t = self.task, self.task.tracker
        ids = task.all_ids
        nframes = len(self.seconds)
        repeated = ids.repeat_interleave(nframes)
        times = (t.plan_t[:, None] + self.seconds[None, :]).reshape(-1)
        pose = plan_pose(t, repeated, times).reshape(self.worlds, nframes, 36)
        ready = t.has_plan & t.has_full
        pose = torch.where(ready[:, None, None], pose, task.data.qpos[:, None, :36])
        # The planner's joint interpolation is piecewise linear. Use the derivative
        # over one control tick in that reference rather than actual robot velocity.
        later = plan_pose(t, repeated, times + task.dt).reshape_as(pose)
        later = torch.where(ready[:, None, None], later, pose)
        qpos = pose[..., 7:][..., self.obs_order]
        qvel = (later[..., 7:][..., self.obs_order] - qpos) / task.dt
        pos, rot = self.fk(pose[..., self.reference_order])
        return qpos, qvel, pos[..., self.body_order, :], quaternion_xyzw(rot[..., self.body_order, :, :])

    def _inputs(self):
        task, n = self.task, self.worlds
        current = {
            "dof_pos": task.data.qpos[:, 7:][:, self.obs_order],
            "dof_vel": task.data.qvel[:, 6:][:, self.obs_order],
            "anchor_rot": task.data.xquat[:, self.anchor_live, :][:, [1, 2, 3, 0]],
            "root_local_ang_vel": task._sensor("gyro_pelvis", task.all_ids),
        }
        reset = task.info["step"] == 0
        self.action_history[reset] = 0.
        self.last_target[reset] = 0.
        for name in CURRENT_FIELDS:
            value = current[name]
            if name not in self.history:
                self.history[name] = value[:, None].expand(-1, 25, -1).clone()
            self.history[name][reset] = value[reset, None]
        q, v, pos, rot = self._reference()
        identity = rot.new_zeros(n, 4)
        identity[:, 3] = 1.
        mode = rot.new_zeros(n, 2)
        mode[:, 1] = 1.  # zero-wrench compensation, with no additional brace displacement
        feed = {f"current_{name}": value for name, value in current.items()}
        feed.update({f"historical_{name}": value for name, value in self.history.items()})
        feed.update(
            hand_force_dof_priv_delta=q.new_zeros(n, 29), hand_force_x_priv_bodies=pos[:, 0],
            hand_force_x_priv_rot=rot[:, 0], hand_force_xpriv_anchor_rot_delta=identity,
            historical_processed_actions=self.action_history,
            mimic_future_anchor_pos=pos[:, 1:, self.anchor], mimic_future_anchor_rot=rot[:, 1:, self.anchor],
            mimic_future_dof_pos=q[:, 1:], mimic_future_dof_vel=v[:, 1:], mimic_future_pos=pos[:, 1:],
            mimic_future_rot=rot[:, 1:], mimic_ref_state_rigid_body_pos=pos[:, 0],
            mimic_ref_state_rigid_body_rot=rot[:, 0], task_mode_force_cmd_eff=q.new_zeros(n, 2, 3),
            task_mode_mode_onehot=mode, task_mode_torque_cmd_eff=q.new_zeros(n, 2, 3), initial_noise=self.noise,
        )
        if set(feed) != self.expected:
            raise ValueError(f"BRACE input contract differs: {set(feed) ^ self.expected}")
        return feed, current

    @torch.no_grad()
    def __call__(self):
        feed, current = self._inputs()
        if self.provider == "cuda":
            binding = self.session.io_binding()
            keep = {name: value.contiguous() for name, value in feed.items()}
            device_id = self.device.index or 0
            for name, value in keep.items():
                binding.bind_input(name, "cuda", device_id, np.float32, tuple(value.shape), value.data_ptr())
            outputs = [torch.empty_like(self.last_target) for _ in range(2)]
            for name, value in zip(("actions", "joint_pos_targets"), outputs):
                binding.bind_output(name, "cuda", device_id, np.float32, tuple(value.shape), value.data_ptr())
            self.session.run_with_iobinding(binding)
            action, target = outputs
        else:
            arrays = {name: value.cpu().numpy() for name, value in feed.items()}
            values = self.session.run(["actions", "joint_pos_targets"], arrays)
            action, target = (torch.from_numpy(value).to(self.device) for value in values)
        if not bool(torch.isfinite(target).all() & torch.isfinite(action).all()):
            raise RuntimeError("BRACE produced non-finite outputs")
        self.action_history = torch.cat((self.last_target[:, None], self.action_history[:, :-1]), 1)
        self.last_target = target
        for name, value in current.items():
            self.history[name] = torch.cat((value[:, None], self.history[name][:, :-1]), 1)
        return action[:, self.action_order], target[:, self.action_order]
