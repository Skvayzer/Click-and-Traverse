"""Frozen NVIDIA GEAR-SONIC (default release) tracking a shared CAT plan in the common G1 harness.

Contract source: the official C++ deployment, gear_sonic_deploy/src/g1/g1_deploy_onnx_ref (policy_parameters.hpp,
g1_deploy_onnx_ref.cpp) at the release vendored in GRAIL aa31d82, with the Hugging Face release nvidia/GEAR-SONIC
revision 6733128a3d8a523b1418b06bca3cdf61c8b0987f ("Default SONIC": model_encoder.onnx, model_decoder.onnx,
observation_config.yaml; PyTorch training config sonic_release/config.yaml).

Encoder (g1 mode, 1762 inputs; only the g1 branch is used):
    [0:4]      encoder_mode_4 = (mode id 0 = g1, 0, 0, 0)
    [4:294]    reference joint positions, 10 future frames at 0.1 s (frame 0 = now), IsaacLab joint order, absolute rad
    [294:584]  reference joint velocities, same frames (all positions first, then all velocities: the release packing)
    [601:661]  reference pelvis orientation in the robot pelvis frame, quat_inv(q_robot) * q_ref, per frame the first
               two rotation-matrix columns flattened row-wise (motion_anchor_ori_b_mf)
    every other block is consumed only by the teleop/smpl encoders and is zero.
Decoder (994): 64 tokens, then 10-sample histories (oldest first) of pelvis gyro (3), q - default (29), qdot (29), last raw
action (29), gravity in the pelvis frame (3).
Action: PD target_mujoco[i] = default[i] + action[isaaclab(i)] * scale[i], scale = 0.25 * effort / stiffness,
Kp = armature * (2 pi 10 Hz)^2, Kd = 4 * armature * (2 pi 10 Hz); 50 Hz control.

The ONNX release has a fixed batch of one. We make the batch dimension symbolic in memory (input/output dims and the
leading 1 of the encoder's constant Reshape shapes -> 0, i.e. "copy the input's dimension"); `verify_batched` checks the
batched graphs against the released graphs on random inputs before any rollout.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from .brace_tracker import plan_pose

RELEASE = "nvidia/GEAR-SONIC@6733128a3d8a523b1418b06bca3cdf61c8b0987f (default variant)"
# variant: (encoder sha256, decoder sha256, encoder width, anchor slice, anchor mode)
#   default: motion_anchor_orientation_10frame_step5 = quat_inv(q_robot) * q_ref       at [601:661] of 1762
#   v1_1:    motion_anchor_orientation_heading_10frame_step5 = quat_inv(heading(q_robot)) * q_ref at [584:644] of 1751
VARIANTS = {
    "default": ("013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3",
                "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed", 1762, (601, 661), "full"),
    "v1_1": ("fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae",
             "34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509", 1751, (584, 644), "heading"),
}

# Policy (IsaacLab) joint order of the released checkpoint.
ISAACLAB_JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
# MuJoCo (URDF) order of the deployment tables below; must equal the harness joint order.
MUJOCO_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
_W = 10 * 2 * math.pi
_MOTORS = {  # armature, effort limit (policy_parameters.hpp)
    "5020": (0.003609725, 25.), "7520_14": (0.010177520, 88.), "7520_22": (0.025101925, 139.), "4010": (0.00425, 5.),
}
# per MuJoCo joint: motor type and gain multiplier (ankles and waist roll/pitch use 2 x 5020 gains)
_JOINT_MOTOR = (["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"] * 2
                + ["7520_14", "5020", "5020"] + ["5020"] * 5 + ["4010"] * 2 + ["5020"] * 5 + ["4010"] * 2)
_GAIN_MULT = [1, 1, 1, 1, 2, 2] * 2 + [1, 2, 2] + [1] * 14
DEFAULT_ANGLES = (-0.312, 0., 0., 0.669, -0.363, 0., -0.312, 0., 0., 0.669, -0.363, 0., 0., 0., 0.,
                  0.2, 0.2, 0., 0.6, 0., 0., 0., 0.2, -0.2, 0., 0.6, 0., 0., 0.)
KPS = tuple(m * _MOTORS[t][0] * _W * _W for t, m in zip(_JOINT_MOTOR, _GAIN_MULT))
KDS = tuple(m * 4. * _MOTORS[t][0] * _W for t, m in zip(_JOINT_MOTOR, _GAIN_MULT))
ACTION_SCALE = tuple(.25 * _MOTORS[t][1] / (_MOTORS[t][0] * _W * _W) for t in _JOINT_MOTOR)
# Effort limits of the SONIC training robot (g1_model_12_dex); ankles and waist roll/pitch are 2 x 25 Nm.
SONIC_EFFORT = tuple(m * _MOTORS[t][1] for t, m in zip(_JOINT_MOTOR, _GAIN_MULT))
FUTURE_FRAMES, FUTURE_DT, HISTORY = 10, 0.1, 10
ACTION_CLIP = 20.                       # env_config.action_clip_value of the training environment


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def batched_models(encoder_path, decoder_path, variant="default"):
    """The released encoder/decoder with a symbolic batch dimension (serialized ONNX bytes)."""
    import onnx
    from onnx import numpy_helper

    out = []
    for path, expected in zip((encoder_path, decoder_path), VARIANTS[variant][:2]):
        if _sha256(path) != expected:
            raise ValueError(f"SONIC file differs from the pinned release: {path}")
        model = onnx.load(str(path))
        for value in list(model.graph.input) + list(model.graph.output):
            value.type.tensor_type.shape.dim[0].dim_param = "N"
        del model.graph.value_info[:]
        constants = {node.output[0]: node for node in model.graph.node if node.op_type == "Constant"}
        for node in model.graph.node:
            if node.op_type == "Reshape" and node.input[1] in constants:
                tensor = constants[node.input[1]].attribute[0].t
                shape = numpy_helper.to_array(tensor).copy()
                if shape.ndim == 1 and len(shape) > 1 and shape[0] == 1:
                    shape[0] = 0
                    tensor.CopyFrom(numpy_helper.from_array(shape, tensor.name))
        out.append(model.SerializeToString())
    return out


def verify_batched(encoder_path, decoder_path, batched, n=6, seed=0):
    """Max |batched - released| over n random rows for the encoder (g1 mode) and the decoder."""
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    errors = []
    for path, blob in zip((encoder_path, decoder_path), batched):
        one = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        many = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
        x = rng.standard_normal((n, one.get_inputs()[0].shape[1])).astype(np.float32)
        if x.shape[1] in (1751, 1762):
            x[:, :4] = 0.
        ref = np.concatenate([one.run(None, {"obs_dict": x[i:i + 1]})[0] for i in range(n)])
        errors.append(float(np.abs(many.run(None, {"obs_dict": x})[0] - ref).max()))
    return errors


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)


def quat_conj(q):
    return q * q.new_tensor([1., -1., -1., -1.])


def quat_matrix(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
        torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
        torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)), -2)


def rotate_inverse(q, v):
    """v expressed in the frame of quaternion q (R(q)^T v)."""
    return (quat_matrix(q).transpose(-1, -2) @ v[..., None])[..., 0]


def plan_reference(task, times):
    """Whole-body plan poses [n, F, 36] (MuJoCo qpos layout) at plan times [n, F]; worlds without a full plan hold their
    measured pose. Also returns the ready mask [n]."""
    t = task.tracker
    n, frames = times.shape
    ids = task.all_ids.repeat_interleave(frames)
    pose = plan_pose(t, ids, times.reshape(-1)).reshape(n, frames, 36)
    ready = t.has_plan & t.has_full
    return torch.where(ready[:, None, None], pose, task.data.qpos[:, None, :36]), ready


class SonicTracker:
    """Callable like BraceTracker: returns (actions [n, 29] MuJoCo order, PD position targets [n, 29])."""

    def __init__(self, task, encoder_path, decoder_path, *, provider="cuda", reference=None, torque_limits="harness",
                 variant="default", ankle_gain=1.):
        import onnxruntime as ort

        names = [task.model.joint(i).name for i in range(1, 30)]
        if tuple(names) != MUJOCO_JOINTS:
            raise ValueError("SONIC deployment tables assume the standard 29-DOF G1 MuJoCo joint order")
        if abs(task.dt - 0.02) > 1e-9:
            raise ValueError("SONIC runs at 50 Hz control")
        if reference is None and task.tracker is None:
            raise ValueError("SONIC needs a plan reference (or an explicit reference function)")
        dev = task.device
        self.task, self.device, self.n = task, dev, task.num_envs
        self.reference = reference or self._plan_reference
        self.mj_to_il = torch.tensor([names.index(j) for j in ISAACLAB_JOINTS], device=dev)   # il i <- mujoco j
        self.il_of_mj = self.mj_to_il.argsort()                                                # mujoco j <- il i
        tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
        self.default = tensor(DEFAULT_ANGLES)
        self.scale = tensor(ACTION_SCALE)
        task.kps, task.kds = tensor(KPS), tensor(KDS)
        if ankle_gain != 1.:          # the v1.1 release's opt-in ankle-pitch tuning (MuJoCo indices 4 and 10)
            task.kps[[4, 10]] *= ankle_gain; task.kds[[4, 10]] *= ankle_gain
        self.variant = variant
        _, _, self.width, self.anchor_slice, self.anchor_mode = VARIANTS[variant]
        if torque_limits == "sonic":
            task.torque_limit = tensor(SONIC_EFFORT)
        elif torque_limits != "harness":
            raise ValueError("torque_limits must be 'harness' or 'sonic'")
        task.upper_gravity = None              # pure PD, as in SONIC training and deployment
        task.upper_average = False
        self.offsets = torch.arange(FUTURE_FRAMES, device=dev) * FUTURE_DT
        self.history = None
        self.last_action = torch.zeros(self.n, 29, device=dev)
        self.provider = provider
        self.latent = None
        if provider == "torch":
            # GPU-batched torch port with the pre-FSQ latent exposed (cat_mjlab/sonic_torch.py); default variant only.
            if variant != "default":
                raise ValueError("The torch backend ports the default release")
            from .sonic_torch import SonicTorch
            self.model = SonicTorch(encoder_path, decoder_path, device=dev)
            self.batched_error = [0., 0.]
            self.ref_q = self.ref_quat = None
            return
        encoder, decoder = batched_models(encoder_path, decoder_path, variant)
        self.batched_error = verify_batched(encoder_path, decoder_path, (encoder, decoder))
        if max(self.batched_error) > 1e-4:
            raise RuntimeError(f"Batched SONIC graph differs from the release: {self.batched_error}")
        self.provider = provider
        opts = ort.SessionOptions()
        if provider == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("SONIC needs an ONNX Runtime CUDA provider")
            stream = torch.cuda.current_stream(dev)
            providers = [("CUDAExecutionProvider", {"device_id": dev.index or 0,
                                                    "user_compute_stream": str(stream.cuda_stream)})]
        else:
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(encoder, sess_options=opts, providers=providers)
        self.decoder = ort.InferenceSession(decoder, sess_options=opts, providers=providers)
        if provider == "cuda" and self.encoder.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("SONIC CUDA provider failed to load")
        self.ref_q = self.ref_quat = None

    # ---- reference -------------------------------------------------------------------------------------------
    def _plan_reference(self):
        """Plan poses at plan_t + 0.1 k (k = 0..9) and their joint velocities (central difference over +-0.05 s)."""
        t = self.task.tracker
        times = t.plan_t[:, None] + self.offsets[None]
        pose, ready = plan_reference(self.task, times)
        h = 0.05
        ahead, _ = plan_reference(self.task, times + h)
        behind, _ = plan_reference(self.task, (times - h).clamp_min(0.))
        span = (times + h) - (times - h).clamp_min(0.)
        qd = (ahead[..., 7:] - behind[..., 7:]) / span[..., None]
        qd = torch.where(ready[:, None, None], qd, torch.zeros_like(qd))
        return pose, qd

    # ---- observation -----------------------------------------------------------------------------------------
    def _proprio(self):
        d, task = self.task.data, self.task
        quat = d.qpos[:, 3:7]
        terms = (task._sensor("gyro_pelvis", task.all_ids),
                 (d.qpos[:, 7:] - self.default)[:, self.mj_to_il],
                 d.qvel[:, 6:][:, self.mj_to_il],
                 self.last_action,
                 rotate_inverse(quat, quat.new_tensor([0., 0., -1.]).expand(self.n, 3)))
        reset = task.info["step"] == 0
        if self.history is None:
            self.history = [term[:, None].repeat(1, HISTORY, 1) for term in terms]
            reset = torch.ones_like(reset)
        for buffer, term in zip(self.history, terms):
            buffer[:] = torch.cat((buffer[:, 1:], term[:, None]), 1)
            buffer[reset] = term[reset, None]
        return torch.cat([buffer.flatten(1) for buffer in self.history], -1)

    def _encoder_input(self, pose, qd):
        n = self.n
        x = pose.new_zeros(n, self.width)
        q = pose[..., 7:][..., self.mj_to_il]                                          # [n, 10, 29] IsaacLab
        x[:, 4:294] = q.reshape(n, -1)
        x[:, 294:584] = qd[..., self.mj_to_il].reshape(n, -1)
        robot = self.task.data.qpos[:, 3:7]
        if self.anchor_mode == "heading":
            forward = quat_matrix(robot)[:, :, 0]
            yaw = torch.atan2(forward[:, 1], forward[:, 0])
            robot = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
        rel = quat_mul(quat_conj(robot[:, None]).expand(-1, FUTURE_FRAMES, -1), pose[..., 3:7])
        a, b = self.anchor_slice
        x[:, a:b] = quat_matrix(rel)[..., :2].reshape(n, -1)
        return x

    def _run(self, session, x, width):
        if self.provider == "cuda":
            x = x.contiguous()
            out = torch.empty(self.n, width, device=self.device)
            binding = session.io_binding()
            device_id = self.device.index or 0
            binding.bind_input("obs_dict", "cuda", device_id, np.float32, tuple(x.shape), x.data_ptr())
            binding.bind_output(session.get_outputs()[0].name, "cuda", device_id, np.float32, tuple(out.shape),
                                out.data_ptr())
            session.run_with_iobinding(binding)
            return out
        return torch.from_numpy(session.run(None, {"obs_dict": x.cpu().numpy()})[0]).to(self.device)

    @torch.no_grad()
    def __call__(self, residual=None, scale=.1):
        task = self.task
        reset = task.info["step"] == 0
        self.last_action[reset] = 0.
        pose, qd = self.reference()
        self.ref_q, self.ref_pose = pose[:, 0, 7:], pose[:, 0]
        proprio = self._proprio()
        x = self._encoder_input(pose, qd)
        if self.provider == "torch":
            a, b = self.anchor_slice
            action, self.latent, _ = self.model.act(x[:, 4:584], x[:, a:b], proprio, residual, scale)
            action = action.clamp(-ACTION_CLIP, ACTION_CLIP)
        else:
            if residual is not None:
                raise ValueError("Latent residuals need the torch backend")
            tokens = self._run(self.encoder, x, 64)
            action = self._run(self.decoder, torch.cat((tokens, proprio), -1), 29).clamp(-ACTION_CLIP, ACTION_CLIP)
        if not bool(torch.isfinite(action).all()):
            raise RuntimeError("SONIC produced non-finite actions")
        self.last_action = action
        mujoco_action = action[:, self.il_of_mj]
        return mujoco_action, self.default + mujoco_action * self.scale
