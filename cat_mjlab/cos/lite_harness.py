"""Lightweight single-robot CPU harness for SONIC feasibility tracking (docs/cos_obstacles.md, G1 at scale).

The full CAT task (feature/sonic-baseline sonic_gates.build) loads a 3,663-scene bank (~22 GB RAM, ~110 s) just to
provide a flat floor. This class gives cat_mjlab/sonic_tracker.SonicTracker the same interface on the bare training
model: 50 Hz control, 10 physics substeps of 2 ms (Euler, Newton, 3 iterations, pyramidal cone - the training
model's own options), PD torque  kp (target - q) - kd qdot + rfi * U(-1, 1)  clipped to the actuator torque limits,
rfi drawn per episode as U(0.5, 1.5) * 0.1 * limit (the harness's dm_rand defaults), no latency. It is checked against
the full harness on the windows tracked both ways (docs/cos_obstacles.md).
"""
from __future__ import annotations

import numpy as np
import torch


class LiteTask:
    def __init__(self, seed=0, rfi=True):
        import mujoco
        from ..model import assemble_training_xml
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(assemble_training_xml())
        self.data = mujoco.MjData(self.model)
        self.dt, self.n_substeps = 0.02, 10
        assert abs(self.model.opt.timestep * self.n_substeps - self.dt) < 1e-9
        self.device, self.num_envs, self.tracker = torch.device("cpu"), 1, None
        self.all_ids = torch.arange(1)
        self.torque_limit = torch.tensor(self.model.actuator_ctrlrange[:, 1], dtype=torch.float32)
        self.kps = torch.zeros(29); self.kds = torch.zeros(29)
        self.upper_gravity, self.upper_average = None, False
        self.sensor_adr = {self.model.sensor(i).name: (int(self.model.sensor_adr[i]), int(self.model.sensor_dim[i]))
                           for i in range(self.model.nsensor)}
        self.rng = np.random.default_rng(seed)
        self.use_rfi = rfi
        self.info = {"step": torch.zeros(1, dtype=torch.long), "rfi": torch.zeros(1, 29)}
        self._sync()

    # ---- the attributes SonicTracker reads ----
    class _D:
        pass

    def _sync(self):
        d = self._D()
        d.qpos = torch.tensor(self.data.qpos[None], dtype=torch.float32)
        d.qvel = torch.tensor(self.data.qvel[None], dtype=torch.float32)
        self._data_view = d

    @property
    def data_t(self):
        return self._data_view

    def _sensor(self, name, ids):
        a, n = self.sensor_adr[name]
        return torch.tensor(self.data.sensordata[None, a:a + n], dtype=torch.float32)

    def reset(self, initial_pose, initial_velocity):
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.asarray(initial_pose, np.float64).reshape(-1)
        self.data.qvel[:] = np.asarray(initial_velocity, np.float64).reshape(-1)
        self.mj.mj_forward(self.model, self.data)
        self.info["step"].zero_()
        lim = self.torque_limit.numpy()
        self.info["rfi"] = torch.tensor(self.rng.uniform(0.5, 1.5, 29) * 0.1 * lim if self.use_rfi else np.zeros(29),
                                        dtype=torch.float32)[None]
        self._sync()

    def step(self, action, position_targets):
        target = position_targets[0].numpy().astype(np.float64)
        kp, kd = self.kps.numpy().astype(np.float64), self.kds.numpy().astype(np.float64)
        lim = self.torque_limit.numpy().astype(np.float64)
        rfi = self.info["rfi"][0].numpy().astype(np.float64)
        for _ in range(self.n_substeps):
            q, qd = self.data.qpos[7:], self.data.qvel[6:]
            tau = kp * (target - q) - kd * qd + rfi * self.rng.uniform(-1., 1., 29)
            self.data.ctrl[:] = np.clip(tau, -lim, lim)
            self.mj.mj_step(self.model, self.data)
        self.info["step"] += 1
        self._sync()
        return {"done": torch.zeros(1, dtype=torch.bool)}


class _TaskAdapter:
    """Exposes LiteTask.data_t as .data (torch views) for SonicTracker while LiteTask keeps the MjData."""

    def __init__(self, lite):
        object.__setattr__(self, "_lite", lite)

    def __getattr__(self, k):
        if k == "data":
            return self._lite.data_t
        return getattr(self._lite, k)

    def __setattr__(self, k, v):
        setattr(self._lite, k, v)


def make_tracker(lite, encoder, decoder, reference, provider="cpu"):
    """provider "cpu" = ONNX Runtime; "torch" = the torch port of the release (cat_mjlab/sonic_torch.py, needs only
    the onnx package to read the weights) on the CPU."""
    from ..sonic_tracker import SonicTracker
    return SonicTracker(_TaskAdapter(lite), encoder, decoder, provider=provider, reference=reference)
