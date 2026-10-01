"""Single-world CPU simulation with the CATSimulation interface, on plain MuJoCo (C).

For interactive use (teleoperation demo): the training task runs unchanged on top of it, so the
policy sees exactly the observations it was trained with. MuJoCo Warp follows MuJoCo's algorithms
and mj_step timing (derived quantities pre-integration, state post-integration), so dynamics match
up to floating point. The task works in float32 like Warp; MjData is float64, so the task-facing
tensors are float32 mirrors with a leading world dimension of 1: inputs (state, controls, applied
forces) are pushed before every step/forward and every field is pulled back after it.
"""
from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from cat_mjlab.model import assemble_training_xml

INPUT_FIELDS = ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat")
VIEW_FIELDS = ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat",
               "xpos", "xquat", "xanchor", "xaxis", "xipos", "ximat", "geom_xpos", "geom_xmat", "site_xpos",
               "subtree_com", "sensordata", "actuator_force", "cvel", "qacc", "qacc_warmstart")


class CPUSimulation:
    def __init__(self, num_envs=1, *, device="cpu", nconmax=None, njmax=None, model=None):
        if int(num_envs) != 1:
            raise ValueError("CPUSimulation runs exactly one world")
        if str(device) != "cpu":
            raise ValueError("CPUSimulation runs on the CPU")
        self.num_envs, self.device = 1, "cpu"
        self.model = model if model is not None else mujoco.MjModel.from_xml_string(assemble_training_xml())
        if (self.model.nq, self.model.nv, self.model.nu) != (36, 35, 29):
            raise ValueError("Unexpected CAT robot model dimensions")
        self.model.opt.timestep = .002
        self.raw = mujoco.MjData(self.model)
        self._shadow = mujoco.MjData(self.model)          # kinematics at the final integrated pose
        f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
        self.data = SimpleNamespace(**{name: f32(getattr(self.raw, name)) for name in VIEW_FIELDS})
        self.data.time = torch.zeros(1, dtype=torch.float32)
        self.data.xmat = f32(self.raw.xmat).reshape(1, self.model.nbody, 3, 3)
        self.data.site_xmat = f32(self.raw.site_xmat).reshape(1, self.model.nsite, 3, 3)
        self._final = SimpleNamespace(xpos=f32(self._shadow.xpos), xmat=f32(self._shadow.xmat).reshape(1, self.model.nbody, 3, 3))
        self._contact_cache = {}

    def _push(self):
        for name in INPUT_FIELDS:
            getattr(self.raw, name)[:] = getattr(self.data, name)[0].double().numpy().reshape(getattr(self.raw, name).shape)

    def _pull(self):
        for name in VIEW_FIELDS:
            getattr(self.data, name)[0].copy_(torch.from_numpy(np.asarray(getattr(self.raw, name)).reshape(getattr(self.data, name)[0].shape)))
        self.data.xmat[0].copy_(torch.from_numpy(self.raw.xmat.reshape(self.model.nbody, 3, 3)))
        self.data.site_xmat[0].copy_(torch.from_numpy(self.raw.site_xmat.reshape(self.model.nsite, 3, 3)))
        self.data.time[0] = self.raw.time

    def step(self):
        self._push()
        mujoco.mj_step(self.model, self.raw)
        self._pull()
        self._contact_cache.clear()

    def reset_data(self, env_ids=None):
        mujoco.mj_resetData(self.model, self.raw)
        self._pull()

    def forward(self, env_ids=None):
        self._push()
        mujoco.mj_forward(self.model, self.raw)
        self._pull()
        self._contact_cache.clear()

    def final_collision_data(self):
        self._shadow.qpos[:] = self.data.qpos[0].double().numpy()
        self._shadow.mocap_pos[:] = self.raw.mocap_pos
        self._shadow.mocap_quat[:] = self.raw.mocap_quat
        mujoco.mj_kinematics(self.model, self._shadow)
        self._final.xpos[0].copy_(torch.from_numpy(self._shadow.xpos))
        self._final.xmat[0].copy_(torch.from_numpy(self._shadow.xmat.reshape(self.model.nbody, 3, 3)))
        return self._final

    def contact_flags(self, pairs):
        key = id(pairs)
        if key not in self._contact_cache:
            pairs_list = np.asarray(pairs.detach().cpu() if isinstance(pairs, torch.Tensor) else pairs).tolist()
            c = self.raw.contact
            touching = {(int(min(g1, g2)), int(max(g1, g2))) for g1, g2, dist in zip(c.geom1, c.geom2, c.dist) if dist < 0}
            self._contact_cache[key] = torch.tensor([[(min(a, b), max(a, b)) in touching for a, b in pairs_list]])
        return self._contact_cache[key]

    def capacity_report(self):
        return dict(contacts=int(self.raw.ncon), overflow_worlds=0)

    def option_contract(self):
        opt = self.model.opt
        return {name: float(getattr(opt, name)) if name in ("timestep", "tolerance", "ls_tolerance", "impratio")
                else int(getattr(opt, name)) for name in
                ("timestep", "integrator", "solver", "iterations", "ls_iterations", "tolerance",
                 "ls_tolerance", "cone", "jacobian", "impratio", "disableflags", "enableflags")}
