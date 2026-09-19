"""mjlab Simulation with CAT's MJCF options and derived-data timing.

The simulation itself is mjlab/MuJoCo Warp. Task tensors are zero-copy Torch
views. A small kinematics-only shadow avoids changing the native pre-integration
observations while checking collision at the final integrated pose.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from cat_mjlab.model import assemble_training_xml

FK_FIELDS = ("xpos", "xquat", "xanchor", "xaxis", "xmat", "xipos", "ximat",
             "geom_xpos", "geom_xmat", "site_xpos", "site_xmat")
OBS_FIELDS = FK_FIELDS + ("subtree_com", "sensordata", "actuator_force", "cvel",
                          "qacc", "qacc_warmstart")


class _PreserveModelOptions:
    def apply(self, model):
        if abs(model.opt.timestep - .002) > 1e-12:
            raise ValueError("CAT physics must retain its 2 ms timestep")


class CATSimulation:
    def __init__(self, num_envs, *, device="cuda:0", nconmax=64, njmax=256, model=None):
        import warp as wp
        import mujoco_warp as mjwarp
        from mjlab.sim import Simulation, SimulationCfg
        self.wp, self.mjwarp = wp, mjwarp
        self.num_envs, self.device = int(num_envs), device
        self.model = model if model is not None else mujoco.MjModel.from_xml_string(assemble_training_xml())
        if (self.model.nq, self.model.nv, self.model.nu) != (36, 35, 29):
            raise ValueError("Unexpected CAT robot model dimensions")
        self.model.opt.timestep = .002
        original_options = self.option_contract()
        cfg = SimulationCfg(nconmax=nconmax, njmax=njmax, mujoco=_PreserveModelOptions())
        self.backend = Simulation(num_envs=self.num_envs, cfg=cfg, model=self.model, device=device)
        # Tensor views below use Torch's current stream. Launch Warp on that
        # same stream; relying on implicit synchronization races PD targets.
        wp.synchronize_device(device)
        if self.option_contract() != original_options:
            raise AssertionError("mjlab initialization changed the CAT solver options")
        raw = self.backend.wp_data
        self.data = SimpleNamespace(**{name: wp.to_torch(getattr(raw, name))
            for name in ("qpos", "qvel", "ctrl", "time", "qfrc_applied", "xfrc_applied", *OBS_FIELDS)})
        self.data.xmat = self.data.xmat.reshape(self.num_envs, self.model.nbody, 3, 3)
        self.data.site_xmat = self.data.site_xmat.reshape(self.num_envs, self.model.nsite, 3, 3)
        # mjwarp.kinematics writes exactly these arrays. All immutable inputs
        # and qpos are shared; no second physics/contact buffer is allocated.
        with wp.ScopedDevice(device):
            self._fk_data = replace(raw, **{name: wp.zeros_like(getattr(raw, name)) for name in FK_FIELDS})
            self._fk_graph = None
            if self.backend.use_cuda_graph:
                with wp.ScopedCapture() as capture:
                    mjwarp.kinematics(self.backend.wp_model, self._fk_data)
                self._fk_graph = capture.graph
        self._final_data = SimpleNamespace(xpos=wp.to_torch(self._fk_data.xpos),
            xmat=wp.to_torch(self._fk_data.xmat).reshape(self.num_envs, self.model.nbody, 3, 3))
        self._contact_cache = {}
        self._pair_cache = {}
        self._contact_index = torch.arange(raw.naconmax, device=device)

    def option_contract(self):
        opt = self.model.opt
        return {name: float(getattr(opt, name)) if name in ("timestep", "tolerance", "ls_tolerance", "impratio")
                else int(getattr(opt, name)) for name in
                ("timestep", "integrator", "solver", "iterations", "ls_iterations", "tolerance",
                 "ls_tolerance", "cone", "jacobian", "impratio", "disableflags", "enableflags")}

    def step(self):
        with self._scope():
            self.backend.step()
        self._contact_cache.clear()

    def reset_data(self, env_ids=None):
        with self._scope():
            self.backend.reset(env_ids)

    def _scope(self):
        if str(self.device).startswith("cuda"):
            return self.wp.ScopedStream(self.wp.stream_from_torch(torch.cuda.current_stream(self.device)))
        return self.wp.ScopedDevice(self.device)

    def forward(self, env_ids=None):
        """Refresh reset rows without changing ongoing task state or warmstart.

        MuJoCo recomputes other solver scratch arrays on the next step. Observable
        fields and persistent solver warmstart are preserved for continuing rows.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            with self._scope():
                self.backend.forward()
            self._contact_cache.clear()
            return
        keep = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        keep[env_ids] = False
        saved = {name: getattr(self.data, name)[keep].clone() for name in OBS_FIELDS}
        contacts = {key: value.clone() for key, value in self._contact_cache.items()}
        with self._scope():
            self.backend.forward()
        for name, values in saved.items():
            getattr(self.data, name)[keep] = values
        self._contact_cache.clear()
        for key, old in contacts.items():
            current = self._compute_contacts(self._pair_cache[key])
            current[keep] = old[keep]
            self._contact_cache[key] = current

    def final_collision_data(self):
        with self._scope():
            if self._fk_graph is None:
                self.mjwarp.kinematics(self.backend.wp_model, self._fk_data)
            else:
                self.wp.capture_launch(self._fk_graph)
        return self._final_data

    def _compute_contacts(self, pairs):
        raw = self.backend.wp_data
        contact = raw.contact
        geom = self.wp.to_torch(contact.geom).long()
        world = self.wp.to_torch(contact.worldid).long().clamp(0, self.num_envs - 1)
        valid = (self._contact_index < self.wp.to_torch(raw.nacon)[0]) & (self.wp.to_torch(contact.dist) < 0)
        result = torch.zeros((self.num_envs, len(pairs)), dtype=torch.bool, device=self.device)
        for index, (first, second) in enumerate(pairs):
            hit = valid & (((geom[:, 0] == first) & (geom[:, 1] == second)) |
                           ((geom[:, 0] == second) & (geom[:, 1] == first)))
            result[:, index].scatter_reduce_(0, world, hit, reduce="amax", include_self=True)
        return result

    def contact_flags(self, pairs):
        key = id(pairs)
        if key not in self._pair_cache:
            self._pair_cache[key] = np.asarray(pairs.detach().cpu() if isinstance(pairs, torch.Tensor) else pairs).tolist()
        if key not in self._contact_cache:
            self._contact_cache[key] = self._compute_contacts(self._pair_cache[key])
        return self._contact_cache[key]

    def capacity_report(self):
        raw = self.backend.wp_data
        contacts = int(self.wp.to_torch(raw.nacon)[0].item())
        constraints = int(self.wp.to_torch(raw.nefc).max().item())
        if contacts > raw.naconmax or constraints > raw.njmax:
            raise RuntimeError("MuJoCo Warp contact/constraint capacity exceeded")
        return dict(contacts=contacts, contact_capacity=raw.naconmax,
                    max_constraints=constraints, constraint_capacity=raw.njmax)

    def state_dict(self):
        # New backend resume keeps its persistent dynamics, never imports a JAX
        # contact solver state into Warp. Derived fields are recomputed on load.
        return {name: getattr(self.data, name).clone() for name in
                dict.fromkeys(("qpos", "qvel", "ctrl", "time", "qfrc_applied", "xfrc_applied", *OBS_FIELDS))}

    def load_state_dict(self, state):
        for name, value in state.items():
            getattr(self.data, name).copy_(value.to(self.device))
        self.forward()
        # Observations in a saved CAT state refer to the last pre-integration
        # pose; preserve that timing rather than leaving forward's newer pose.
        for name in OBS_FIELDS:
            getattr(self.data, name).copy_(state[name].to(self.device))
