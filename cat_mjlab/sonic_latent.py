"""SONIC + latent adaptor (play2perfect/GRAIL frozen latent-adaptor design) in the common G1 harness.

Reference: GaitReference, a GPU-batched equivalent of SONIC's own locomotion generator. Strides of the released
planner_sonic.onnx SLOW_WALK motion (scripts/build_sonic_gait_library.py: phase-folded, phase 0 at maximum left hip
flexion) are replayed phase-indexed per world, blended across speeds, time-scaled below the slowest stride used, with
the pelvis heading integrating a commanded yaw rate; standing pose from the generator's IDLE output, 0.8 s ramps in/out.
SONIC sees only joints, joint velocities and pelvis orientation, so no root translation is needed.
Command: walk toward the field-guidance route point 0.6 m ahead (yaw rate 1.5 x heading error, |.| <= 0.6 rad/s) at
the operator's commanded speed. The adaptor may modulate both (bounded): speed x (1 + 0.5 a_speed) and +0.5 a_turn rad/s.

Adaptor: actions = 64 latent residual + a_speed + a_turn (tanh-Gaussian PPO, each in [-1, 1]). The residual is added to
SONIC's g1 encoder latent before FSQ: z' = z + latent_scale * a (frozen encoder, quantizer and decoder; torch port in
cat_mjlab/sonic_torch.py). The policy's mean head starts at zero, so the initial controller is plain SONIC on this
reference. Observation: our actor observation (13-point field, proprioception, command, route reference block) plus
SONIC's last latent (x 0.5), last raw action (x 0.25), gait phase (sin, cos), commanded gait speed and yaw rate.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

SONIC_DIR = Path(os.environ.get("SONIC_DIR", "/home/konstantinsmirnov/robotics/sonic/hf_gear_sonic"))
LIBRARY = Path(os.environ.get("SONIC_GAIT_LIBRARY", "/tmp/cat-sonic/latent/gait_library.npz"))
EXTRA_OBS = 64 + 29 + 2 + 1 + 1
ACTIONS = 64 + 2
RAMP_S = .8
FRAMES, FRAME_DT = 10, .1


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def _quat_yaw_rp(yaw, roll, pitch):
    """wxyz quaternion of yaw * pitch * roll (ZYX)."""
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    return torch.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1)


class GaitReference:
    def __init__(self, task, library=LIBRARY, *, min_speed=.6):
        lib = np.load(library)
        keep = lib["speeds"] >= min_speed - 1e-6
        dev = task.device
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
        self.task, self.n, self.dt = task, task.num_envs, task.dt
        self.joints, self.roll, self.pitch = t(lib["joints"][keep]), t(lib["roll"][keep]), t(lib["pitch"][keep])
        self.rate = 1. / t(lib["period"][keep])                  # strides per second
        self.fwd = t(lib["forward_speed"][keep])                 # m/s of each stride (kinematic)
        self.stand = t(lib["stand"])
        self.P = self.joints.shape[1]
        self.phase = torch.zeros(self.n, device=dev)
        self.yaw = torch.zeros(self.n, device=dev)
        self.ramp = torch.zeros(self.n, device=dev)
        self.speed = torch.zeros(self.n, device=dev)
        self.yaw_rate = torch.zeros(self.n, device=dev)
        self.offsets = torch.arange(FRAMES, device=dev) * FRAME_DT

    def _blend(self, v):
        """Stride indices, weight and phase rate for kinematic speeds v [n]."""
        F = self.fwd
        k = torch.searchsorted(F, v.clamp(F[0], F[-1]).contiguous()).clamp(1, len(F) - 1) if len(F) > 1 else \
            torch.ones_like(v, dtype=torch.long)
        lo = (k - 1).clamp_min(0); hi = k.clamp_max(len(F) - 1)
        span = (F[hi] - F[lo]).clamp_min(1e-6)
        w = ((v - F[lo]) / span).clamp(0., 1.) if len(F) > 1 else torch.zeros_like(v)
        rate = (1 - w) * self.rate[lo] + w * self.rate[hi]
        # outside the library range the nearest stride is time-scaled (slower or faster cadence, same joint amplitudes)
        rate = torch.where(v < F[0], self.rate[0] * v / F[0], torch.where(v > F[-1], self.rate[-1] * v / F[-1], rate))
        return lo, hi, w, rate

    def _cycle(self, lo, hi, w, phase):
        """Joint angles [n, K, 29], roll and pitch [n, K] at phases [n, K] for stride pair (lo, hi, w)."""
        u = torch.remainder(phase, 1.) * self.P
        i0 = u.floor().long() % self.P; i1 = (i0 + 1) % self.P; a = (u - u.floor())
        def at(table, s):
            x0, x1 = table[s[:, None], i0], table[s[:, None], i1]
            frac = a[..., None] if x0.dim() == 3 else a
            return x0 + frac * (x1 - x0)
        ww = w[:, None, None]
        q = (1 - ww) * at(self.joints, lo) + ww * at(self.joints, hi)
        r = (1 - w[:, None]) * at(self.roll, lo) + w[:, None] * at(self.roll, hi)
        p = (1 - w[:, None]) * at(self.pitch, lo) + w[:, None] * at(self.pitch, hi)
        return q, r, p

    @torch.no_grad()
    def __call__(self, speed, yaw_rate):
        """speed [n] (kinematic m/s, < 0.05 = stand) and yaw rate [n] -> (pose [n, 10, 36], qd [n, 10, 29]); advances."""
        task = self.task
        fresh = task.info["step"] == 0
        robot_yaw = task._root_yaw(task.all_ids)
        self.phase = torch.where(fresh, 0., self.phase)
        self.ramp = torch.where(fresh, 0., self.ramp)
        self.yaw = torch.where(fresh, robot_yaw, robot_yaw + _wrap(self.yaw - robot_yaw).clamp(-.4, .4))
        moving = speed > .05
        self.speed, self.yaw_rate = speed, yaw_rate
        lo, hi, w, rate = self._blend(speed.clamp_min(.05))
        tk = self.offsets[None]
        direction = torch.where(moving, 1., -1.)[:, None]
        ramp_k = (self.ramp[:, None] + direction * tk / RAMP_S).clamp(0., 1.)
        phase_k = self.phase[:, None] + rate[:, None] * tk
        q, r, p = self._cycle(lo, hi, w, phase_k)
        h = .5 / self.P
        qa, _, _ = self._cycle(lo, hi, w, phase_k + h); qb, _, _ = self._cycle(lo, hi, w, phase_k - h)
        qd = (qa - qb) / (2 * h) * rate[:, None, None] * ramp_k[..., None]
        q = (1 - ramp_k[..., None]) * self.stand + ramp_k[..., None] * q
        yaw_k = self.yaw[:, None] + yaw_rate[:, None] * tk
        quat = _quat_yaw_rp(yaw_k, r * ramp_k, p * ramp_k)
        pose = torch.zeros(self.n, FRAMES, 36, device=q.device)
        pose[..., 2] = .79
        pose[..., 3:7] = quat
        pose[..., 7:] = q
        self.ramp = (self.ramp + torch.where(moving, 1., -1.) * self.dt / RAMP_S).clamp(0., 1.)
        self.phase = torch.remainder(self.phase + rate * self.dt * (self.ramp > 0).float(), 1.)
        self.yaw = _wrap(self.yaw + yaw_rate * self.dt)
        return pose, qd


SPEED_FLOOR = float(os.environ.get("SONIC_GAIT_FLOOR", ".25"))


def route_heading_speed(task, gait, a_speed=None, a_turn=None, *, lookahead=.6, gain=1.5, max_rate=.6, floor=None):
    """Commanded kinematic gait speed and yaw rate from the field-guidance route and the operator command."""
    t = task.tracker
    ids = task.all_ids
    xy = task.data.qpos[:, :2]
    s, _ = t.project(ids, xy)
    goal = t._point(ids, s + lookahead)
    heading = torch.atan2(goal[:, 1] - xy[:, 1], goal[:, 0] - xy[:, 0])
    rate = (gain * _wrap(heading - gait.yaw)).clamp(-max_rate, max_rate)
    speed = torch.linalg.vector_norm(task.info["command"][:, 1:3], dim=-1)
    if a_speed is not None:
        speed = speed * (1 + .5 * a_speed)
        rate = rate + .5 * a_turn
    # Below ~0.25 m/s the time-scaled stride becomes slow motion that SONIC does not carry forward (0.06 m/s achieved
    # at a 0.15 m/s gait command on flat ground), so walking commands are floored; < 0.05 m/s means stand.
    floor = SPEED_FLOOR if floor is None else floor
    speed = torch.where(speed > .05, speed.clamp(floor, .8), torch.zeros_like(speed))
    return speed, rate.clamp(-.8, .8)


class LatentAdaptorTask:
    """Training wrapper (attribute delegation to the CATTask). step(action [n, 66])."""

    _own = {"_task", "gait", "sonic", "latent_scale", "obs", "observation_size", "privileged_observation_size",
            "_mod", "_last_action", "_steps", "soft_steps", "contact_cost", "rate_weight", "timing"}

    def __init__(self, task, *, latent_scale=.3, soft_steps=0, contact_cost=2., rate_weight=.02, library=LIBRARY,
                 sonic_dir=SONIC_DIR):
        from .sonic_tracker import SonicTracker

        if task.tracker is None:
            raise ValueError("The latent adaptor needs the tracker's field route")
        object.__setattr__(self, "_task", task)
        self.gait = GaitReference(task, library)
        self._mod = torch.zeros(task.num_envs, 2, device=task.device)
        self.sonic = SonicTracker(task, sonic_dir / "model_encoder.onnx", sonic_dir / "model_decoder.onnx",
                                  provider="torch", reference=self._reference)
        self.latent_scale = float(latent_scale)
        self.observation_size = task.observation_size + EXTRA_OBS
        self.privileged_observation_size = task.privileged_observation_size + EXTRA_OBS
        self._last_action = torch.zeros(task.num_envs, ACTIONS, device=task.device)
        self._steps, self.soft_steps, self.contact_cost, self.rate_weight = 0, int(soft_steps), float(contact_cost), float(rate_weight)
        if self.soft_steps > 0:
            task.config["contact_penalty"] = self.contact_cost
            task.config["contact_soft_hands"] = True
        self.timing = dict(steps=0)
        self._observe()

    def __getattr__(self, name):
        return getattr(self._task, name)

    def __setattr__(self, name, value):
        if name in self._own:
            object.__setattr__(self, name, value)
        else:
            setattr(self._task, name, value)

    def _reference(self):
        speed, rate = route_heading_speed(self._task, self.gait, self._mod[:, 0], self._mod[:, 1])
        return self.gait(speed, rate)

    def _observe(self):
        task, s = self._task, self.sonic
        z = s.latent if s.latent is not None else torch.zeros(task.num_envs, 64, device=task.device)
        extra = torch.cat(((z * .5).clamp(-5, 5), (s.last_action * .25).clamp(-5, 5),
                           torch.sin(2 * math.pi * self.gait.phase)[:, None], torch.cos(2 * math.pi * self.gait.phase)[:, None],
                           self.gait.speed[:, None], self.gait.yaw_rate[:, None]), -1)
        fresh = (task.info["step"] == 0)[:, None]
        extra = torch.where(fresh, torch.zeros_like(extra), extra)
        self.obs = {"state": torch.cat((task.obs["state"], extra), -1),
                    "privileged_state": torch.cat((task.obs["privileged_state"], extra), -1)}
        return self.obs

    def reset(self, *args, **kwargs):
        out = self._task.reset(*args, **kwargs)
        self._observe()
        return out

    def step(self, action, **_):
        task = self._task
        action = action.clamp(-1., 1.)
        self._mod = action[:, 64:66]
        raw, target = self.sonic(residual=action[:, :64], scale=self.latent_scale)
        out = task.step((raw * .25).clamp(-1., 1.), position_targets=target)
        rate = -self.rate_weight * (action - self._last_action).square().sum(-1)
        done = out["done"].bool()
        self._last_action = torch.where(done[:, None], torch.zeros_like(action), action)
        out["reward"] = out["reward"] + rate
        out["metrics"]["reward/adaptor_action_rate"] = rate
        out["metrics"]["sonic/latent_residual_abs"] = (self.latent_scale * action[:, :64]).abs().mean(-1)
        out["metrics"]["sonic/speed_mod"] = action[:, 64]
        out["metrics"]["sonic/turn_mod"] = action[:, 65]
        self._steps += 1
        if self.soft_steps and self._steps == self.soft_steps:
            task.config["contact_penalty"] = None
            task.config["contact_soft_hands"] = False
            print(f"SONIC latent adaptor: contacts terminal from control step {self._steps}", flush=True)
        out["obs"] = self._observe()
        return out


class LatentController:
    """Evaluation controller (BraceTracker interface): SONIC on the batched gait reference along the field route, with
    an optional trained latent adaptor (None = plain SONIC on this reference)."""

    def __init__(self, task, *, adaptor=None, latent_scale=.3, library=LIBRARY, sonic_dir=SONIC_DIR):
        from .sonic_tracker import SonicTracker

        self.task, self.adaptor, self.latent_scale = task, adaptor, float(latent_scale)
        self.gait = GaitReference(task, library)
        self.mod = torch.zeros(task.num_envs, 2, device=task.device)
        self.sonic = SonicTracker(task, sonic_dir / "model_encoder.onnx", sonic_dir / "model_decoder.onnx",
                                  provider="torch", reference=self._reference)
        self.extra = None

    def _reference(self):
        speed, rate = route_heading_speed(self.task, self.gait, self.mod[:, 0], self.mod[:, 1])
        return self.gait(speed, rate)

    @torch.no_grad()
    def __call__(self):
        task = self.task
        if self.adaptor is None:
            raw, target = self.sonic()
            return (raw * .25).clamp(-1., 1.), target
        z = self.sonic.latent if self.sonic.latent is not None else torch.zeros(task.num_envs, 64, device=task.device)
        extra = torch.cat(((z * .5).clamp(-5, 5), (self.sonic.last_action * .25).clamp(-5, 5),
                           torch.sin(2 * math.pi * self.gait.phase)[:, None], torch.cos(2 * math.pi * self.gait.phase)[:, None],
                           self.gait.speed[:, None], self.gait.yaw_rate[:, None]), -1)
        extra = torch.where((task.info["step"] == 0)[:, None], torch.zeros_like(extra), extra)
        obs = {"state": torch.cat((task.obs["state"], extra), -1),
               "privileged_state": torch.cat((task.obs["privileged_state"], extra), -1)}
        action = self.adaptor.act(obs, policy_ids=0, deterministic=True)["action"].clamp(-1., 1.)
        self.mod = action[:, 64:66]
        raw, target = self.sonic(residual=action[:, :64], scale=self.latent_scale)
        return (raw * .25).clamp(-1., 1.), target
