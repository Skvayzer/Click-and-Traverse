"""SONIC + residual reflex: frozen GEAR-SONIC walks with its own gait; a small PPO policy adds joint-target residuals.

Per control step (50 Hz):
    SONIC's locomotion generator (planner_sonic.onnx) is commanded along our field guidance route (route_command:
    walk toward the route point 0.6 m ahead at the operator's commanded speed, IDLE below 0.05 m/s) and the frozen
    SONIC tracker turns its reference into PD targets q_S.
    The residual policy sees our observation (13-point potential field, proprioception, command, route reference block)
    plus SONIC's current raw action (x 0.25) and outputs a in [-1, 1]^29 (tanh-Gaussian PPO).
    PD target = q_S + scale * mask * a, with SONIC's PD gains; mask = all 29 joints ('whole') or the 14 arm joints ('arms').
    Reward = the task's reward (environment from a COMBO_FIX checkpoint, gait-shaping terms removed, see
    residual_environment) - penalty * sum(residual^2) (rad^2), so the motion stays SONIC's unless an obstacle pays.
The wrapper forwards every other attribute to the wrapped CATTask, so the native PPO runner is unchanged.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import torch

SONIC_DIR = Path(os.environ.get("SONIC_DIR", "/home/konstantinsmirnov/robotics/sonic/hf_gear_sonic"))
ARM_JOINTS = tuple(range(15, 29))
# Terms kept from the COMBO_FIX reward: potential-field guidance/clearance, contact and clearance costs, progress
# (route tracking, goal progress/bonus, stall), the speed gate, uprightness, joint limits and residual smoothness.
KEEP = {"feetdf", "feetgf", "handsdf", "handsgf", "headdf", "headgf", "kneesdf", "shldsdf",
        "hand_body_contact", "self_clearance", "wholebody_arm_clearance", "wholebody_hand_clearance",
        "track_pos", "track_lead", "goal_progress", "goal_bonus", "stall", "tracking_root_field",
        "overspeed", "upright", "joint_limits", "action_rate", "smoothness_action"}


def residual_environment(contract_env, planner_path):
    """COMBO_FIX environment for residual training: field-route references only, nominal physics (SONIC saw no
    latency/pose-noise robustness), gait/posture/energy shaping removed (SONIC supplies the style)."""
    env = copy.deepcopy(contract_env)
    for key in ("robustness", "refusal_training", "sgs"):
        env.pop(key, None)
    env["tracker"]["mix"] = [["field", 1.]]
    env["tracker"]["planner"]["path"] = str(planner_path)
    scales = env["reward_config"]["scales"]
    removed = sorted(k for k in scales if k not in KEEP)
    for key in removed:
        scales[key] = 0.
    env["upper_gravity_compensation"] = False
    env["upper_target_average"] = False
    return env, removed


class SonicResidualTask:
    _own = {"_task", "sonic", "generator", "mask", "scale", "penalty", "obs", "observation_size",
            "privileged_observation_size", "_target", "_raw", "last_residual", "timing"}

    def __init__(self, task, *, mode="whole", scale=.3, penalty=1., sessions=4, threads=2, provider="cuda",
                 generator_provider="cpu", sonic_dir=SONIC_DIR):
        from .sonic_locomotion import SonicGenerator, route_command
        from .sonic_tracker import SonicTracker

        if task.tracker is None:
            raise ValueError("The residual task needs the tracker's field route")
        object.__setattr__(self, "_task", task)
        self.generator = SonicGenerator(task, Path(sonic_dir) / "planner_sonic.onnx", lambda ids: route_command(task),
                                        sessions=sessions, threads=threads, provider=generator_provider)
        self.sonic = SonicTracker(task, Path(sonic_dir) / "model_encoder.onnx", Path(sonic_dir) / "model_decoder.onnx",
                                  provider=provider, reference=self.generator.reference)
        mask = torch.zeros(29, device=task.device)
        if mode == "whole":
            mask[:] = 1.
        elif mode == "arms":
            mask[list(ARM_JOINTS)] = 1.
        else:
            raise ValueError("mode must be 'whole' or 'arms'")
        self.mask, self.scale, self.penalty = mask, float(scale), float(penalty)
        self.observation_size = task.observation_size + 29
        self.privileged_observation_size = task.privileged_observation_size + 29
        self.last_residual = torch.zeros(task.num_envs, 29, device=task.device)
        self.timing = dict(sonic=0., physics=0., steps=0)
        self._propose()

    def __getattr__(self, name):
        return getattr(self._task, name)

    def __setattr__(self, name, value):
        if name in self._own:
            object.__setattr__(self, name, value)
        else:
            setattr(self._task, name, value)

    def _propose(self):
        """SONIC's PD targets for the current state (after resets); refresh the augmented observation."""
        import time
        tic = time.perf_counter()
        raw, target = self.sonic()
        if self._task.device.type == "cuda":
            torch.cuda.synchronize()
        self.timing["sonic"] += time.perf_counter() - tic
        self._raw, self._target = raw, target
        block = (raw * .25).clamp(-5., 5.)
        task = self._task
        self.obs = {"state": torch.cat((task.obs["state"], block), -1),
                    "privileged_state": torch.cat((task.obs["privileged_state"], block), -1)}
        return self.obs

    def reset(self, *args, **kwargs):
        out = self._task.reset(*args, **kwargs)
        self._propose()
        return out

    def step(self, action, **_):
        import time
        residual = action.clamp(-1., 1.) * self.scale * self.mask
        tic = time.perf_counter()
        out = self._task.step(action, position_targets=self._target + residual)
        if self._task.device.type == "cuda":
            torch.cuda.synchronize()
        self.timing["physics"] += time.perf_counter() - tic
        self.timing["steps"] += 1
        if self.timing["steps"] % 25 == 0:
            k = self.timing["steps"]
            print(f"SONIC timing per control step over {k}: sonic+generator {self.timing['sonic'] / k * 1e3:.1f} ms, "
                  f"physics {self.timing['physics'] / k * 1e3:.1f} ms, generator calls {self.generator.calls}", flush=True)
        cost = -self.penalty * residual.square().sum(-1)
        out["reward"] = out["reward"] + cost
        out["metrics"]["reward/sonic_residual"] = cost
        out["metrics"]["sonic/residual_abs_mean"] = residual.abs().sum(-1) / self.mask.sum()
        self.last_residual = residual
        out["obs"] = self._propose()
        return out


def zero_residual_mean(learner):
    """Residual starts as SONIC: zero the mean rows of the actor's output layer (the sigma rows are untouched)."""
    layer = learner.model.actor.layers[-1]
    with torch.no_grad():
        layer.weight[:learner.config.action_size].zero_()
        layer.bias[:learner.config.action_size].zero_()


class SonicController:
    """Evaluation controller with the BraceTracker interface: SONIC driven by its generator along the field route, plus
    an optional trained residual policy (None = zero residual, the plain SONIC baseline). Returns (action, targets)."""

    def __init__(self, task, *, residual=None, mode="whole", scale=.3, provider="cuda", sessions=8, threads=2,
                 sonic_dir=SONIC_DIR):
        from .sonic_locomotion import SonicGenerator, route_command
        from .sonic_tracker import SonicTracker

        if task.tracker is None:
            raise ValueError("SONIC evaluation needs the tracker's field route (--tracker-mix field=1)")
        self.task, self.residual = task, residual
        self.generator = SonicGenerator(task, Path(sonic_dir) / "planner_sonic.onnx", lambda ids: route_command(task),
                                        sessions=sessions, threads=threads)
        self.sonic = SonicTracker(task, Path(sonic_dir) / "model_encoder.onnx", Path(sonic_dir) / "model_decoder.onnx",
                                  provider=provider, reference=self.generator.reference)
        self.mask = torch.zeros(29, device=task.device)
        self.mask[list(range(29)) if mode == "whole" else list(ARM_JOINTS)] = 1.
        self.scale = float(scale)

    @torch.no_grad()
    def __call__(self):
        raw, target = self.sonic()
        if self.residual is None:
            return torch.zeros_like(target), target
        block = (raw * .25).clamp(-5., 5.)
        obs = {"state": torch.cat((self.task.obs["state"], block), -1),
               "privileged_state": torch.cat((self.task.obs["privileged_state"], block), -1)}
        action = self.residual.act(obs, policy_ids=0, deterministic=True)["action"]
        return action, target + action.clamp(-1., 1.) * self.scale * self.mask
