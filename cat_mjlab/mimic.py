"""Full-body motion tracker ("mimic") on mjlab's G1 tracking task (BeyondMimic-style), for clip libraries.

mjlab's MotionCommand plays one long motion. A clip library (scripts/mimic_library.py) stores many separate clips back
to back plus their starts (segment_starts); SegmentMotionCommand treats a clip's end like the motion's end, so the world
is re-sampled and re-initialised on a reference state there and the reference never jumps into the next clip.
Importing this module registers the task "Cat-Mimic-G1" (mjlab's G1 flat tracking task with this command); train it with
scripts/mimic_train.py.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg

TASK_ID = "Cat-Mimic-G1"


class SegmentMotionCommand(MotionCommand):
    """MotionCommand over a library of separate clips: a world that reaches the end of its clip is re-sampled."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        total = int(self.motion.time_step_total)
        with np.load(cfg.motion_file) as f:
            starts = f["segment_starts"].astype(np.int64) if "segment_starts" in f.files else np.zeros(1, np.int64)
        ends = np.append(starts[1:], total)
        frame_end = np.empty(total, np.int64)
        for s, e in zip(starts, ends):
            frame_end[s:e] = e
        self.frame_end = torch.as_tensor(frame_end, device=self.device)
        self.segment_end = self.frame_end[self.time_steps.clamp(0, total - 1)].clone()

    def _resample_command(self, env_ids: torch.Tensor):
        super()._resample_command(env_ids)
        self.segment_end[env_ids] = self.frame_end[self.time_steps[env_ids]]

    def _update_command(self, env_ids: torch.Tensor | None = None):
        # Worlds about to step past their clip are put on the motion's last frame, so the base class's own wrap-around
        # (time step >= total -> re-sample and re-initialise) handles them; _resample_command then sets their new clip.
        if env_ids is None:
            ending = torch.nonzero(self.time_steps + 1 >= self.segment_end).squeeze(-1)
        else:
            ending = env_ids[self.time_steps[env_ids] + 1 >= self.segment_end[env_ids]]
        if ending.numel():
            self.time_steps[ending] = self.motion.time_step_total - 1
        super()._update_command(env_ids)


@dataclasses.dataclass(kw_only=True)
class SegmentMotionCommandCfg(MotionCommandCfg):
    def build(self, env):
        return SegmentMotionCommand(self, env)


def mimic_env_cfg(play: bool = False, motion_file: str = ""):
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)
    old = cfg.commands["motion"]
    new = SegmentMotionCommandCfg(**{f.name: getattr(old, f.name) for f in dataclasses.fields(old)})
    if motion_file:
        new.motion_file = motion_file
    cfg.commands["motion"] = new
    return cfg


def mimic_rl_cfg():
    from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    cfg = unitree_g1_tracking_ppo_runner_cfg()
    cfg.experiment_name = "cat_mimic"
    cfg.logger = "tensorboard"
    return cfg


def register():
    from mjlab.tasks.registry import list_tasks, register_mjlab_task
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
    if TASK_ID in list_tasks():
        return
    register_mjlab_task(task_id=TASK_ID, env_cfg=mimic_env_cfg(), play_env_cfg=mimic_env_cfg(play=True),
                        rl_cfg=mimic_rl_cfg(), runner_cls=MotionTrackingOnPolicyRunner)


register()
