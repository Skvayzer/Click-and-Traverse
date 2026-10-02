# cat_g1_rooms_keyboard_v9_u492

Unitree G1 (29 DoF, fixed Dex3 hands) whole-body navigation policy: rooms + table edges + keyboard
(joystick-style) control. Saved 2026-10-02 from run `outputs/expert_rooms_tables_v9_20261001`,
update 492 (config `configs/pilots/expert_rooms_tables_v9_20261001.sh`, code at commit 732d4d8).

## Files
- `actor.pt` (1.2 MB): the policy only -- `actor` state dict (292,474 parameters, MLP
  226 -> 512 -> 256 -> 128 -> 64 -> 58 = mean and scale for 29 joints), `learner_config`,
  `environment_config` (observation/command settings), `contract`. Verified to give identical
  actions to the full checkpoint (max difference 0.0).
- Full training snapshot (919 MB: optimizer, critics, style discriminators, simulator state of 40,960
  worlds) is kept on dep-1 at `outputs/saved_checkpoints/cat_g1_rooms_keyboard_v9_u492_full.pt`
  (hard link); sha256 in `full_checkpoint.sha256`.

Load the policy:
```python
import torch
from cat_mjlab.learning import ActorCritic, LearnerConfig, gaussian_parameters
a = torch.load("checkpoints/cat_g1_rooms_keyboard_v9_u492/actor.pt", map_location="cpu", weights_only=True)
model = ActorCritic(LearnerConfig(**a["learner_config"])); model.load_state_dict(a["actor"], strict=False)
mean, _ = gaussian_parameters(model.logits(obs_226, 0)); action = mean.tanh()
```

## Inputs it needs (226 per step, 50 Hz)
Proprioception (IMU gyro and gravity, joint positions/velocities, last action, motor targets, gait
phase, foot height), the command `[move, vx, vy (heading frame), gated heading error]`, and obstacle
information at 11 body points (distance field, obstacle normal, guidance field) plus elbows and the
clearance-rate features. Observation layout: `cat_mjlab/task_math.py::observations`.

## Measured (simulation)
- Route following (point navigation), training metrics at update ~490: clutter_dense 76%,
  furniture_dense 76%, clutter/furniture pilot rooms ~90%, falls 0.1%; table edges 23% (degraded
  in this run: hesitation at narrow table gaps, see the pipeline doc).
- Clutter videos (deterministic, no pushes): 8/8 successful (`outputs/videos_20261002_clutter`).
- Keyboard control (`scripts/demo/command_following_test.py`, flat scene): turn in place 0.68 / -0.61
  rad/s of 0.8 (25-35 cm drift); forward 0.36 m/s of 0.5; forward + turn 0.48 of 0.5 rad/s;
  walking backward and sideways not learned (it turns to face the direction instead).
- Joystick episodes in training: 31% end in obstacle contact (from 55%).

## Not yet validated
No real-robot or other-simulator test. The obstacle fields come from the scene map in simulation; a
robot needs them from a map + localization or onboard perception.
