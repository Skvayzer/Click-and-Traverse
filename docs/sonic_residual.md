# SONIC + residual reflex

Started 2026-11-24 (Claude, branch `feature/sonic-baseline`). In this project the frozen NVIDIA GEAR-SONIC tracker
(Default release, `nvidia/GEAR-SONIC@6733128a`) is driven through an interface it already handles well. A small residual
policy will then add obstacle corrections from our potential-field inputs. Gates G0–G3; work stops at any failed gate.
Background: E6b in `docs/overnight_20261123.md`. SONIC fails to track our planner's whole-body plans because their
swing-foot lift is only 3–4.5 cm.

## G0: command interfaces of the released Default SONIC (2026-11-24 18:20 +04)

Sources: the official deployment `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref` (vendored in GRAIL `aa31d82`):
`localmotion_kplanner.hpp`, `localmotion_kplanner_onnx.hpp`, `g1_deploy_onnx_ref.cpp` and `input_interface/*`. We also
checked the release's `observation_config.yaml` and the ONNX graphs.

1. **The policy has no velocity, heading or joystick input.** The SONIC encoder accepts only three reference types:
   - g1 (mode 0): 10 future whole-body frames at 0.1 s plus pelvis orientation.
   - teleop (mode 1): 10 future frames of the 12 lower-body joints, VR 3-point wrist and torso position and
     orientation targets, and pelvis orientation.
   - smpl (mode 2): SMPL joints.
   Locomotion from sparse commands always goes through a separate kinematic generator whose output SONIC tracks in g1
   mode.
2. **Kinematic locomotion generator `planner_sonic.onnx`** (774 MB, SHA256 `39b553e1…edea`, interface version 1, batch 1).
   - Inputs:
     - the last 4 frames of whole-body qpos at 30 Hz as context;
     - `mode`, one of 27 styles (IDLE, SLOW_WALK 0.1–0.8 m/s, WALK 0.8–2.5, RUN, squat, kneel, crawl, boxing and styled
       walks; the keyboard clamps SLOW_WALK to 0.2–0.8 m/s);
     - `target_vel` (−1 selects the mode default);
     - `movement_direction` and `facing_direction`, as unit 3-vectors in the motion's world frame;
     - `height` (−1 selects the default);
     - `random_seed`;
     - waypoint inputs: `has_specific_target`, `specific_target_positions [4×3]`, `specific_target_headings [4]` and
       `allowed_pred_num_tokens [11]`.
   - Output: up to 64 frames at 30 Hz (about 2.1 s) of MuJoCo qpos (root and 29 joints). The deployment resamples them to
     50 Hz and feeds them to SONIC as the g1 reference.
   - Replanning: every 1.0 s while walking (0.1 s running, 0.2 s crawling), and at once when the mode, direction,
     facing, speed or height changes.
   - The context comes from the generator's own previous output, sampled 2 control ticks ahead, not from the measured
     robot. The generator is therefore open-loop in global position.
   - **The waypoint inputs exist in the model but the deployment always sets `has_specific_target = 0`.** Their frame and
     semantics are undocumented.
3. **Upper-body override.** When external upper-body data arrives over ZMQ or ROS2 (`has_upper_body_data_`), the 17
   waist and arm joint positions and velocities in the g1 reference are replaced by the streamed targets. The legs and
   root still come from the generator. This is how the released VLA and teleoperation stacks command the arms while
   SONIC walks.
4. **Streamed whole-body motion** (`streamed_motion_merger.hpp`): arbitrary whole-body reference streams over ZMQ.
   This is the same g1 tracking interface that failed with our plans in E6b.

**G1 driving plan.** Port the generator loop to Python. At each replan (1.0 s, and also on large direction changes),
send SLOW_WALK with `target_vel` set to the operator speed, `movement_direction` set to the route tangent at a
look-ahead point, and `facing_direction` set to the same tangent. Close the global loop by computing these directions
from the measured robot pose relative to our route. The route is either our planner's pelvis waypoints or the field
guidance path. Our planned waist and arm targets can be sent through the upper-body override. The generator stays
frozen; SONIC tracks its output.
Open questions for G1:
- whether 0.15 m/s lies inside SLOW_WALK's trained range (the keyboard minimum is 0.2 m/s);
- whether the open-loop generator plus our closed-loop direction command meets ≤ 10 cm path error;
- batching the 774 MB batch-1 generator for the 384- and 72-scene benchmarks, which needs the same symbolic-batch
  patch as the encoder.

## G1: driving SONIC from our route (quick flat-ground check, 2026-11-24 18:20 +04)

`cat_mjlab/sonic_locomotion.py` ports the deployment's generator loop to Python. Each call is canonicalized. Replanning
happens every 1 s, at once when the direction turns by more than 20° or the speed or mode changes, and when the buffer
runs low. The splice follows the deployment: 2-frame look-ahead and an 8-frame blend. Worlds are planned one call at a
time on a thread pool of batch-1 ONNX sessions (27–34 ms per call). `route_command` commands SLOW_WALK toward the
field-guidance route point 0.6 m ahead of the robot's projection, facing the same way, at the operator speed. Below
0.05 m/s it commands IDLE. The direction is computed from the measured robot pose, which closes the position loop that
the generator leaves open.

Flat scene `flat-balance-walk-v1`, one world, CPU MuJoCo, harness torque noise on, 12 s per run (`scripts/sonic_gates.py loco`).

| Commanded generator speed | Net travel (m) | Mean speed (m/s) | Median route error (cm), excluding the first 2 s | p90 (cm) | Max foot lift L/R (cm) | Fall |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.30 | 2.58 | 0.215 | 3.1 | 6.8 | 6.0 / 6.8 | no |
| 0.15 | 0.02 | 0.002 | 2.5 (stood still on the route) | 3.9 | 1.5 / 1.7 | no |

At 0.30 m/s SONIC walks a natural gait along our route with a 3 cm median error. At 0.15 m/s the generator's own
SLOW_WALK is a shuffle that SONIC does not execute; the deployment keyboard also never commands less than 0.2 m/s.
**The decision is therefore to clip the generator command to 0.30–0.80 m/s.** The robot then walks at about
0.21 m/s, which stays under the harness overspeed limit for a 0.15 m/s operator command
(1.25 × 0.15 + 0.05 = 0.24 m/s). Upper-body override is not used yet; the residual handles the arms.

## G3: residual reflex training (launched 2026-11-24 18:23 +04, user override: before G2)

**Design** (`cat_mjlab/sonic_residual.py`; runner hooks `--sonic-residual`; eval `scripts/eval_per_scene.py --sonic`):

- **Control.** PD target = q_SONIC + 0.3 rad × mask × a, with a = tanh-Gaussian PPO action and SONIC's PD gains. The
  `whole` variant masks in all 29 joints; the `arms` variant masks in the 14 arm joints. The generator and SONIC are
  frozen in the loop.
- **Observation.** Our COMBO_FIX actor observation (13-point potential field, proprioception, command and the field-route
  reference block; 267 actor and 351 critic inputs) plus SONIC's current raw action × 0.25 (29), giving 296 actor and
  380 critic inputs.
- **Initialization.** PPO trains from scratch with default network sizes (actor 512-256-128-64). The mean head is
  initialized to zero, so the policy starts as plain SONIC. Initial σ is 0.3, i.e. ±0.09 rad of residual noise.
- **Environment.** COMBO_FIX seed-0 environment, with the following changes:
  - tracker mix field=1, so the route is the guidance path;
  - nominal physics (no latency or pose-noise robustness);
  - pushes kept;
  - upper-body gravity feed-forward off;
  - bank `tablegap_v1` and operator speeds 0.1–0.6 m/s, both unchanged.
- **Reward.** The COMBO_FIX terms are kept for:
  - potential field (feet, hands, head, knees, shoulders);
  - contact and clearance: hand-body contact, self-clearance, whole-body arm and hand clearance;
  - progress: track_pos, track_lead, goal_progress, goal_bonus, stall, tracking_root_field;
  - the speed gate (overspeed);
  - upright, joint limits, action rate and action smoothness.

  All gait, posture and energy shaping terms are set to 0, because SONIC supplies the style. A residual cost of
  −1 per rad² completes the reward.
- **Runs (dep-1 GPU under the shared lock, both variants in parallel).** Seed 0, 1024 worlds, unroll 32, batch
  256 × 4 minibatches, 300 updates, checkpoints every 10 updates, snapshots every 50. Outputs are
  `~/robotics/cat-energy-local-runs/sonic_residual_{whole,arms}_seed0` and logs are
  `/tmp/cat-sonic/train/{whole,arms}.log`. Throughput is about 20 s per update with both runs active (SONIC and the
  generator take about 0.3 s per control step), giving an ETA of about 1 h 45 min (around 20:15 +04).
- **Smoke.** 2 updates each at 64 and 256 worlds completed, writing `best.pt` and `resume.pt`.

**G2 zero-residual baseline (cluster, running in parallel).** Code `91a5467` is in `~/cat_worktrees/sonic_91a5467` and
the release files are in `/l/users/konstantin.smirnov/sonic_release`. Smoke 322018 (3 scenes) gates 322019 (384 narrow
scenes at 0.15 and 0.30 m/s, 2 episodes, 7 h limit) and 322020 (72 E4 layouts at 0.15, 0.30 and 0.50 m/s, the same
certified starts, 5 h limit). All three run on ws-ia or gpu with 1 GPU, 12 CPUs and 54 GB. Outputs are
`/l/users/konstantin.smirnov/runs/sonic_G2_{smoke,384,E4}_<job>`.
