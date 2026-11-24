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
