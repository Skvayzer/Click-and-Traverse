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

## Results so far (2026-11-24 21:30 +04)

**G2: plain SONIC with zero residual** (cluster jobs 322019 and 322020, code `91a5467`, 2 episodes per scene, every
episode finished; full JSON in `/l/users/konstantin.smirnov/runs/sonic_G2_{384_322019,E4_322020}`). The generator
command is at least 0.30 m/s, so SONIC walks at about 0.21 m/s at every operator speed.

| Benchmark | Speed (m/s) | Clean % | Contact % | Refused % | Hands % | Fall % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 384 narrow scenes | 0.15 | 12.89 | 87.11 | 0.00 | 19.40 | 0 |
| 384 narrow scenes | 0.30 | 12.89 | 86.33 | 0.78 | 20.96 | 0 |
| E4, 72 layouts | 0.15 | 0.00 | 100.00 | 0.00 | 37.50 | 0 |
| E4, 72 layouts | 0.30 | 0.00 | 100.00 | 0.00 | 39.58 | 0 |
| E4, 72 layouts | 0.50 | 0.00 | 100.00 | 0.00 | 50.69 | 0 |

The reference rates on the same protocols are 61–62% clean on the 384 scenes (planner–reflex, three-seed mean). On E4
they are 47–55% clean for COMBO_FIX seed 0 and 52–58% for V7r. On E4, SONIC's episodes end after a mean of 0.6–0.7 m of
path.

**G3: residual training at u300** (both variants completed 300 updates at 20:05–20:07 +04).
- Training clean-goal rate went 0 → 1.3% (whole) and 0 → 1.4% (arms).
- CAT-course obstacle-contact rate stayed at 100%.
- Residual cost fell from −0.20 to −0.05 per step (whole) and from −0.10 to −0.02 (arms).
- Action σ went 0.30 → 0.13 (whole) and stayed at 0.25 (arms).
- No falls.

The residual did not learn obstacle avoidance at this budget, and the u300 checkpoints have not yet been evaluated on G2.

**Obstacle videos.** `artifacts/sonic-videos/obstacles/` holds 10 scene types: hurdle, overhead beam, side gap 0.53 m,
cabinet passage 0.58 m, table gap, mixed CAT course, cluttered workshop, and the living, dining and meeting rooms. Each
scene has four 1280×720 clips (plain SONIC, residual whole-body u300, residual arms-only u300, planner–reflex), all at
0.30 m/s from the same certified start, plus a 2×2 side-by-side clip. Clips are at most 15 s; episodes longer than 15 s
play at 2×. Script `scripts/sonic_obstacle_videos.py`, commit `5fc5d13`.

| Policy | Clean of 10 | Contact or timeout |
| --- | ---: | --- |
| Plain SONIC | 0 | Contact in all 10 (side gap: hand), after 2.8–4.3 s on CAT, cabinet and table scenes and 8–18 s in rooms |
| Residual whole u300 | 0 | Contact in all 10 (side gap: hand) |
| Residual arms u300 | 0 | Contact in all 10 (side gap: hand) |
| Planner–reflex (COMBO_FIX s0) | 6 | Contact on hurdle, side gap and CAT course; 30 s timeout on table gap and dining room |

## SONIC + latent adaptor (overnight program approved 2026-11-24 ~21:45 +04)

This follows the play2perfect/GRAIL frozen latent-adaptor design, not joint residuals. The policy outputs a 64-D
residual that is added to SONIC's g1 encoder latent before FSQ, z' = z + 0.3·a. It also outputs 2 bounded command
modulations: gait speed × (1 + 0.5 a_s) and yaw rate + 0.5 a_t rad/s. The SONIC encoder, quantizer and decoder are
frozen. The mean head is zero-initialized, so training starts as plain SONIC.

**GPU port (`cat_mjlab/sonic_torch.py`).** The g1 encoder MLP (640→2048→1024→512→512→64, SiLU), the FSQ (2 tokens × 32
dimensions, 32 levels, `tanh(z+0.032237)·15.5155−0.5`, round, /16; straight-through) and the g1_dyn decoder
(994→2048→2048→1024→1024→512→512→29) are rebuilt from the hash-checked release ONNX initializers. `SonicTracker` has a
`provider="torch"` backend that exposes the latent.

**Batched reference (`cat_mjlab/sonic_latent.GaitReference`).** The batch-1, 774 MB generator (27–34 ms per call)
cannot serve 4096 worlds, so it is replaced by a phase-indexed replay of its own strides:
- `scripts/build_sonic_gait_library.py` rolls out `planner_sonic.onnx` open-loop exactly as deployed. For each speed it
  phase-folds the steady strides; empty bins are circularly interpolated.
- The output is `data/sonic_gait_library_v1.npz`. It contains strides for generator commands 0.3, 0.4, 0.5, 0.6 and 0.8:

  | Generator command (m/s) | Stride period (s) | Forward speed (m/s) | Fold RMS residual (rad) |
  | ---: | ---: | ---: | ---: |
  | 0.3 | 1.76 | 0.26 | 0.074 |
  | 0.4 | 1.28 | 0.35 | 0.070 |
  | 0.5 | 1.02 | 0.46 | 0.070 |
  | 0.6 | 1.00 | 0.52 | 0.003 |
  | 0.8 | 1.00 | 0.69 | 0.003 |

- Only the clean 0.6 and 0.8 strides are used. Slower commands time-scale the 0.6 stride, with a floor of 0.25 m/s
  kinematic; IDLE gives the standing pose. Pelvis heading integrates the yaw rate, the pelvis keeps the stride's
  roll/pitch, and 0.8 s ramps take the robot in and out of standing.
- Route command: walk toward the field-route point 0.6 m ahead, with yaw rate 1.5 × heading error (capped at 0.6 rad/s).

**Gate 1: PASS.** `tests/test_sonic_latent.py`, together with `tests/test_sonic_tracker.py`, passes 5/5.
- On 256 random inputs the torch port reproduces every FSQ code of the release (0 mismatches). The action chain matches
  ONNX within 3.7e-6.
- A zero residual is bitwise identical to plain SONIC.
- Gradients reach the residual through the straight-through FSQ and reach the adaptor's zero-initialized head. SONIC's
  weights receive no gradient and are `requires_grad=False`.

**Flat-ground check of the batched reference** (1 world, CPU MuJoCo, harness torque noise on, 12 s, plain SONIC):

| Operator command (m/s) | Gait command (kinematic, m/s) | Mean speed achieved (m/s) | Median route error (cm) | p90 (cm) | Max foot lift (cm) | Fall |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.15 | 0.25 (floor) | 0.184 | 4.6 | 8.6 | 8.0 / 8.5 | no |
| 0.30 | 0.30 | 0.28 | 4.6 | 8.0 | 8.5 / 8.9 | no |
| 0.50 | 0.50 | 0.48 | 2.8 | 5.3 | 9.7 / 9.8 | no |

Without the floor, a 0.15 m/s gait command became slow motion, and SONIC moved only 0.06 m/s. The floor gives 0.184
m/s, which stays under the overspeed limit for a 0.15 m/s operator command. This driving tracks better than the ONNX
generator in G1, which gave 0.215 m/s at its 0.30 command.

Diagnosis log: the first library left half of the phase bins empty, because a 1.0 s stride sampled at 50 Hz fills only
50 of 100 bins. Open-loop replay at exactly native cadence never sampled those bins, while closed-loop speeds did. That
produced 0.2 m/s walking with a 0.15 rad heading error. Interpolating the empty bins fixed it. Backend (torch vs ONNX),
environment, ramp length and steering were ruled out first.

**Reward and schedule.**
- The task terms are the COMBO_FIX ones kept for the residual project: field, contact/clearance, progress, speed gate,
  upright and joint limits. The task's own action-rate terms are removed, because the task's "action" is now SONIC's.
- The wrapper charges −0.02·‖Δa‖² on the adaptor's 66 actions.
- Falls always end an episode. For the first third of the requested updates every contact, including hand contact,
  costs 2 per step and is not terminal (new task flag `contact_soft_hands`). After that, contacts are terminal again.
- There is no latent-size penalty.
- The observation adds SONIC's last latent (×0.5), last raw action (×0.25), gait phase sin/cos, gait speed and yaw rate,
  97 extra inputs on top of the actor observation.

**Gate 2 (throughput): pending.** Cluster job 322754 (4096 worlds, unroll 16, 3 updates; code `457e442` in
`~/cat_worktrees/sonic_457e442`) waits on the per-user CPU limit. The same test is queued on dep-1 behind three
lock-holders' jobs.

**Gate 2 (throughput): FAIL on the stated criterion; the harness is the cause, not SONIC.** Measured on cluster
gpu-04 (RTX 5000 Ada 32 GB), job 322876, code `910307a`.

| Pipeline | Worlds | Control steps per second | Seconds per update | Transitions per update | Peak VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SONIC + latent adaptor | 4096 | 7,203 | 8.8–9.3 | 65,536 (16 steps) | 21.0 |
| Native planner–reflex (E2 322679, same GPU type) | 12,288 | 5,206 | 75.5 | 393,216 (32 steps) | 24.7 |

At 4096 worlds × 16 steps, ≤ 2 s per update would need about 33,000 control steps per second. That is 6× what the
harness reaches without SONIC. The only lever inside the criterion is a 4-step unroll, which trades samples per update
rather than fixing speed, so it was not adopted. The torch SONIC path and the batched reference make the pipeline faster
than the native one (7.2k vs 5.2k steps per second).

**Gate 3 (training): running.** Cluster job 322876 runs 600 updates. Contacts are soft for the first 200 updates
(3,200 control steps). Settings: 4096 worlds, unroll 16, batch 1024 × 4 minibatches, latent scale 0.3, initial σ 0.3,
seed 0. Snapshots every 50 updates go to `/l/users/konstantin.smirnov/runs/sonic_latent_seed0_322876`. ETA about
03:35 +04 on 2026-11-25. Before it started, the job waited about 3 h for a GPU because of the per-user GPU quota and
the dep-1 lock queue.

Zero-adaptor evaluations (plain SONIC on the batched reference) are queued: 322877 (384 scenes) and 322878 (E4).

**Gate 3 (training, u600): done.** Job 322876, 1.6 h on gpu-04.

| Update | Clean overall | Clutter rooms | CAT courses | Falls | Action σ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0% | 0% | — | 0 | 0.30 |
| 300 | 1.9% | 15% | — | 0 | — |
| 450 | 4.2% | 24% | — | 0 | — |
| 600 | 5.8% | 36% | ~1% | 0 | 0.19 |

**Gate 4 (partial).** All evaluations are on the cluster with 2 episodes per scene and the same certified E4 starts as
the other baselines. The 72-layout evaluation finished; the 384-scene one is still running.

| Controller | Benchmark | Speed (m/s) | Clean % | Contact % | Hands % |
| --- | --- | ---: | ---: | ---: | ---: |
| Zero adaptor, gait reference (322877/322878) | 384 scenes | 0.15 | 12.89 | 86.98 | 19.79 |
| Zero adaptor, gait reference | 384 scenes | 0.30 | 12.63 | 87.11 | 22.27 |
| Zero adaptor, gait reference | E4, 72 layouts | 0.15 | 0.00 | 100.00 | 40.97 |
| Zero adaptor, gait reference | E4, 72 layouts | 0.30 | 0.00 | 100.00 | 36.11 |
| Zero adaptor, gait reference | E4, 72 layouts | 0.50 | 0.00 | 100.00 | 37.50 |
| Adaptor u600 (323467) | E4, 72 layouts | 0.15 | 2.78 | 97.22 | 58.33 |
| Adaptor u600 | E4, 72 layouts | 0.30 | 3.47 | 96.53 | 46.53 |
| Adaptor u600 | E4, 72 layouts | 0.50 | 2.78 | 97.22 | 61.81 |

| Adaptor u600 (323466) | 384 scenes | 0.15 | **41.67** | 57.94 | 17.71 |
| Adaptor u600 | 384 scenes | 0.30 | **43.62** | 55.73 | 15.36 |

The zero-adaptor numbers reproduce plain SONIC driven by the ONNX generator (G2: 12.9% on 384 scenes, 0% on E4).
- **384 scenes:** u600 lifts clean traversal from about 13% to 42–44% (+29–31 pp) and lowers hand contact by 2–7 pp. It
  still trails planner–reflex (61–62% clean, 2.6–2.9% hands in the three-seed round 4).
- **E4, 72 layouts:** u600 reaches only 3% clean, with more hand contact than the zero adaptor. Planner–reflex gets
  47–55% clean there.
- **Caveat:** u600 is 1 training seed, and the 95% intervals are not yet computed.

**Gate 5 (u600): done.** Videos are in `sonic-videos/obstacles/`:
- `<scene>__sonic-latent.mp4` (u600) and `<scene>__plain-sonic-gait.mp4` (zero adaptor on the batched gait). The gait
  clips were recorded on CPU MuJoCo and rendered on the cluster.
- `<scene>__latent-side-by-side.mp4`, a 2×2 grid: plain SONIC on gait | SONIC + latent | planner–reflex | plain SONIC on
  the ONNX generator.

u600 outcomes on the 10 scenes:
- **Clean (1 scene):** cabinet passage 0.58 m.
- **Contact (5):** hurdle, beam, side gap, table gap, mixed CAT course.
- **30 s timeout (4):** cluttered room and the living, dining and meeting rooms. The robot moves slowly without touching.

Plain SONIC on gait contacts in all 10.

**Continuation (user request, 04:00 +04).**
- **Run 1 (323791)** resumes 322876 from u600 to u2000 with its exact code (`910307a`) and contract, snapshots every
  100 updates.
- **Run 2 (323792)** warm-starts from the u600 snapshot. It resets σ to 0.3 and keeps the latent scale at 0.3, since σ
  had not collapsed (0.19).
  - Bank: the beam/bar bank `tablegap_bb_v1`, with the height curriculum (advance 0.5, window 300).
  - Curricula: the side-gap width curriculum and the success-guided sampler (target 2/3, without the stale prior
    files).
  - Contacts stay soft until the CAT-course clean EMA exceeds 10% over at least 500 CAT episodes.
  - Length: 1400 updates.

Both runs wait on the per-user GPU and memory quota; their memory request was lowered to 40 GB.
