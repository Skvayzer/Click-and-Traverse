---
title: Style-prior pipeline (task RL + context-gated human motion prior)
tags: [robotics, humanoid, unitree-g1, amp, t-gmp, motion-prior, pipeline]
status: living document — update with every change
related: [[CAT-WholeBody-Project-State]], [[Motion-Prior-Pipeline]]
---

# Style-prior pipeline

Goal: the reward-trained CAT whole-body policy keeps deciding **what** to do (navigation,
collision avoidance, hand protection, posture); a learned motion prior decides **how** it moves,
so motion looks human. Approach: **AMP discriminators, gated by scene context (the T-GMP idea)**.
Approaching-object (reactive) episodes are **out of scope for now** (2026-09-29).

## Change log
| date | commit | change |
|---|---|---|
| 2026-09-28 | 2474ce0, 6dc89ac | PHUMA / OmniRetarget streamed fetch (range reads), skill tagging |
| 2026-09-29 | e73aad7, d44c6d9 | AMASS filter (run on the laptop; only filtered 62 MB transferred) |
| 2026-09-29 | 1cfc154 | AMASS (SMPL-X) → G1 retargeting with GMR, 938 clips, 0 errors |
| 2026-09-29 | 6bc331a | library v1: per-frame tags, 50 Hz, 89-D features, preview renderer |
| 2026-09-29 | 6ea4ffa | obstacle groups only from obstacle-selected clips; kick-proof step-over |
| 2026-09-29 | ee03436 | Stage 2 code: style prior, style critic, gates, schedule, tests |
| 2026-09-29 | c397cff | offline discriminator check; pilot config `configs/pilots/style_20260929.sh` |
| 2026-09-29 | 34918b1 | gates from geometry probes instead of guidance-field z; config checks accept pre-style checkpoints |
| 2026-09-29 | — | pilot `cat_style_20260929` launched (40,960 envs, warm start pilot 3, no reactive episodes); stopped at update 40 — discriminators trained (d_human +0.7…+0.9, d_robot −0.8…−0.9) |
| 2026-09-29 | d3930cb | switch to **experts + distillation**; expert `expert_rooms_passages_20260929` launched |
| 2026-09-29 | 4b5d12b | DAgger distillation implemented and CPU-smoke-tested |
| 2026-09-29 | c28fb5b | rooms+passages expert split; `--scene-group-override`; rooms+tables expert launched |
| 2026-09-30 | (this) | **goal hold**: room episodes now end 1 s after the goal (see §7a); rooms+tables expert v2 launched |

---

## 1. Motion library (`data/motion_library/library_v1.{npz,json}`)

**Sources**
| source | how obtained | licence |
|---|---|---|
| PHUMA (G1-native) | streamed subset, everyday categories (humanml, idea400, EgoBody, game_motion, custom, humman, GRAB) | Apache-2.0 |
| OmniRetarget (G1-native) | 200 unique box-carry clips (augmented copies removed) | MIT |
| AMASS CMU / KIT / BMLmovi / SFU | filtered on the laptop (`filter_amass_local.py`), retargeted with GMR (`retarget_amass.py`) | non-commercial |

**Processing**: joint order verified identical to our G1 model; per-frame skill tags from our
G1 kinematics (`tag_clips.py`): walk, sidle, duck (head < 1.02 m while travelling, hands off the
floor), crawl, step-over (0.22–0.55 m lift with forward travel), stand, carry; resampled to the
50 Hz control rate; 89-D features (`build_library.py`).

**Style groups** (after cleaning): locomotion 270 min, sidle 7.0 min, duck/step 3.0 min,
crawl 7.7 min (disabled), protect 0 (reserved for own VR capture). Obstacle groups take frames only
from obstacle-selected AMASS clips, never from fighting/kick/dance/jump or box-pickup clips.

**Features** (heading frame, yaw removed): pelvis height; gravity in the pelvis frame; root linear
velocity; root angular velocity (pelvis frame); 29 joint angles; 29 joint velocities; positions of
head, 2 palms, 2 ankles, 2 elbows relative to the pelvis. The simulator computes the same vector
(`style_prior.robot_features`); parity is unit-tested.

## 2. What is trained

| component | trained | how |
|---|---|---|
| policy (actor, 226 inputs → 29 joint targets) | yes, warm start from pilot 3 | PPO |
| task critic | yes, warm start | value loss on task reward |
| style critic (new value head) | yes, fresh | value loss on style reward |
| 3 discriminators (locomotion, sidle, duck/step) | yes, fresh, online | LSGAN + gradient penalty (5), 8 steps × 4096 per update |
| library, gates, task rewards, scenes | fixed | — |

## 3. Context gating (the T-GMP idea)

Every control step, per environment:
- **gap width** from the heading probes (free space at ±0.16 m beside the shoulders, here and one stride ahead);
- **duck** when geometry is within 25 cm of where an upright head goes (distance field sampled at
  1.30 m height, here and 0.3 m ahead); **step** when geometry is at shin height (0.25 m) 0.25–0.45 m
  ahead. (First version used the guidance fields' vertical components; measured non-zero almost
  everywhere → duck/step gate open in 57–69% of steps. Geometry probes: 0% in rooms/tables,
  13–38% in CAT hurdle/beam scenes, sidle gate 43% in narrow passages.)

Gate weights (sum to 1): `w_sidle` ramps 0 → 1 as the gap narrows 0.75 → 0.55 m;
`w_duck_step` = max(duck, step) ramps (overhead distance 0.25 → 0.05 m, shin distance 0.15 → 0.05 m); `w_locomotion` = the rest, so a
discriminator is always active (fallback = everyday human motion; never "task only").

Style reward: `r_style = Σ_g w_g · max(0, 1 − ¼ (D_g(f_t, f_t+1) − 1)²)` ∈ [0, 1].
Each D_g trains on human transitions of its group vs robot transitions from steps where
`w_g ≥ 0.3` (replay buffer, 10% of steps stored).

T-GMP's generative part (a terrain-conditioned CVAE that *generates expert samples for the robot's
current terrain*) is planned for when paired (motion, geometry) data exists, i.e. after our own VR
capture in the procedural scenes. It does not change the action space; it only replaces the
discriminator's expert source.

## 4. Combining with the task

- Two critics; total advantage `A = norm(A_task) + λ · norm(A_style)`.
- λ schedule: 0 for 20 updates (discriminators and style critic warm up), linear ramp to 0.3 over
  50 updates. **Guard**: while CAT navigation, narrow-zone, protected-zone or clutter-room success
  is > 5 points below its value at ramp start, λ backs off by 10% per update instead of rising.
- λ = 0 leaves the PPO loss exactly unchanged (unit-tested).

## 5. Checks before training (done)

- Feature parity simulator ↔ library: < 5e-3 on 25 sampled frames.
- **Offline discriminator check** (`offline_discriminator_check.py`), locomotion group vs 11
  recordings of the current policy (held-out data):

  | discriminator input | human reward | robot reward |
  |---|---|---|
  | all features | 0.999 | 0.158 |
  | positions only (no velocities) | 0.979 | 0.203 |

  Largest differences robot vs human: lateral velocity (sideways walking), waist pitch (hunch),
  elbow / palm positions (arm posture), ankle roll — the defects seen in the videos.

## 6. Metrics (wandb)

`style/reward_mean`, `style/lambda`, `style/guard_held_metrics`, per group
`style/<g>/gate_share`, `style/<g>/reward_when_gated`, `style/<g>/d_human`, `style/<g>/d_robot`,
`style/<g>/d_loss`, `style/<g>/replay`, `learner/style_v_loss`, plus all existing task metrics.

## 7. Experts + distillation (decided 2026-09-29)

Joint training of all tasks shares one reward-weight set and one experience budget. Measured cost:
hand-clearance −60 protected hands but cut narrow-passage success 53% → ~30%; starved groups
eroded (CAT 18% → 7%, narrow 63% → 45%); CAT success flat at ~12–14% for all pilots, while the
released CAT generalist (itself experts + distillation) beats ours on CAT scenes (0.235 vs 0.193,
3,200 paired episodes). Gradient cosines between tasks are positive, so the conflict is in shared
weights and data, not gradients.

| expert | scenes | start | status |
|---|---|---|---|
| CAT navigation | CAT procedural / original / published | released CAT generalist | not retrained |
| ~~rooms + passages~~ | (combined; stopped at update 36) | style pilot, update 40 | split: one weight set was a compromise between tucked hands (passages) and raised/kept-away hands (tables) — weakest skills table edges 30%, narrow 41% at update 30 |
| **rooms + tables** | clutter, furniture, table edges | combined expert, update 30 | **training** (`configs/pilots/expert_rooms_tables_20260929.sh`): hand clearance −40; furniture .25, clutter .43, table edges .30 (own group), flat .02 |
| passages | narrow / protected / transition / open | combined expert, update 30 | next: mild hand clearance, sidle style |
| standing / reactive | later (after VR capture) | — | — |

Expert config: experience masses 0 for CAT groups; furniture .15, clutter+passages+tables .45,
narrow .22, flat .02, protected .08, transition .08; same rewards and style prior as the pilot.

### Distillation (implemented: `cat_mjlab/distill.py`, config `configs/pilots/distill_cat_rooms_20260929.sh`)

- **Experts** (frozen): `cat` = released CAT generalist (`.npz`, uses the first 222 observation
  features), `rooms` = rooms+passages expert (native checkpoint, all 226 features).
- **Routing** by scene type: CAT procedural / original / published → `cat`; everything else → `rooms`
  (`--distill-route NAME=BUCKET,...` to change).
- **Student**: the ordinary native actor, warm-started from the rooms expert, full scene mix.
- **Each step**: all experts compute their deterministic action means; the env's owner provides the
  label. The executed action is the expert's with probability β (drawn per env at episode start),
  else the student's mean + small noise. β decays 1 → 0 over `--distill-beta-decay` updates (60).
- **Each update**: supervised MSE of the student's action mean onto the labels, over the newest
  rollout + 2 previous (replay), 2 epochs; no reward is used for learning, but all task/success
  metrics are still logged. Metrics: `distill/<expert>/action_mse`, `distill/<expert>/sample_share`,
  `distill/beta_next`, `distill/expert_driving_share`.
- **Checks**: unit tests (label bookkeeping, β = 1 executes the expert, regression reduces the gap);
  CPU smoke test (64 envs, 3 updates): CAT action MSE 0.082 → 0.065, β schedule as configured.
- **Not run yet on the GPU** (the rooms+passages expert is training there).

The same DAgger machinery later distils into the LiDAR voxel student.

## 7a. Room episodes never ended at the goal (found 2026-09-30)

Rooms+tables expert v1 (`outputs/expert_rooms_tables_20260929`): clutter success looked like 81% and then
"fell" to ~50%. Cause: reaching the goal did not end a room episode, so a robot that arrived at step ~500
stood parked at its goal (route complete -> zero guidance, stand command) until the 4000-step horizon.
- `reward_term/stand_still` grew to -3.1 per step = at least 64% of all robot-steps were robots parked on a stand command;
- fresh clutter_dense attempts fell from 91 to ~5 per update (navigation learning starved);
- check: clutter_pilot 77% success x 4000 steps + 23% x ~500 = 3195 predicted vs 3198 logged mean length;
- the start offsets are randomized over 0-1000 steps, so the first parked cohort timed out together at
  update ~95 (timeouts 40 -> 830 per update) and the success metric dropped while new attempts restarted.
- ruled out: robots losing the route ("blocked") -- 0 blocked robots in a CPU diagnostic
  (`scripts/diagnose_route_blocking.py`); the opt-in `--route-recovery-speed` stays off.

Fix: `--goal-hold-seconds 1` ends the episode 1 s after the goal as a truncation (value bootstrapped, so
arriving early loses nothing). Smoke test (4096 envs, 20 updates): room episodes end ~1 s after the goal,
stand_still -0.003 vs -0.64 at the same point in v1. New metrics per scene: `route_lost_rate`,
`route_blocked_share`, `episode_over_1000_steps_share`. Run: `configs/pilots/expert_rooms_tables_v2_20260930.sh`
(identical to v1 otherwise), `scripts/keep_snapshots.sh` keeps resume.pt every 20 updates.
Still open: table edges -- heading gate and style gates probe only shoulder/head/shin height, blind to 0.71 m tabletops.

## 7b. Hands vs the rest of the robot (2026-09-30)

Before: a hand could only physically hit the thighs/shins (10 explicit pairs); the other hand, torso,
pelvis, head and other arm had no collision, so hands passed through them and nothing priced it.
Measured on the v2 update-380 recordings with the new pairs: hand-hand contact in 3.5% of frames,
hand vs other arm's wrist ~1%, hand vs pelvis 0.3%; 3 of 12 episodes, the other 9 none.

`--hand-body-contact-weight W` (opt-in):
- physics: the collision-proxy shapes of the trunk (4 boxes), head (1 box) and arms (capsules) become
  massless non-colliding geoms, with explicit contact pairs against each hand's envelope box (every Dex3
  mesh triangle + 5 mm): trunk, head, the OTHER arm, and hand vs hand (25 pairs; own arm excluded);
- reward: W per touching pair, CONTACT ONLY (penetration, no distance margin) -- tucking a hand close
  to the body or near the other hand stays free; the 4 cm anticipation margin stays for the legs only
  (they swing into the hands while walking);
- metrics: `self_clearance/hand_contact_{trunk,head,arm,hand}` (share of steps), episode flag
  `hand_body_contact`. Cost: +1% simulation step time (CPU benchmark).

## 7c. Style vs task, smoothness, table gates (v3-v5, 2026-09-30)

- v2 analysis: robot joint accelerations 3-14x human; the discriminator scored vibration (style reward
  flat ~0.3 at every speed). v3: action std cap 0.25, action-rate -0.02, joint-acc -5e-6, discriminator
  on pose features with 0.1 s stride, gradient penalty 10, 4 steps. Jitter only -17%; style reward held.
- v4 (warm start v2 update 200, + hand-body contact -2): success fell during the style ramp again
  (rooms 81 -> 70%, tables 76 -> 42%); third run with the same pattern; the guard never fired.
- v5 (warm start v4 update 20): style weight 0.1; guard on table_edges/clutter_dense/furniture_dense
  success averaged over 10 updates (`--style-guard-watch/--style-guard-window`), warmup 30;
  hand-body contact -10; `--heading-hand-probes` (0.44 m table gaps: shoulder clearance +0.42 m kept the
  forward gate open, hand clearance -0.08 m now closes it; 1.5 m gaps unchanged).

## 7d. Hunching near room goals (2026-10-01)

Recorded rollouts: median torso pitch 1-2 deg in every version, but within 1 m of the goal the torso
exceeds 15 deg in 54-60% of frames (p95 ~54 deg); beyond 2 m: 0%. Room fields were generated toward a
goal at z=0.75 m, so at head height the field points down near every goal (z -0.86 at 0.25 m, -0.4 at
1 m). That opened the crouch gate (upright/stand_tall off: 17-21% of all training steps) and headgf paid
for moving the head down toward the goal. Likely worse since goal hold: the approach zone is now ~1/4
of room experience instead of a sliver next to 70 s of parking.
Fix `--head-guidance-near-sdf 0.3`: in rooms the head field keeps its vertical component, and the crouch
gate its field condition, only with geometry within 0.3 m of the head (gate open near goals 100% -> 9%,
40 random rooms). Ducking under real geometry is unchanged. v6 = v5 + this, warm start v5 update 340.

## 7e. Teleoperation: recordings, live demo, joystick training (2026-10-01)

- `task.joystick` [N,2] (world-frame velocity) replaces the room route; recorder `--joystick`,
  `--joystick-toward top`, `--joystick-speed`, `--joystick-seconds` (then release).
  v5 final, 4.5 s straight at a table: 0.5 m/s no contact in 2/5 scenes (closest 0.14-0.18 m, deflects
  sideways), 0.8 m/s 0/5. Not a trained skill yet.
- `cat_mjlab/sim_cpu.py`: one world on plain MuJoCo with the CATSimulation interface (float32 mirrors);
  the unchanged training task runs on it. Parity vs MuJoCo Warp from the same state: observations
  agree to 6e-8, root paths 3 mm apart after 1 s, 4 cm after 2 s; both reach the same contact.
- `scripts/demo/teleop_demo.py`: keyboard in the terminal (ssh), robot + furniture in viser
  (`ssh -L 8080:localhost:8080 dep-1`, http://localhost:8080), on-screen buttons/slider/scene list,
  live clearance and contact readout. 18 ms per control step = 1.1x real time (obstacle check every
  other step, rewards skipped, no pushes/noise/PD randomization, contact shown not terminal).
- `--teleop-fraction F`: F of room episodes follow random joystick commands for 20 s (0.3-0.8 m/s,
  15% stops, half aimed at the nearest obstacle, new command every 1.5-4 s); contact ends them.
  Own `teleop` bucket: `scene/teleop/obstacle_contact_rate`, `mean_episode_length`, `fall_rate`.
  Current policy under random commands: 5 of 8 episodes contact within ~4 s.

## 7f. Joystick interface (v7, 2026-10-01)

How the robot was told where to go: CAT computes a world-frame walking velocity from the goal field
(or our room route); the command is [move, vx, vy, yaw_rate] but CAT always zeroed yaw_rate in the
actor observation, so the policy never had a turn input and chose its facing itself.
Fixes found on the way:
- gait restart: standing sets both foot phases to 0; resuming walking commanded a synchronized hop and
  the robot stayed in place (0.5 m/s after a 2 s stand -> 0.01 m/s). restart_phase now applies to every
  route/joystick world. After the fix (v6 upd 60): 0.3-0.5 m/s followed at 0.96-1.23x, top ~0.6 m/s.
- forward-facing reward used pelvis_rpy yaw, which is relative to the heading frame (~0 by
  construction): it compared the command with the world x axis and never depended on the body's facing.
  Now uses the world yaw from the heading frame.
v7 = v6 final (update 98) + `--teleop-fraction 0.25 --teleop-body-commands --tracking-yaw-weight 1`:
body-frame commands (forward -0.5..0.8, sideways +-0.3, turn +-1 rad/s; 20% turn in place, 15% stop,
30% toward the nearest obstacle), turn rate kept in the actor observation (`yaw_command`), turn-rate
tracking reward, no forward-facing bonus in joystick episodes, contact ends them.
Baseline (v6 final, body-frame commands, flat scene): turn-in-place 0.8 rad/s -> 0.32 rad/s and 0.37 m
drift; turn right -> turned left; backward -> turned around 140 deg; sideways -> turned 37-158 deg.
Test: `scripts/demo/command_following_test.py` (body segments automatic for v7+). Demo keys for v7+:
W/S forward/back, A/D turn, Q/E sideways.

## 7g. Joystick interface v8: reference heading + clearance gate + obstacle-safe tracking (2026-10-01)

v7 (turn-rate command, stopped at update 12) had an ambiguity: turn rate 0 meant "choose your facing"
in route episodes and "do not rotate" in joystick episodes. v8 uses one command for both:
- walking velocity fixed in the world; reference heading psi_ref (route: direction of travel, i.e.
  CAT-style point navigation; joystick: integrated from the operator's turn rate);
- the policy's turn slot (command[3], always zeroed by CAT) carries gate * wrap(psi_ref - facing);
  gate = would a body facing psi_ref fit (shoulder probes +-0.16 m, hand probes +-0.22 m minus the
  hand radius, beside the root and 0.3 m ahead along the travel direction; 0 at <= 5 cm, 1 at >= 15 cm).
  Open space: orientation held (strafe, walk backward, turn in place). Narrow gap: orientation free,
  the policy may turn sideways to pass, then returns to psi_ref;
- velocity reward target = command minus its component into obstacles within 0.25-0.6 m of the pelvis,
  hands, knees or shoulders (CAT's command projection used as the reward target): driven at a table,
  stopping or sliding along it earns the reward; along a gap the command survives;
- heading reward = gate x 0.5(1 + cos(psi_ref - facing)), weight 1.0 (replaces the forward-facing bonus
  for route/joystick worlds);
- joystick episodes 50% (forward -0.5..0.8, sideways +-0.3 m/s relative to psi_ref, turn +-1 rad/s,
  20% turn in place, 15% stops, 30% toward the nearest obstacle), contact ends them.
CPU check (v6 policy, 8 worlds): gate mean 1.00 flat, 0.92 open tables, 0.82 furniture, 0.68 narrow
tables (closed 32% there). Metrics: heading/heading_gate, heading/heading_error_abs.
Demo (v8+ automatically): W/S forward/back, A/D turn the reference heading (also in place), Q/E
sideways; yellow arrow = reference heading; status shows whether orientation is held or free.

## Milestone 2026-10-02: rooms + keyboard control (tag `milestone-2026-10-02-rooms-keyboard-v9`)

Policy `checkpoints/cat_g1_rooms_keyboard_v9_u492/` (v9 update 492; actor 292,474 parameters, 1.2 MB,
same size as released CAT). Route following: clutter/furniture dense 76%, pilot rooms ~90%, falls 0.1%;
8/8 clutter videos without pushes. Keyboard (heading-reference interface): turn in place ~80% of the
commanded rate (25-35 cm drift), forward + steer good; backward/sideways not learned; table edges
degraded to 23% in v9 (obstacle-safe target applied to route worlds incl. hands -- fix planned).
Tooling: CPU single-robot simulator (parity with Warp), browser + keyboard demo, command-following
test, teleop recordings, no-push recordings.

## 7h. Voxel student for the real G1 (head Mid-360 only), 2026-10-02

Gallant (CVPR 2026) reimplemented in our mjlab stack (its code is Isaac-based, no checkpoints):
- `cat_mjlab/lidar.py`: the G1's head Mid-360 (MJCF `mid360_site`, mounted upside down: vertical field of
  view 52 deg down .. 7 deg up; floor visible beyond ~0.96 m, shin-height obstacles from ~0.7 m). 10 Hz
  scans of 2048 random rays in the field of view (non-repetitive pattern), sphere-traced on the scene SDF,
  plus the floor plane; the robot's collision-proxy primitives occlude rays (body hits dropped, as the real
  pipeline filters them). Gallant randomisation: sensor pose N(0,1 cm)/N(0,1 deg) per episode, hits
  N(0,1 cm), latency 1-2 scans, 2 % voxel dropout. Points of the last 10 scans kept in world coordinates
  and voxelised in the robot frame every step (needed with one head sensor: the feet area is seen only
  while approaching). Grid: 32x32x40, 0.05 m, x/y +-0.8 m, z +-1.0 m around the pelvis, yaw-aligned.
- `cat_mjlab/student.py`: proprioception + command (teacher features 0..130, raw heading error instead of
  the map-based gated one) + z-grouped 2D CNN on the voxels; 497k parameters.
- `scripts/student/train_voxel_student.py`: DAgger from the field experts (rooms+keyboard v9 for rooms/
  tables/flat/joystick, CAT expert for CAT scenes), beta 1 -> 0.1 over 150 updates, replay of 3 rollouts,
  per-scene success/contact of student-driven episodes. CPU smoke: loss 0.16 -> 0.06 in 2 updates.
- `scripts/student/lidar_preview.py`: images of LiDAR memory + voxel grid (outputs/lidar_preview).
- Real robot (to build): Mid-360 -> FAST-LIO2 (odometry + deskewed world points) -> drop points on the
  robot (kinematics) -> keep the last ~1 s -> same VoxelMemory.grid in the pelvis frame at 50 Hz -> student.
  Route/goal heading from a map-based planner or the keyboard (same command as in training).

## 7i. CAT expert and keyboard v10 (2026-10-02/03)

**CAT expert.** v1 started from rooms+keyboard v9: 3% CAT success after 60 updates, 91-97% of episodes
ending in contact (v9 had forgotten CAT). v2 starts from the released CAT generalist, converted with
`scripts/convert_released_cat_to_native.py` (identical actions after widening 222 -> 226 inputs), body motion
back to -0.5, upright on pitch only. Training success reached ~34% in all three CAT groups by update 616.
Released CAT under our rule (paired eval, 3200 episodes, deterministic) scores 23-25%: it "fits" mostly by
touching -- side scenes 15% clean / 72% contact (hands 33%), crouch 7% clean / 87% contact (head 60%).

**Demo course** (`scripts/demo/build_obstacle_course.py`, bank `data/furniture/course_v1_*`): 25 m corridor,
hurdles 10/20 cm, beams 1.16/1.02 m, side gaps 0.44/0.34 m, combinations, tables with a 0.60 m gap. CAT-type
scene (no room route certificate, which would forbid gaps < 0.46 m). Field-driven, the CAT expert passed
hurdles and both beams and stopped at the 0.44 m gap; keyboard-driven it stalled before the 1.02 m beam.

**Diagnosis.**
- Safe velocity target normalised the 3-D field's horizontal part to unit length: a beam overhead or a hurdle
  underfoot became "wall ahead", target stop.
- Heading gate (shoulder/hand probes on the 3-D SDF) read a beam at shoulder height as "too narrow to face",
  released the heading, the robot turned sideways (~70 deg) in front of the beam.
- Teleop worlds were drawn only from scenes with room navigation: in the CAT expert that is the flat scene,
  ~44 episodes/update. It never drove a CAT obstacle by keyboard.
- Idle stepping: the gait flag turned on above 0.05 rad heading error; with no command it was on 100% of
  the time (28 touchdowns, 1.2 m drift in 20 s). With the error held at 0: 2-4 touchdowns.
- Side gaps: 9 cm hand target with 20 cm anticipation cannot be held on both sides of a 0.34-0.44 m gap.

**v10 (`--keyboard-v10`, `--hand-clearance-tight 0.10 0.40 0.3`, run `expert_cat_v3_20261003`).**
- `cat_mjlab/blocking.py`: per-scene 2-D blocking footprint, occupied anywhere in 0.25-0.95 m; 2-D signed
  distance + direction, built from the bank SDF at load. Used by the safe target and the heading gate.
- Safe target in joystick worlds only.
- Heading deadband 0.20 on / 0.08 off rad (hysteresis); a held turn key turns at once.
- Teleop in CAT scenes; THROUGH commands (heading and walk along the stored goal field at the root) with
  p 0.7 in CAT scenes, 0.3 elsewhere; p_stop 0.15 -> 0.30, a stop sets the reference heading to the facing.
- Hand/elbow clearance targets x room scale: pelvis distance to the blocking footprint, 0.3 at <= 0.10 m,
  1 at >= 0.40 m. Contact still terminates.
- New metrics bucket `scene/teleop_cat/*` (keyboard episodes in CAT scenes, success = clean crossing):
  6.9% success, 51% contact at the start of v3. Tests: `tests/test_keyboard_v10.py`.

## 8. Not yet done / next

- Reference-state initialisation from sidle / duck clips (planned; not in the first pilot).
- Protect group and approaching-object episodes (after own VR capture; out of scope now).
- Full T-GMP CVAE (after paired capture).
- Crawl group (simulation only).
- LiDAR voxel student (Gallant-style) via DAgger for hardware.
