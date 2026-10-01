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

## 8. Not yet done / next

- Reference-state initialisation from sidle / duck clips (planned; not in the first pilot).
- Protect group and approaching-object episodes (after own VR capture; out of scope now).
- Full T-GMP CVAE (after paired capture).
- Crawl group (simulation only).
- LiDAR voxel student (Gallant-style) via DAgger for hardware.
