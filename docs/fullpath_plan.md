# Full-path planning for the planner–reflex system (log, opened 2026-11-25 09:40 +04)

Branch `feature/fullpath-plan` (from `feature/energy-price` 16ad2fe, with `feature/eval-overnight` c9d981d merged for
`--subset-fields`, value selection and `cat_mjlab/diagnose.py`). Worktree `~/robotics/cat-energy-fullpath`.
Durable copy: `~/.local/share/openresearch/files/humanoid-protection/fullpath/fullpath_plan.md` (copied after each result).

Diagnosis to test (user, 2026-11-25): a 2 s chunk (0.3–0.6 m at 0.15–0.3 m/s) is too short to plan over an obstacle; the plan
must cover the whole traversal; legs must be followed at hurdles; being close to an obstacle is fine while following a verified
plan and dangerous only when deviating from it.

Stages: S1 diagnostics (evaluation only) → S2 oracle full-path upper bound (gate) → S3 tracker v2 fine-tune → S4 learned
full-path planner. Every experiment is pre-registered here before it runs.

## Compute situation at 09:40

dep-1 GPU: held under `/tmp/cat-energy-dep1-gpu.lock` by another session's SONIC training (440/1400 updates, ≈2 h left), with
two more batches (improve D2, eval failure diagnosis) waiting on the lock. Cluster: the per-user limits are full (ws-ia 2 jobs,
gpu 1 GPU, all held by other sessions' jobs, five arrays pending). S1 and S2 therefore run on **CPU MuJoCo** (one world per
process, `cat_mjlab/sim_cpu.py`, ≈25 control steps/s with the planner on the CPU, 2.2 GB RSS per process), 18 processes in
parallel with a free-RAM guard. CPU and Warp physics differ in floating point and the CPU task checks body collisions once
per control step instead of every substep, so every comparison below is made **inside the CPU protocol** (same simulator for
all arms), and the CPU plan-on rates are checked against the GPU benchmark of the same checkpoint (sanity check, not a gate).

## Benchmarks (fixed for S1–S2)

* **F144** – the 144 scenes of the 384-scene list in the families the brief names: 40 side gaps (`sidegap*`) and 104 CAT
  courses (`D8*`, `procedural-*`, `published-*`: hurdles, beams/crouch bars, side hurdles, multi-hurdles), bank `tablegap_v1`,
  goal-column cache, nominal, deterministic, 2 episodes per scene (trial seeds 1 and 2 of the task's own reset), speeds 0.15 and
  0.30 m/s → 576 episodes per policy.
* **TG24** – the 24 held-out table-gap layouts of the E4 planning benchmark (`/tmp/cat-overnight/planning_v1`), both certified
  starts, 0.15 and 0.30 m/s → 96 episodes per policy.
* Outcomes as in `eval_per_scene.py`: clean (goal, no contact), contact (any body–obstacle contact of the 35-part proxy),
  refused (timeout without goal/contact/fall), fall.

## S1 – diagnostics of planner–reflex seed 0, plan on (pre-registered 09:40, before any S1 number)

Policy: COMBO_FIX seed 0 (`COMBO_FIX_315808_u150.pt`), `--mix plan=1`, planner v1 as trained (4 candidates, clearest rule,
continuity). Recorder: `scripts/fullpath_rollout.py` (one episode per process) stores per control step the robot qpos,
plan clock, plan validity and reflex release, and per replan the chosen 21-frame whole-body chunk and all candidates
(world qpos), and the first contact (step, regions, obstacle box). Analysis: `scripts/fullpath_audit.py`.

Measures:
* **(a) Lookahead.** At the first contact (contact episodes) and at the stall onset (refused episodes: the last step at which
  the best route progress improved by ≥ 5 cm), along the start→goal direction: `need` = distance from the pelvis to the far
  end of the obstacle (the contacted box; for stalls the box with the smallest exact body separation), `reach` = distance from
  the pelvis to the last frame of the executing plan. Reported: median need, median reach, share of events with
  reach < need ("the plan ends before the obstacle does").
* **(b) Plan audit under exact geometry.** Every chosen chunk and every candidate is checked with the exact episode-ending
  geometry (35 proxy primitives — boxes, capsules, spheres — against the scene's collision boxes, `cat_mjlab/collision.py`
  separations over all boxes of the scene) at all 21 frames, and over the executed part (frames up to the next replan).
  Reported: share of chosen chunks that collide (min separation < 0), share whose executed part collides, distribution of
  the min margin (p5/p50), share of replans where all 4 candidates collide, and the sphere-cover score vs exact agreement.
* **(c) Tracking error vs plan.** At every step with an active plan: pelvis (xy and z), hands (left/right `*_wrist_yaw_link`
  bodies… the hand bodies used by the collision proxy), feet (ankle-roll links) world positions of the robot vs the plan pose
  at the current plan time (FK of the interpolated plan frame). Reported p50/p95, all steps and steps within 15 cm of an
  obstacle.
* **(d) Failure split.** For each contact episode, at the first contact, with the plan pose at that plan time:
  *bad plan* = the executing plan's exact separation over ±0.2 s around the contact plan time is < 0 (the plan itself goes
  through the obstacle), or every candidate of the last replan collides; *bad tracking* = plan separation ≥ 0 and the pelvis
  or hand error exceeds its p95 from (c) (measured on clean episodes); *conflict / untracked deviation* = safe plan, tracked
  errors ≤ p95, contact anyway (reported with the contacting region and whether the reflex was releasing an arm);
  *no plan* = no valid plan at contact. Refusals are split into *plan stops* (executing plan advances < 5 cm over its next
  1 s), *robot stops on a moving plan*, and *no plan*, each with (a).

Predictions (the user's diagnosis): (a) reach < need in ≥ 60% of contact/stall events on courses and side gaps;
(b) ≥ 20% of executed chunks before a contact collide under exact geometry, and ≥ 10% of replans near obstacles have no
collision-free candidate; (d) among course (hurdle/beam) contacts, the largest share is legs/feet contacts on a safe plan
(untracked deviation). Falsifiers: reach ≥ need in most events (the horizon is not the limit), or bad tracking of tracked
channels dominates (the plan is fine and the tracker misses it), or bad plans are < 10% of failures.
S1 has no gate; it decides how S2 is built (margin = p95 tracking error from (c)).

Additional S1 rollouts on the same benchmarks (needed as S2 baselines and posture library, recorded with the same tool):
the same network with `--mix none=1` (plan-off), and the reactive E2 u300 policy (`stationary_hands_E2_u300.pt`, 43% of
courses clean in mid-November) whose clean traversals seed the posture library.

## Run log

(entries appended below)

### 09:20–09:45 setup notes
* The recorder also logs the first terminating obstacle cause: besides the 35-primitive proxy contact, an episode is a
  contact when one of the 13 policy body points reads the scene field below −4 cm, an elbow probe does, or a routed world
  leaves its corridor (`task._termination`). Such events take the terminating pose and the nearest box. The first batch
  (09:10, 36 episodes) lacked this and was discarded and restarted at 09:20 (`/tmp/cat-fullpath/batch_s1.log`).
* Added to (c), descriptive and not pre-registered: a **path-relative** tracking error (error to the plan frame nearest the
  robot's pelvis, i.e. without along-track lag), because with plan continuity the time-indexed error mostly measures how far
  the plan has run ahead of a stalled robot. S2's distance-indexed feed is judged by this one.
* Smoke values (17 episodes, not results): time-indexed pelvis error p50 ≈ 15 cm, hands ≈ 35 cm; path-relative pelvis ≈ 8 cm,
  hands ≈ 35 cm. The arms of the current tracker do not follow the plan's arm targets closely.

## S2 – oracle full-path upper bound (pre-registered 09:45, before any S1 or S2 result)

Planner (`cat_mjlab/fullpath.py`): weighted A* (ε = 4; ε = 2 was too slow in a pilot on six scenes, 10:10) over a motion graph of 0.4 s whole-body clips (21 frames, 50 Hz) cut
every 0.1 s from the **clean** S1 traversals of COMBO_FIX seed 0 plan-on, plan-off and E2 u300 on F144 + TG24 (posture
library; clips are stored in their start pelvis frame and labelled sideways / crouch / high step / hands raised / hands
tucked). From each episode's actual start state, successors are the 32 clips nearest in a continuity feature (leg and
arm joints, pelvis height/roll/pitch, local velocity), placed at the current pelvis frame and accepted only if the exact
separation (35 proxy primitives vs every collision box of the scene) is ≥ margin at every other frame and at the end. The
plan is the full start→goal whole-body trajectory (pelvis SE(2) path, posture schedule, and the clips' own foot placements
and swing apexes), re-checked end-to-end at every frame. Margins tried in order 8, 6, 4, 2 cm (150 s search budget each);
the first success is used. The pre-registered rule "margin ≥ p95 tracking error" cannot hold if (c) gives p95 errors of
tens of cm; the achieved margins and the oracle-feed tracking errors are reported side by side.
Deviation from the brief, stated up front: no foot IK — the clips carry the policies' own foot trajectories and the current
network receives no leg targets anyway; leg references are S3's job.

Feed (`OracleFeeder`): every 0.1 s, the oracle time whose (0.5 s-smoothed) pelvis is nearest the robot, searched forward
from the last match, gives a 21-frame chunk re-timed to the operator speed; it enters through the tracker's plan source
(pelvis waypoints 0.1–1 s, heading, height, upper-body targets at 0.3 s): the current COMBO_FIX network unchanged. If no
margin gives a plan the episode runs without a reference (reactive mode of the same network) and counts in the oracle arm.

Arms (CPU protocol, seed 0, F144 + TG24, 0.15 and 0.30 m/s, two trials): oracle-guided vs plan-off and plan-on from S1,
paired by scene × speed × trial.
Metrics: clean, contact, refused; per family (side gaps, courses split hurdle / beam / both, table gaps); oracle success
rate and achieved margin; path-relative tracking error under the oracle feed.
**Gate (as the brief):** longer-horizon planning is validated if oracle-guided clean ≥ plan-off clean + 5 pp pooled over
F144 + TG24, with the paired scene-cluster bootstrap 95% CI of the difference above 0; reported per family. Seeds 1–2 of
COMBO_FIX are run on the same protocol if seed 0 passes or misses the bar by < 2 pp.
Prediction (user's diagnosis): the gate passes, mostly through side gaps and table gaps (where the plan-on refusals and
contacts come from short, colliding chunks); courses gain less because the current network cannot follow leg plans.
Falsifier: oracle-guided clean < plan-off + 5 pp, or the oracle plans are verified collision-free yet contacts stay at the
plan-off level (the network cannot exploit a correct full plan through its current 41 inputs).

## S1 results (10:30, COMBO_FIX seed 0 plan-on, CPU protocol, 672 episodes; `/tmp/cat-fullpath/audit_fix0_plan.json`)

Exact-geometry check validated first: over 40 clean plan-on trajectories the robot's own poses never read < 0 (min 1.4 mm,
median of episode minima 5.3 cm) and every proxy first contact reads ≤ 0 (−0.2 to −1.4 cm).
CPU vs GPU (same checkpoint, same scenes; sanity only): clean F144 courses 21.4% CPU vs 13.0/15.9% GPU (0.15/0.30), side
gaps 42.5% vs 31.2/37.5%, table gaps 63.5% vs 58.3/54.2%. The CPU protocol (collision checked once per control step) is
more lenient by ~5–8 pp; all S1/S2 comparisons stay inside it.

| | F144 (576) | side gaps (160) | courses (416) | table gaps (96) |
| --- | --- | --- | --- | --- |
| clean / contact / refused % | 27.3 / 55.9 / 16.3 | 42.5 / 38.1 / 19.4 | 21.4 / 62.7 / 15.1 | 63.5 / 32.3 / 4.2 |
| (b) chosen chunk collides (exact, any of 21 frames) | 70.5% | 73.0% | 69.4% | 6.2% |
| (b) executed part collides / near obstacles (<15 cm) | 45.8% / 58.4% | 50.2% / 57.3% | 43.8% / 59.6% | 3.2% / 4.5% |
| (b) all 4 candidates collide / near obstacles | 69.9% / 80.7% | 72.5% / 80.7% | 68.7% / 80.7% | 5.6% / 7.2% |
| (b) executed-part min margin p50 | +0.9 cm | 0.0 cm | +1.6 cm | +7.9 cm |
| (b) sphere score ≥ 0 but exact < 0 | 0.6% of candidates | | | 0.07% |
| (a) contacts: need / reach p50; reach < need | 0.23 / 0.47 m; 22.6% | 0.28 / 0.57 m; 14.8% | 0.21 / 0.44 m; 24.4% | 0.60 / 0.64 m; 45.2% |
| (a) stalls: need / reach p50; reach < need | 0.38 / 0.50 m; 51.1% (94) | 0.47 / 0.37 m; 58.1% | 0.38 / 0.52 m; 47.6% | 0.84 / 0.40 m; 4 of 4 |
| (c) pelvis xy error p50 / p95, time-indexed | 18.1 / 50.1 cm | 18.3 / 50.9 | 18.0 / 49.7 | 16.1 / 50.3 |
| (c) path-relative pelvis / hands / feet p50 | 11.6 / 33.8 / 21.2 cm | 11.2 / 36.4 / 20.8 | 11.8 / 32.9 / 21.4 | 9.2 / 51.3 / 20.1 |
| (c) path-relative pelvis / hands / feet p95 | 42.7 / 88.2 / 61.1 cm | 43.6 / 92.9 / 61.7 | 42.4 / 84.8 / 60.7 | 42.3 / 103.8 / 57.0 |
| (d) contacts: bad plan / bad tracking / conflict / no plan | 81.5 / 3.4 / 15.0 / 0% (319) | 68.9 / 11.5 / 19.7 / 0 (61) | 84.5 / 1.6 / 14.0 / 0 (258) | 25.8 / 35.5 / 38.7 / 0 (31) |
| (d) refusals: plan stops / robot stops on a moving plan | 14.9 / 85.1% (94) | 6.5 / 93.5 | 19.0 / 81.0 | 0 / 100 (4) |

Contacting regions on bad plans (F144): feet 114, legs 57, head 33, hands 27, arms 20, trunk 13 — 70% lower body.
Descriptive (not pre-registered): 80.6% of F144 contacts come right after a replan in which **every** candidate collided;
of the 59 contacts on a safe plan, all 58 with a primitive-level record had the touching part farther from its planned
position (median 32 cm) than the plan's own margin (median 4.2 cm) — i.e. they happen while deviating from the plan.

Against the predictions: (b) **met** (≥ 20% of executed chunks collide: 46–50%; ≥ 10% of replans with no collision-free
candidate: 70–81%). (a) **not met**: at the first contact the plan's reach already passes the obstacle's far end in 77% of
events (contacts happen at the obstacle, where 2 s still covers it); at stalls reach < need in 51% (F144) and in all table-gap
stalls. (d) **not met** as worded: course contacts are mostly *bad plans* (84.5%), not untracked deviations on safe plans
(14%) — though 70% of bad-plan contacts are feet/legs, i.e. the plans collide with their legs, which the tracker does not
follow anyway. Falsifier "bad plans < 10% of failures" not triggered; bad tracking of the tracked channels is rare (3%).
Reading: the dominant failure is that planner v1 cannot produce a collision-free 2 s whole-body chunk in clutter, side gaps
and hurdles from where the robot stands (all 4 samples collide 4 times out of 5 near obstacles), and the clearest-but-
colliding chunk is executed; the horizon itself mainly matters for stalls. Tracking of the plan is loose (pelvis 12 cm,
hands 34 cm median path-relative), so the S2 margin rule "≥ p95 tracking error" (pelvis 43 cm) is infeasible; S2 keeps the
pre-registered 8→2 cm margin schedule and reports the oracle-feed tracking error.

### S2 amendment (11:20, after the first 55 oracle searches, before any oracle-guided rollout)
The motion-graph search finds plans for only 8 of the first 55 starts (dense D8 clutter courses; all margins exhausted in
most). Since starts are identical across speeds and policies for a scene and trial (checked on 60 pairs), every clean S1
traversal from the same start (pelvis within 1 cm; E2 u300 resets differently and mostly does not qualify) is itself a whole-traversal, collision-free whole-body path. The oracle's candidate set is
therefore the searched plan **plus** the recorded clean traversals of the three S1 policies from the same start (any
speed), each re-verified with the exact geometry; the one with the largest verified margin is used if that margin is
≥ 2 cm (searched plan preferred on ties). The source of every plan (searched / recorded, and which policy) is reported, and
the gate is also reported on the searched-only subset. This makes the oracle a hindsight upper bound (it may reuse a
traversal that succeeded at the other speed or with another policy); episodes without any plan still run reactive.

### S2 search fix (11:50, before any oracle-guided rollout)
The first search pass (11:03–11:42, 179 of 336 starts) failed in every procedural course for two reasons found in the logs:
(1) the field-grid bound also constrained x with a 22 cm pad, which rejected the start pose itself in scenes whose start lies
near the grid's x edge — but the episode's own goal check bounds only y in these plane-goal scenes (`task._goal_status`);
(2) ~1 s per expansion in dense scenes. Fix: lateral-only bounds (pelvis 15 cm inside) for plane-goal scenes, xy only for the
table-gap layouts; collision check in two stages (4 frames, then every 2nd frame of the survivors). The 67 plans already
found are kept (found under the stricter bound); the 141 failures and the unfinished starts are searched again.

## S3 – tracker v2 (pre-registered 12:05, before S2's oracle-guided rollouts; training is launched only after the S2 gate)

Code (`cat_mjlab/fullpath_tracker.py`, hooks in `cat_mjlab/task.py`, `--tracker-fullpath JSON` in `train_cat_mjlab.py`,
`--start-replay/--fullpath-plans` in `scripts/eval_per_scene.py`). Opt-in: configs without `tracker.fullpath` are unchanged.
* **Plans.** A bank of S2 oracle plans (search + recorded alternatives) for 652 training scenes that are **not** in the 384
  evaluation list (320 procedural courses, 160 side gaps, 96 table-edge passages, 76 cabinets; one task-reset start each,
  `scripts/fullpath_starts.py`). At a reset, a share of 0.5 of the worlds becomes full-path worlds: a random plan, its scene
  and its start pose. Moving plan-source worlds without a full plan act without a reference (reactive), as the brief asks.
* **Feed.** Every step the robot's match on its plan (nearest smoothed pelvis, searched forward) and a reference clock that
  advances at the operator speed but at most 0.3 m of plan arc length ahead of the match (a purely distance-indexed
  reference pays full pelvis tracking to a robot that stands still: CPU smoke 11:45); the tracker's existing 41-input block
  is filled from the clock every 0.1 s.
* **New inputs (46, appended, zero-widened warm start):** the path at 0.25/0.5/1/1.5/2/3 m of arc length ahead of the match
  (dx, dy, dz, sin/cos Δyaw), posture-mode code (sideways, crouch, high step, hands raised, hands tucked) + distance to the
  next high step / crouch, and leg references near hurdles (each foot's planned position 0.3 s ahead relative to the pelvis,
  the planned swing apex of the next second, leg mask = a high step within 1 m).
* **Rewards.** `track_feet` = exp(−mean_feet |p_foot − p̂_foot|² / 0.06²) under the leg mask, weight 2 (track_pos is 4);
  plan-relative clearance: on a world following a full plan, every field/clearance cost sees a body point only when its
  clearance is below min(planned clearance − 3 cm, 2 cm), elbows below 2 cm; deviation reflex: an arm is released when its
  hand is closer than planned by more than 3 cm or closer than 2 cm (hard floor); the band rule stays elsewhere. "Obstacles
  absent from the plan map" do not occur in the static training scenes and are not modelled. All other rewards as COMBO_FIX.
* **Training.** Warm start from COMBO_FIX seeds 0, 1, 2 (u150), fresh optimizer, 150 updates at 12,288 worlds (the E2
  fine-tune budget), cluster (ws-ia/gpu, 1 GPU, ≤ 12 CPUs, ≤ 54 GB) or dep-1 under the GPU lock.
* **Evaluation (GPU, identical starts for every arm via `--start-replay`):** the 384 scenes and the 72 held-out layouts,
  2 episodes, 0.15 and 0.30 m/s; arms: tracker v2 with oracle plans, plan-off (COMBO_FIX none=1), plan-on (COMBO_FIX
  planner v1), reactive V7r; 3 seeds where the arm has seeds; paired scene-bootstrap 95% CIs.
* **Success criterion:** tracker v2 clean ≥ plan-off + 5 pp on F144 + TG24 (CI lower bound > 0) **and** contact not higher
  than plan-off on the 384 scenes; seed-level sign agreement in all 3 seeds. Prediction: hurdles/beams gain most (leg
  references), side gaps and table gaps gain through the plan-relative costs. Falsifier: no family gains ≥ 5 pp, or the
  gain comes only from scenes whose plan is a recorded traversal (reported separately).

### S2 feed correction (13:45, before any complete oracle-guided result)
The first oracle-guided batch (12:58–13:35, 410 of 672 episodes) was **discarded**: its feed indexed the oracle by the oracle's
own time from the matched point, so where the oracle trajectory dwells (recorded traversals hesitate for seconds before a
gap) the 0.1–1 s waypoints collapsed onto the robot and the nearest-point match got trapped in the dwell loop — table gaps
then refused 27 pp more than plan-off with the robot never leaving the first 1–2 m. That feed was not distance-indexed as
pre-registered. Fixed feed (`OracleFeeder`): the oracle is indexed by progress u = smoothed pelvis arc length + 0.15 m per rad
of heading change; the robot's match is monotone in u; chunk frames are at u + v_op·(0.1 k).
A pilot of 3 table-gap episodes showed a second effect: with the reference exactly at the robot's progress, the current
network stops in front of the gap where the reference turns sideways, at no tracking cost (it was trained with commit=True
plan clocks that keep running). S2 therefore runs **two feeds on the same 672 paired episodes**: *primary* — a reference
clock advancing at the operator speed, at most 0.3 m (leash) ahead of the robot's match (`--oracle-leash 0.3`; the feed S3
uses); *secondary* — purely distance-indexed (leash 0). The gate is evaluated on the primary feed; both are reported.
Pilot episodes (9, not part of the results): leash 0.3 → table gaps 1 clean / 1 contact / 1 refused vs 3 refused.
* 13:58 S3 GPU smoke 324990 (gpu partition, 3 updates, 4096 worlds, F144 plans as a stand-in bank; code d7787cf at ~/cat_worktrees/fullpath_d7787cf)
* 14:04 S3 GPU smoke 325002 passed: 3 updates, 124 plans in 70 scenes loaded, track_feet active, peak 29.4 GB at 4096 worlds (first attempt 324990 failed on a missing untracked outputs/ file; clones now copy outputs/).
