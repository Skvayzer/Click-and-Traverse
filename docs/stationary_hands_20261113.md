# Stationary hand protection: overnight experiments, 2026-11-13

**Question.** A standing G1 is approached by objects aimed at its hands. Can it learn to protect the hands by moving
only the arms (raise or tuck), without stepping aside? Moving the hands is cheaper than moving the whole body (less
inertia, fewer motors, less power), so we test that idea in two ways: as a hard rule and as an energy price.

**Start point.** `generalist_robust_delay_v1_u150` fine-tuned in its own setup (robust v1 randomisation, waist
upper-target averaging), anchored to itself everywhere except the flat standing scene. Approaching-object standing
episodes (bank `reactive_standing_v2_20260923`: 2000 rows, danger / anticipation / negative, objects re-aimed at the
live hand) are 50% of experience. Config `configs/pilots/stationary_hands_20261113.sh`, job `hpc/stationary_hands.sbatch`
(MBZUAI Student HPC, ws-ia, RTX 5000 Ada 32 GB, 12288 envs, ~62 s/update).

| run | what changes | Slurm job |
|---|---|---|
| S1 | feet-planted rule `--reactive-stationary 0.05 5 30 0.2`: after settling, a foot > 5 cm from its settled spot, a foot off the floor > 5 steps, or the pelvis > 0.2 m ends the episode as a failure; training pushes suspend the rule and the spots are re-taken once the robot is calm | 295682, stopped at update ~45 (resume point: update 30) |
| S1b | S1 + `--reactive-feet-hold-weight -10`: dense cost (foot travel / 5 cm)^2 while the rule is active | 295727, from 05:11 |
| S2 | energy price `--reactive-power-weight -0.02`: cost per watt of joint power sum abs(tau*qdot) in standing episodes; no rule (old non-terminal 0.3 m spot flag); the feet rule is only measured (shadow mode) | 295683, from 04:21 |

**Evaluation.** `scripts/eval_standing_hands.py`: all 2000 bank rows, one env each, 800 steps (16 s), deterministic
actions, no pushes, no randomisation, feet rule in shadow mode (measured, never ending an episode) for every policy.
An event succeeds when the robot stayed within 0.3 m, never touched the object, and retreated the hand >= 5 cm from a
threatening object (danger, anticipation) or did not flinch for one that passes by (negative).

## Results

Clean deterministic evaluation, 2000 rows x 16 s (about 6400 approach events per policy):

| | u150 (start) | S1b @ update 50 | S2 @ update 90 | **S1b @ update 150** | **S2 @ update 180** |
|---|---|---|---|---|---|
| episodes where a foot moved > 5 cm (or lifted > 0.1 s) | 47.7% | 25.5% | 7.7% | **6.8%** | **6.0%** |
| settled steps with feet displaced | 18.7% | 7.9% | 1.8% | 1.2% | 1.4% |
| mean foot travel from the settled spot | 3.8 cm | 2.6 cm | 1.5 cm | 1.5 cm | 1.4 cm |
| hand-protection success | 0.71 | 0.72 | 0.73 | 0.74 | 0.73 |
| success danger / anticipation / negative | 0.87 / 0.73 / 0.56 | 0.88 / 0.73 / 0.57 | 0.90 / 0.78 / 0.55 | 0.91 / 0.76 / 0.58 | 0.93 / 0.83 / 0.47 |
| hand-object contact per event | 0.0002 | 0.0003 | 0.0003 | 0.0005 | 0.0005 |
| joint power, standing: total | 43.7 W | 40.5 W | 36.6 W | 37.4 W | 33.5 W |
| ... legs / waist / arms | 25.9 / 7.3 / 10.5 | 26.1 / 5.3 / 9.1 | 22.5 / 4.7 / 9.4 | 23.6 / 4.9 / 8.9 | 20.0 / 4.2 / 9.2 |
| falls | 0 | 0 | 0 | 0 | 0 |

Reading (09:30, runs still training): both routes reach the same place -- the feet stay planted in ~94% of episodes
(u150: 52%) and threatening objects are handled better (danger 0.87 -> 0.91-0.93, anticipation 0.73 -> 0.76-0.83).
The energy price gets there without any hand-written rule and with the lowest power (-23%): the saving is in the legs
(-23%) and waist (-42%) while arm power is unchanged -- the robot protects the hands with the arms and keeps the body
still, which is the "moving the hands is cheaper than moving the body" idea. Cost: S2 flinches more at objects that
would pass by (negative 0.56 -> 0.47); hand contacts stay rare (~3 in 6400 events, u150 ~1).

Under training randomisation (latency, torque offsets, pose noise, friction, pushes) the hard rule remains strict:
S1b still ends ~66% of 16-s training standing episodes by foot creep at update 200 (91% at update 50).

Traversal retention (training metrics, noisy): original CAT ~0.31-0.37, published ~0.55-0.6, side gap ~0.28 in both
runs, stable over the night, about 0.05-0.1 below u150's own fine-tune at the same update counts (half the experience
is standing). Not yet evaluated on held-out traversal.

### Under training noise (deployment-like), 11:40

Same 2000 rows x 16 s, deterministic actions, but with the training randomisation (20 ms latency, pose/mount error,
friction x0.5-1.2, mass x0.9-1.1, torque offsets, kp/kd scaling) and pushes; the feet rule re-settles after a push.

| | u150 | S1b @ update 290 | S2 @ update 330 |
|---|---|---|---|
| episodes where a foot moved > 5 cm | 44.0% | 21.0% | 29.6% |
| settled steps with feet displaced | 12.2% | 5.6% | 7.1% |
| mean foot travel | 2.8 cm | 1.9 cm | 2.1 cm |
| hand-protection success | 0.60 | 0.63 | 0.62 |
| success danger / anticipation / negative | 0.74 / 0.63 / 0.46 | 0.79 / 0.74 / 0.39 | 0.80 / 0.75 / 0.34 |
| **robot-initiated object contact per event** | **0.0031** | **0.0072** | **0.0066** |
| joint power: total (legs / waist / arms) | 52.0 W (33.1 / 7.8 / 11.1) | 50.4 W (34.0 / 6.7 / 9.7) | 44.8 W (29.3 / 5.3 / 10.2) |
| falls | 0 | 1 | 0 |

Reading: under noise both still roughly halve foot movement and handle threatening objects better, but far less than
in clean conditions, and robot-initiated contacts with the object more than double (0.3% -> 0.7% of events): with the
feet planted the arms do all the avoiding, and with latency and torque noise they hit the object more often. They also
flinch more at objects that pass by. Not ready for a real-robot hand-protection trial: next, a stronger contact cost in
standing episodes and/or the rule + energy combined (S3), then re-check under noise. S1b's training-condition violation
rate plateaued at ~62% of standing episodes from update ~220.

### Hand retreat distance (S2far, S2far2) and table passages (T1), 2026-11-14 ~01:00

S2far = S2 + standing hand clearance target 0.20 / anticipation 0.40 m and >= 0.15 m retreat for threatening objects;
S2far2 = S2far u200 + 0.30 / 0.50 m and >= 0.25 m. Clean evaluation (2000 rows x 16 s; "threatening" = danger +
anticipation events):

| | u150 | S2 u430 | S2far u200 | S2far u550 (final) | S2far2 u330 |
|---|---|---|---|---|---|
| mean hand retreat, threatening objects | 12.9 cm | 16.1 cm | 22.1 cm | 25.2 cm | 26.8 cm |
| retreat >= 15 cm / >= 25 cm | 37% / 10% | 56% / 13% | 84% / 35% | 90% / 52% | 93% / 62% |
| danger / anticipation handled | 0.87 / 0.72 | 0.94 / 0.87 | 0.99 / 0.96 | 0.99 / 0.98 | 0.99 / 0.99 |
| pass-by object, no flinch | 0.59 | 0.46 | 0.15 | 0.10 | 0.07 |
| episodes with a foot moved > 5 cm | 49% | 2.6% | 5.0% | 3.3% | 5.8% |
| standing power | 43.7 W | 31.5 W | 32.1 W | 30.8 W | 31.1 W |

Table-edge passages (scripts/eval_table_style.py, 8 episodes per scene, clean). The 16 forward_protected scenes are
passages between table pairs: 0.44 m gap at tabletop height (0.71 m), tall baffles block detours, 3 pairs along 7 m.
T1 = S2far u200 + face-the-route cost -3 and leg+waist power -0.005/W in table scenes, table mass 0.25, standing share
0.3, 30% raised-hand starts, no anchor in table scenes.

| | S2 u430 | T1 start (S2far u200) | T1 u130 |
|---|---|---|---|
| open table scenes: goal / sideways near tables | 100% / 25% | 100% / 25% | 100% / 3% |
| between tables: goal | 0% | 0% | 0% |
| between tables: sideways / hand raised / tucked (walking near tables) | 89% / 33% / 33% | 90% / 39% / 38% | 89% / 54% / 32% |
| between tables: episodes still running at 40-60 s (stalled) | 40 of 128 | 101 of 128 | 83 of 128 |

Cause found: table scenes are room worlds, where the anti-refusal terms (stall, goal progress, goal bonus) were not
computed (x-plane goals only) and the room reward group zeroes them -- stopping in front of the gap cost nothing,
squeezing through cost hand clearance (-40). T1b (from T1 u210, 2026-11-14 02:30, gpu partition): --table-progress
gives table scenes those terms with CAT weights (passages run along +x).

Traversal retention (eval_per_scene, 384 fixed non-table scenes x 4 episodes, clean; clean success):
T1 start 0.60 -> T1 u130 0.59 overall; per type within +-0.03 (furniture 0.90/0.92, clutter 0.88/0.87, original CAT
0.22/0.24, procedural 0.41/0.39, published 0.41/0.42, side gap 0.31/0.29, narrow 0.35/0.38) except transition passages
0.63 -> 0.53; refusals 5.5% -> 9.5%.

Videos: outputs/stationary_hands_20261113/videos_S2_vs_S2far, videos_0045 (S2far vs S2far2; table start vs T1).

### Table passages: width curriculum (T2) and arm exploration (T3), 2026-11-15

T1, T1b and T1c (progress terms in table scenes, facing cost -6, hands-up bonus 2) never passed the 0.44 m
forward_protected pairs: their certificate leaves the body 4 cm at the tightest point and only the raised posture
clears (tucked hands still hit), with no intermediate widths to learn from. T2 (from T1c u640, 2026-11-15 02:04, job
299749) trains on bank `tablegap_v1` = sidegap_v2 + 64 protected table pairs with gaps 0.46-0.70 m (hanging hands hit,
a raised or tucked posture clears; `scripts/generate_table_gaps.py`), with the table scenes dropped from the anchor
(`hpc/route_anchor_no_reactive_no_tables_v2.json`). T3 = T2 + arm exploration sigma 0.25 (legs/waist stay 0.10;
2026-11-15 04:24, job 299888).

`scripts/eval_table_style.py --bank data/furniture/tablegap_v1`, 8 deterministic episodes per scene (one per env, up to
60 s), clean. Sideways / raised are shares of walking-near-table steps; goal and stalled (still in front of or inside
the passage at 60 s) are shares of all episodes (the script's own `goal` divides by finished episodes only, which
flatters runs that stall):

| gap (episodes) | T1c u640 (start) | T2 u120 | **T2 u410** | T3 u280 |
|---|---|---|---|---|
| open table scenes: sideways / goal | 2% / 100% | 1% / 100% | 1% / 100% | 0% / 100% |
| 0.62-0.70 m (168): sideways / raised / goal / stalled | 53 / 20 / 84 / 6% | 25 / 27 / 96 / 0% | **4 / 61 / 99 / 0%** | 19 / 38 / 98 / 0% |
| 0.54-0.62 m (168) | 73 / 19 / 37 / 49% | 51 / 27 / 64 / 8% | **10 / 68 / 95 / 0%** | 43 / 40 / 72 / 4% |
| 0.46-0.54 m (176) | 79 / 19 / 5 / 44% | 71 / 17 / 15 / 28% | **29 / 61 / 51 / 0%** | 61 / 35 / 15 / 7% |
| original 0.44 m (128) | 83 / 18 / 0 / 42% | 77 / 14 / 0 / 26% | **46 / 55 / 2 / 2%** | 57 / 46 / 0 / 2% |

What ends the narrow passages (share of episodes; the hand violation is a palm 4 cm inside a table, the regions are the
body-collision proxies): T2 u410 at 0.44 m -- legs 57%, hand violation 36%, feet 3%, hands 2%; at 0.46-0.54 m -- hand
violation 32%, legs 14%. T3 u280 at 0.46-0.54 m: hand violation 70%, legs 9%.

Retention (eval_per_scene, 384 fixed non-table scenes x 4): T1c u640 clean 0.576 (contact 0.28, refused 0.14) vs T2
u120 0.571 (0.29, 0.14).

Reading: the width curriculum works -- T2 walks forward with the hands raised wherever the gap leaves a few centimetres
(0.54-0.70 m: 95-99% goal, 4-10% sideways) and is still improving at the narrow end (0.46-0.54 m goal 15% -> 51%
between u120 and u410); it no longer stalls in front of the gaps. T3's arm exploration shows no benefit at its update count (behind T2's trajectory). Videos (T1c
vs T2 u200 at 0.66 / 0.58 / 0.50 / 0.44 m): outputs/stationary_hands_20261113/videos_T2.

Where the narrow passages fail (`scripts/diagnose_table_failures.py`, T2 u410, 8 deterministic episodes per scene;
"between" = pelvis within 0.40 m of an obstacle while walking; palms "above" = more than 8 cm over the tabletop):

| gap | goal | failures touch (share of failed episodes) | at failure: yaw / off route line / lowest palm | between the tables, successes vs failures: off line / yaw / both palms above |
|---|---|---|---|---|
| original 0.44 m | 2% | legs 62%, hand 33%, feet 6% | 7 deg / 3.7 cm / 0.83 m | 2.4 vs 4.2 cm / 9 vs 9 deg / 81 vs 87% |
| 0.46-0.50 m | 27% | hand 50%, legs 40% | 9 deg / 4.9 cm / 0.78 m | 2.8 vs 4.5 cm / 8 vs 11 deg / 72 vs 73% |
| 0.50-0.54 m | 67% | hand 66% (left 72%), legs 25% | 15 deg / 4.9 cm / 0.73 m | 3.1 vs 4.8 cm / 9 vs 15 deg / 66 vs 59% |

Between the tables T2 already does the asked-for thing -- it never turns sideways there (the sideways share above comes
from the approach) and holds both palms over the tabletop -- and fails on precision: 4-5 cm off the route line where
the raised-hand body has 4 cm per side at 0.44 m, 9-15 deg of yaw (which swings the forward-held hands 7-11 cm
sideways), and sagging palms at 0.50-0.54 m. T4 (from T2 u450, 2026-11-15 11:28, gpu partition job 300255, 8 h)
adds, at the protected pairs only: with the pelvis between the tables (+-0.10 m of a pair along the route), a
cross-track cost -2 x (excess / 5 cm)^2 on the distance from the route line beyond the slack a wider gap adds,
(gap - 0.437 m) / 2, saturating at 10 cm; from 0.45 m before a pair (the hands are held ~0.43 m forward) to 0.15 m
after it, a yaw cost -2 x (sin err / sin 10 deg)^2 saturating at 20 deg; and the hands-up band moves to palm z
0.80 -> 0.92 m (the certified raised posture holds the palms at ~0.94 m). Two earlier starts were stopped after
13 and 11 updates: they also priced the open pairs, then the whole hand-contrast zones (1.65 m around each 0.65 m
pair), where the mean cross-track was ~10 cm (the approach) and the cost saturated.

### Regression found: narrow room passages (2026-11-15 afternoon)

The overall retention number hid it (S2far u200 0.60 -> T2 u120 0.57 clean). Per passage type (same 384 scenes x 4,
clean success / refused = stopped in front until the time limit):

| narrow cabinet passages ("narrow" role) | S2far u200 (before tables) | T1 u130 | T1c u640 | T2 u120 |
|---|---|---|---|---|
| < 0.50 m (52 episodes) | 0.02 / 33% | 0.00 / 87% | 0.00 / 100% | 0.00 / 98% |
| 0.50-0.60 m (52) | 0.35 / 10% | 0.35 / 25% | 0.04 / 90% | 0.06 / 88% |
| 0.60-0.70 m (24) | 0.50 / 4% | 0.67 / 8% | 0.79 / 4% | 0.58 / 0% |
| >= 0.70 m (32) | 0.78 / 6% | 0.84 / 6% | 0.91 / 0% | 0.75 / 0% |

Transition passages the same (0.50-0.60 m: 0.57 / 5% -> 0.23 / 62%); CAT wall gaps that need sideways walking are
unchanged (side gaps 0.31 -> 0.34, published "side" scenes 0.31 -> 0.41). In the T2 u650 video of a 0.55 m cabinet
passage the robot stands at the entrance for 60 s, yaw 0-43 deg from the route, never sideways. Cause: the facing lesson
of the table stages generalised to cabinet passages that look alike to the policy; room worlds zero stall / goal
progress / goal bonus (stopping is free there); each stage was anchored to the previous stage, so the drift added up.

T5 (first started from T2 u650, 2026-11-15 ~15:50, ws-ia job 300518; see below): T2's recipe, but the non-table scenes are anchored to S2far
u200 (the last policy before the table stages) instead of the start checkpoint, and the hand-contrast passages and width
scaffolds get the anti-refusal terms at CAT weights (`--passage-progress`). T3 was stopped at u610 to free the slot.

### T2 final vs T4, and T5 restarted from T2 u410 (2026-11-15 16:30)

Same table protocol (goal / stalled over all episodes; sideways / raised over walking-near-table steps):

| gap | T2 u410 | T2 u650 (final) | T4 u170 (T2 u450 + pair terms) |
|---|---|---|---|
| 0.62-0.70 m: sideways / raised / goal / stalled | 4 / 61 / 99 / 0% | 9 / 47 / 98 / 0% | 8 / 34 / 98 / 0% |
| 0.54-0.62 m | 10 / 68 / 95 / 0% | 18 / 54 / 85 / 0% | 37 / 31 / 82 / 4% |
| 0.46-0.54 m | 29 / 61 / 51 / 0% | 42 / 43 / 32 / 0% | 69 / 18 / 23 / 26% |
| original 0.44 m | 46 / 55 / 2 / 2% | 60 / 37 / 1 / 2% | 81 / 14 / 0 / 25% |

T2 peaked near u410 and drifted back towards sideways passing at the narrow end (between the tables its yaw grew from
8-15 to 19-27 deg). T4's pair terms did tighten the passage itself (yaw 7-14 deg, 3 cm off the line) but it now stops
in front of a quarter of the narrow pairs and reaches fewer goals -- stopped (job 300255). Retention (384 x 4): T2 u650
clean 0.578 / refused 0.14 (narrow cabinet passages still 0.24 / 68% refused), T4 0.565 / 0.16. Videos T2 u650 vs T4
u170 at 0.44 and 0.50 m: runs/eval_T2final_T4_1115_1445/videos (cluster).

T5 restarted from T2 u410 (16:40, job 300639; the first start from u650 was stopped after ~45 updates), with snapshots
every 50 updates (`--snapshot-interval-updates`) so the best checkpoint can be picked instead of the last.

T5 snapshots (table goal over all episodes / sideways near tables; cabinet passages clean / refused; all 384 scenes):

| | tables 0.46-0.54 | tables 0.54-0.62 | narrow < 0.60 m | narrow >= 0.60 m | transition < 0.60 m | width scaffolds | all |
|---|---|---|---|---|---|---|---|
| S2far u200 (before tables) | - | - | 0.17 / 22% | 0.78 / 2% | 0.42 / 10% | 0.72 / 4% | 0.602 / 5% |
| T2 u410 (T5 start) | 51% / 29% | 95% / 10% | 0.00 / 98% | 0.75 / 0% | 0.05 / 90% | 0.68 / 5% | 0.591 / 15% |
| T5 u50 | 62% / 24% | 95% / 8% | 0.01 / 95% | 0.78 / 0% | 0.14 / 81% | 0.77 / 7% | 0.587 / 13% |
| T5 u100 | 55% / 11% | 85% / 7% | 0.06 / 79% | 0.55 / 0% | 0.24 / 64% | 0.58 / 7% | 0.589 / 12% |

Refusals in the narrow passages are coming down and the tables hold; the wider cabinet passages and scaffolds dipped at
u100 -- watching the next snapshots (`~/jobs/summarize_snapshots.py T5` on the cluster prints this table).

### E1: energy only at the tables (2026-11-15 20:00, ws-ia job 300873)

Hypothesis (from the 2026-11-13 framing "moving the hands is cheaper than moving the body"): an energy price alone,
without posture terms, makes the robot pass table gaps facing forward with raised or tucked hands. E1 starts from S2far
u200 (before any table stage; also its anchor outside the table scenes) on the width curriculum `tablegap_v1` with the
anti-refusal terms, and prices the mechanical power of all 29 joints in table scenes, `--table-power-weight -0.02`
(the standing price): no facing cost, no hands-up bonus, no leg-power price, no passage progress. Snapshots every 50
updates, evaluated like T5.

The energy term: per control step the reward carries -lambda * sum_j |tau_j * qdot_j| * dt, i.e. -lambda per joule of
absolute mechanical work; tau = the PD torque the simulation applies (kp/kd from the training contract, gravity
feedforward, robustness noise, clipped to the G1 limits 139/88/50/25/5 Nm), qdot from MuJoCo; sampled once per 50 Hz
step (the last of 10 substeps). It is mechanical, not electrical: holding a pose (raised arms) costs nothing, where a
real motor dissipates R (tau/K)^2 -- a Joule term needs the per-motor constants K, R (not in the MJCF).

### Overnight 15-16 Nov: energy price at the tables (E1) and everywhere (E2) -- morning evaluation

E2 = E1 + -0.01/W in every other environment (from S2far u200, 23:10, ws-ia; two starts on the gpu partition died of
OOM: its GPUs were filled by processes outside Slurm). T5 paused at u360 for E2's slot. At 05:30 E1 and E2 were paused
for two evaluation jobs (`~/jobs/morning_*.sh`, results `runs/morning_eval_1116`, local copy
`outputs/stationary_hands_20261113/videos_morning_1116`), then resumed (jobs 301589, 301590). Evaluations now measure
energy (joint work per episode, mean power, cost of transport; commit 33ebff2).

Table passages (goal over all episodes / sideways near tables / hands raised; mean power at the tables):

| | 0.44 m | 0.46-0.54 m | 0.54-0.62 m | 0.62-0.70 m |
|---|---|---|---|---|
| S2far u200 (no table training) | 1 / 90 / 38%, 105 W | 12 / 87 / 20%, 108 W | 61 / 77 / 10%, 131 W | 81 / 67 / 5%, 145 W |
| E1 u500 (energy price at the tables only) | 0 / 75 / 13%, 92 W | 37 / 68 / 21%, 96 W | 62 / 57 / 26%, 118 W | 93 / 42 / 22%, 126 W |
| E2 u300 (energy price everywhere) | 0 / 51 / 50%, 101 W | 26 / 44 / 50%, 109 W | 77 / 38 / 49%, 123 W | 96 / 37 / 36%, 124 W |
| T5 u350 (facing cost + hands-up bonus) | 3 / 29 / 62%, 132 W | 49 / 17 / 65%, 155 W | 82 / 7 / 71%, 168 W | 92 / 4 / 64%, 165 W |

Retention, 384 non-table scenes x 4 (clean / refused, mean power, cost of transport): E1 u500 0.562 / 15%, 107 W, 0.891;
E2 u300 **0.617** / 13%, 108 W, **0.875**; T5 u350 0.553 / 13%, 118 W, 0.893 (S2far u200 0.602 / 5%; T2 u410 0.591 /
15%, no energy measured). E2 is best on cabinet passages >= 0.60 m (0.95), protected passages (0.88-0.91), scaffolds
(0.87), furniture (0.89). Narrow cabinet passages < 0.60 m, refused: E1 96%, E2 86%, T5 u350 91% (T5 u250: 61%).

Reading: (1) an energy price alone at the tables improves passage over no table training (0.46-0.54 m: 12% -> 37%) at
the lowest power, but the robot mostly still turns sideways (68%) and seldom raises the hands (21%); the facing cost
gives forward passage with raised hands at ~60% more power at the tables (155 vs 96 W). With mechanical work as the
price, turning costs little and raising the arms is not rewarded. (2) The price everywhere (E2) raises the hands at the
tables 2.4x as often as E1, has the best retention of all policies and the lowest cost of transport. (3) Every
table-trained policy now refuses the narrow cabinet passages, with or without the facing cost: the regression is not
the facing lesson spreading but the refusal equilibrium of the room passages (stopping is free there, per-step rewards
keep flowing while standing) winning over the anchor during long fine-tunes; T5's anti-refusal terms helped at u250 and
relapsed by u350.

### Why the robot stops in front of narrow room passages (2026-11-16) and the fix (E3)

`scripts/diagnose_stalling.py` (narrow cabinet passages < 0.60 m, 36 scenes x 4 deterministic episodes; per phase the
scaled reward terms, the route command and the clearances):

| | outcome clean / contact / refused | standing in front: share of steps, reward/step | inside the passage: reward/step |
|---|---|---|---|
| S2far u200 (before the table stages) | 0.17 / 0.69 / 0.14 | 61%, 7.4 | 11.7 |
| T2 u410 | 0.02 / 0.06 / 0.92 | 94%, 9.3 | 13.5 |
| E2 u300 | 0.05 / 0.05 / 0.90 | 91%, 8.4 | 12.1 |

The route command says go (0.6 m/s, route never blocked) in every phase, and being inside pays MORE per step than
standing in front (guidance terms feetgf / headgf / handsgf outweigh the clearance costs there). The refusal is risk
aversion: ~+9 per step is earned standing still (upright, stand_tall, tracking_orientation, feetgf), any touch ends the
episode and forfeits that stream, and with discount 0.98 the policy weighs ~1 s ahead. S2far still attempted (69%
contact); fine-tuning taught every later policy, with or without table or energy terms, that entering usually ends
the episode -- and a policy that stops entering never practises the passage. In room passages stopping was also free
(the room group zeroes stall / progress / goal bonus; room timeouts are truncations, not failures).

E3 (from E2 u340, 2026-11-16 ~12:40, job 302057; control E2c = E2 continued from the same checkpoint, job 302043):
`--passage-soft-contact 0.5` (in hand-contrast passages and width scaffolds only a hand contact, or the root leaving the
navigation corridor, ends the episode; other touches cost 0.5 per step; success stays strict), `--passage-progress`
with `--passage-stall-factor 1.5` (stalling stays net-positive, so ending the episode on purpose never pays), and
`--hand-tight-passages` (room-scaled hand / elbow margins between full-height cabinets; tabletops keep them).

Also fixed: no run could resume -- the environment config recorded the collision proposal's absolute path inside the
job's private code copy (/tmp/slurm-<job id>/code), different on every resubmission (now repository-relative). E1 and E2
died on resume this morning; their checkpoints are kept (`checkpoints/stationary_hands_E1_u510.pt`, `_E2_u340.pt`).

E3 made it worse (narrow cabinet passages < 0.60 m at update 200, `runs/diag_fix_u0200`): E2c clean 0.035 / contact
0.076 / refused 0.889, E3 0.000 / 0.014 / 0.986 (mean furthest progress 1.03 vs 0.80 m). With body touches soft, the
existing collision-event penalty (1 per step) plus the soft penalty (0.5) is paid on every step of a touch -- 6-8x the
whole positive per-step reward -- so a sustained brush cost more than ending the episode had. E3 stopped at u240.

E4 (from E2 u340, 2026-11-16 ~17:00, job 302277; control E2c): `--passage-stall-gate` -- in room passages, while the route
says go and the robot is not advancing (the stall test), no positive term is paid; costs stay, so stalling earns ~0
(not a large negative that would make ending the episode pay) and entering (~+13 per step) wins even if a touch later
ends it. Telemetry `passage/passage_stalled`.

J1 (2026-11-16, commit 0ea44b6; starts in E2c's slot after the E4/E2c update-200 evaluation, by the user's choice):
E4 with energy priced per metre instead of per second. `--route-energy -0.01 234 0.6 2` replaces table_power +
global_power: under a go command the price is P - q*v (v = pelvis speed along the commanded world-frame direction,
capped at 0.6 m/s; q = 234 J/m, E2's clutter-room energy per metre), under a stop command P alone; table steps x2 as
in E2/E4, standing episodes keep reactive_power. Dinkelbach form of a ratio objective: at q = the policy's own energy
per metre it minimises energy per metre, so standing under a go command costs its full power and walking at that ratio
costs about nothing. Hypothesis: the per-second price's preference for standing still is gone, so hesitation at tables
and passages falls further than with the gate alone, at equal or lower energy per metre and with the arm-first
standing behaviour unchanged. Expected effect is modest at 0.01/J (the progress credit is ~1.4 reward units/s at
cruise against ~9/s of posture terms); a null result would mean the currency matters only at stronger prices, the next
test (per-second against per-metre at ~5x the price). Telemetry: energy/energy_progress_mps, energy/energy_go.
Measured vs E4 and E2c (same start, E2 u340) at matched updates: narrow / transition passages (refused, contact,
clean), table forward success, 384-scene retention, energy per metre and COT, stalled share, standing foot motion.

Evaluation of E4 at update 200 (`~/jobs/eval_e4.sh`) also runs the stalling diagnostic on the anchor (S2far u200) and
on the start (E2 u340): the self-anchor covers every cabinet passage (route_anchor_no_reactive_no_tables_v2 drops only
tables and reactive scenes), so if the anchor itself refuses, it holds the stall in place (Wolczyk et al. 2024) and
the next arm (J2) masks it off the passages.

E4 at update 200 (`runs/eval_E4_vs_E2c_u200`, 2026-11-17 01:18-02:29; E2c stopped for good at that point, its
snapshots to update 650 remain the control):

| narrow cabinet passages < 0.60 m | clean | contact | refused | stalled share of steps |
| --- | --- | --- | --- | --- |
| E2 u340 (start of E4, E2c, J1) | 0.000 | 0.090 | 0.910 | 0.94 |
| E2c u200 (control) | 0.035 | 0.076 | 0.889 | 0.94 |
| E4 u200 (stall gate) | 0.028 | 0.104 | 0.868 | 0.92 |
| S2far u200 (the anchor) | 0.222 | 0.639 | 0.139 | 0.61 |

- The gate works as designed: while stalled, E4's positive terms are zero and the scaled terms sum to about -3.6/s (energy,
  self-clearance, guidance fields), against about +19/s inside (feetgf 6.2, headgf 4.1, handsgf 3.3, posture). The
  command says go throughout (command_stop 0, speed 0.6 m/s, route never blocked). Still, by update 200 the policy has
  not learned to enter: refusals 87% against 89%. With a 1-s horizon even certain contact after 0.5 s inside would be
  worth about +7 against -3.6 for stalling, so a converged critic should enter; the policy rarely tries (0.7% of steps
  inside), which points at exploration rather than incentives. Later snapshots (update 400, 600) are evaluated next.
- The anchor does not refuse: S2far enters (refused 14%) and touches (64%). The self-anchor on passages pulls toward
  entering, so it does not hold the stall; masking it there (planned J2) is not indicated.
- Retention (384 scenes): E2c 0.606 clean / 16% refused, 98.7 W, COT 0.84; E4 0.589 / 14%, 106 W, 0.85. Transition
  passages < 0.60 m improve with the gate (clean 0.08 -> 0.14, refused 86% -> 76%); clutter and furniture lose 4-5
  points (0.96 -> 0.91, 0.95 -> 0.91).
- J1's first start (job 302895) stopped at config time: the student's inherited global_power/table_power tripped the
  conflict check; fixed (c6f3ad2, inherited per-second prices are dropped), resubmitted as job 302978 (03:55).

E4 against E2c later in training (`runs/eval_E4_vs_E2c_late`, E4 ended at its 12 h limit at update 640):

| narrow cabinet passages < 0.60 m | u200 refused / clean | u400 refused / clean | u600 refused / clean |
| --- | --- | --- | --- |
| E2c (control, no gate) | 0.889 / 0.035 | 0.451 / 0.410 | 0.701 / 0.181 |
| E4 (stall gate) | 0.868 / 0.028 | 0.993 / 0.000 | 0.965 / 0.014 |

- Negative result for the gate as built: the control learned on its own to enter (refusals 89% -> 45% by update 400,
  then 70% at 600), the gated run did not (99%, 97%). At u400 E4's stalled scaled terms sum to -6.6/s (the step reward is clipped at 0 apart from hand clearance, so a stalled step earns about 0, not a negative amount) and its steps inside
  +15.2/s, against +7.8 and +10.8 for E2c, yet E4 enters 0.2% of steps against 4.5%.
- Retention at u600 (384 scenes): E2c 0.594 clean / 12% refused, COT 0.81; E4 0.573 / 15%, COT 0.83. Transition
  passages < 0.60 m 0.35 vs 0.06, scaffolds 0.78 vs 0.66; clutter and furniture slightly better with the gate.
- Likely mechanism: the gate zeroes every positive term while progress is slow, including the guidance-field terms
  (feetgf/headgf/handsgf: +2.4 to +6.6/s) that reward each small step along the field. The slow preparation before an
  entry (aligning, raising the hands) then earns exactly what stalling earns, so nothing in the immediate reward
  separates preparing from stalling, and with a ~1 s horizon the payoff of the entry seconds later is discounted
  away. The gate flattened the approach gradient instead of removing the stall payoff (Miller: the approach gradient is
  what has to win). Candidate fix (E5, not run): subtract a constant opportunity cost while stalled (average-reward
  view, Niv et al. 2007) instead of zeroing, so that every difference between actions survives.
- J1 (gate + energy per metre) inherits the gate; it still isolates the energy currency against E4.
- Learning curve over the snapshots (E2c and E4 at u250-u500, E2c u650): `runs/eval_narrow_curve`.

Deploy bundles (2026-11-17, `scripts/export_deploy_bundle.py`, CPU, on dep-1): `outputs/deploy/hands_S2far_u200`,
`hands_T2_u410`, `hands_E2_u300` (checkpoints in `outputs/saved_checkpoints/stationary_hands_*.pt`, sha256 d29faa3f,
966c5892, 5933b120). Policy I/O, timing, joints, action rule, actuation, observation layout, fields, gait clock and
training randomisation are identical to `generalist_robust_delay_v1_u150`, the bundle already run on the G1; ONNX
matches PyTorch to <= 1.0e-6. Not yet done: evaluation under the training randomisation, and the moving-object
field latency for standing ball trials.

V1, slow traversal (2026-11-17 ~20:30, commit 62ac46a, job 304204, from E2 u300 = the all-round bundle on the robot
branch; anchor E2 u300 except in episodes slower than 0.4 m/s). The robot was repelled at narrow obstacles and passed
only with momentum. Causes in code: the field command (compute_cmd_from_rtf) brakes near obstacles and is zero below
0.2 m/s, a zero command latched the robot for the rest of the episode, the speed was always ~0.525 m/s, and the
reward made slow passage worse than none (clearance costs per second, progress capped in absolute speed, tracking
kernel barely charging a stop at low commands). V1 changes:
1. operator speed: field direction (the same limb projection) at an operator speed, log-uniform 0.1-0.6 m/s per
   episode, a slower speed within 0.35 m of obstacles in 60% of episodes, re-drawn every 4-10 s; stop only below 0.05
   m/s of projected command, no latch; room routes at the same speed; episode time x 0.525/speed;
2. 25% of side-gap episodes start at rest just before or inside the gap;
3. clearance costs per metre (x commanded/0.525), tracking, goal progress and the stall test relative to the
   commanded speed; no gate.
Hypothesis: success no longer depends on speed (the speed sweep is flat from 0.1 to 0.525 m/s) at no more contacts.
Test: `~/jobs/eval_sweep.sh` (eval_per_scene --operator-speed S on 303 side-gap / passage / CAT / scaffold scenes, 2
episodes each) for E2 u300 (baseline, job 304207) and V1 snapshots; CPU check of the code passed (commanded speed =
operator speed, rest starts at zero velocity, no stops, finite rewards).

Correction (2026-11-17): the step reward is clamp(sum of scaled terms x dt, 0) plus the hand clearance term
(`reward_floor.py`, end of `task._rewards`), so the "-3.6/s" and "-6.6/s" stalled figures above are sums before that
clip; the stalled steps of E4 earned about 0. The E4 reading stands, sharper: the gate turned the approach zone into a
flat zero-reward plateau with no gradient toward the passage.

Baseline speed sweep, E2 u300 with the operator-speed command (`runs/speed_sweep/E2_u300`, 303 obstacle scenes, 2
episodes each): clean 0.53 / 0.53 / 0.54 / 0.54 at commanded 0.1 / 0.2 / 0.35 / 0.525 m/s, contacts ~0.29, refusals
0.16-0.19 -- but the mean speed is 0.22 m/s when told 0.1 (it ignores slow commands). The command change alone (no
latch, 0.05 m/s cut-out, constant speed) lifts narrow cabinet passages from 0.11 to 0.55 clean.

V2 (2026-11-18 00:20, commit 6246f42, job 304806, from E2 u300), after the review of 2026-11-17: V1 + overspeed cost
(-60/s on speed above 1.2x the command, relative, outside the reward clip so it can outweigh the goal bonus's
early-arrival premium), goal bonus 500 (was 2000), relative progress / stall in room passages, 35% of episodes at
0.45-0.6 m/s (anchored), rest starts inside a gap only where >= 0.5 m wide and in the nominal pose. Speed sweeps of
E2 (with gap metrics: speed ratio and flight fraction within 15 cm of obstacles), V1 and V2 at u200 / u400 run
overnight on the gpu partition (`~/jobs/eval_sweep.sh`).

Robot side (humanoid_navigation `feature/g1-slow-traversal`, e555064 / e31ba19): `--speed` on the executor and in
wholebody_sim for policies whose contract records the operator-speed rule; `command_from_fields(speed=...)` matches
`task._operator_command` on 400 recorded field samples; no stop latch; arrival stop within 0.25 m of the final goal /
hold point (release 0.45 m), because the field no longer slows the robot at its goal.

Speed sweeps, 2026-11-18 (`runs/speed_sweep/<label>`, 303 obstacle scenes x 2 episodes, `~/jobs/summarize_sweep.py`;
"near" = steps with any body point within 15 cm of an obstacle, ratio = mean planar pelvis speed / commanded):

| policy | command m/s | clean | contact | refused | near-obstacle speed (ratio) |
|---|---|---|---|---|---|
| E2 u300 | 0.1 / 0.2 / 0.35 / 0.525 | 0.53 / 0.53 / 0.54 / 0.53 | 0.28-0.32 | 0.15-0.19 | 0.43 / 0.45 / 0.46 / 0.47 m/s (4.31 / 2.26 / 1.32 / 0.89) |
| V1 u200 | same | 0.47 / 0.47 / 0.46 / 0.49 | 0.37-0.38 | 0.13-0.17 | ratio 3.83 / 2.15 / 1.33 / 0.95 |
| V1 u400 | same | 0.48 / 0.48 / 0.46 / 0.47 | 0.38-0.41 | 0.12-0.14 | ratio 4.59 / 2.43 / 1.42 / 0.97 |
| V2 u200 | 0.1 / 0.2 | 0.12 / 0.09 | 0.48 / 0.55 | 0.40 / 0.36 | 0.15 / 0.15 m/s (1.49 / 0.77) |
| E2 u300, one-process sweep (`E2s`, `--operator-speeds`) | 0.1 / 0.2 / 0.35 / 0.525 | 0.52 / 0.51 / 0.53 / 0.52 | 0.28-0.30 | 0.17-0.20 | 0.4 s mean 0.32 / 0.34 / 0.36 / 0.37 m/s (3.19 / 1.71 / 1.03 / 0.70) |

The one-process sweep (all speeds in parallel, 2560 envs) reproduces the per-speed runs within 0.02 and adds the 0.4 s
mean speed, which leaves out the gait's sway: E2 travels at 0.32-0.37 m/s within 15 cm of obstacles at any command.
It took 1 h 50 min on one GPU, mostly a tail of 0.1 m/s room episodes (length x 5.25).

Every policy except V2 crosses within 15 cm of an obstacle at 0.43-0.47 m/s whatever it is told (the momentum
passage seen on the robot), and V1's training did not change that. V2 slows down but collapsed: training success 0.29
(update 34) -> 0.05 (u200), flat to u490; stopped at u490 (2026-11-18 08:50). Why V2 collapsed: its overspeed cost
read the instantaneous pelvis speed with 20% headroom over the command, and at 0.1 m/s the gait's own sideways sway
exceeds 0.12 m/s, so every walking step paid it, outside the clip at 0. Walking slowly then paid less per step than
standing (V2 vs V1 at ~u390: stall -1.14 vs -0.03 /s, headgf 0.32 vs 1.62, handsgf 0.47 vs 1.99, steps clipped at 0
27% vs 3%), and ending the episode on a touch paid most (contacts 0.48-0.55).

Why E2 / V1 rush near obstacles (found 2026-11-18): headgf / handsgf / feetgf (scales 1 / 1 / 2, up to 5 per point)
score the cosine between the instantaneous head / hand / swing-foot velocity and the guidance field, and act only
within tau = 0.5 / 0.5 / 0.3 m of an obstacle. A slow pass reads as misaligned because sideways sway is a large part
of a slow velocity: 0.1 m/s forward with 0.15 m/s sway scores 2.77 of 5, a 0.45 m/s pass with the same sway 4.74
(`gf_reward` on these vectors). Near obstacles these terms outweigh the relative tracking term (~0.5/s) by 10x, so the
fastest pass scores best exactly where speed matters.

V3 (2026-11-18 08:50, commit 9a0054f, from E2 u300, anchor E2 u300 except episodes slower than 0.4 m/s; V3a job
306164, V3b job 306165, ws-ia), `--operator-v3` = V2's episode settings (35% anchored near-nominal episodes, safe rest
starts, passage progress) plus:
1. headgf / handsgf / feetgf score velocity along the field relative to the operator speed, capped at +-1 (both
   passes above score 5.0; at the nominal speed close to the cosine: 4.76 vs 4.79);
2. tracking (relative, as V1) on the 0.4 s mean pelvis velocity, so sway is not tracking error;
3. goal bonus back to 2000;
4. V3b only: overspeed -10/s on the 0.4 s mean speed above 1.25x the command + 0.05 m/s, relative to
   max(command, 0.2 m/s), inside the clip (it can take a step's reward to 0, never below).
Hypotheses: (H1, V3a) removing the speed premium in the guidance terms alone brings the near-obstacle speed close to
the command; (H2, V3b) a bounded speed cost does it without V2's collapse. Pass: near-obstacle ratio <= 1.5 at 0.1 and
0.2 m/s with clean >= 0.45 and contacts <= 0.35 (E2: 0.53 / 0.28-0.32).

Robot-side baseline (humanoid_navigation `cat/wholebody_sim.py`, plain MuJoCo, `--speed`, `--robot-perception`, the
shipped E2 u300 bundle; near = any body point within 15 cm in the policy's SDF observation; mean = 0.4 s EMA of the
pelvis velocity): told 0.15 or 0.25 m/s, E2 u300 crosses gap_044_exact / sidegap_plain_047 / sidegap_corridor_048 in
4.1-5.9 s without contact, at 0.4 s mean speed 0.67-0.73 m/s near the walls (peaks 0.99-1.24 m/s; 0.48-0.52 m/s
over the whole run): it lunges through the gap whatever it is told, the momentum passage seen on the robot.
V2 u450 in the same harness: 0.02 m/s average told 0.15 or 0.25 m/s, never reaches the goal in 80 s (one fall at
0.25 in gap_044): the collapse is a robot that has stopped walking. (Snapshots are stripped to schema / contract /
learner without the optimizer, 9.4 MB, with `scripts/strip_snapshot.py` on the cluster, then exported here on CPU.)

V3 at update 50, same harness: V3a / V3b still approach at 0.3-0.4 m/s and leave the gap at 1.1-1.4 m/s when told 0.15
(0.4 s mean within 15 cm of the walls 0.61-0.66 / 0.57-0.67 m/s, E2 0.67-0.73), no contacts registered (robot-side sim walls
never collide, see 2026-11-18 sim bug below); training success 0.31
(V1 0.32, V2 0.29 and falling). The traces show the lunge is the passage itself: the robot turns sideways at the mouth
and bursts out, while the command stays at 0.15 m/s. Two incentives V3 left in place:
1. inside a narrow room passage, hand / arm clearance (outside the clip, -40/s in rooms, pressure 0.27 at 5 cm) and the
   SDF terms (up to -14 per point at 5 cm) are charged per second; V1-V3 scale them by commanded/nominal speed, which
   keeps them per second, so leaving the gap sooner costs less (the 0.98 discount, ~1 s horizon, values exactly that);
2. the anchor (E2 u300) still holds gap scenes in episodes at >= 0.4 m/s, and E2 lunges through gaps at any command.
V4 (2026-11-18 10:07, commit 601fd5f, job 306281, replaces V3a at u59; V3b continues): V3b + clearance and SDF costs at
max(commanded, actual 0.4 s mean speed)/nominal -- per metre of travel when faster than told, at the commanded rate when
slower -- and side-gap / hand-passage / width-scaffold scenes anchor-free.

The larger incentive (found 10:10): every CAT-type scene of tablegap_v1 (2226 procedural, 640 side gaps, original /
published CAT) runs from (0, 0) to a goal plane at x = 2 m with the obstacle in between, so the goal plane lies 0.4-1.4
m past it. The goal bonus (2000) is paid once at the plane and discounted at 0.98 per step: ~730 one second out, ~35
four seconds out. Sprinting the last metre is worth ~700 more than walking it at 0.15 m/s, against at most ~0.3 per
step that a clipped cost can take away -- the burst at the gap exit in the robot-side traces, ~1 m before the goal.
V2's goal bonus 500 still left a ~180 premium.
Correction (13:05): the bonus is not paid at x = 2 m. `_goal_status` counts the goal reached when the pelvis and both
feet are past `_crossed`'s line, x > 1.5 m in CAT scenes (the fields still point at the goal point x = 2 m), so the
success line is 0-0.4 m past the walls of the three robot-side gap scenes. Walking the last 0.5 m at 0.15 m/s (3.3 s)
leaves the bonus worth ~69, sprinting it in 0.5 s ~1,200: the premium is ~1,100, larger than estimated above. V5 (2026-11-18 10:14, commit 030a2f1, job 306294, replaces V3b at u71;
V4 continues as the ablation): V4 + goal bonus x clamp(1 - f / 0.25, 0, 1), f = share of the last ~2 s (EMA) with the
0.4 s mean speed above the overspeed limit (1.25x command + 0.05 m/s): about half after 0.25 s above the limit, 0.1
after 0.5 s, nothing after a 1 s sprint. Prediction: V4 still sprints at the gap exit (H3: the goal premium dominates),
V5 does not.

Robot-side check, told 0.15 / 0.25 m/s (0.4 s mean within 15 cm of the walls / over the run / peak, m/s):

| policy | near the walls | whole run | peak |
|---|---|---|---|
| E2 u300 | 0.67-0.73 | 0.48-0.52 | 0.99-1.24 |
| V3a / V3b u50 | 0.61-0.66 / 0.57-0.67 | 0.43-0.49 / 0.40-0.47 | 0.88-1.21 / 0.92-1.16 |
| V4 u50 / u100 | 0.59-0.73 / 0.51-0.61 | 0.42-0.47 / 0.40-0.45 | 0.89-1.24 / 0.94-1.13 |
| V5 u50 / u100 | 0.46-0.63 / 0.45-0.59 | 0.34-0.40 / 0.34-0.42 | 0.91-1.21 / 0.81-1.05 |

All reach the goal without contact. Along the course every policy speeds up toward the goal line: ~0.2 m/s in the
first 0.5 m, 0.7-1.1 m/s in the last, peak at x = 1.4-1.96 m. V5's gate works in training (goal-bonus term 0.03-0.04
vs V4's 0.29-0.35: most arrivals still forfeit it) but the sprint is unlearned slowly. One goal attraction it left:
gf_reward pays a constant 4 per head / hand / foot point past the goal line (`task._crossed`: x > 1.5 m in CAT scenes,
right at the gap exit) and ~0 in the open just before it, ~+16/s for being across sooner. The anchor touches only 8.5%
of samples, but there the policy equals E2 (action MSE 0.002); the AMP style term is off near obstacles.
V6 (2026-11-18 12:30, commit 85e694f, job 306586, replaces V4 at u120; warm start V5 u100, anchor E2 u300): V5 +
that constant x the V5 compliance factor (stance feet keep 4), every goal-line (CAT) scene anchor-free (the anchor
keeps rooms), overspeed -30/s (inside the clip).

Slow-passage tests, 2026-11-18 afternoon (robot-side sim, `--speed`, `--robot-perception`; 0.4 s mean speed):
- Told 0.1-0.2 m/s in three 0.47-0.48 m gaps (27 runs, no contacts registered; see the sim bug below): E2 u300 approach 0.10-0.28, at the gap
  0.65-0.86, peak 0.93-1.31 m/s, 0.5-1.3 s from 0.3 m before to 0.3 m after the wall; V5 u150 0.43-0.56 / 0.73-0.93,
  0.9-2.4 s; V6 u50 approach 0.10-0.18, gap 0.39-0.58, peak 0.64-0.94 m/s, 0.9-2.7 s. Later checkpoints near the
  walls (told 0.15 / 0.25): V5 u200 0.30-0.41 (peaks 0.64-0.89), V6 u100 0.33-0.42 (peaks 0.69-0.97) m/s.
- Command field (operator rule, nominal pose at each spot, facing +x or sideways): with the goal behind the gap the
  command keeps its full 0.52-0.55 m/s field speed and points into the gap everywhere; nowhere does the projection cancel
  it. A forward-facing body does not fit the gap; a sideways one fits only through the opening. The command slows only
  within ~0.1-0.3 m of the goal. The observation clips every site's SDF at 0.5 m and zeroes its normal beyond, so a
  wall is seen only within 0.5 m of a body point.
- Goal placed in front of / at / inside the opening (told 0.15): both policies stop 0.38-0.64 m short and march in place
  (never within the executor's 0.25 m arrival radius). On open floor with no walls they also stop short: V6 0.35-0.38 m
  before goals 0.5-2.0 m away, E2 ~0.4 m before goals 0.5-0.8 m away; no retreat (0.00-0.04 m). Near the wall (goal
  within ~0.35 m of the opening) they stop 0.1-0.2 m farther, back off 0.08-0.14 m after the closest approach and stand
  half-turned (50-65 deg). E2 with the goal inside the gap lunges through it.
- Why ~0.4 m: CAT-type episodes do not end at success (successful CAT episodes in the eval all run the full 1000 steps;
  the 1 s goal hold applies to route worlds only). Success counts once the pelvis and both feet pass x = 1.5 m; the
  trailing foot crosses when the pelvis is at ~1.60-1.65 m, i.e. 0.35-0.40 m before the goal ball at x = 2.0 m, and the
  robot then stands for the rest of the 20 s: nothing pays for going closer (progress and bonus over, the past-line
  constant is the same anywhere past 1.5 m, energy is priced, the field flips near the ball, the grid ends at ~2.6 m).
  The policy recognises that distance from the 3-D guidance pattern around the 0.75 m-high goal ball (head arrow steeply
  down, feet steeply up): a vertical-column goal seed (`scratchpad sim_column_goal.py`) lets E2 reach open-floor goals
  (0.15-0.23 m), not V6. Robot-side consequence: the 0.25 m arrival radius is inside the learned stop distance, so the
  robot marches in place ~0.4 m before a goal; use ~0.45-0.5 m or place goals ~0.4 m beyond the intended stop.
- Prior-work report: `reports/Slow passage and field repulsion.md` (classical potential fields stall at doorways by a
  position-only force balance; learned policies rush wherever time near danger is priced; remaining per-second
  pressures in V6: contact termination, pushes 0.1-1.0 m/s every 5-10 s in gaps too, gamma 0.98).

Ground-projected goal (2026-11-18 afternoon). The guidance was seeded from a 12 cm ball at the goal point, z = 0.75
m: head arrows tilt down and foot arrows up by angles that depend only on the distance to the goal (open floor: -17 /
+24 deg at 1.5 m, -50 / +60 deg at 0.4 m), a hidden distance gauge; with success counted past x = 1.5 m (0.35-0.40 m
before the ball) the policies learned to stop ~0.4 m before any goal and to sprint the last metre. Column seed = goal
projected onto the ground (free voxels within 12 cm of the goal xy, 0.10-1.40 m). Robot side: humanoid_navigation
a7deccf (`--goal-seed`, contract "goal_seed"). Robot-side simulator, told 0.15 m/s, no retraining: E2 u300 gap speed
0.75-0.94 -> 0.23-0.36 m/s, peaks 1.01-1.24 -> 0.47-0.60 m/s, wall zone 0.5-1.3 -> 2.7-3.7 s, open-floor goals reached
(0.15-0.23 m), goals within 0.35 m of an opening still 0.28-0.34 m short; V5 u200 / V6 u100 reach worse with it (trained
on the ball). Training side 781ce07: cache `data/furniture/tablegap_v1_goal_column` built on the cluster (3591 scenes,
685 s, 1 scene keeps the ball: no free voxel in its column; cache_sha256 23148ab5...), SceneBank check: SDF and normals
identical, 85.2% of guidance entries change (mostly their vertical part). V7 (sbatch) = V6 + cache + `--goal-ground`,
from V6 u100, waiting for a training slot. The command rule (task._operator_command / compute_cmd_from_rtf) adds no
braking on the way into a gap and a 15-33% speed-up after it (limbs past the wall add forward corrections: 0.60 m/s
inside sideways, 0.70 m/s just past it); the operator rule keeps only the direction.

Robot-side simulator bug and near-wall goals (2026-11-18 evening).
- humanoid_navigation cat/wholebody_sim.build_model sets geom_contype on the robot geoms AFTER compiling; MuJoCo 3.16
  filters collisions by body_contype / body_conaffinity, computed at compile time. Only `debug_axis` has the flag, so every
  robot link passes through the walls and contact_steps is always 0 (pelvis placed inside a wall: 0 contacts; 67 after
  recomputing the body flags). Every robot-side "no contacts" so far is void. Tracked-point distances instead: a palm's
  10.3 cm sphere overlapped a wall in 9/9 ground-goal passages (0.4-7.9 cm; E2 up to 3.5 cm) and 2/9 ball-goal ones.
  Fixed in humanoid_navigation 98a6576 (wholebody_core.enable_obstacle_contacts, used by wholebody_sim and FakeG1;
  regression test). Rerun with solid walls, told 0.15 m/s: ball-goal passages 0/9 touched; ground-goal passages 6/9
  touched, mostly the left wrist (E2 0.5 s in sidegap_plain_047; V5 u200 0.3 / 2.3 s; V6 u100 0.1-0.7 s), no falls,
  all reached. Near-wall goals unchanged (stand-off 0.27-0.35 -> 0.40-0.51 m, no contact; goal inside sidegap_plain_047:
  0.6 s right-ankle contact, no fall).
- Near-wall goals, E2 u300, ground goal, 0.15 m/s, 30 s (no early stop; the sim's stop-when-reached rule had cut the
  earlier runs): closest 0.27-0.35 m, then backs off to 0.40-0.51 m, never inside the 0.25 m arrival zone; open-floor
  control 0.18 m. The nearest body part at the closest point is always a palm, 0.10-0.18 m from the wall surface.
  Wall inputs blanked (distance 0.5, normal 0, rates 0; guidance and command unchanged; slowtest/sim_full.py BLIND=1):
  reaches 0.17-0.25 m in 2-4 s and stays (7/8), but the palm spheres then sit 0-7 cm from the wall (overlap up to 12 cm).
  So the stand-off is learned hand clearance, not field repulsion: reaching those goals in its posture means touching.
- E2 with the goal inside sidegap_plain_047 arrived, stood in the gap and fell 3 s later (1 run; walls non-colliding).

Real-robot logs, 17 Nov 2026 (Artem's archive /l/users/artem.voronov/g1_real_robot_data, checksums OK; unpacked under
runs/real_g1 on the cluster; scripts ~/jobs/real_g1_analysis.py, real_g1_extra.py). u150 sessions A-D, S2far2 session E,
executor = humanoid_navigation feature/g1-real-robot (19 commits beyond feature/g1-slow-traversal's merge base).
52,030 of 54,351 POLICY steps matched exactly to the 1 kHz motor log (identical q / targets).
- Timing: step 20.0 ms (p99 23.4, 226 steps > 25 ms); executor compute 4.7 ms median (p99 12); LowState -> /lowcmd at the
  logger 7.0 ms median (p95 12, p99 16, max 45); /lowcmd period 19.9 ms (p05 15.5, p95 24). Command -> torque: the motors'
  PD law on the logged commands explains the estimated torque with a 5 ms lag (legs rms 0.98 vs 7.6 Nm signal).
  Sensor-to-torque ~12 ms median, ~20 ms p99: inside the trained 0-20 ms action delay.
- Gains exactly the contract's. Estimated / PD-model torque slope: legs 0.98, arms 0.94, waist 0.82. Saturation ~0.
- SLAM: new poses 48-50 Hz, yaw 30 ms behind the pelvis IMU at the executor; stamps on a clock 1294.07 s (A, B) /
  1294.46 s (C-E) away from the robot's. 5 relocalisation steps of 3.6-5.8 cm while walking (one 1.6 s before the
  episode-3 palm contact / hip-roll DAMP).
- Upper body: the robot filters waist/arm motor targets with EMA alpha 0.30 (reconstructed exactly). Training: u150 no
  filter; E2 / S2far2 / T2 / V7 the mean of the last two targets (--upper-target-average); the robot branch offers
  'avg2'. feature/g1-slow-traversal applies no filter at all. The 2 rad/s upper rate limit binds on 26-36% of joint
  steps on the robot and 31-35% in the training rollouts: the step-to-step dithering is the policy's, not sim-to-real.
- Guidance-speed "fix" on the robot branch (|gf| 0.6, (d/0.3)^3 within 0.3 m) feeds only the stop latch; our training
  banks (sidegap_v2_packed for u150, tablegap_v1_packed for E2/V*) store unit directions, so in training the latch
  command was 0.525 m/s nominal, never slowed at goals. Robot latch stops at goals: 0.17-0.42 m away after the change.
  Risk: the latch now fires after a 37% field-command cut near obstacles instead of 62%.
- Walking (1 s windows): median 0.20-0.28 m/s, p90 0.38-0.55, max 0.54-0.95 (short goals, turns, obstacles, gantry);
  robot-side sim, u150, 2 m open-floor goal: mean 0.59, p90 0.99 -- possibly 2x slower in reality, needs a matched replay.
  Gait (knee) 1.40-1.42 Hz.
- Both right-hip-roll DAMPs (episodes 3, 10): hip roll -0.54 / -0.56 rad past the -0.436 limit, pelvis 0.73-0.81 m/s in
  the last second, right after a palm reached a box: the no-contact training showing.

Narrow-passage retention, V7 vs V6 vs E2 (2026-11-18, `~/jobs/eval_narrow.sh`, `runs/speed_sweep/narrow_*`, the 384
fixed scenes x 2 episodes, deterministic, `~/jobs/narrow_summary.py`). Clean / contact / refused told 0.15 m/s (all-scenes
clean at 0.3 m/s in brackets). V7 is scored under its own rule (ground goal, within 0.30 m after a 2 s hold); V6 under
its own easier rule (past the goal line) and under V7's (`env_from` V7 u150); E2 under its own rule only.

| scenes | V7 u150 | V6 u300 | V6 u300, V7 rule | E2 u300 |
|---|---|---|---|---|
| side gaps 0.3-0.6 m (40) | 0.38 / 0.51 / 0.11 | 0.38 / 0.60 / 0.03 | 0.10 / 0.84 / 0.06 | 0.44 / 0.55 / 0.01 |
| cabinet passages < 0.60 m (103) | **0.78 / 0.03 / 0.18** | 0.57 / 0.01 / 0.41 | 0.53 / 0.12 / 0.35 | 0.52 / 0.03 / 0.45 |
| cabinet passages >= 0.60 m (57) | 0.98 / 0.00 / 0.02 | 0.96 / 0.04 / 0.00 | 0.89 / 0.08 / 0.03 | 0.94 / 0.01 / 0.05 |
| CAT courses (104) | 0.17 / 0.75 / 0.07 | 0.25 / 0.66 / 0.07 | 0.04 / 0.88 / 0.08 | 0.36 / 0.55 / 0.08 |
| rooms (80) | 0.88 / 0.12 / 0.00 | 0.94 / 0.06 / 0.00 | 0.89 / 0.10 / 0.01 | 0.87 / 0.13 / 0.00 |
| all 384 | 0.62 / 0.29 / 0.08 (0.61) | 0.60 / 0.26 / 0.13 (0.61) | 0.48 / 0.39 / 0.13 (0.51) | 0.60 / 0.24 / 0.15 (0.61) |

Under the same (V7) rule V7 beats V6 by 14 points overall, most in side gaps (0.38 vs 0.10), narrow cabinet passages
(0.78 vs 0.53) and CAT courses (0.17 vs 0.04). V7's clearest gain over every policy is narrow cabinet passages: 0.78
clean with 18% refused, against 0.52-0.57 with 35-45% refused. The rule change alone costs V6 12 points overall and 21
on CAT courses, so V7's CAT-course gap to E2 (0.17 vs 0.36) is at least partly the stricter rule; E2 under V7's rule has
not been run. Told 0.3 m/s, V7's side gaps drop to 0.21 clean (contact 0.70), against 0.26-0.34 for E2 and V6 under their
own rules.

Whole-body planner v1, overnight 2026-11-19 (cluster job 307723, run /l/users/konstantin.smirnov/runs/wbp/planner_v1,
code 2d4d40f; probe 5d677bc, job 308515; report https://claude.ai/artifact/XsUF4SgoZDtLyF89nuBWwZ).
- Data: clean 2 s windows (no contact or fall until 0.5 s after) from V7 u450 (200k, mean speed 0.24 m/s), E2 u300 and
  T2 u650 (150k each) on tablegap_v1; 374 training / 112 held-out scenes; ~90 min of collection per source at 4096 worlds.
- Training: 100k steps in 129 min; validation loss 0.29 (5k) -> 0.18, flat after 70k.
- Held-out, 4200 situations of 14 types: best of 16 collision-free 99.6% (the policies' own motion 99.5%, blind 98.4%);
  one sample 97.7% (blind 93.6%); goal error 0.6 cm; foot skate 2.0 cm (executed 1.75 cm); samples 2.4 cm apart. The
  hindsight goal (where the policy ended 2 s later) implies a free route, hence the high blind score.
- Inserted-column probe (scripts/planner_probe.py): a 10 cm column 25 cm beside the executed pelvis path 1 s in, cutting
  the executed motion, start and goal unchanged (1514 situations): best of 16 clear 10% blind, 54% with the map, 70%
  with classifier-free guidance 2 (one sample 2 / 29 / 50%); the pelvis passes 25 -> 41 cm from the column; worst table
  edges 41%, original CAT 61%, protected / transition passages 63-64%. Guidance 2 costs nothing on ordinary situations
  (one sample 98.1% clear, goal error 2.1 cm): make it the default.
- Next: hindsight obstacles (SWAP) so plans learn to go around obstacles the goal does not reveal (gate >= 90% of
  probes), goals from a route instead of hindsight, more diverse samples; the tracker after that.

V7 resume failure, V7r (2026-11-19). V7's next 12 h block (job 307690, queued afterany:306915) failed at 03:40:
task.load_state_dict refused the snapshot ("Changed task state keys: info"; info gains keys while stepping, a fresh task
lacks them). Any fix changes source_sha256 in the resume contract, so V7 continues as V7r: warm start from V7's u600
snapshot with a fresh optimizer and task state, code 672352c (branch v7-resume-fix = b028157 + info restored by name, so
V7r's own resumes should pass), job 308522 from 07:41.

Full-body motion tracker v1 and plan execution, 2026-11-19 (cluster job 308916, run
/l/users/konstantin.smirnov/runs/mimic/mimic_v1; code 9725c42, evaluation af552e0). mjlab's G1 tracking task
(BeyondMimic-style PPO, flat ground) on 120 min of the policies' own clean motion: 1138 clips stitched from the planner
v1 datasets (scripts/mimic_library.py), clip boundaries handled by cat_mjlab/mimic.py. 4096 worlds, 0.7 s/iteration,
20000 iterations in 4.4 h; at the end episodes run the full 10 s, body position error 3.9 cm. (SuperMimic: code not
public. NablaMimic: installing it on the cluster needs the user's approval.)
Plan execution (scripts/planner_track_eval.py, 150 held-out situations per type, 2 s open loop, plans = best of 16 with
guidance 2, executed motion scored against the map like a plan):
- model_19999: plans executed clean 93.6% (the policies' own motion 92.9%), falls 0.1%, pelvis tracking error 5.6 cm;
  worst original CAT 76%, procedural CAT 85%, side gaps 89%, table edges 91%. model_5000: 93.0%, 7.5 cm.
- Column probe (752 situations): executed plans clear the column 39% (model_5000 30%), the policies' motion 2%;
  71% of the plans clear it kinematically, so tracking error (~10 cm on these less typical motions) eats the margin.
- Choosing plans by clearance up to 8 cm instead of 2 cm: 93.8% / 40.6%. No gain, since the 16 samples are ~2.4 cm apart.
- Executed clean by the plan's own minimum clearance: under 3 cm 54% (114 plans), 3-6 cm 73% (154), 6-10 cm 87% (192),
  over 10 cm 99% (1635). The planner needs about a 10 cm margin for this tracker.
- Videos (scripts/planner_track_video.py, b71f522): 28, in the planner report (version 3).
- Bug found and fixed in the evaluation: the reset advances the reference one frame, so the clip-end wrap teleported
  robots on the last recorded step (a97bbfb).

Planner with a field reflex, COMBO1 (2026-11-20; branch feature/plan-reflex 1f35ff3, cluster job 310027, run
/l/users/konstantin.smirnov/runs/stationary_hands_20261113_COMBO1). The user asked for the combined system: the planner
plans ahead and replans periodically, the potential-field policy is the fast reflex. One network: TRACK1 (V7 + the
41-input reference block) with planner v1 in the loop as a fourth reference source ('plan', cat_mjlab/planner/online.py):
from the measured state, toward a goal 2 s ahead on the guidance path at the operator speed, end heading along the path
or turned 90 deg either way (4 samples, 6 Euler steps, bf16, guidance 2), the sample with the most body-sphere clearance
up to 10 cm, minus 0.02 for a sideways pass and 0.1 per m (plus 0.2 m per rad) of disagreement with the replaced chunk;
replanned every 0.3-1.0 s (random), the chunk played on its own waiting clock. The reflex rule: an arm whose hand is
within 15 cm of an obstacle (thrown objects included) drops out of track_upper (its share is not paid); the pelvis term
tolerates max(0, min(0.2, 0.35 m - body clearance)) of error. Sources plan 0.5, field 0.15, random 0.1, none 0.25; half
the reactive standing worlds on a stand-still plan. Warm start V7r u400 (first launch: u600; zero-weight reference inputs), 50 critic-only
updates, anchor E2 u300 on reference-free worlds. Evaluations: scripts/eval_per_scene.py --tracker-mix plan=1 | none=1,
scripts/eval_standing_hands.py --hold-plan 1 (cluster ~/jobs/eval_combo.sh).
- Checks: local env with V7r driving (32 worlds): every plan world gets a chunk at reset, pelvis error ~7 cm, the reflex
  rule active in 1.5-4% of steps, 29% of replans pick a sideways goal heading. Plan quality with 6-8 Euler steps in
  bf16 equals 16 steps in fp32 (600 held-out situations: 100% clear, goal error 1.5 cm).
- Smoke (3 updates): 74 s/update (V7r 69 s), GPU peak 30.1 GB (V7r 30.9 GB). Two failed smokes first: the new worktree
  lacked untracked outputs/distill_v1_prep files (SGS init), then a CUDA OOM at Warp graph creation after the first
  reset planned ~6,900 worlds at once (fix: blocks of 256 worlds, grid and scoring in small pieces: 0.4 GiB peak).
- First launch (job 310027) stopped after 3 critic-only updates: track/track_released was 0.0 because --tracker-reflex
  passes {} (the default bands) and the config builder dropped it as falsy, so the reflex rule never ran (fixed in
  3cbb69d, with a test). Its run dir is kept as stationary_hands_20261113_COMBO1_noreflex_aborted. Relaunched as job
  310037 from worktree ~/cat_worktrees/3cbb69d.
- Baseline for COMBO1, V7r u600 on the 384 narrow scenes (~/jobs/eval_combo.sh, runs/speed_sweep/V7r_u600; clean /
  contact / refused at 0.15 m/s, 0.3 m/s in brackets): side gaps 0.42 / 0.51 / 0.06 (0.31 / 0.55 / 0.14), cabinet
  passages < 0.60 m 0.63 / 0.30 / 0.07 (0.74 / 0.19 / 0.07), >= 0.60 m 0.83 / 0.14 / 0.03 (0.89 / 0.11 / 0.00), CAT
  0.23 / 0.67 / 0.07 (0.25 / 0.65 / 0.08), rooms 0.89 / 0.11 / 0.01 (0.92 / 0.08 / 0.00), all 0.58 / 0.36 / 0.05
  (0.62 / 0.32 / 0.05). Against V7 u150 at 0.15 m/s: fewer refusals, but contacts in narrow cabinet passages rose from
  3% to 30%. The robot runs V7r u400 (dynamic_V7r_u400): its narrow evaluation is queued (job 310075).
- V7r u600 standing test (2000 rows, nominal, tablegap_v1 static bank; eval_standing_hands.py now defaults to it for
  goal-seed checkpoints): events handled 0.68 (danger 0.98, anticipation 0.98, pass-by 0.11: it flinches), contact per
  event 0.000, mean threat retreat 0.257 m, threats with >= 15 cm retreat 0.886, >= 25 cm 0.576.
- V7r u400 (the robot's dynamic_V7r_u400) on the same 384 scenes is the best of the line: all 0.67 / 0.26 / 0.07 at
  0.15 m/s (0.65 / 0.29 / 0.06 at 0.3), cabinet passages < 0.60 m 0.80 / 0.06 / 0.14, >= 0.60 m 0.99 / 0.01 / 0.00,
  side gaps 0.46 / 0.46 / 0.06, CAT 0.22 / 0.67 / 0.11, rooms 0.96 / 0.04 / 0.00. From u400 to u600 cabinet-passage
  contacts rose fivefold (6% to 30% at 0.15 m/s). COMBO1 was therefore stopped after 77 updates (run dir kept as
  stationary_hands_20261113_COMBO1_from_u600_stopped) and restarted from V7r u400 (job 310177): the strongest base,
  and the policy the robot runs, so the comparison is against the robot's policy.
- COMBO1 u200 (from V7r u400), every episode on planner plans (eval_per_scene --tracker-mix plan=1), 384 narrow
  scenes, clean / contact / refused at 0.15 m/s against V7r u400: side gaps 0.46 / 0.38 / 0.16 (0.46 / 0.46 / 0.06),
  cabinet < 0.60 m 0.77 / 0.04 / 0.19 (0.80 / 0.06 / 0.14), >= 0.60 m 1.00 / 0.00 / 0.00 (0.99 / 0.01 / 0.00), CAT
  0.19 / 0.47 / 0.34 (0.22 / 0.67 / 0.11), rooms 0.62 / 0.14 / 0.24 (0.96 / 0.04 / 0.00), all 0.58 / 0.20 / 0.21
  (0.67 / 0.26 / 0.07). Contacts fall everywhere, refusals rise, rooms collapse. Standing with a stand-still plan:
  danger 0.991, anticipation 0.981, contact 0.0003 per event, threat retreat 0.248 m (V7r u600: 0.984 / 0.980 / 0 /
  0.257 m).
- Rooms diagnosis: plan and field references were built along the stored goal field (gf), but room worlds navigate a
  route. Traced from the start pose in three rooms that refused, the gf streamline settles 3-7 m away and oscillates
  in place (it never reaches the route's goal), so plans kept aiming at the wrong spot until the 280 s timeout. Fix
  (c2ab017): route worlds build references along their route polyline (navigation.route_direction; reaches the goal
  within 2-4 cm in those rooms). COMBO1 was stopped at update ~312 and continued as COMBO1b (job 310504) from COMBO1
  u300 with the fix, 10 critic-only updates (smoke 310500: 73 s/update, 29.2 GB).
- COMBO1 u200 with the plan input off (--tracker-mix none=1): all 0.66 / 0.26 / 0.07 at 0.15 m/s and 0.65 / 0.28 /
  0.07 at 0.3 m/s, the same as V7r u400 (0.67 / 0.26 / 0.07, 0.65 / 0.29 / 0.06): learning to follow plans cost no
  reactive skill; rooms 0.93 (V7r u400 0.96), so the room collapse is in plan following only. Side gaps at 0.3 m/s:
  plans 0.45 / 0.38 / 0.17, no plan 0.38 / 0.50 / 0.12, V7r u400 0.33 / 0.54 / 0.14. Standing without a plan: danger
  0.987, anticipation 0.979, contact 0.0002 per event, feet violations per settled step 0.011 (V7r u600 0.027).
- COMBO1b u150 (route fix), plans on, 0.15 m/s: all 0.61 / 0.19 / 0.20 (COMBO1 u200 0.58 / 0.20 / 0.21; V7r u400
  0.67 / 0.26 / 0.07); rooms 0.93 / 0.07 / 0.01 (fixed; was 0.62 / 0.14 / 0.24); CAT 0.13 / 0.45 / 0.42; cabinet
  < 0.60 m 0.70 / 0.05 / 0.25; side gaps 0.46 / 0.36 / 0.17. At 0.3 m/s: all 0.64 / 0.18 / 0.18 (V7r u400 0.65 /
  0.29 / 0.06). No plan: 0.63 / 0.28 / 0.09 (0.65 / 0.26 / 0.07 at 0.3). Standing with a stand-still plan: danger
  0.986, no contact, threat retreat 0.255 m.
- Why it refuses with plans (recording of COMBO1 u300 + route fix, runs/videos/COMBO1_u300_plan): in a hurdle course
  that stalled for 60 s the plans moved 0.45 m per 2 s while the robot moved 0.08 m/s. Replans restart the chunk at the
  robot and the clock waits for a lagging robot, so the reference point stayed 4 cm from the pelvis and standing still
  earned 91% of track_pos (clean runs: 2-3 cm, 95%); near obstacles the deadband forgave lag as well. The refusal
  equilibrium of H2 reappears inside the planner-reflex split. COMBO2 (00d3931, --tracker-commit): plan clocks do not
  wait and the deadband forgives sideways error only; staged in ~/cat_worktrees/00d3931, not launched (needs the
  user's go).
- 2026-11-21 overnight (user: "do experiments overnight, work autonomously"): COMBO2 launched (job 311901, from COMBO1b
  u500); H1 seed replicates (S2 seeds 1-2 to update 180, S1b seeds 1-2 to update 150, on the original commits 85d311a /
  0a6697f plus a run-directory suffix: seeds/S2 fa7996c, seeds/S1b 67f3676; jobs 311902-311905); COMBO3 (protection
  precision: --reactive-far-threat-only, e7cc12f) staged.
- COMBO1b u500 (end of its block), plans on, 0.15 m/s: all 0.56 / 0.17 / 0.27; cabinet < 0.60 m 0.55 / 0.01 / 0.44;
  side gaps 0.36 / 0.28 / 0.36. Refusals grow with training under the waiting clock (0.07 -> 0.20 at u150 -> 0.27 at
  u500) while contact falls (0.26 -> 0.19 -> 0.17). No plan: 0.64 / 0.29 / 0.07 (= V7r u400). Standing with a
  stand-still plan: danger 0.983, anticipation 0.973, no contact, pass-by 0.124.
- Released CAT generalist (Xue et al.; outputs/mjlab_migration_20260919/original-cat-expanded.npz) on the 384 scenes in
  our task (eval_per_scene with an .npz policy and --env-from V7r u400: native 222/310 inputs, same commands and
  success rules), clean / contact / refused (hand contact): 0.3 m/s all 0.33 / 0.64 / 0.03 (0.15), side gaps 0.01 /
  0.99 / 0.00 (0.70), cabinet < 0.60 m 0.33 / 0.66 / 0.02, >= 0.60 m 0.43 / 0.53 / 0.04, CAT courses 0.19 / 0.75 / 0.06
  (0.23), rooms 0.60 / 0.38 / 0.02; 0.525 m/s all 0.31 / 0.65 / 0.04 (0.13). V7r u400 at 0.525: 0.62 / 0.31 / 0.08
  (0.02); COMBO1b u500 plans at 0.525: 0.59 / 0.17 / 0.24 (0.02).

## Measurement notes
- Standing power, u150 clean: 26-31 W at rest (CPU trace), recovery steps after a shove 170-570 W.
- Training metrics `reactive/*` average over all env-steps: divide by `reactive/reactive_standing` (power) or
  `reactive/reactive_settled` (feet). In S1/S1b a violation ends the episode; in S2 it persists while the feet stay away,
  and S2 also walks back to its reset spot after pushes (spot_hold), which the shadow rule counts -- use the evaluation.
- Known limits: the anchor is dropped per scene, so flat-scene walking/keyboard episodes (~1.5% of transitions) are also
  unanchored; on resume, StandingObjects is rebuilt (reactive_fraction 0.25 until the next re-solve) and in-flight
  standing episodes re-settle.
