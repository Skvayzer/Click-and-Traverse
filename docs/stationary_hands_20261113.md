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

- The gate works as designed: while stalled, E4's positive terms are zero and the step pays about -3.6/s (energy,
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
  then 70% at 600), the gated run did not (99%, 97%). At u400 E4's stalled steps pay -6.6/s and its steps inside
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

## Measurement notes
- Standing power, u150 clean: 26-31 W at rest (CPU trace), recovery steps after a shove 170-570 W.
- Training metrics `reactive/*` average over all env-steps: divide by `reactive/reactive_standing` (power) or
  `reactive/reactive_settled` (feet). In S1/S1b a violation ends the episode; in S2 it persists while the feet stay away,
  and S2 also walks back to its reset spot after pushes (spot_hold), which the shadow rule counts -- use the evaluation.
- Known limits: the anchor is dropped per scene, so flat-scene walking/keyboard episodes (~1.5% of transitions) are also
  unanchored; on resume, StandingObjects is rebuilt (reactive_fraction 0.25 until the next re-solve) and in-flight
  standing episodes re-settle.
