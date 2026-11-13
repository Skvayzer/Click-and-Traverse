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

## Measurement notes
- Standing power, u150 clean: 26-31 W at rest (CPU trace), recovery steps after a shove 170-570 W.
- Training metrics `reactive/*` average over all env-steps: divide by `reactive/reactive_standing` (power) or
  `reactive/reactive_settled` (feet). In S1/S1b a violation ends the episode; in S2 it persists while the feet stay away,
  and S2 also walks back to its reset spot after pushes (spot_hold), which the shadow rule counts -- use the evaluation.
- Known limits: the anchor is dropped per scene, so flat-scene walking/keyboard episodes (~1.5% of transitions) are also
  unanchored; on resume, StandingObjects is rebuilt (reactive_fraction 0.25 until the next re-solve) and in-flight
  standing episodes re-settle.
