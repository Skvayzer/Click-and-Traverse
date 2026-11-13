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

| | u150 (start) | S1b @ update 50 | S2 @ update 90 |
|---|---|---|---|
| episodes where a foot moved > 5 cm (or lifted > 0.1 s) | 47.7% | 25.5% | 7.7% |
| settled steps with feet displaced | 18.7% | 7.9% | 1.8% |
| mean foot travel from the settled spot | 3.8 cm | 2.6 cm | 1.5 cm |
| hand-protection events (success) | 6376 (0.71) | 0.72 | 0.73 |
| success danger / anticipation / negative | 0.87 / 0.73 / 0.56 | 0.88 / 0.73 / 0.57 | 0.90 / 0.78 / 0.55 |
| hand-object contact per event | 0.0002 | 0.0003 | 0.0003 |
| joint power, standing: total (legs / waist / arms) | 43.7 W (25.9 / 7.3 / 10.5) | 40.5 W (26.1 / 5.3 / 9.1) | 36.6 W (22.5 / 4.7 / 9.4) |
| falls | 0 | 0 | 0 |

Reading (mid-run, 06:20): pricing energy alone already keeps the feet still in 92% of episodes (u150: 52%) while hand
protection improves slightly; the saving comes from legs and waist, arm power is unchanged -- the robot protects its
hands with the arms and keeps the body still. Under training randomisation (latency, torque offsets, pose noise,
friction, pushes) the hard rule is much harder: S1/S1b still end ~90% of standing episodes by foot creep at update
50, falling slowly.

Traversal retention (training metrics, noisy): original CAT ~0.35, published ~0.55-0.6, side gap ~0.3 in both runs,
about 0.05-0.1 below u150's own fine-tune at the same update counts (half the experience is standing).

## Measurement notes
- Standing power, u150 clean: 26-31 W at rest (CPU trace), recovery steps after a shove 170-570 W.
- Training metrics `reactive/*` average over all env-steps: divide by `reactive/reactive_standing` (power) or
  `reactive/reactive_settled` (feet). In S1/S1b a violation ends the episode; in S2 it persists while the feet stay away,
  and S2 also walks back to its reset spot after pushes (spot_hold), which the shadow rule counts -- use the evaluation.
- Known limits: the anchor is dropped per scene, so flat-scene walking/keyboard episodes (~1.5% of transitions) are also
  unanchored; on resume, StandingObjects is rebuilt (reactive_fraction 0.25 until the next re-solve) and in-flight
  standing episodes re-settle.
