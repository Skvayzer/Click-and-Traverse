# CAT clutter checkpoint evaluation — 16 September 2026

Videos show actual deterministic policy rollouts of the saved best checkpoint at **157,286,400 training steps**, from W&B run `de6ae369`. Each video is at real time, with a two-second freeze at the terminal pose, a room map and the recorded outcome. Furniture is the exact primitive geometry used to construct the training obstacle fields.

## Which checkpoint

The 31.25% W&B point occurred at 288,358,400 steps. That exact model was not retained: stochastic clutter success was 14.0625%, so it did not replace the selected best. The retained model had logged 29.6875% deterministic and 26.5625% stochastic clutter success. These videos use that retained model, whose identity and file hashes were frozen before evaluation.

## Recorded evaluation

We evaluated the original fixed 16 layouts × 16 seeds in both action modes (512 episodes). All 64 deterministic clutter episodes were recorded. Their observed goal success was **20/64 = 31.25%**. The separately compiled stochastic replay achieved 12/64 = 18.75%. These are fresh replay results, distinct from the checkpoint’s stored training-validation scores.

| Layout | Successful deterministic episodes |
| --- | --- |
| Furniture 4001 | 5/16 |
| Furniture 4002 | 10/16 |
| Mixed clutter 5001 | 5/16 |
| Mixed clutter 5002 | 0/16 |

The 44 deterministic clutter failures were obstacle-field clearance violations. No clutter episode fell, timed out, left the room bounds or produced nonfinite state. Hand violations occurred in 29/64 episodes and elbow violations in 13/64; causes can overlap. The CAT retention portion reached 111/192 goals (57.8125%) without action noise and 108/192 (56.25%) with noise, and passed the existing retention gates.

These four clutter layouts belong to the training bank. The result measures this fixed benchmark and does not establish performance in unseen rooms. The six displayed clips deliberately show three successes and three failures; their 50% split is not the success-rate estimate.

## Evaluation-path sensitivity

A separate check used the unmodified `RetentionValidator` with its usual 25-step chunks on the same frozen weights, byte-identical scene fields and scene/seed identities. It scored **15/64 = 23.4375%** deterministic clutter success and **11/64 = 17.1875%** stochastic success. The recorder, which collects poses through a separately compiled graph calling the same step implementation, scored 20/64 and 12/64. The checkpoint’s stored in-training scores were 19/64 and 17/64.

| Evaluation | Deterministic clutter | Stochastic clutter |
| --- | ---: | ---: |
| Stored training validation | 29.6875% | 26.5625% |
| Recorded rollouts | 31.25% | 18.75% |
| Unmodified standalone validator | 23.4375% | 17.1875% |

Parameter loading preserved bitwise-identical float32 values. This establishes sensitivity to the evaluation path; the exact cause has not been isolated. The stored validation uses the full bank, while both standalone checks use the verified subset and different compiled execution graphs. Treat the 31.25% as the observed rate of these recorded episodes, not a reliably reproduced checkpoint score. The unmodified check still passed the existing CAT retention gates (113/192 deterministic and 108/192 stochastic CAT goals). No further evaluation jobs remain running.

## Video examples

The example links below open contact sheets. Full MP4s and an offline `START_HERE.html` player are on the Mac in `/Users/konstantinsmirnov/Downloads/CAT-Clutter-Rollouts-20260916`.

| Video | Seed | Episode length | Outcome |
| --- | --- | --- | --- |
| [01_furniture4001_success_seed09.mp4](assets/clutter-checkpoint-evaluation-20260916/01_furniture4001_success_seed09.contact-sheet.png) | 9 | 9.98 s | Goal reached |
| [02_furniture4001_failure_seed00.mp4](assets/clutter-checkpoint-evaluation-20260916/02_furniture4001_failure_seed00.contact-sheet.png) | 0 | 6.36 s | Hand clearance violation |
| [03_furniture4002_success_seed02.mp4](assets/clutter-checkpoint-evaluation-20260916/03_furniture4002_success_seed02.contact-sheet.png) | 2 | 8.22 s | Goal reached |
| [04_furniture4002_failure_seed08.mp4](assets/clutter-checkpoint-evaluation-20260916/04_furniture4002_failure_seed08.contact-sheet.png) | 8 | 5.16 s | Elbow clearance violation |
| [05_mixed_clutter5001_success_seed14.mp4](assets/clutter-checkpoint-evaluation-20260916/05_mixed_clutter5001_success_seed14.contact-sheet.png) | 14 | 6.84 s | Goal reached |
| [06_mixed_clutter5002_failure_seed12.mp4](assets/clutter-checkpoint-evaluation-20260916/06_mixed_clutter5002_failure_seed12.contact-sheet.png) | 12 | 3.68 s | Hand clearance violation |

Success clips use the lower-median successful duration per scene. Failures prefer runs lasting at least five seconds, then hand/elbow faults and longer duration; mixed clutter 5002 had no failure that long. Selection rules and hashes are in `selection.json`.

## Visible limitation

In clip 04, the robot visibly intersects the tabletop before the elbow-field violation terminates the episode. This is present in the saved rollout, not introduced by video rendering. CAT-style sparse field checks and field-only obstacle physics do not guarantee full-body mesh collision avoidance. A goal-success result alone should therefore not be interpreted as a physically collision-free traversal.

## Reproducibility

The checkpoint was copied under the selection lock and verified against its manifest. A 51-scene evaluation bank retains all 37 mandatory original slots plus the fixed validation scenes, using hardlinks to the original immutable arrays. It loads approximately 0.575 GiB of fields. Scene records, field files and robot/policy contracts were verified. The frozen training source is commit `67339e3aff66c467059e1fc3fbab4dd0d8ea4b52`; actor/critic observations are 222/310 with 29 actions and the saved bounded upper-body action distribution.

The recorder calls the frozen `RetentionValidator` step implementation, preserving all 16 scene ordinals for reset/noise keys, native disturbances, first clean root-and-both-feet goal, failure precedence, 50-step collision grace and 4000-step room horizon. Recording uses a separate compiled evaluation graph, so it is not a claim of bit-identical replay of the earlier in-training validation.

Video replay calls only MuJoCo `mj_forward` on saved poses; no motion is generated or corrected during rendering. Walls and collision-debug geometry are hidden for visibility. The furniture boxes remain opaque and keep their original dimensions and poses. Clearance violations refer to the training potential fields and protective spheres, not measured physical contacts with furniture meshes.

Raw trajectories, per-episode metadata and scene JSON are stored on this Mac under `/Users/konstantinsmirnov/research/CAT-Clutter-Rollouts-20260916/episodes`. `evaluation.json` contains every benchmark outcome; `preparation.json` and `validation_history.json` explain checkpoint provenance. Training continued while evaluation ran.
