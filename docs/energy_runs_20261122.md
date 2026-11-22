# Energy pricing runs, 22 November 2026

The deployed training code is `bc51c10308c91bb78bbf96b9492c1efa7e4c38e3` on `feature/energy-price`, based on `feature/plan-reflex` at `5e6079e`. The isolated cluster checkout is `~/cat_worktrees/energy_price_bc51c10`. Its bank, prior, and checkpoint links point to the existing shared inputs.

## Submitted jobs

| Job | Recipe | Partition | GPUs | CPUs | RAM | Limit | Expected runtime | After | Output directory |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| 314865 | S2H smoke, 3 updates, 12,288 worlds, seed 0 | ws-ia | 1 | 12 | 54000M | 00:40:00 | 5–15 min | | `/l/users/konstantin.smirnov/runs/smoke_hands_S2H_314865` |
| 314946 | Standing u150, S1b seed 1 u150, S2 seeds 1–2 u180 | ws-ia | 1 | 8 | 48000M | 02:00:00 | 15–45 min |  | `/l/users/konstantin.smirnov/runs/energy_20261122_standing_314946` |
| 314947 | Tables T1c u640, T2 u410, E1 u500 | ws-ia | 1 | 8 | 48000M | 02:00:00 | 15–40 min |  | `/l/users/konstantin.smirnov/runs/energy_20261122_tables_314947` |
| 314948 | 384 scenes, V7r u600 | ws-ia | 1 | 8 | 48000M | 03:00:00 | about 50–65 min | 314946 | `/l/users/konstantin.smirnov/runs/energy_20261122_v7r_314948` |
| 314949 | 384 scenes, COMBO1b u500, plans on | ws-ia | 1 | 8 | 48000M | 03:00:00 | about 75–100 min | 314947 | `/l/users/konstantin.smirnov/runs/energy_20261122_combo1b_314949` |
| 314950 | S2H, 50 updates from S2 seed 1 u180, seed 0 | ws-ia | 1 | 12 | 54000M | 03:00:00 | about 70–95 min | 314948 | `/l/users/konstantin.smirnov/runs/stationary_hands_20261113_S2H` |
| 314951 | COMBO_NOPRICE, 50 updates from COMBO1b u500, seed 0 | ws-ia | 1 | 12 | 54000M | 03:00:00 | about 65–95 min | 314949 | `/l/users/konstantin.smirnov/runs/stationary_hands_20261113_COMBO_NOPRICE` |

Slurm initially requeued and held smoke job 314865 with `user_env_retrieval_failed_requeued_held`. After the shell profile fix and release, the same job completed all three updates in 15m 40s. The training metrics recorded finite policy and applied energy values for both models. The smoke log is `/l/users/konstantin.smirnov/runs/logs/hands_energy_smoke_314865.out`.

The follow-up jobs use two dependency chains with `afterany`. The runtimes are estimates after allocation and exclude queue and dependency waits. All jobs use W&B offline. Evaluation stdout goes to `/l/users/konstantin.smirnov/runs/logs/energy_<case>_<job>.out`. The eval batch script is `~/jobs/energy_eval_bc51c10_20261122.sbatch`. The 384-scene jobs use the retention list, two episodes per scene, and speeds of 0.15 and 0.3 m/s.

The submitted training jobs override the recipe defaults with 50-update weights-only fine-tunes. S2H loads `/l/users/konstantin.smirnov/checkpoints/stationary_hands_S2_seed1_u180.pt` and keeps the original u150 anchor. COMBO_NOPRICE loads `/l/users/konstantin.smirnov/checkpoints/plan_reflex_COMBO1b_u500.pt` and keeps the E2 u300 anchor. Both use fresh optimizer state with pretrained policy weights.

Standing evaluation uses the available checkpoints at the requested updates. A search of the shared runs and checkpoint archives found no seed-0 S1b u150 or S2 u180 snapshot. The Nov. 13 evaluation JSON references temporary copies that are absent. S1b seed 2's current checkpoint is u220, and its u150 file was also unavailable.

## Recipes and measurements

- S2H uses S2's standing price of −0.02 with the heat model, alpha 500 W, and the u150 warm start. The full budget is 180 updates, matching the S2 seed reruns.
- COMBO_NOPRICE uses COMBO1b's 500-update recipe, warm-started from `plan_reflex_COMBO1_update_0300.pt` with 10 critic-only updates and the E2 u300 anchor. It removes the explicit standing, table, leg-and-waist, global, and route energy prices, including inherited terms. It retains the shared native joint-torque regularizer.
- Existing launch recipes omit `--energy-model` and retain the historical final-sample mechanical reward. Explicit `mech`, `heat`, and `metab` charge clipped PD plus gravity feed-forward, timed over each physics substep. The route offset `--route-energy-p0` defaults to zero. The heat model uses joint torque limits with a shared alpha because motor masses were unavailable.
- Energy telemetry reports policy and applied torque for both mechanical and heat models, by legs, waist, and arms. Heat's positive net work is allocated across groups with positive net work in proportion to their contribution. Standing events cover the active approach through return to the object start. A step requires at least 5 cm horizontal travel and 2 cm lift of the same foot. The arm-first retreat measurement projects the targeted hand's torso-relative displacement onto the initial retreat direction.

## Round 2 — motor-size heat and standing response classes

Code `f93f1c2` is deployed in `~/cat_worktrees/energy_price_f93f1c2`. The original equal-heat S2H job 314950 had already completed 50 updates. The new jobs write separate outputs under `/l/users/konstantin.smirnov/runs/`. All use one GPU and W&B offline.

| Job | Purpose | Partition | CPUs | RAM | Limit | ETA after allocation | Output directory relative to runs |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| 315346 | Standing re-score of u150, S1b seed 1, S2 seeds 1 and 2 | ws-ia | 8 | 48000M | 02:00:00 | 20–45 min | `energy_round2_standing_315346` |
| 315347 | Retry COMBO1b 384 scenes at 0.15 and 0.3 m/s | gpu | 8 | 48000M | 03:00:00 | 75–100 min | `energy_round2_combo1b_315347` |
| 315348 | Motor-weighted S2H, 50 updates | ws-ia | 12 | 54000M | 03:00:00 | 70–95 min | `stationary_hands_20261113_S2H_peak` |

Retry 315347 uses a partition with disjoint nodes and explicitly excludes `ws-l1-[001-021],ws-l2-[001-021]`. The failed 314949 log had no hostname, the controller had purged the job, and the accounting service was unavailable. There was no exact node name to recover.

The old standing JSON saved group-level equal heat and movement labels without hand side or event success. Round 2 reruns the same 2,000 rows and 800 control steps. Completed successes are classified by torso-relative hand retreat, stepping and pelvis displacement. Active approaches cut short by a fall or contact are failures. Events still open at a time limit are censored.

| Actuator class | Joint torque limit (N·m) | Speed limit (rad/s) | Peak (W) | Heat weight |
| --- | ---: | ---: | ---: | ---: |
| 7520-22 | 139 | 20 | 2780 | 1.000000 |
| 7520-14 | 88 | 32 | 2816 | 1.012950 |
| 5020 | 25 | 37 | 925 | 0.332734 |
| 4010 | 5 | 22 | 110 | 0.039568 |
| Two parallel 5020 motors | 50 | 37 | 1850 | 0.665468 |

The class mapping follows mjlab 1.6.0's G1 actuator expressions. Ankles and waist pitch/roll use its parallel 5020 assignment at nominal 1:1. The reference peak is 2780 W and alpha stays 500 W. `heat_eq` preserves the old uniform normalized price. Fifteen focused CPU tests passed.

The matched baseline for each bucket and target side is the still-success energy divided by still-success duration. It is multiplied by each response event's duration before subtraction. The comparison weights both classes to the same observed bucket/side mix on common support. The k* estimator divides total policy fixed-heat extra joules per body-first event by the arm-first estimate. The 95% percentile interval uses 2,000 resamples of entire benchmark rows. S2 seed 1 is the calibration policy because it is the flip warm start.

### Flip pilot plan

All arms start from `/l/users/konstantin.smirnov/checkpoints/stationary_hands_S2_seed1_u180.pt`, use the fixed heat reward for 50 updates and keep the u150 anchor, seed 0 and the same standing benchmark. The fixed-heat S2H run at unit prices is the control. The arm multipliers are k*/3, k* and 3k*, with a mirror leg multiplier of 1/k*. Use arm multipliers 2, 8 and 30 and mirror 1/8 if k* is undefined or the CI is missing or more than fourfold wide. The planned effect is at least a five-percentage-point decrease in arm-first share near k*, with at least half that change absorbed by body-first successes. An increase in failures that absorbs the shift refutes the predicted response substitution. This is a single-seed pilot on the calibration benchmark.

### Standing re-score and flip submissions

Standing re-score 315346 completed. The detailed report with count, mean watts, joules per event and matched excess by actuator group is `/l/users/konstantin.smirnov/runs/energy_round2_standing_315346/baseline_matched_summary.json`. The raw approach export is `standing.json` in the same directory.

S2 seed 1 had −7.84 J of matched policy fixed-heat excess per arm-first event and +86.43 J per body-first event across four common-support strata. The negative arm-first estimate leaves k* undefined as a positive price ratio, so there is no applicable 95% interval. Only 18 body-first events contributed. The fallback schedule uses arm prices 2, 8 and 30 and mirror leg price 0.125. This benchmark describes matched event energy and cannot on its own establish a causal effect of price.

| Policy | Arm-first | Body-first | Still success | Failure |
| --- | ---: | ---: | ---: | ---: |
| u150 | 626 (9.8%) | 246 (3.8%) | 3690 (57.5%) | 1858 (28.9%) |
| S1b seed 1 u150 | 1462 (22.9%) | 35 (0.5%) | 3159 (49.6%) | 1715 (26.9%) |
| S2 seed 1 u180 | 1306 (20.4%) | 18 (0.3%) | 3359 (52.4%) | 1725 (26.9%) |
| S2 seed 2 u180 | 1481 (23.5%) | 17 (0.3%) | 3218 (51.0%) | 1597 (25.3%) |

All jobs below use `ws-ia`. Each training job uses 1 GPU, 12 CPUs and 54000M RAM with a 3 h limit and a 70–95 min expected runtime. Each evaluation uses 1 GPU, 8 CPUs and 48000M RAM with a 75 min limit and a 7–15 min expected runtime. All four trained flips use the S2 seed 1 u180 warm start, the u150 anchor, and 50 updates. An eval runs after successful training. Two dependency chains limit concurrent jobs. Outputs below are relative to `/l/users/konstantin.smirnov/runs/`.

| Job | Purpose | After | Arm price | Leg price | Output |
| --- | --- | --- | ---: | ---: | --- |
| 315381 | Standing eval S2H_peak | 315348 |  |  | `energy_round2_trained_315381` |
| 315382 | Train Flip_arm_low | 315381 | 2 | 1 | `stationary_hands_20261113_Flip_arm_low` |
| 315383 | Standing eval Flip_arm_low | 315382 |  |  | `energy_round2_trained_315383` |
| 315384 | Train Flip_arm_high | - | 30 | 1 | `stationary_hands_20261113_Flip_arm_high` |
| 315385 | Standing eval Flip_arm_high | 315384 |  |  | `energy_round2_trained_315385` |
| 315386 | Train Flip_arm_center | 315383 | 8 | 1 | `stationary_hands_20261113_Flip_arm_center` |
| 315387 | Standing eval Flip_arm_center | 315386 |  |  | `energy_round2_trained_315387` |
| 315388 | Train Flip_leg_mirror | 315385 | 1 | 0.125 | `stationary_hands_20261113_Flip_leg_mirror` |
| 315389 | Standing eval Flip_leg_mirror | 315388 |  |  | `energy_round2_trained_315389` |

Video array 315390 runs after 315387:315389. Each task uses 1 GPU, 8 CPUs and 48000M RAM, with a 2 h 30 min limit and an estimated 15–45 min runtime. The array allows two concurrent tasks and writes each policy to its own subdirectory under `/l/users/konstantin.smirnov/runs/energy_videos/round2_315390/`. Every policy uses seed 0 and the same files by scenario: `danger_row0_seed0.mp4`, `anticipation_row1_seed0.mp4`, `negative_row2_seed0.mp4`. These are planned outputs until the array completes.

## Round 3 — planner refusals and longer price flips

Code `3a6f7db` is deployed at `~/cat_worktrees/energy_price_3a6f7db`. The original queued jobs 315781–315788 were canceled before startup and replaced after a CPU regression identified and fixed a scalar handsdf-knee input where a per-world tensor was required. The running center-flip and video array were left untouched. The no-price benchmark uses the same two-speed, 384-scene protocol and original evaluator as 315347. No matched lower-refusal COMBO4/5 checkpoint was found in the run log or retained cluster paths. Both COMBO trainings use `/l/users/konstantin.smirnov/checkpoints/plan_reflex_COMBO1b_u500.pt` and the original E2 u300 anchor.

| Job | Purpose | Partition | GPU / CPUs / RAM | Limit | ETA after allocation | Waits for | Output under runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 315778 | COMBO_NOPRICE final u50, 384 scenes | gpu | 1 / 8 / 48000M | 3 h | 1–2 h | — | `energy_round3_noprice_combo1b_315778` |
| 315808 | COMBO_FIX, 150 updates | ws-ia | 1 / 12 / 54000M | 6 h | 3–4 h | 315390 | `stationary_hands_20261113_COMBO_FIX` |
| 315810 | COMBO_FIX, 384 scenes | ws-ia | 1 / 8 / 48000M | 3 h | 1–2 h | 315808 | `energy_round3_COMBO_FIX_315810` |
| 315809 | COMBO_NOREFLEX, 150 updates | ws-ia | 1 / 12 / 54000M | 6 h | 3–4 h | 315390 | `stationary_hands_20261113_COMBO_NOREFLEX` |
| 315811 | COMBO_NOREFLEX, 384 scenes | ws-ia | 1 / 8 / 48000M | 3 h | 1–2 h | 315809 | `energy_round3_COMBO_NOREFLEX_315811` |
| 315812 | ×30 exact checkpoint continuation, +100 updates | ws-ia | 1 / 12 / 54000M | 6 h | 2–3 h | 315810 | `stationary_hands_20261113_Flip_arm_high_u150` |
| 315814 | ×30 u150 standing eval | ws-ia | 1 / 8 / 48000M | 90 min | 7–15 min | 315812 | `energy_round3_Flip_arm_high_u150_315814` |
| 315813 | ×100 warm-start from S2 seed 1 u180, 150 updates | ws-ia | 1 / 12 / 54000M | 6 h | 3–4 h | 315811 | `stationary_hands_20261113_Flip_arm100` |
| 315815 | ×100 u150 standing eval | ws-ia | 1 / 8 / 48000M | 90 min | 7–15 min | 315813 | `energy_round3_Flip_arm100_315815` |

Both COMBO variants use the .5 m continuing-plan rule and COMBO2 committed clock with lateral-only pelvis deadband. FIX releases arm tracking within .05 m. NOREFLEX keeps arms tracked globally at hand band zero while retaining the same lateral pelvis deadband. The hand and elbow margins are .05 and .04 m, and the handsdf knee is .03 m before existing room-width scaling. Traversal evaluations save transition-averaged reward terms for eventually refused episodes. Standing evaluations save threatened-event displacement quantiles and raw per-event values. The ×30 run copies the original u50 runtime into its own output directory and uses the original `f93f1c2` trainer snapshot for strict optimizer and simulation-state restoration. The ×100 run uses `stationary_hands_S2_seed1_u180.pt` plus the u150 anchor.

### E — CPU standing-energy reference

Job 315819 is a CPU-only allocation on `gpu` with `--gres=none`, 12 CPUs, 60000M RAM and a six-hour limit. Expected runtime is 30–90 minutes after allocation. Code `e73bf29` is deployed in `~/cat_worktrees/energy_price_e73bf29`. Output is `/l/users/konstantin.smirnov/runs/energy_round3_reference_315819` and stdout is `/l/users/konstantin.smirnov/runs/logs/energy3_reference_315819.out`. The inputs are S2 seed 1 u180, fixed-heat S2H u50, and the existing four-checkpoint standing export. All 20 focused tests passed. The original CPU job 315797 was canceled while pending and replaced after a single-support regression exposed the limits of the initial outward-only step templates.

The CPU replay uses the same authored rows and seed 0 as benchmark 315346, sampling up to 60 rows per threat bucket. The smaller CPU batch has its own realized trajectories. Each motor comparison uses the replayed event's initial pose, targeted hand, approach duration until object reversal, and achieved clearance in an enclosing swept-capsule proxy. When clearance did not improve relative to the initial hand position, the optimization targets at least .10 m clearance. The policy efficiency comparison excludes those substituted targets. Policy excess is relative to still-success W matched by policy, bucket and hand side, scaled to the same approach window.

Arm endpoints are searched with bounded IK and joined to the initial pose by minimum-jerk joint interpolation with both feet welded and the pelvis fixed. The body-only search holds arm joints fixed, varies pelvis and waist, and uses leg IK to preserve both feet. The step search checks inward and outward scripted lateral swings on each side across three step lengths, with stance-foot support until landing. MuJoCo inverse dynamics and the weld reactions estimate torques, with a minimum-norm correction for residual floating-base wrench. Candidate rejection checks torque and speed limits, net friction and the ZMP against rectangular support polygons. The estimate assumes rest at each endpoint and uses finite differences, simplified support geometry, and an endpoint clearance target. Feasible energies are the lowest among the searched candidates. These finite searches do not establish a global minimum or certify collision-free real-world motion. The planned sample is 75 completed events per policy per threat bucket. `reference.json` contains the event-level estimates, feasibility summaries, k_phys and same-event efficiency ratios.

### Baseline threatened-event motion for the extensions

Quantiles below are cm and use all danger and anticipation entries, including failures. Values are median / 90th percentile of the recorded maximum displacement within the event. The earlier episode-level feet-violation rate measures a different population.

| Checkpoint | Eval | Events | Feet | Pelvis | Torso-relative hand retreat |
| --- | ---: | ---: | ---: | ---: | ---: |
| S2 seed 1 u180 | 315346 | 4010 | .74 / 1.65 | 2.89 / 4.87 | 8.39 / 14.46 |
| Fixed-heat S2H u50 | 315381 | 3972 | .65 / 1.38 | 2.86 / 4.66 | 8.89 / 14.37 |
| Arm ×2 u50 | 315383 | 4030 | .70 / 1.61 | 2.94 / 4.99 | 8.61 / 14.34 |
| Arm ×30 u50 | 315385 | 4017 | .75 / 1.81 | 3.05 / 5.12 | 8.70 / 16.25 |
| Leg ×0.125 u50 | 315389 | 4029 | .74 / 1.57 | 2.84 / 4.65 | 8.67 / 14.56 |

### ×100 moved to dep-1

The ×100 train/eval pipeline started on dep-1 at 2026-11-23 01:33 Dubai time. It uses the same trainer commit `3a6f7db` in detached worktree `/tmp/cat-energy-dep1-3a6f7db`, with the dedicated CUDA environment `/tmp/cat-energy-test-env`. It runs one process chain under a local GPU lock on an RTX 6000 Ada (48 GB), 12,288 worlds, seed 0, 150 updates, W&B offline, then the same 2,000-row standing evaluation. The student is the cluster copy of `stationary_hands_S2_seed1_u180.pt` (SHA-256 `4e228d3ce6901b3dbb329e86e362b53d52ae3f1d2f23d8c5cc3299dc6c430147`, matching the prior S2H run contract) with the local u150 anchor. Both input names and the effective launch options are recorded by the pipeline.

Output is `/home/konstantinsmirnov/robotics/cat-energy-local-runs/Flip_arm100/` with `launch.log`, `run/status.json`, and a final `standing.json` on successful completion. The launch PID was 207579. Estimated training is 3–4 hours plus standing evaluation. Only the pending cluster counterparts 315813 and 315815 were canceled. The evaluated dependency list showed that 315815 was the sole downstream consumer of 315813.

The CPU reference 315819 was still pending at this check due to `QOSMaxCpuPerUserLimit`, with no output at `/l/users/konstantin.smirnov/runs/energy_round3_reference_315819/` yet.
