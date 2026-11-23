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

## Reference videos and matched standing contact controls

Reference job 315819 completed 300 threatened events. Arm-only was feasible on 298/300, lean/twist on 288/300, and one side step on 10/300. Median mech/heat J among feasible events were 0.565/31.12, 2.047/39.68, and 22.86/307.82 respectively. Event-matched median lean/arm ratios were 3.65 (mech) and 1.16 (heat). The signed same-event S2 and S2H mean-baseline excess ratios are in `/l/users/konstantin.smirnov/runs/energy_round3_reference_315819/reference.json`.

The renderer reconstructed the minimum-heat candidate for each feasible arm, lean/twist, or step and checked its heat against the saved result. Both numbers shown describe the same rendered trajectory. The shared scene uses the authored primitive dimensions and rotation and the stored sweep endpoints. Object translation during the clip reconstructs the sweep because frame-level object motion was not saved. Verified 1280×720 H.264 deliverables are under `/home/konstantinsmirnov/.local/share/openresearch/files/humanoid-protection/energy-videos/`.

- `reference-anticipation-row-0895-arm-lean-step.mp4`, 4.0 s
- `reference-anticipation-row-0778-arm-lean.mp4`, 3.0 s
- `reference-danger-row-1752-arm-lean.mp4`, 3.0 s

The standing contact comparison runs pin `01dd3c8` in `~/cat_worktrees/energy_contact_01dd3c8`. Both load `generalist_robust_delay_v1_u150.pt` as a native warm start with fresh optimizer, seed 0, 12,288 worlds and 150 updates. Both use the S2 threat bank, 800-step horizons, nominal standing resets, mixed threat buckets at training, and the S2 latency, mount, friction, mass and pose randomization. All resets are standing reactive to concentrate both arms on the intervention. There is no feet-planted rule, drift or stillness shaping.

| Job | Purpose | Partition | GPU / CPUs / RAM | Limit | Output below shared runs |
| --- | --- | --- | --- | --- | --- |
| 317952 | ENERGY_ONLY training and 2,000-row standing eval | ws-ia | 1 / 12 / 54000M | 8 h | `energy_contact_ENERGY_ONLY_317952` |
| 317953 | CONTACT_ONLY training and 2,000-row standing eval | ws-ia | 1 / 12 / 54000M | 8 h | `energy_contact_CONTACT_ONLY_317953` |
| 317954 | Four matched videos, array with at most two tasks running, after both evals succeed | ws-ia | 1 / 8 / 48000M per task | 90 min | `energy_contact_videos_317954` |

Both retain the existing collision event penalty −1 and terminal collision, hand self-contact, fall and numerical safeguards. The shared contact/stability rewards are `hand_body_contact`, `foot_balance`, `foot_slip`, `straight_knee`, `joint_limits`, `smoothness_action`, `smoothness_joint`, and `action_rate`. A constant `standing_alive` weight +12/s supplies a survival incentive without a pose target. ENERGY_ONLY adds only `reactive_power=-0.02` with `heat` and alpha 500 W. The heat term is applied outside the native positive reward floor. CONTACT_ONLY has no energy price term.

The profile zeros clearance, direction and retreat rewards (`handsdf`, `handsgf`, `wholebody_hand_clearance`, `wholebody_arm_clearance`, `reactive_event`), upper and torso posture shaping, arm-specific smoothness, ordinary torque cost, route progress and motion targets, and every other reward outside the listed shared whitelist. It removes inherited group-scale overrides, retreat-specific configuration, teleop and SGS. The launcher supplies no stance rule, hands-up term, teacher imitation or style prior. The first log line labeled `Standing contact profile` prints the full effective active weights and exact zeroed key list for each checkpoint contract.

The prediction is higher arm-first and lower body-motion rates with energy pricing at comparable event success. The shared no-price arm and u150 checkpoint provide controls. A lower apparent body-first rate caused by failures or lower successful response is inconclusive for arm substitution. This is a single-seed, 150-update pilot. The videos use authored danger rows 0 and 3, anticipation rows 1 and 4, and seed 0 for all three policies.

Jobs 317952 and 317953 failed during device initialization on `ws-l5-004` and `ws-l5-005`. In each error the CUDA driver reported a separate process using 28.63 GiB of the 31.46-GiB card. Neither policy reached an update. Slurm canceled their dependent array 317954. The retries pin `c115ef1` in `~/cat_worktrees/energy_contact_c115ef1`, exclude those two nodes, and log assigned-card memory use before launching CUDA. Both controls use 8,192 worlds and eight 1,024-world minibatches per update. The reward comparison, 150-update budget, warm start, seed, benchmark and recordings are as above.

| Job | Purpose | Partition | GPU / CPUs / RAM | Limit and expected runtime | Output below shared runs |
| --- | --- | --- | --- | --- | --- |
| 317956 | ENERGY_ONLY training and 2,000-row standing eval | ws-ia | 1 / 12 / 54000M | 8 h limit, roughly 2–4 h | `energy_contact_ENERGY_ONLY_317956` |
| 317957 | CONTACT_ONLY training and 2,000-row standing eval | ws-ia | 1 / 12 / 54000M | 8 h limit, roughly 2–4 h | `energy_contact_CONTACT_ONLY_317957` |
| 317958 | Four matched video tasks after 317956 and 317957, maximum two concurrent | ws-ia | 1 / 8 / 48000M per task | 90 min limit, roughly 15–35 min per wave | `energy_contact_videos_317958` |

The two retry logs report the same active training weights, apart from the priced arm's `reactive_power`. Retained in both: `action_rate=-0.02`, `foot_balance=-30`, `foot_slip=-0.5`, `hand_body_contact=-10`, `joint_limits=-1`, `smoothness_action=-0.001`, `smoothness_joint=-0.000005`, `standing_alive=12`, and `straight_knee=-30`. The external `collision_event` is −1 for both. ENERGY_ONLY has `reactive_power=-0.02`. CONTACT_ONLY omits `reactive_power`.

The emitted zero-weight list for both is `body_motion`, `body_rotation`, `feetdf`, `feetgf`, `foot_clearance`, `foot_contact`, `foot_far`, `goal_bonus`, `goal_progress`, `handsdf`, `handsgf`, `headdf`, `headgf`, `heading_align`, `joint_torque`, `kneesdf`, `self_clearance`, `shldsdf`, `sideways_bonus`, `spot_hold`, `stall`, `stand_still`, `stand_tall`, `torso_rate`, `tracking_orientation`, `tracking_root_field`, `upright`, `wholebody_arm_clearance`, `wholebody_hand_clearance`, `wholebody_hand_contrast_heading`, `wholebody_hand_contrast_region`, `wholebody_upper_clear_posture`, `wholebody_upper_target_acceleration`, and `wholebody_upper_target_velocity`. Additional optional rewards absent in both include `reactive_event`, `reactive_feet_hold`, the hand-up and far-clearance terms, style score, and distillation loss.

## Current 5 cm planner/reflex traversal videos

Job 317981 renders two policies on `gpu` with 1 GPU, 12 CPUs, 54000M RAM and a 2 h limit. Estimated runtime is 20–45 min. It pins `bca153c` in `~/cat_worktrees/energy_reflex_video_bca153c`. Cluster output is `/l/users/konstantin.smirnov/runs/videos/current_5cm_band_317981`, with stdout in `/l/users/konstantin.smirnov/runs/logs/current_5cm_video_317981.out`. Local destination is `/home/konstantinsmirnov/videos/planner_reflex/current_5cm_band/`, with `planner_reflex` and `reactive` subfolders.

The planned arm uses final `stationary_hands_20261113_COMBO_FIX/run/resume.pt` (job 315808, 150 updates), with continuing plans and a .05 m palm-clearance reflex band. The reactive arm uses the existing `slow_V7r_u400.pt` checkpoint with no tracker. Both use `~/jobs/video_scenes_v2.txt`, the same 11 `COMBO2_u500_v2/plans` scene IDs, tablegap_v1, seed 1, nominal deterministic actions, 3,000 control steps, and a .15 m/s command. The original camera geometry, 25 fps, blue 0.3 s look-ahead ghost and candidate lines are reused. Red arm coloring denotes active release from plan tracking. The shared clip window includes the planner's earliest active release, or its closest hand approach when every arm stays tracked. Where the reactive episode ends before the chosen window, its final segment is shown with its actual elapsed time in the overlay. Each clip is capped at 15 s.

Job 317981 completed in about 24 min. All 22 local outputs passed H.264, 1280×720, ≤15 s and pairing checks. The two `index.json` files give the matched filenames, scene IDs, elapsed-time ranges, and simulation outcomes. Four planner clips contain active release. The red right-arm highlight and 5 cm overlay were visually confirmed in `planner_reflex/02-side-gap-0-40-m.mp4`. The 0.58 m cabinet reactive episode ended before the planner window and uses its final 15 s. On these 11 illustrative scenes the planner rollout outcomes were 5 clean, 4 contact, 1 refusal and 1 still running at 60 s. Reactive outcomes were 8 clean and 3 contact.

## Round 3 completed outcomes

The two-speed evaluations completed 768 episodes per speed. Entries are percentages of clean / contact / refused episodes. Falls account for the remainder up to rounding. These are checkpoint comparisons with one training seed per ablation.

| Policy | Eval | 0.15 m/s | 0.30 m/s |
| --- | ---: | --- | --- |
| COMBO1b u500 | 315347 | 55.99 / 18.49 / 25.52 | 59.51 / 17.19 / 23.31 |
| COMBO_NOPRICE u50 | 315778 | 54.43 / 16.02 / 29.43 | 57.16 / 15.89 / 26.82 |
| COMBO_FIX u150 | 315810 | 59.77 / 30.08 / 10.03 | 61.59 / 25.52 / 12.37 |
| COMBO_NOREFLEX u150 | 315811 | 61.98 / 29.95 / 8.07 | 63.02 / 28.12 / 8.72 |

Refusal decreased under COMBO_FIX while contact increased. Turning off the 5 cm band yielded a further modest increase in clean rate, with hand-contact rates rising from 2.60% to 2.86% at 0.15 m/s and 2.34% to 2.73% at 0.30 m/s. `speed_0.15.json` and `speed_0.3.json` in both evaluation folders include transition-averaged refused-episode reward terms. For COMBO_FIX the logged `stall` term averaged −3.19 and −3.05, and `goal_progress` averaged +1.49 and +0.90 at the two speeds. The separate no-price fine-tune modestly reduced contacts and increased refusals at both speeds.

The ×30 continuation finished on the cluster and the ×100 run finished on dep-1 in 2 h 20 min of training plus about 5 min of eval. Both completed 150 updates and 2,000 standing worlds. Threat classes here exclude the negative bucket and include aborted failures. Native event success uses completed events across all three buckets. Foot/pelvis/hand entries are p50 / p90 / p95 maximum displacements in cm.

| Policy | Threat events | Arm-first / body-first / still-success / failure | Native event success | Worlds with feet violation | Foot motion | Pelvis motion | Torso-relative hand retreat |
| --- | ---: | --- | ---: | ---: | --- | --- | --- |
| ×30 u150 (315814) | 4035 | 29.64% / 1.96% / 51.95% / 16.46% | 74.48% | 25.80% | 0.92 / 2.57 / 4.05 | 3.17 / 5.78 / 6.93 | 8.29 / 15.51 / 17.25 |
| ×100 u150 (dep-1) | 3901 | 34.66% / 2.28% / 54.06% / 9.00% | 75.88% | 28.50% | 1.24 / 2.97 / 4.07 | 3.70 / 6.59 / 7.98 | 10.14 / 19.02 / 20.92 |

For these two runs, more foot and pelvis motion at ×100 coincided with more hand retreat and a larger arm-first share. The predicted arm-to-body substitution did not appear. The maximum foot displacement reached 5 cm on 3.99% and 3.67% of threats, respectively, so the contrast depends on the motion statistic.

The inverse reference median [p25, p75] J for arm, lean/twist and side-step was 0.565 [0.421, 0.764], 2.047 [1.239, 3.103], and 22.862 [22.094, 24.128] under mech, and 31.122 [21.364, 44.574], 39.681 [27.689, 59.606], and 307.818 [281.975, 360.692] under size-weighted heat. Paired lean/arm energy-ratio medians [p25, p75] were 3.653 [2.429, 5.081] mech and 1.162 [0.989, 1.636] heat. Mean-baseline excess / optimal arm-only ratios for S2 (52 eligible threats) had signed medians −1.76 mech and −0.093 heat. For S2H (64 threats) they were −1.71 and −0.232. Negative values indicate measured consumption below the matched still-success baseline over that window. The signed ratios do not measure an absolute efficiency gain.

## Round 4 paired training seeds and heat sensitivity

All tasks warm-start with fresh optimizer state for 150 updates, 12,288 worlds, and the prior recipe's remaining settings. The planner tasks exactly reuse trainer `3a6f7db`, `plan_reflex_COMBO1b_u500.pt` and the E2 u300 anchor. Training seeds 1 and 2 are matched across COMBO_FIX and COMBO_NOREFLEX, with jobs 315808 and 315809 supplying the existing seed 0 pair. The standing tasks reuse `f93f1c2`, `stationary_hands_S2_seed1_u180.pt` and the u150 anchor. The four standing tasks share training seed 1, leg multiplier 1, and λ = .02. The arm multiplier and α variations are given below. The two α runs share the same 150-update α500 arm×1 comparison, with the same seed and start.

Wrapper `68206db` is deployed as `~/jobs/energy_round4_68206db.sbatch` with SHA-256 `34edd4cbe6ad43c17a4e7c1685a32d1401e2c1389f0031a29993eb22cda16ede`. Each array permits two concurrent tasks. Array 318295 waits for standing comparison video array 317958 to reach a terminal state so its tasks do not compete for `ws-ia`. Each task evaluates its own final checkpoint after verifying 150 updates. Planner eval uses the same 384 scene IDs, two episodes per scene, both .15 and .30 m/s and forced planning. Standing eval uses the same 2,000 deterministic rows over 800 control steps. All tasks use 1 GPU and 54000M RAM. Expected from allocation is 5–6 h for planner tasks (8 h limit), and 3.5–4.5 h for standing tasks (6 h limit).

| Job | Purpose | Partition / CPUs | Training seed | Eval output below `/l/users/konstantin.smirnov/runs` |
| --- | --- | --- | ---: | --- |
| 318295_0 | COMBO_FIX | ws-ia / 12 | 1 | `energy_round4_COMBO_FIX_seed1_318295` |
| 318295_1 | COMBO_NOREFLEX | ws-ia / 12 | 1 | `energy_round4_COMBO_NOREFLEX_seed1_318295` |
| 318295_2 | COMBO_FIX | ws-ia / 12 | 2 | `energy_round4_COMBO_FIX_seed2_318295` |
| 318295_3 | COMBO_NOREFLEX | ws-ia / 12 | 2 | `energy_round4_COMBO_NOREFLEX_seed2_318295` |
| 318296_0 | S2H arm×30, α500 W | gpu / 8 | 1 | `energy_round4_Flip_arm30_seed1_318296` |
| 318296_1 | S2H arm×1, α500 W | gpu / 8 | 1 | `energy_round4_Flip_arm1_seed1_318296` |
| 318296_2 | S2H arm×1, α250 W | gpu / 8 | 1 | `energy_round4_Heat250_seed1_318296` |
| 318296_3 | S2H arm×1, α1000 W | gpu / 8 | 1 | `energy_round4_Heat1000_seed1_318296` |

Training directories use `stationary_hands_20261113_R4_<case>_<array-id>/run`. Report planner mean and sample s.d. across the three independent training seeds at each speed for clean/contact/refused/hand-contact rates, and seed-wise reflex-on minus reflex-off differences before averaging. For standing, compare threatened-event response shares and foot motion at matched seed 1, with the existing seed 0 ×30 result for context. The sensitivity comparison holds arm and leg prices equal and changes only α. The evaluator inherits each checkpoint’s α for heat telemetry, so compare motion and mechanical energy on common scales.

## Standing contact-control evaluations

Threats include danger and anticipation. The four response classes include terminal failures. "Standing eval event success" uses the native completed-event denominator across all three buckets. The feet violation column is the fraction of worlds ever tripping the shadow feet rule. The step column requires ≥5 cm foot travel and ≥2 cm lift on the same foot.

| Policy | Threat events | Arm-first / body-first / still-success / failure | Standing eval event success | Classified threat success | World feet violation | Per-threat step |
| --- | --- | --- | --- | --- | --- | --- |
| u150 | 4064 | 623 (15.3%) / 242 (6.0%) / 2356 (58.0%) / 843 (20.7%) | 71.1% | 79.3% | 48.8% | 4.8% |
| CONTACT_ONLY | 3824 | 877 (22.9%) / 11 (0.3%) / 2034 (53.2%) / 902 (23.6%) | 71.6% | 76.4% | 57.0% | 0.2% |
| ENERGY_ONLY | 2216 | 421 (19.0%) / 3 (0.1%) / 1272 (57.4%) / 520 (23.5%) | 68.9% | 76.5% | 7.6% | 0.1% |

Max displacement p50 / p90 / p95 in cm. Standing policy-attributable group powers in legs / waist / arms order.

| Policy | Feet | Pelvis | Torso-relative hand retreat | Mech W | Size-weighted heat W |
| --- | --- | --- | --- | --- | --- |
| u150 | 1.65 / 5.24 / 10.93 | 4.36 / 7.84 / 10.01 | 6.60 / 13.58 / 15.77 | 17.65 / 7.42 / 9.37 | 28.96 / 14.98 / 16.84 |
| CONTACT_ONLY | 1.39 / 2.99 / 3.69 | 3.09 / 4.78 / 5.47 | 6.12 / 15.14 / 18.01 | 15.16 / 5.23 / 8.14 | 25.79 / 10.57 / 16.06 |
| ENERGY_ONLY | 0.82 / 1.96 / 2.47 | 2.84 / 4.57 / 5.23 | 6.06 / 12.61 / 14.65 | 12.03 / 4.17 / 8.52 | 14.80 / 6.86 / 12.96 |

| Policy / bucket | Threat events | Arm-first / body-first / still-success / failure counts | Feet p50/p90/p95, cm | Pelvis p50/p90/p95, cm | Hand retreat p50/p90/p95, cm |
| --- | --- | --- | --- | --- | --- |
| u150 / danger | 1980 | 389 / 173 / 1177 / 241 | 1.98 / 7.90 / 14.13 | 4.71 / 8.84 / 11.23 | 7.99 / 15.37 / 16.80 |
| u150 / anticipation | 2084 | 234 / 69 / 1179 / 602 | 1.35 / 3.66 / 5.69 | 3.97 / 7.10 / 8.46 | 5.36 / 11.57 / 13.21 |
| CONTACT_ONLY / danger | 1821 | 557 / 8 / 989 / 267 | 1.60 / 3.30 / 4.07 | 3.34 / 5.09 / 5.98 | 7.66 / 16.83 / 19.46 |
| CONTACT_ONLY / anticipation | 2003 | 320 / 3 / 1045 / 635 | 1.23 / 2.68 / 3.29 | 2.86 / 4.40 / 5.01 | 5.04 / 12.66 / 15.68 |
| ENERGY_ONLY / danger | 1065 | 282 / 1 / 639 / 143 | 0.87 / 2.09 / 2.67 | 3.03 / 4.76 / 5.38 | 7.46 / 13.32 / 15.63 |
| ENERGY_ONLY / anticipation | 1151 | 139 / 2 / 633 / 377 | 0.77 / 1.79 / 2.24 | 2.68 / 4.30 / 5.03 | 4.85 / 10.74 / 13.36 |

Per-response threat energy uses policy-attributable torques. Mean W is energy divided by aggregate event duration. Each cell is legs / waist / arms.

| Policy / class | Threat events | Mean mech J | Mech W | Mean heat J | Heat W |
| --- | --- | --- | --- | --- | --- |
| u150 / arm_first | 623 | 54.20 / 23.58 / 27.33 | 17.58 / 7.65 / 8.86 | 77.60 / 48.47 / 49.04 | 25.17 / 15.72 / 15.90 |
| u150 / body_first | 242 | 108.64 / 27.02 / 31.72 | 29.51 / 7.34 / 8.62 | 169.84 / 51.25 / 51.59 | 46.14 / 13.92 / 14.01 |
| u150 / still_success | 2356 | 58.81 / 24.45 / 28.77 | 17.76 / 7.38 / 8.69 | 95.20 / 49.51 / 51.62 | 28.75 / 14.95 / 15.59 |
| u150 / failure | 843 | 43.95 / 19.43 / 22.79 | 16.52 / 7.30 / 8.57 | 70.90 / 39.42 / 41.74 | 26.64 / 14.81 / 15.69 |
| CONTACT_ONLY / arm_first | 877 | 45.03 / 15.72 / 24.38 | 14.45 / 5.04 / 7.82 | 73.83 / 30.54 / 50.87 | 23.68 / 9.80 / 16.32 |
| CONTACT_ONLY / body_first | 11 | 90.72 / 18.50 / 30.88 | 22.98 / 4.68 / 7.82 | 159.14 / 32.71 / 60.79 | 40.31 / 8.28 / 15.40 |
| CONTACT_ONLY / still_success | 2034 | 50.05 / 17.51 / 25.73 | 14.96 / 5.23 / 7.69 | 86.27 / 35.61 / 53.54 | 25.79 / 10.65 / 16.01 |
| CONTACT_ONLY / failure | 902 | 35.74 / 13.03 / 18.62 | 14.76 / 5.38 / 7.69 | 62.49 / 27.11 / 38.73 | 25.82 / 11.20 / 16.00 |
| ENERGY_ONLY / arm_first | 421 | 36.15 / 13.04 / 21.29 | 11.88 / 4.29 / 7.00 | 42.61 / 21.84 / 39.17 | 14.01 / 7.18 / 12.88 |
| ENERGY_ONLY / body_first | 3 | 139.54 / 32.07 / 52.12 | 22.46 / 5.16 / 8.39 | 194.07 / 52.92 / 82.13 | 31.24 / 8.52 / 13.22 |
| ENERGY_ONLY / still_success | 1272 | 34.00 / 12.93 / 20.87 | 11.20 / 4.26 / 6.87 | 45.12 / 22.35 / 37.22 | 14.86 / 7.36 / 12.26 |
| ENERGY_ONLY / failure | 520 | 25.70 / 9.97 / 15.73 | 11.18 / 4.34 / 6.85 | 33.43 / 17.50 / 28.47 | 14.55 / 7.62 / 12.39 |

Across the full benchmark, adding the energy term reduced policy-attributable heat from 52.42 to 34.61 W and max per-threat foot displacement p90 from 2.99 to 1.96 cm. Shadow feet violations fell from 56.95% to 7.65% of worlds. Arm-first decreased from 22.93% to 19.00% across all recorded threats, with similar classified failure (23.59% vs 23.47%). Body-first was rare in both (11 vs 3 events). The main shift was toward still-success. These are single-seed results and the u150 checkpoint is contextual.

Exposure differed substantially. Of 1,334 authored threat rows, CONTACT_ONLY recorded an approach in 1,308 and ENERGY_ONLY in 1,098. Across all 2,000 worlds, 2,039 vs 7,310 standing episodes ended, with zero falls in both evaluations. On the 1,090 rows that both recorded, the paired first-event arm-first share was 12.20% vs 15.14%. The ENERGY_ONLY minus CONTACT_ONLY difference was +2.94 percentage points (95% paired row bootstrap interval +0.64 to +5.32, 5,000 draws, seed 0). In those same rows, first-event p90 foot motion fell from 2.96 to 1.95 cm and p90 arm retreat fell from 12.11 to 11.19 cm. The bootstrap conditions on recording a threat in both policies and describes benchmark-row variation for this seed. The exposure imbalance and opposite shifts across event populations make the proposed arm-first substitution inconclusive.

All four matched standing videos were delivered under `/home/konstantinsmirnov/.local/share/openresearch/files/humanoid-protection/energy-videos/`. Each is an 8 s H.264 1280×720 three-panel clip at the same authored row and seed 0. Array 317958 delivered `standing-controls-danger-row-0003.mp4` and `standing-controls-anticipation-row-0001.mp4`. Tasks 317958_0 and 317958_3 completed the recordings, then failed when FFmpeg on `ws-l2-010` loaded conflicting XCB libraries. The two failed clips were recomputed and encoded one at a time on dep-1, with the same three checkpoints, recording arguments and `c115ef1` code. Local row 0 ran 15:57–16:05 Dubai and row 4 ran 16:05–16:13, each on one RTX 6000 Ada. The local u150 file hash matches the warm start recorded in both training contracts. Delivered `standing-controls-danger-row-0000.mp4` and `standing-controls-anticipation-row-0004.mp4` passed the same checks.

The pending Round 4 `gpu` tasks 318296_1, 318296_2 and 318296_3 were updated to 24000M host RAM each based on <12 GB measured host peaks in the prior same-recipe runs. 318296_0 retains its original 54000M request. All GPU, CPU, time, training and evaluation settings were retained. With two `ws-ia` planner tasks and one `gpu` standing task running, the remaining GPU tasks report a scheduler GRES quota.

## Furnished traversal clips for Keynote

Rendered on dep-1's RTX 6000 Ada under the single-GPU lock on 2026-11-23. The first pair completed at 17:59 Dubai and the remaining pairs at 18:16. All 12 files are under `/home/konstantinsmirnov/videos/cluttered/{planner_reflex,reactive}/`. The policies are final COMBO_FIX from 315808 and reactive `slow_V7r_u400.pt`. Every pair uses the same scene, seed 1 and a 0.15 m/s operator command. The living room uses the exact two rollouts captured by 317981. Five new pairs were recorded on dep-1 using the same planner weights and exact goal-column cache fields for their selected scenes, for up to 2,000 control steps. Code is `6e9cf84`.

| Filename in both folders | Authored scene ID |
| --- | --- |
| `scene_01_living_room.mp4` | `random-furniture-dense-train-004004-b43029cf3054` |
| `scene_02_dining_room.mp4` | `random-furniture-dense-train-004008-73ed3af356e1` |
| `scene_03_cabinet_room.mp4` | `contrast-train-20260920-r2-w0.580000-forward_protected-96a57ca86054` |
| `scene_04_meeting_room.mp4` | `random-furniture-dense-train-20261213-a64e30780040` |
| `scene_05_cluttered_workshop.mp4` | `random-generic_clutter-dense-train-20261045-50861c5526b3` |
| `scene_06_storage_room.mp4` | `random-generic_clutter-pilot-train-20261128-d0a7e2a3e991` |

The MuJoCo camera uses colored furniture boxes from the authored geometry, a tiled floor material, shadow-casting lighting and low cutaway walls for visibility. The captured physical trajectories use the original walls and fields. The live plan is blue and arms currently released inside the 5 cm band turn orange. Living room and meeting room contain active-release frames. The paired cameras use the same settings and planner time-window anchor, with shorter reactive clips where their episodes end sooner. All 12 clips were checked for 1280×720 H.264 yuv420p, 48 kHz stereo silent AAC, faststart atom order, and durations of 3.48–14.88 s.

### Round 4 arm×30 seed 1 result

Response shares use danger and anticipation events, including truncated failures. Feet violation uses all 2,000 worlds and `≥5 cm` is the threatened-event max foot-displacement statistic.

| Case | Threat events | Feet violation | Feet ≥5 cm | Body-first | Arm-first | Failure | Foot p50/p90/p95, cm | Pelvis p50/p90/p95, cm | Hand retreat p50/p90/p95, cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Flip_arm30_seed1 | 4077 | 2.45% | 0.27% | 0.25% | 31.71% | 18.79% | 0.72 / 1.54 / 1.95 | 2.81 / 4.72 / 5.53 | 7.54 / 17.27 / 19.18 |

Task 318296_0 completed all 150 updates and the 2,000-row standing evaluation. Native completed-event success was 75.42%, with zero recorded falls. The seed 0 ×30 run had 25.80% worlds with a shadow feet violation, versus 2.45% for seed 1. The matched seed 1 arm×1 evaluation is required for the pricing contrast.

### Round 4 planner seed 1 pair

Tasks 318295_0 and 318295_1 completed both speeds. Rates use the same 384 scenes and 768 episodes per policy per speed. Entries are clean / contact / refusal / hand contact percentages and paired differences in percentage points.

| Speed | Reflex on | Reflex off | On − off |
| --- | --- | --- | --- |
| 0.15 m/s | 62.63 / 31.38 / 5.99 / 2.60 | 62.76 / 27.99 / 9.24 / 2.34 | −0.13 / +3.39 / −3.26 / +0.26 |
| 0.30 m/s | 63.54 / 27.34 / 8.98 / 2.86 | 63.41 / 26.95 / 9.51 / 4.04 | +0.13 / +0.39 / −0.52 / −1.17 |

The clean count differs by one episode per speed for this pair. The seed 1 refusal contrast favors reflex on at both speeds and reverses the seed 0 direction. Tasks 318295_2 and 318295_3 are the third pair used in the final mean and sample s.d.

### Round 4 matched standing price flip

Response shares use danger and anticipation events, including truncated failures. Feet violation uses all 2,000 worlds and `≥5 cm` is the threatened-event max foot-displacement statistic.

| Case | Threat events | Feet violation | Feet ≥5 cm | Body-first | Arm-first | Failure | Foot p50/p90/p95, cm | Pelvis p50/p90/p95, cm | Hand retreat p50/p90/p95, cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Flip_arm30_seed1 | 4077 | 2.45% | 0.27% | 0.25% | 31.71% | 18.79% | 0.72 / 1.54 / 1.95 | 2.81 / 4.72 / 5.53 | 7.54 / 17.27 / 19.18 |
| Flip_arm1_seed1 | 4026 | 2.05% | 0.30% | 0.35% | 38.85% | 11.90% | 0.64 / 1.37 / 1.68 | 2.93 / 4.91 / 5.92 | 9.32 / 14.43 / 15.93 |

At fixed seed 1 and α500, arm×30 versus arm×1 changed arm-first by −7.14 pp, body-first by −0.10 pp, and failure by +6.89 pp. Per-threat feet p90 rose 0.17 cm while pelvis p90 fell 0.19 cm. These threat outcomes support an increase in failures in place of arm-first successes, with foot motion changes below the threshold for a class shift. Native completed-event success across all buckets was 75.42% versus 74.99%, and neither run fell. The higher arm-first share at seed 0 under the longer, higher price experiments and the very different seed 0 world-violation rate limit generalization from this one matched pair.

## Round 5 plan-blind obstacles and standing flip replication

The test asks whether the 5 cm reflex limits hand contacts when the route planner never receives the obstacle in its SDF. The test uses the fixed 384-scene traversal benchmark, two matched episodes per scene at each of 0.15 and 0.30 m/s, for both static reveal and moving-obstacle variants. All six frozen COMBO_FIX/COMBO_NOREFLEX checkpoints are evaluated. Seed 0 COMBO_FIX initializes the common starting-plan guide, without running or fitting a policy action. Every arm gets identical ordered scene IDs, initial qpos/qvel, fixed world-space box schedule, and two deterministic episodes per scene. The evaluator checks the starting-state match before measuring. A 5.4 × 5.4 × 17 cm mocap box is added to MuJoCo and potential-field observations. Only the unmodified packed bank is given to the planner. The static box appears 0.7 s before an initial-plan hand intercept. The moving box takes 0.7 s to approach its final intercept candidate from 15 cm away. The initial guide tests candidate clearance against static scene samples and the approved body envelope. The planned intercept does not depend on a tested policy's later action. Actual box reveal and physical inserted-object hand/body contacts are logged for every arm.

The primary prediction is at least a 3 percentage-point reduction in inserted-object hand contact under reflex on versus off at each speed, with fewer than five percentage points more refusals. A nonnegative hand-contact difference, or a large clean-traversal loss through extra refusals, fails this prediction. Static-bank runs from Rounds 3 and 4 are the on-map reference. Analyze all assigned episodes, and report placement coverage plus realized reveal before comparing conditional outcomes. Compute mean and sample s.d. over the three matched training seeds and average seed-wise on-minus-off differences. Use three paired seed 0 videos from predefined trace rows on the narrow-scene list. Training-free video traces use exactly the evaluated worlds.

The standing replication uses training seed 2 for α500 arm×30 and arm×1, 150 warm-start updates each from `stationary_hands_S2_seed1_u180.pt` with the u150 anchor, recipe `f93f1c2`; eval uses `3a6f7db`. Reuse the same standing motion and response outcomes. The seed 1 contrast shifted arm-first into failures, so an increase in failures again falsifies arm-to-body substitution for this seed.

The CPU benchmark regressions passed (15 tests). The short 12-world GPU smoke created 12 placements, exercised 11 actual reveals and recorded one physical inserted hand contact. dep-1 generated all four full schedules with 768/768 valid placements per speed/variant and exact shared initial states across static and moving. The evaluator pins `a197178` and the guide manifest pins bank `0ebc2cf93afba59319d6ec0306985e0e499e213ed388833a706ddffc9740dca2`. No blind training is involved.

| Job | Purpose | Partition/resources | Limit / ETA after allocation | Eval output |
| --- | --- | --- | --- | --- |
| dep-1 local, 23:25 Dubai | COMBO_FIX seed 0, all four blind protocols and video traces | one RTX 6000 Ada, dedicated CUDA environment | 4–6 h expected | `/home/konstantinsmirnov/robotics/cat-energy-local-runs/PlanBlind_COMBO_FIX_seed0` |
| 319349_0 | COMBO_NOREFLEX seed 0, plus traces | `ws-ia`, 1 GPU / 12 CPUs / 54000M | 8 h / 4–6 h | `energy_round5_blind_COMBO_NOREFLEX_seed0_319349` |
| 319349_1 | COMBO_FIX seed 1 | `ws-ia`, same | 8 h / 4–6 h | `energy_round5_blind_COMBO_FIX_seed1_319349` |
| 319349_2 | COMBO_NOREFLEX seed 1 | `ws-ia`, same | 8 h / 4–6 h | `energy_round5_blind_COMBO_NOREFLEX_seed1_319349` |
| 319349_3 | COMBO_FIX seed 2 | `ws-ia`, same | 8 h / 4–6 h | `energy_round5_blind_COMBO_FIX_seed2_319349` |
| 319349_4 | COMBO_NOREFLEX seed 2 | `ws-ia`, same | 8 h / 4–6 h | `energy_round5_blind_COMBO_NOREFLEX_seed2_319349` |
| 319350_0 | Standing arm×30, α500 W, seed 2, 150 updates then eval | `gpu`, 1 GPU / 8 CPUs / 24000M | 6 h / 3.5–4.5 h | `energy_round5_Flip_arm30_seed2_319350` |
| 319350_1 | Standing arm×1, α500 W, seed 2, 150 updates then eval | `gpu`, same | 6 h / 3.5–4.5 h | `energy_round5_Flip_arm1_seed2_319350` |

Cluster paths are relative to `/l/users/konstantin.smirnov/runs`. Training output tags are `stationary_hands_20261113_R5_<case>_319350/run`. The cluster deploy is `~/cat_worktrees/energy_plan_blind_a197178` with four exact local guide files staged in `blind_inputs/`. Array 319349 waits for 318295 to finish so the final seed 2 snapshots are available and the two-per-user `ws-ia` limit is respected. Both arrays allow two concurrent tasks within the scheduler's GPU quota. Round 4 α250 and α1000 proceed in 318296_2 and 318296_3. The 12 urgent furnished clips are complete.
