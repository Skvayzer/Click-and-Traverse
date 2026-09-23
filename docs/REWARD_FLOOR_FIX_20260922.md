> **Historical record — superseded.** Native training now uses actor 222 / critic 310, no authored box features or rewards, and starts from scratch. Old expansion utilities are retired; no checkpoint trim is supported or requested. Commands and old test paths below describe the historical experiment, not the current workflow. See [current removal report](NATIVE_CLEARANCE_REMOVAL_20260922.md).

# Protected reward floor — measured fix, no training

CPU-only, no policy/checkpoint loaded, no banks/checkpoints or observation definitions edited. Reproduce with `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv-mjlab/bin/python scripts/audit_reward_floor_cpu.py`. Machine-readable evidence: `outputs/protected_reward_floor_cpu/audit.json`. The existing PD audit now accepts a sample callback and bank path, retaining its original default behavior.

## Weight sensitivity (the main result)

The v6 bank has the same protected geometry/fields as v5. This audit replays all 12 protected scenes at their first-zone midpoint: 20 actual CPU MuJoCo PD steps per nominal/compliant hold, with stored fields, production reward evaluation and ten-substep collision checks. Identical trajectories at every weight; heading fixed at -5. No learned policy, synthetic walking velocity, or training-distribution estimate. Nominal holds terminate on their first step; subsequent frames are explicitly counterfactuals. The episode-valid population contains 12 terminal nominal frames plus 240 compliant-hold frames.

**Shaped sum below is before multiplying by dt=.02, and excludes collision events.** Clipping means sum < 0 under the old hard floor.

| Region weight | Valid mean | Valid p05 | Valid median | Valid p95 | Valid clipped / 252 | Full holds clipped / 480 |
|---|---:|---:|---:|---:|---:|---:|
| -1 | -0.787 | -4.479 | 3.100 | 9.544 | 60 / 252 = **23.81%** | 288 / 480 = **60.00%** |
| -3 | -1.517 | -5.164 | 2.636 | 8.400 | 75 / 252 = **29.76%** | 303 / 480 = **63.13%** |
| -5 | -2.246 | -5.848 | 2.140 | 7.256 | 96 / 252 = **38.10%** | 324 / 480 = **67.50%** |
| -10 | -4.070 | -7.824 | 0.947 | 5.273 | 111 / 252 = **44.05%** | 339 / 480 = **70.63%** |
| -20 | -7.719 | -12.579 | -1.937 | 4.559 | 165 / 252 = **65.48%** | 393 / 480 = **81.88%** |

| Region weight | Full mean | Full p05 | Full median | Full p95 |
|---|---:|---:|---:|---:|
| -1 | -32.246 | -84.661 | -19.572 | 8.498 |
| -3 | -33.504 | -86.497 | -20.837 | 7.417 |
| -5 | -34.761 | -88.336 | -22.102 | 6.475 |
| -10 | -37.904 | -92.933 | -25.397 | 4.561 |
| -20 | -44.190 | -102.127 | -32.388 | 4.194 |

The weight theory is **partly confirmed**: -3 to -20 causes another **90/252 = 35.71 percentage points** of valid clipping, and **90/480 = 18.75 points** of full-hold clipping. It is not the sole cause. Even -1 misses the historical 2.19% global rate by a large margin. The valid sample mean region cost is .364825; full-hold mean .628624, nominal-hold mean .920040. These populations differ from the previously reported policy cost ~.77. At cost .77, -20 contributes -15.4 to the pre-dt sum, or **-.308 per control step**, not -15.4 reward per control step. Mean valid hand-SDF contribution is -.095106/step; static nominal hand-SDF was -1.650595/step versus region -.366744/step.

## Implemented choice: (b), a local soft floor

Use **region -3, heading -5, `--hand-reward-soft-floor .2`**. The option defaults to zero for compatibility and is saved in the environment contract. It applies only to active, valid hand zones in protected/transition roles, including the existing .40 m protected approach ramp. Flat, clutter, CAT, narrow and inactive zones keep their exact old floor. Existing phase ramps blend the correction continuously. Weights remain signed costs; their sign is not reversed.

For shaped reward x after dt and width t=.2:

- x < t: f(x) = t² / (2t - x)
- x >= t: f(x) = x

This joins with matching value and slope at t; the negative tail is algebraic. The existing upper cap remains. At core phase, every finite negative reward in the measured range has strictly positive sensitivity. The added reward is at most t/2 = .1 per step. Collision events are still subtracted afterward. Since t < the production collision penalty of 1, smoothing cannot turn a collision-negative reward positive at fixed input. The floor never produces negative shaped rewards.

A positive region term inside the old clamp is **not sufficient by construction**: replacing -3*c with +3*(1-c) in these core samples still clips **39/252 = 15.48%** valid samples and **267/480 = 55.63%** full-hold samples. Other costs can erase it. The flat bonus analogy identifies a useful reward direction but cannot establish a positive total budget. Placing a bonus after the floor would preserve that term's signal but leave the other hand-clearance improvements clipped. A huge offset would alter survival/occupancy incentives more strongly. This fix instead retains ordering of the complete shaped ledger at fixed zone phase.

**Tradeoff:** this is not policy-invariant reward shaping. It adds a zone-local survival/occupancy incentive, changes value targets there, and compresses differences in negative sums. It does not make the derivative unity. Width .2 keeps the measured worst valid sensitivity about .009 while bounding the lift at .1; this is an engineering choice, not a fitted optimum. Do not infer preserved ordering across different zones/phase weights. No termination, truncation, discount, or bootstrap code changed; true failures still terminate and timeouts retain the existing bootstrap treatment. Removing the floor globally would change every task's return scale and failure-versus-survival tradeoff, with potential value-learning instability; that was not necessary here.

## Re-measurement and weight recommendation

At -3/-5 and width .2:

- **0/252 = 0%** valid protected-core hard-floor plateaus; **0/480 = 0%** full-hold plateaus.
- Raw sum remains negative on **75/252 = 29.76%** and **303/480 = 63.13%**. The fix removes the plateau; it does not claim to make those sums positive.
- Valid floor sensitivity: min **.008806**, median **.323579**, mean **.364037**. Full-hold min **.008669**. No measured negative-tail underflow or zero derivative.
- Mean lift over the hard floor at the same -3 weight: **+.055269/step** valid, **+.039978/step** full holds.
- Static compliant-minus-nominal ledger at -3: **+1.723701/step pre-floor**, **+1.113146/step final** (including the +1 collision-event advantage). Compare prior -20: **+2.035433**, **+1.099245**. The pre-floor raising advantage was already positive without escalating to -20.
- Full PD-hold advantage at -3: **+1.440267 pre-floor**, **+.460385 final**; prior -20 final **+.384849**. These post-terminal counterfactual deltas are not episode returns.
- Zero measured final reward-sign inversions across all 720 nominal/compliant/slew physical frames, both versus the -3 hard floor and versus the original -20 configuration.

Choose -3 rather than -20 because it preserves a moderate direct region signal without the additional 35.71-point clipping burden in the unsmoothed budget. This does not prove -3 optimal versus -1. Keep heading -5 because this aligned-pose audit does not calibrate heading strength, and changing it would also change the working narrow objective. Heading's static paired delta is zero. No reason to strengthen it was measured.

The historical global 2.19% and this 0% use different populations and are not a policy-level apples-to-apples comparison. On an authorized run, inspect `training/protected_core_reward_floor_clipping_fraction` with its sample count, plus the newly separate `training/protected_core_reward_pre_floor_negative`, `training/protected_core_reward_soft_floor_lift`, and `training/protected_core_reward_floor_slope`. PPO uses reward differences/advantages; these slopes diagnose reward sensitivity, not backpropagation through the simulator.

## Retention and launch

All **2351 non-protected/non-transition v6 rows** have no active region objective (including all 12 narrow rows), checked against actual scene metadata. Thus -20 to -3 changes none of their region contributions. Heading stays -5; flat bonus stays 1 with radius .40; foot-balance stays -30. Soft floor role gating preserves their rewards exactly. CPU tests cover default no-op, roles, inactive/phase masks, monotonic real hand-pose costs, derivatives, collision-event signs, all retention scene metadata, compiled graph capture, production audit results, configuration and launch parsing. Learning retention rates **cannot be guaranteed**: the shared policy can experience interference, and a from-scratch network does not inherit the old flat .30 / clutter .70 / CAT .135 / narrow .56 performance.

Prepared only: `bash configs/pilots/hand_reward_floor_v6_50.sh`. It uses v6's **30% protected + 15% transition** mixture; 50% bounded-IK seeding of protected resets leaves half unseeded; actor 254 / critic 342 from scratch; 60/20/10/10 selection; uncapped std; 50 updates; online W&B CAT-wholebody/skvayzer. No speed/tolerance ladder is added to this floor pilot. Startup uses existing command-bounded IK certification; banks are read-only. No training was launched.

## Honest outlook

Subjective, uncalibrated estimate: **about 50% (plausible range 30–70%)** that 50 updates produce repeatable nonzero strict unseeded protected-core compliance, rather than just one accidental frame. For sustained unseeded compliant passage acceptance, **about 20%**. These are judgment, not measured success probabilities. Explicit observations and restored reward sensitivity remove identifiable blockers, but certified holds still drift, commandable poses saturate joints, and learning walking/control from scratch is a harder problem than repairing an existing gait.

Watch evidence in this order: sufficient unseeded core exposure and termination mix; unseeded region cost/distance trending down; strict compliance recurring across multiple scenes/windows; seeded-after-100-step compliance rising above zero; clean unseeded passage acceptance and forward progress (reject standing reward exploitation). Verify clipped fraction near zero **and** useful floor sensitivity. Monitor value loss/target scale, KL, upper-body std/saturation and task-conditioned flat/clutter/CAT/narrow outcomes. A falling region cost with no strict compliance can be early progress; seeded-only gains or near-zero unseeded sample counts are not success. By updates 20–30, flat unseeded cost plus persistent post-seed collapse would reduce my confidence; do not respond by restoring -20. The 60% selection score includes seeded core steps, so the selected checkpoint alone is insufficient evidence.

## Verification results

**87 distinct CPU tests passed** across the targeted reward/observation/approach/heading/logging/runner/flat/IK suites. The 10 floor-specific tests also passed after the final upper-cap telemetry correction. Four existing reset integration fixtures skipped for unavailable local banks. Seven legacy JAX-reference checks in `test_mjlab_task.py` could not run: five failures and two setup errors are missing `ml_collections`, `jaxlie`, or `mujoco_playground`, not numerical mismatches. No dependency installation attempted on the shared nearly full disk. `git diff --check` and launch `bash -n` passed. The retained audit JSON occupies about 225 KiB on disk; redundant baseline/temporary measurement files were removed. Existing user changes in the dirty worktree were preserved.
