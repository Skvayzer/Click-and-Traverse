# Flat raised-walking isolation experiment — built, not launched

The live hand-posture run was not stopped, signalled, modified, or accessed through GPU tools. Construction and verification used CPU only. The replacement has passed CPU correctness checks, not CUDA startup/capacity validation or a learning/acquisition test.

## Exact composition and reset masses

| Population | Scenes | Reset mass |
|---|---:|---:|
| Original CAT | 37 | Combined with published below |
| Published CAT | 27 | Original + published: 20% |
| Procedural CAT | 2,226 | 40% |
| Ordinary furniture | 24 | 6.25% |
| Ordinary generic clutter | 24 | 3.75% |
| Narrow replay, widths 0.70 / 0.64 / 0.58 m | 4 / 4 / 4 | 5% total |
| Flat raised-walking | 1 | 25% |
| Hand-table / hand-shelf tasks | 0 / 0 | 0% |
| **Total** | **2,351** | **100%** |

The 2,338 retention records and all 12 narrow records are byte-for-byte equal as JSON records to their pinned source records. Fields and geometry are unchanged. Retention fields remain referenced on their original filesystem; narrow fields are hardlinked. Reset pools reuse exactly the corresponding source rows. There is no hand or width advancement curriculum. Difficulty adaptation remains within each bucket, never between bucket masses. These are reset probabilities, not guaranteed transition shares; `balance/leader_flat_step_fraction` reports actual exposure.

## Paths

- Fields/metadata: `data/furniture/cat_flat_balance_v1_20260920/manifest.json`
- Collision: `data/furniture/cat_flat_balance_v1_20260920_collision/manifest.json`
- Resets: `data/furniture/cat_flat_balance_v1_20260920_resets/manifest.json`
- Prepared launch: `configs/pilots/flat_balance_v1_30720.sh`
- CPU result: `outputs/flat_balance_build_validation/cpu-smoke.json`
- Builder: `scripts/build_flat_balance_bank.py`
- Verification: `scripts/verify_flat_balance_cpu.py`

All manifests/pools and referenced source hashes were checked. The new flat record has an empty obstacle list and collision index. Its tiny constant field has SDF=10 m, BF=0, and GF=[0.6,0,0]; the physical simulator floor is unchanged. The route runs along +X to a distant endpoint. Runtime flat episodes last 500 control steps (10 s), explicitly set in the validated experiment settings; no goal/qualification gate ends the episode. Existing numerical, self-contact and fall safety terminations remain enabled. Standard reset randomization remains; initial horizon staggering is disabled for flat rows so their first episode is not prematurely truncated.

## Objective and structural isolation

For the flat task only, add raw reward `3 * (1-C) * clip(v_route/0.6,0,1)` before the native sum's 0.02 timestep scaling and clamp. Thus the added reward is at most +0.06 per control step. C is the existing mean two-hand box-distance cost d²/(d²+0.15²). No binary compliance gate appears in reward.

The only pair has centers `[0.429,+0.125,0.870]` and `[0.429,-0.125,0.870]`, half sizes `[0.018,0.010,0.018]`, route-relative XY and absolute world Z. Compliance requires both hands within 0.05 m of their respective boxes. There is no tucked pair or extra heading objective.

A dedicated manifest validator requires the exact pinned composition, settings and empty flat geometry/fields. `SceneBank.flat_balance` is derived from that validated unique task record. The reward kernel uses `torch.where(flat_mask, bonus, 0)`; the existing arm-posture regularizer suppression is also restricted to that mask. All retention/narrow reward definitions remain unchanged. The CPU mixed-task probe found bit-identical non-flat rewards with the feature enabled versus disabled on the same physical states.

The +0.03483/step reward ordering established by the prior FK audit is not evidence of a successful dynamic raised gait. The experiment tests whether the policy can learn one.

## Leader-only metrics

All new metrics are under `balance/leader_` and describe one update's flat transitions unless named `all_task`.

- `walking_steps_compliant_fraction`: walking-and-compliant / walking steps; walking means measured forward route speed >=0.2 m/s.
- `all_steps_walking_compliant_fraction`: walking-and-compliant / all flat steps, including acquisition and standing.
- Explicit `step_count`, `walking_step_count`, `walking_compliant_step_count`, `all_task_step_count`, `flat_step_fraction`.
- `fall_rate_per_episode`, `fall_rate_per_simulated_minute`, `completed_episode_count`, `fall_episode_count`, `simulated_minutes`.
- `longest_raised_walking_seconds`: maximum uninterrupted streak observed this update. Streak carries across updates, resets at noncompliance/nonwalking/termination, and is checkpointed for this bank only.
- `route_speed_mean_mps`, `posture_bonus_mean_per_step`.
- Root, left-hand and right-hand height min/P10/P50/P90/max, in metres.
- `foot_balance_raised_reward_per_step` and `foot_balance_nominal_reward_per_step`, with explicit raised/nominal counts. Raised means both-hand compliance; nominal means arm-joint RMS offset from nominal <0.15 rad and not compliant. Intermediate postures belong to neither bucket.
- Walking-only counterparts `foot_balance_walking_raised_reward_per_step` and `foot_balance_walking_nominal_reward_per_step`, with walking subgroup counts.

Foot-balance metrics include the configured scale and 0.02 timestep; negative numbers are actual reward contributions. Undefined ratios are omitted, not reported as zero; their denominators are always present.

Flat outcomes are excluded from retention goal counters. The inherited hand-navigation group now contains only narrow replay, so its W&B metric is renamed `success/narrow_replay_clean_goal_success_rate`; it must not be interpreted as hand protection. Existing CAT and ordinary-clutter retention success metrics remain present.

Best checkpoint selection for this experiment uses the leader all-step walking-and-compliant fraction, not a goal or posture-qualification gate. It is a one-rollout training statistic attached to post-update weights, not an independent evaluation. Durable `resume.pt` is the full run state.

## Validation and limits

- 43 targeted tests passed: flat objective isolation/speed/absolute Z, CPU Inductor parity, sampling masses, source/reset identity, leader metric denominators, and existing runner, logging, sigma, passage-heading and hand-curriculum checks.
- Full-bank construction and all field hashes checked on CPU.
- Six-world real MuJoCo CPU smoke: 15 control steps per world, frozen native warm-start weights, no optimizer update. Includes flat, original CAT, procedural CAT, ordinary room, and narrow tasks.
- Non-flat reward difference exactly 0 in the mixed-state check.
- Fresh optimizer has zero state entries; trained scale-head weights are retained, with ceiling 0.15. No sigma reset.
- Streak save/restore and flat horizon/autoreset verified.
- Full task pure-kernel graph capture checked on CPU with the eager backend; the new objective also ran through CPU Inductor. No CUDA compilation or capacity claim is made.
- Shell syntax, Python compilation and diff whitespace checks passed.
- Strict source_sha256 resume-contract comparison remains unchanged. This launch is a fresh weights-only warm start, not a resume. Old-bank task snapshots do not gain a required balance-streak field.

## Launch and resources

Only after the user reviews composition and frees the GPU:

```bash
bash configs/pilots/flat_balance_v1_30720.sh
```

This uses `outputs/cat_hand_posture_30720_20260919/resume.pt`, a fresh optimizer, SAPG, 30,720 environments, batch 768, 40 minibatches, unroll 32, sigma ceiling 0.15, `--compile-task`, W&B online, a separate run directory, 300 updates and checkpoint interval 50. Gamma and epochs are inherited; no `--init-action-std`, `--resume`, or training launch occurred during the build. There is no provisional `--balance-experiment` flag: the validated bank marker selects the task.

Field storage is 13.5467 GiB shared at runtime; collision arrays are 0.06569 GiB, max candidate count 64. Combined field/collision memory is only ~5 MiB larger than the current bank. Expect roughly 39–41 GiB device usage based on the current run's ~39 GiB, not a new GPU measurement. CPU validation used the full bank successfully.

New unshared logical files total about 95 MiB (~31 MiB allocated on this compressed filesystem); the 13+ GiB retention bank is not duplicated. Runtime checkpoint size is expected near the current 0.567 GiB, with temporary overlap during atomic replacement. Keep 2–3 GiB free for checkpoints, logs and compiler cache. Checkpoints overwrite rather than accumulating every interval.
