# Stabilized whole-body CAT fine-tuning

This restarts from released CAT revision `46ce4b57ba0639168d51741b661ff62f7ce6f045`, with fresh Adam. No parameters from the failed fine-tuning run are reused. The compact architecture remains 222 actor observations, 310 critic observations, and 29 body actions. The same 2,338-scene bank contains released, published, procedural, randomly placed furniture and generic clutter scenes.

## Changes motivated by the paired noise ablation

The 16-layout, 16-seed ablation found both excessive upper-body exploration and harmful learned upper-body targets. With the trained weights fixed, capping upper Gaussian standard deviation at 0.1 increased goal reaching from 69/256 to 87/256 and reduced falls from nine to one. Removing all action noise still underperformed holding upper targets nominal. These findings motivate the following candidate fix; they do not prove that the new training will converge.

- The original 12 leg actions keep their native CAT Gaussian distribution and entropy objective. The 17 waist/arm/wrist scales are bounded to [0.02, 0.10], initialized at 0.05. Upper-action entropy weight is zero. Sampling, PPO log probabilities, and distribution-based diagnostics all use the same bounded scales. Values are pre-tanh action standard deviations, not angles.
- Add physical costs for the **filtered targets actually sent to PD control**: target velocity, target acceleration, and departure from nominal upper posture in clear space. Independent target history prevents native bookkeeping from zeroing the first difference. No observation dimensions are added.
- Velocity is normalized by 2 rad/s, acceleration by 20 rad/s², and posture offset by 0.8 rad. Their reward scales are −0.05, −0.02, and −0.05, respectively, before CAT's existing timestep multiplication. Waist dimensions receive weight 4; arm dimensions weight 1, normalized by total weight.
- The nominal-posture cost fades to zero when a hand is within 0.12 m or an elbow within 0.08 m of an obstacle; it reaches full strength at 0.24 m and 0.20 m respectively. Thus arms remain free to move near obstacles. Hand/elbow clearance rewards and collision termination stay enabled.
- Retain gentle PPO: learning rate 3e−5, clip 0.1, and coefficient 0.05 for the original leg-policy reference KL on CAT scenes. The shared network is trainable; this is not a frozen-policy adapter. Native leg rewards and scene/reset rules remain unchanged.

## Validation and checkpoint selection

At step zero and every 50 PPO updates (26,214,400 transitions at batch size 256), evaluate the same 16 layouts × 16 seeds in both deterministic and stochastic modes: 512 episodes per validation. Evaluation shares the existing field buffers; it does not load another bank. Original CAT scenes retain their 1,000-step horizon, rooms 4,000 steps.

A validation episode ends at the first clean whole-body goal, native failure, or horizon. Goal completion requires the root and both feet to cross the CAT goal plane, or enter the room goal region. Boundary bypass is rejected, and failure on the completion step takes precedence. This differs from the diagnostic ablation, which continued after reaching a goal. These fixed training-bank layouts monitor retention; they do not measure unseen-layout generalization.

The initial expanded released policy is the baseline in the **same whole-body robot and compact observations**, not an untouched original robot benchmark. A candidate must remain within five percentage points of baseline CAT success in each action mode, and within two of 16 seeds on each CAT scene, with zero numerical failures. Eligible candidates rank by worst-mode clutter success, then lower clutter hand violation, CAT success, and hand clearance. These tolerances accommodate the finite benchmark; passing does not guarantee universal retention.

One best model is retained. Ineligible candidates never replace it. A separate overwritten `resume.msgpack` preserves the complete latest learner state for recovery. Validation results are overwritten in `validation_latest.json`; the original baseline remains in `validation_baseline.json`. Training continues indefinitely until manually stopped, apart from execution errors.

## W&B metrics

All metrics share one run and use `global_step` (training transitions) on the x-axis. Validation episodes do not advance that axis.

| Metric | Interpretation |
| --- | --- |
| `validation/cat_goal_success_rate` | Primary retention measure. Compare against `baseline/cat_goal_success_rate`; it should remain stable or improve. |
| `validation/clutter_goal_success_rate` | Primary new-task measure. It should increase. |
| `validation/hand_violation_rate`, `validation/clutter_hand_violation_rate` | Fraction of episodes with a hand-sphere field violation after CAT's grace period. Lower is better; these are field-based clearance violations, not instrumented physical obstacle contacts. |
| `validation/fall_rate`, `validation/obstacle_violation_rate` | Failure diagnostics; lower is better. Causes can overlap. |
| `validation/stochastic/...` | The same benchmark with training-time exploration. A widening gap from deterministic performance indicates exploration sensitivity. |
| `validation/retention_eligible` | 1 means this evaluation passes all checkpoint-retention gates. |
| `training/goal_success_rate` | Fraction of episodes ending in the latest rollout that previously reached a clean goal. Training retains native CAT termination, so an episode can reach the goal and later fail. Adaptive scene sampling changes its difficulty mix. |
| `training/upper_std_mean`, `training/upper_std_max` | Upper action noise; the maximum must remain ≤0.10. `training/upper_std_bounds_violation_rate` must be zero. |
| `training/mean_upper_target_velocity_rms`, `training/mean_upper_target_acceleration_rms` | Actual filtered-target motion, in rad/s and rad/s². Watch for growth toward saturation alongside worsening success, rather than expecting zero motion. |
| `training/reference_kl`, `training/reference_kl_loss` | Drift from the initial leg policy on CAT observations; interpret together with actual retention validation. |
| `health/nonfinite` | Must remain zero. |

`training/completed_episode_count` gives the denominator for the latest rollout rates. No completion means the undefined rates are omitted. `training/timeout_rate` means survival to the horizon, **not success**. Episode reward remains useful within this run, but added penalties, changing scene mix and episode duration prevent direct reward comparison with older setups. Reward is secondary to goal completion and safety.

The stabilized run suppresses thousands of per-scene rollout chart series; periodic validation and aggregate current-rollout metrics provide a compact view. Existing historical rolling metrics remain available, but the explicit `training/*_rate` values above describe the latest completed rollout.

## Deployment

Use a frozen source checkout, shared environment/assets, `--finetuning stabilized --num-envs 16384 --batch-size 256`, and online W&B. This retains the previous practical Ada 6000 parallelism; new runtime memory and initial updates must be checked on the GPU. Startup performs the baseline benchmark before continuous PPO.

The custom action-distribution settings are serialized in native checkpoint network kwargs and the runtime compatibility contract. Stochastic inference must reconstruct the bounded factory; loading weights with an unbounded factory would change behavior. Deterministic actions remain `tanh(mean)`.

## Running experiment

Started on ws008090 at 2026-09-16 09:09:22 UTC, PID 208438, from frozen commit `67339e3aff66c467059e1fc3fbab4dd0d8ea4b52`. Source SHA256: `1cb2a96ad6c3bf396fba32cebc50ac5fb2854ab33713e4c29e2b2b8ddd851ca3`. W&B: <https://wandb.ai/skvayzer/CAT-wholebody/runs/de6ae369>.

The original actor and critic mapping reported zero maximum error. The initial deterministic fixed-scene success is 52.604% for CAT and 9.375% for clutter; stochastic success is 52.604% and 7.8125%. Neither baseline mode recorded a fall or numerical failure. These values apply to this selected benchmark and modified whole-body robot, not the original paper's overall success rate.

Startup verification observed over 19.9 million transitions online, upper standard deviation mean 0.0491, maximum 0.10 (float32), bounds-violation rate zero, and `health/nonfinite=0`. GPU memory was 37,582 MiB used and 11,051 MiB free; one warm PPO update reported approximately 50,800 transitions/s of learner compute. Checkpoint I/O and validation add elapsed time. These checks establish correct execution and logging, not learning convergence. The first post-training fixed-scene check is scheduled at 26,214,400 transitions.

Launch and initial benchmark provenance are stored in `docs/assets/cat-stabilized-20260916/`. The previous stopped experiment remains preserved. This experiment continues without a configured step limit.
