# Flat balance v2: per-run reward overrides, no bank rebuild

Prepared only; live v1 run untouched. Use `bash configs/pilots/flat_balance_v2_30720.sh` only after manually freeing the GPU.

New CLI flags are `--flat-bonus-scale 10.0 --flat-region-scale 0.40`. Values must be finite and positive, require a validated flat-balance bank, and resolve CLI > inherited environment config > immutable bank defaults (3.0/.15). Effective values are stored under `contract.environment_config.flat_balance_reward`; the strict source/contract resume checks remain intact. The bank, collision and reset manifests are unchanged. Target boxes, 5-cm qualification tolerance, .6-m/s speed cap, sampling masses and retention rewards are unchanged.

At 20 ms the bonus is at most +.20/step, versus the 10,000 upper reward clamp. Increasing its nonnegative scale cannot introduce zero-floor clipping at a fixed state. Rational proximity remains bounded by one and the speed factor remains clipped to [0,1]; neither receives an extra scale-dependent clipping stage. Action tanh, sigma ceiling, PPO ratio clipping, advantage normalization and global gradient clipping are unchanged and remain relevant constraints. This is not a claim that arbitrary future physical states never hit the native lower clamp.

Warm start from `outputs/cat_flat_balance_v1_30720_20260920/resume.pt`, not the older hand-posture checkpoint. Preserve the current experiment's learned adaptation and observed hand drift. Use a fresh optimizer, retain learned per-dimension scale-head weights with ceiling .15, and use the separate v2 output directory/W&B run. No proportional acceleration in learning is promised by the larger reward-distance slope.

## New diagnostics

`learner/diagnostics/task_share/{flat,retention,narrow}/...` measures the SAPG augmented batch once before optimization, including relabeled follower samples. Labels follow the actual pre-step scene, including resets within trajectories. Metrics include sample count/fraction; reward mean and absolute reward share; raw advantage mean/std; normalized advantage mean, positive fraction, absolute mean and absolute share; initial value-error MSE and share of squared error. Global raw-advantage mean/std report the common normalization population. Truncation masking follows the actual optimizer inputs. These are observational quantities and never change targets, advantages or loss weighting.

`learner/diagnostics/task_updates/{flat,retention,narrow}/ppo_ratio_clip_fraction` measures clipping during optimizer minibatches. Counts include repeated epochs; clip fractions average nonempty minibatches, matching the existing diagnostics convention.

Read these alongside `success/cat_goal_success_rate`, ordinary-clutter retention success, collisions and narrow replay metrics. Rising flat advantage/error share plus falling retention influence is early evidence of reward reweighting; it is not a causal attribution of a later skill regression or a per-task gradient measurement.

## Validation

30 targeted CPU tests passed, including override validation/inheritance, recorded environment config, unchanged bank defaults, new-scale CPU Inductor equivalence, bonus isolation, label preservation through SAPG relabeling, unchanged computed targets/advantages, a tiny CPU optimizer test, and existing runner/logging/sigma tests. The full-bank real CPU MuJoCo smoke used the current flat checkpoint and effective settings 10/.4; non-flat reward difference was exactly zero and optimizer state was empty. Results: `outputs/flat_balance_v2_validation/cpu-smoke.json`.

No GPU launch, bank regeneration, live-process action, or W&B run creation occurred. CUDA startup remains for the manually initiated relaunch.
