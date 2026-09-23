PPO replacement is prepared, CPU validated, and **not launched**. The live v3 process, tmux session, outputs, and GPU were not modified. Existing unrelated working-tree changes were preserved.

1. **Warm start: use v3's next durable `resume.pt`.**

The pre-existing native loader attempted strict model loading and could not convert SAPG to PPO. `cat_mjlab/conversion.py:24` now extracts policy 0 explicitly. For each actor/critic first layer, it keeps `W_obs` and replaces `b` with `b + W_embedding @ embedding[0]`. This preserves the entire leader function, including the learned scale head. The six-row table and conditioning columns are removed after folding; follower-specific conditioning is discarded. No retained model layer is reinitialized. Adam, counters, simulator, sampler, RNG and success windows start fresh; use `--fresh-optimizer`, never `--resume` across algorithms. The target configuration records one policy (`cat_mjlab/runner.py:470`).

At the recorded audit, v3 had logged **20 updates / 19,660,800 transitions**, but its durable resume was still **update 0** because the old interval is 50. Its saved weights were exactly equal to v2's **update 13 / 12,779,520-transition** checkpoint. Thus the current v3 resume is structurally suitable but contains no v3 learning yet. Use v3 after its next periodic or graceful-stop snapshot; if a replacement must start from currently durable weights, explicitly select `outputs/cat_flat_balance_v2_30720_20260920/resume.pt`. Do not use the stale zero-score `best.pt`. The launch script rejects an update-0 resume (`configs/pilots/flat_balance_ppo.sh:11`).

The real checkpoints were converted and strictly loaded on CPU. On 1,024 stored observations, v3 leader versus PPO max absolute differences were **6.56e-7 action, 7.15e-7 mean, 7.45e-8 sigma, 5.25e-6 value**; v2 max value error was **6.20e-6**. These are float32 summation-order effects. Fresh Adam had zero state entries. Evidence: `docs/assets/ppo-flat-restart-20260920/audit.json:3`; reproducible read-only probe: `scripts/audit_ppo_flat_restart_cpu.py`.

2. **Checkpoint selection now has a useful signal.**

Correction to the original diagnosis: the code already had a flat-bank override, and v3's `run.json` names its one-rollout strict walking-compliance score. Role balancing was not selecting v3's checkpoint. The strict flat statistic itself was zero; absent role evidence returns `None`, not an invented zero (`cat_mjlab/runner.py:110`).

The new flat-bank score is:

`S = 0.50 * strict_walking_compliance + 0.25 * soft_posture_speed_progress + 0.25 * cat_goal_success`

Strict compliance is both hands within 5 cm of their boxes while route speed is at least 0.2 m/s, divided by **all flat steps**. Soft progress is `(1 - mean_hand_box_cost) * clip(route_speed / 0.6, 0, 1)`, averaged over all flat steps. It is obtained by dividing the existing measured bonus by `bonus_scale * dt`, so it is bounded in [0,1] and does not merely increase with reward scale. CAT-goal success is successful/resolved first outcomes; flat episodes are excluded (`cat_mjlab/task.py:428`). All components use the same **100-update window**, summed numerators and denominators, rather than equally averaging unequal rollouts (`cat_mjlab/runner.py:143`, `:318`).

The strict objective has the largest weight. Smooth progress provides information before the hard tolerance is reached. Retention contributes independently, unlike a product score that stays zero while either component is zero. Each component is monotone with the others held constant; tradeoffs remain intentional (a 0.10 retention loss offsets 0.05 strict-compliance gain, or 0.10 soft-progress gain). This is a selection rule, not a guarantee that all skills improve or a retention gate. Ordinary-clutter and narrow-replay success remain visible but are not additional selection terms.

Missing flat samples or CAT outcomes defer publication. `best.pt` updates only on strict score improvement, and `training/best_checkpoint_selection_score` records the monotone incumbent. The current score can fluctuate. All score components and sample counts are logged. These are training-window statistics attached to post-update weights, not independent evaluation of those exact weights. The rolling window deliberately reduces one-rollout noise and may lag changes.

Replaying the new formula on the first 20 v3 log rows gives **0.05559–0.14772**, with **14 record improvements** despite zero strict compliance. This shows the signal is non-degenerate on actual data; it cannot reconstruct historical best weights. Regression coverage also runs a two-update native SAPG→PPO runner and verifies `best.pt` advances to update 2 (`tests/test_ppo_restart.py:67`).

3. **PPO geometry and metric populations.**

At 30,720 envs: `30,720 * 32 = 983,040` physical samples; `768 * 32 = 24,576` per minibatch; **40 minibatches**, four epochs, **160 optimizer steps**. `batch_size` is trajectories, not flattened samples. At 37,632 envs: **49 minibatches**, same 768 trajectories / 24,576 samples each, four epochs / **196 optimizer steps**. PPO has no six-policy divisibility requirement. `batch_size * num_minibatches` still must be a multiple of num_envs (`cat_mjlab/runner.py:483`).

SAPG currently selects one follower group, not all five, for relabeling (`cat_mjlab/learning.py:245`). Its total training batch is **7/6** of the physical rollout, and its actual optimizer minibatch is **896 * 32 = 28,672 samples** at 40 minibatches. It doubles the leader-targeted data, not the entire training batch. PPO's requested 24,576 matches the original physical minibatch geometry.

All PPO environment IDs are 0. Consequently, former leader masks cover every environment: navigation success, role counts, balance fractions/heights/streaks, leader collision/timeout rates, and rollout reward now describe the single policy population. Episode return/length, unprefixed episode counts, and action-std metrics already covered all physical policies and remain population aggregates. New PPO logs rename `leader_` to `policy_`, `diagnostics/on_policy_leader/` to `diagnostics/on_policy/`, and `success/*` to `success/policy_*` to avoid silently overlaying success histories. Counts remain essential: a larger population alone increases raw counts and may increase observed maximum streaks. `training/metric_population_envs`, physical samples/update, optimizer samples/epoch and run metadata explicitly identify scope (`cat_mjlab/runner.py:555`, `:639`). Rollout reward retains its existing key; its population is explicitly recorded.

Follower/relabeled metrics disappear. On-policy importance weights are zero in log space and ESS fraction is 1; PPO ratio clipping remains informative. Task-update clipping/count metrics remain, now from actual PPO samples. The old SAPG `task_share` advantage/error diagnostics are intentionally absent: PPO recomputes GAE and normalization per minibatch/epoch (`cat_mjlab/learning.py:331`), and there is no single frozen global advantage population to report (`:349`). Do not interpret absent metrics as zero or overlay their SAPG values onto PPO.

The reported 655,360 relabeled samples include **four epochs over 163,840 physical follower samples**, not 655,360 unique transitions. ESS values are averages of minibatch ESS, not an exact full-rollout ESS (`cat_mjlab/learning.py:278`). The small ESS fraction still supports the algorithm switch. SAPG also shares trunk weights across policies: follower gradients are not literally discarded, even though the leader has only 5,120 directly on-policy environments.

4. **Capacity: start 37,632, fallback 33,792, then measure before increasing.**

A GPU-verified maximum is **unknown**; no GPU tools or GPU workloads were used. The estimate combines recorded device usage, documented fixed arrays, and exact CPU tensor-byte accounting. It is not a measured PPO peak or a statistical confidence interval.

The actual v3 logs show **38.404–38.426 GiB** device use. The earlier measured usable device capacity was **47.492 GiB**, not 48.000; retaining its **4.784 GiB** reserve gives a **42.708 GiB working ceiling** (`docs/MJLAB_CONTRASTIVE_RESTART_20260919.md:29`). Fixed field/collision arrays total **13.61239 GiB** (`docs/FLAT_BALANCE_EXPERIMENT_20260920.md:85`). nconmax=64 and njmax=256 stay unchanged.

At 30,720 envs, the physical rollout is exactly **4.03198 GiB** (4,404 bytes per transition including int64 policy/task IDs). SAPG creates a distinct augmented copy while the caller retains the original rollout: **4.73389 GiB** persists through optimization, or **4.74243 GiB** including temporary task-share arrays. A selected follower copy additionally occupies **0.67200 GiB** during preparation. PPO's detach-only dictionary aliases original storage and creates no augmented copy (`cat_mjlab/learning.py:409`). Thus **4.73389 GiB of simultaneous learner tensor storage disappears**, plus SAPG preparation temporaries; optimizer minibatches also shrink from 28,672 to 24,576. This is more than just saving the appended 1/6 because the original and augmented buffers coexist. Exact scaled accounting is in `audit.json:48`, generated by executing real SAPG preparation on a small CPU batch.

Use `F=13.61239`, `B=38.42554`, `D=4.73389`, all GiB:

- Central storage estimate: `F + (B - F - D) * N / 30720`.
- Conservative planning budget: `F + (B - F - 0.5*D) * N / 30720 + 1.0`.

Only half the exact storage reduction is credited in the planning budget, with another 1 GiB uncertainty allowance, because device-wide usage includes allocator caches, Warp, compiler workspaces, and buffers whose lifetimes differ from tensor sums. These are explicit engineering allowances, not measured allocator recovery. Treating all non-field baseline usage as scaling with env count also errs conservatively for fixed model/minibatch costs. The 4.784 GiB operational reserve is preserved **in addition** to the uncertainty allowance.

| Env count | Minibatches | Samples/update | Central GiB | Planning GiB | Free GiB after planning |
|---:|---:|---:|---:|---:|---:|
| 30,720 | 40 | 983,040 | 33.692 | 37.059 | 10.433 |
| **33,792 fallback** | **44** | **1,081,344** | **35.700** | **39.303** | **8.189** |
| **37,632 initial** | **49** | **1,204,224** | **38.209** | **42.109** | **5.383** |
| 38,400 next step | 50 | 1,228,800 | 38.711 | 42.670 | 4.822 |
| 39,168 | 51 | 1,253,376 | 39.213 | 43.231 | 4.261 |
| 49,152 | 64 | 1,572,864 | 45.739 | 50.526 | -3.034 |

Every row uses batch 768 and unroll 32. **38,400 is the model's maximum grid count under the conservative budget**, but only 0.038 GiB separates its planning budget from the working ceiling. Start one step below at **37,632**, with 0.599 GiB additional margin beyond the required reserve. The **33,792 fallback** would budget 41.907 GiB even with **zero credited SAPG savings** plus the 1 GiB allowance. Return to 30,720 if observed behavior exceeds that envelope. Neither fallback is a GPU-verified guarantee.

The central estimate may leave substantially more unused memory than the planning budget. After the new run has completed compilation, representative resets, updates and a runtime checkpoint, use real `performance/device_vram_used_gib` to recalibrate. The next 768 envs cost about **0.502 GiB** centrally / **0.561 GiB** in the planning slope. Keep measured device use plus a step's allowance below 42.708 GiB. New free/total-device and PyTorch peak allocated/reserved metrics help observe this; PyTorch peaks do not include all Warp/device allocations, and end-of-update device samples do not capture every transient (`cat_mjlab/runner.py:610`, `:652`). A further count increase requires a fresh warm start because environment count is part of the strict resume contract. Do not reduce contacts or consume the reserved 4.784 GiB to reach nominal 48 GiB usage.

At 37,632, one policy gets **1,204,224 unique transitions/update**, versus **163,840 direct leader transitions** now: **7.35x**. Total physical data/update rises **22.5%**. At 30,720 the corresponding gain is 6x; at 38,400 it is 7.5x. Relative to the current augmented SAPG learner batch, 37,632 PPO processes only **5% more samples/epoch**, with more, smaller minibatches and no relabel preparation.

Wall-clock throughput is not measured for PPO. At the logged **11,650.8 physical transitions/s**, unchanged simulator throughput would make the new update about **103.36 s**, versus ~84.38 s now. Direct single-policy data/s would be ~11,650.8 rather than ~1,941.8: **6x at equal aggregate throughput**, not 7.35x per second. PPO may improve learner time; extra worlds may improve utilization or increase simulator cost. No numerical speedup beyond the sample arithmetic is promised.

The reward clamp, gamma **0.98**, entropy **0.003**, sigma cap **0.15**, bonus **10.0**, region scale **0.40**, task compilation, online W&B, and continuous max-updates=0 are retained. There is no evidence here requiring a reward/horizon change. The sigma cap bounds sampling spread but does not prove mean-action or gait stability. Checkpoint interval is shortened from 50 to **10** to reduce durable-state lag; atomic CPU checkpoint copies remain unchanged. No optimizer or task reward behavior was otherwise changed.

Launch after the live run is switched over under user control and v3 has a fresh durable snapshot:

```bash
bash configs/pilots/flat_balance_ppo.sh
```

Fallback count:

```bash
bash configs/pilots/flat_balance_ppo.sh 33792
```

Explicit currently durable source fallback:

```bash
CHECKPOINT_NATIVE=outputs/cat_flat_balance_v2_30720_20260920/resume.pt bash configs/pilots/flat_balance_ppo.sh
```

The complete command, all flags and output naming are in `configs/pilots/flat_balance_ppo.sh:26`. It creates a fresh W&B lineage and output directory. It contains no stop/kill commands and has not been executed.

For a measured step-up, use the new PPO run's durable weights so its learning is retained:

```bash
CHECKPOINT_NATIVE=outputs/cat_flat_balance_ppo_37632_20260920/resume.pt bash configs/pilots/flat_balance_ppo.sh 38400
```

Validation completed: **48 CPU tests passed**, covering native conversion, nonzero learned conditioning, unchanged source tensors, fresh optimizer, score missing-data behavior/monotonicity/count weighting/restore, two-update best publication, PPO metric naming, existing runner/resume, flat rewards/overrides, logging, pilot and curriculum checks. Shell syntax and `git diff --check` passed. Real-checkpoint CPU equivalence and memory accounting results are recorded in the audit JSON. CUDA startup, compiled GPU kernels, GPU peak memory, and PPO wall-clock throughput remain unverified.
