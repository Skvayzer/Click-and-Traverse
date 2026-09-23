# Passage acceptance contract v1 — preregistered measurement, not a reward

Implemented in `cat_mjlab/acceptance.py`, wired into the mjlab collector. No training or GPU work was launched to implement it. The existing checkpoint selector, rewards, terminations, sampler, reset choices and optimizer remain unchanged. Existing compliance/success curves remain historical diagnostics; they are not renamed into the new metric.

**S = safe, on-time clean-goal trials completing every required passage / all assigned upstream trials.** Pending trials remain in the denominator and are never passes. A zero-denominator rate is absent, not zero or one. Do not report a partial cohort as a completed evaluation.

## Frozen rules before observing a rollout

- An assignment specifies the scene/gates, initial state/cohort, policy, and overall deadline before its first action. Main-cohort trials must start strictly upstream of the first entry plane. Explicitly seeded starts and all other non-upstream starts are quarantined in `seeded_inside` diagnostics and cannot contribute to S. Non-passage CAT/flat/retention scenes are not passage assignments.
- For each straight contrastive scene, gates are world-space planes at the declared `start_m` and `end_m`. Every declared zone is required, even without a posture target. Root XY must cross entry then exit in order. Positions are projected onto the fixed world tangent, **not** the navigation system's route-progress estimate.
- The root corridor is the module opening width, centered on the fixed route. It extends across each entire declared zone. Connecting corridors between zones use the narrower adjacent opening. This conservative corridor definition is fixed in advance; it cannot expand to accommodate an observed bypass. Between sampled root positions, the swept segment is clipped against each longitudinal slab and both clipped endpoints are checked. Unsupported bent routes fail validation rather than receive a chord approximation.
- Entry/exit timestamps are interpolated across the observed segment. Each crossing must satisfy `t_exit - t_entry <= L / 0.2`. **0.2 m/s is an explicit design choice, not a validated finger-protection requirement.** The v5 bank audit found **36 contrastive scenes, 108 zones, each 1.65 m long: 8.25 s per zone** (12 forward-protected, 12 narrow, 12 transition scenes).
- A pass also requires clean goal completion by the overall deadline, no hand collision from actual reset through completion, no body collision, fall, bypass or other existing safety failure. A slow crossing cannot be repaired by a later goal. A fall before entry fails rather than becoming a collision-free success.
- Reset collision evidence uses the existing detector on the actual post-replacement reset, not the rejected reset proposal. Runtime hand evidence is `body_collision_hands OR hand_violation`; other safety faults also fail closed. There is no new collision detector or reward. Root corridor checks and the existing collision model are complementary; they do not certify unmodelled finger geometry or continuous body contact between simulation observations.
- Geometric failures latch immediately, but observation continues until physical termination, clean goal, or deadline, so subsequent collision evidence is retained. Successful trials freeze at clean completion; later activity in the same physical episode does not retroactively change them. Non-entry includes missing **any** required zone when observation ends.

## Two uses of the same state machine

`evaluate_assigned_trials(assignments, rollout)` is the CPU reference interface for a finite preregistered cohort. It materializes and validates **all** assignments before calling the first rollout. Each assignment has `trial_id`, `room`, `start_xy`, `deadline_s`, `cohort` (`upstream` or `seeded_inside`), `reset_body_collision`, and `reset_hand_collision`. The callback supplies every control-step observation (`time_s`, `xy`, `clean_goal`, `done`, `hand_collision`, `body_collision`, `fall`, `other`). Bind a frozen checkpoint and preregistered seeds in that callback when performing an actual policy evaluation. The implementation does not load or train a policy. It consumes every assignment through completion/physical failure/deadline, without filtering successes. Truncated evidence produces PENDING and `complete=False`; it never fabricates an unobserved deadline outcome. Results include individual trial statuses, separate cohort statistics, and completeness. No actual frozen-policy evaluation was run here.

`TrainingAcceptance` observes the **existing** reset stream before actions and follows every assigned trial across rollout/update boundaries. It preserves the existing sampler and startup horizon staggering. Its assigned overall deadline is the remaining physical wrapper horizon, frozen before the first action. State, counters and reset evidence persist in runtime checkpoints. A legacy mid-episode observation is not reconstructed from missing history: it waits until the next reset and logs `unobserved_initial_count`.

Training metrics are cumulative from this observer's creation, not the old 100-update success window. The training policy changes and scene sampling can be adaptive: these are **training telemetry**, not an evaluation of the final `best.pt`. At the 50-update boundary, remaining trials stay pending/unfinished in the saved runtime; the collector does not add training updates or shorten deadlines to make a favourable final score. Continue observation to deadlines for a complete fixed-cohort evaluation. Do not certify a policy from the pilot's partial aggregate.

## Exact keys for the supplied PPO pilot

The existing PPO convention rewrites `leader_` to `policy_`. SAPG retains `leader_`. Acceptance lives under `training/` to distinguish it from the existing selection-oriented `success/` namespace.

Main finger-protection stratum (watch this together with transition and narrow strata; do not replace it with a favourable aggregate):

```text
training/passage_acceptance_policy_forward_protected_success_rate
training/passage_acceptance_policy_forward_protected_assigned_count
training/passage_acceptance_policy_forward_protected_pass_count
training/passage_acceptance_policy_forward_protected_fail_count
training/passage_acceptance_policy_forward_protected_pending_count
training/passage_acceptance_policy_forward_protected_unfinished_count
training/passage_acceptance_policy_forward_protected_complete
training/passage_acceptance_policy_forward_protected_hand_collision_incidence
training/passage_acceptance_policy_forward_protected_zone_entry_rate
training/passage_acceptance_policy_forward_protected_trial_entry_rate
training/passage_acceptance_policy_forward_protected_crossing_count
training/passage_acceptance_policy_forward_protected_crossing_time_mean_s
```

`zone_entry_rate` is entered required zones / all assigned required zones; `trial_entry_rate` is trials entering any zone / assigned trials. Each rate includes failed and pending assignments. `complete` is 1 only when no assigned trial still needs observation. `unfinished_count` also includes already-failed trials still being observed; `pending_count` counts trials whose pass/fail outcome is not yet known. `pass_count + fail_count + pending_count = assigned_count`.

Failure breakdown uses these exact suffixes under the same prefix:

```text
failure_non_entry_count             failure_non_entry_rate
failure_bypass_count                failure_bypass_rate
failure_timeout_count               failure_timeout_rate
failure_fall_count                  failure_fall_rate
failure_body_collision_count        failure_body_collision_rate
failure_hand_collision_count        failure_hand_collision_rate
failure_other_count                 failure_other_rate
```

Categories are **multi-label**, not mutually exclusive. For example a hand/body collision before entry records all relevant flags. Their counts need not sum to `fail_count`. `other` retains existing obstacle/self-contact/numerical/out-of-bounds/elbow safety failures, preventing unclassified early termination from passing.

The bounded crossing-time distribution contains all completed geometric crossings, including those from ultimately failed trials. Its exact suffixes are:

```text
crossing_time_bin_le_0p5_s_count
crossing_time_bin_le_1p0_s_count
crossing_time_bin_le_2p0_s_count
crossing_time_bin_le_4p0_s_count
crossing_time_bin_le_8p0_s_count
crossing_time_bin_le_16p0_s_count
crossing_time_bin_le_32p0_s_count
crossing_time_bin_le_inf_s_count
```

These are disjoint intervals `(previous edge, edge]`, with the first including all durations up to 0.5 s and the last above 32 s. Also log `crossing_time_sum_s`, `entered_zone_count`, `required_zone_count`, and `hand_collision_count`. Always inspect entry/non-entry, crossing count and unfinished counts alongside the distribution; non-crossings cannot masquerade as fast crossings.

Replace `policy_forward_protected_` with `policy_posture_transition_` or `policy_narrow_passage_` for those strata, `policy_` for all policy-0 passage assignments, or remove it for all physical policies. Thus the global S key is **`training/passage_acceptance_success_rate`**. The diagnostic prefix is `training/passage_acceptance_policy_forward_protected_seeded_inside_` (or its corresponding aggregate/role prefix); these counts never enter upstream S. Inside-start diagnostics cannot satisfy an upstream entry requirement and must not be presented as evidence of full traversal success.

Existing per-episode collision logging additionally emits, for each of `feet`, `legs`, `trunk`, `head`, `arms`, `hands`:

```text
training/body_collision_hands_count
training/body_collision_hands_rate
training/policy_body_collision_hands_count
training/policy_body_collision_hands_rate
training/policy_forward_protected_body_collision_hands_count
training/policy_forward_protected_body_collision_hands_rate
training/policy_forward_protected_seeded_body_collision_hands_rate
training/policy_forward_protected_unseeded_body_collision_hands_rate
training/policy_forward_protected_seeded_after100_body_collision_hands_rate
```

Counts accompany every rate, with the same existing policy/role/table/shelf/seeded masks. These per-episode rates use physically completed episodes in the current rollout, not S's assigned-trial denominator. The inherited `seeded_after100` mask is a survival diagnostic, never an acceptance cohort. No collisions-among-successes metric is used.

## Prepared command — not launched

```bash
bash configs/pilots/hand_v5_acceptance_50.sh
```

The script pins the requested `outputs/cat_hand_feasible_v5_seeded_pilot50_20260921/best.pt`, writes a distinct run directory, uses 50 updates, online W&B project `CAT-wholebody`, entity `skvayzer`, and explicitly preserves 0.5 seeding so both upstream and seeded diagnostics are exercised. It retains the previous pilot's optimizer/reset/reward options. No acceptance score influences training or checkpoint selection. The prepared command requests CUDA for a future user launch; verification here used CPU only.

## Verification and interpretation

CPU command:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  .venv-mjlab/bin/python -m pytest -q \
  tests/test_mjlab_acceptance.py tests/test_mjlab_logging_diagnostics.py \
  tests/test_mjlab_runner.py tests/test_mjlab_raised_reset.py tests/test_mjlab_pilot.py
```

**Final result: 61 passed in 11.26 s**, with CUDA hidden and JAX forced to CPU. `bash -n` for the prepared command and `git diff --check` also passed.

The tested gate includes refusal, crawling, lateral bypass, inter-zone escape, backwards escape, fall-first, hand/body collisions, other failures, missing zones, reset collisions, deadline and crossing-budget boundaries, ordered multiple zones, seeded isolation, pending denominators, resume, terminal-before-autoreset evidence, and valid safe traversal. Observer-on/off collection is bit-for-bit identical for every learning tensor and Torch RNG state in the CPU fixture. A separate real CPU MuJoCo comparison across five control steps (including forced hand collision and resets) also produced exactly equal rewards, observations, termination flags, physics tensors and task RNG states with the observer on/off. Existing loss/gradient invariance, reset, runner and pilot-contract tests also pass. This is CPU evidence, not a claim of measured GPU bitwise determinism or throughput.

**Preregistered interpretation:** genuine success means **S = 1.0** on a completed, representative, fixed upstream cohort, with zero hand-collision incidence and no failing required stratum. Neither zero collisions with zero entries nor a good seeded or easy-scene aggregate qualifies. Finite simulation trials cannot prove zero real-world finger risk; no validated lower safety target is being invented here.

**Unverified prediction:** current forward-protected/transition policies may score **S approximately 0**, including the current seeded pilot. The gate may be far away. At environment step 28,508,160, the live pilot logged 0 forward-protected successes / 3,463 resolved trials under the *old posture-qualified criterion*, and 0.4666666667 forward-protected body-collision rate in that rollout. These are not measurements of the new S, and the old prescribed-box failure alone does not prove failure under the new criterion. Narrow/retention success does not establish finger protection. No checkpoint's S was measured during this implementation.

## Protected-file and live-run audit

A before/after metadata snapshot covered **23,967 files**, including **23,940 bank files** and every existing `outputs/**/*.pt` checkpoint. **23,966 were unchanged** in size, mtime_ns and inode. The only changed file was the running pilot's `outputs/cat_hand_feasible_v5_seeded_pilot50_20260921/resume.pt`, consistent with its own checkpoint at update 30. The requested source `best.pt`, all other snapshotted checkpoints and every bank file were unchanged. This is a metadata audit, not a content-hash comparison. No implementation command wrote to those files or the live output directory.

The live metrics file grew from **410,072 to 608,853 bytes**; its latest inspected record reached **36,372,480 environment steps (37 updates)**. Status remained `running`. The tmux socket itself is inaccessible in the sandbox, so liveness is confirmed from advancing output rather than a pane inspection. No process was signalled, training launched, GPU used, or existing policy checkpoint loaded for this task. Synthetic CPU learner/runner unit tests use temporary toy data only. Verification temporaries were removed.
