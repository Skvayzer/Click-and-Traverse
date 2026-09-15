# CAT setup correction — 15 September 2026

This report records the stopped state at the end of implementation. The subsequent authorized restart and completed capacity check are recorded in [CORRECTED_RUN_20260915.md](CORRECTED_RUN_20260915.md).

The earlier whole-body task was not a faithful extension of CAT. This correction restores the released task and generalist training behavior, retaining explicit whole-body, hand-protection and scene-size extensions. Historical runs/checkpoints are preserved. Training remains stopped.

## Implementation

- The original `env_cat.py` reset/step implementation remains the base. Small default-preserving hooks support expanded targets, relocated room starts, longer room episodes and geometric goals.
- `env_cat_wholebody.py` adds fixed Dex3 hands, 29 actions, 22 probes and two additive clearance penalties. It retains CAT's flat training physics, without physical furniture obstacles or extra self-contact pairs. The complete original reward and 0.0 m/50-step collision rule remain.
- `generalist_fields.py` preserves source fields and the released sampler in a compact ragged bank. Dense rooms use CAT's original 3D FMM at 4 cm. The unavailable configured scene is explicitly reconstructed: [provenance](CAT_FIELD_BANK.md).
- `generalist_training.py` preserves CAT's adaptive reset stack and Brax timeout semantics. Original scenes have 1000 steps, new rooms 4000. Probe-history state resets with scenes; a collision coinciding with a timeout remains a termination.
- `generalist_config.py` checksum-pins the released config. Non-resource PPO settings are inherited. The 32 GB profile explicitly reduces batch size; simulator parallelism no longer implicitly changes optimizer batches.
- Native PPO has one continuing learner and full atomic recovery snapshots. One best model stays separate from one overwritten current-state file.
- `train_cat_wholebody.py` restores the released final generalist, uses one W&B ID with pooled plots and scene diagnostics, disables automatic evaluation and runs until stopped or an error occurs. Nonfinite losses are explicitly reported.
- Retired single-scene/sequential training CLIs reject launches. Historical evaluation tools remain.

## Validation and limits

Tests compare complete reset and actual simulated-step trees in original 12-action compatibility mode, mapped whole-body features, base rewards, motor targets, collision thresholds/grace, timeout bootstrapping, ragged interpolation and Dex3 thumb containment. Small deterministic CPU PPO tests compare uninterrupted versus interrupted/saved/resumed updates; parameters, Adam, normalizer, RNG, environment/sampler and metric buffers must match exactly. Mocked launcher tests verify native arguments, fixed batches, same W&B ID and original initialization provenance.

The actual 39-scene bank is verified against file/source hashes. Random interior and boundary samples compare exactly with CAT's sampler for all three arrays in every original scene slot (111 comparisons). The bank contains 36 byte-verified source scenes, one reconstructed missing slot, and two dense rooms.

A native forward-kinematics check sampled 100 randomized resets per dense room. None of these 200 samples had negative body/probe clearance or physical robot–obstacle contact; minimum probe clearance was 29.39 mm. This finite sample is not an exhaustive guarantee. A real 10-environment mixed-bank CPU check passed reset, physics stepping, forced falls, original/new-room timeouts, scene resampling and a subsequent finite step.

Implementation checks are not learned success. Corrected training has not been restarted. The provisional 2048-env/524288-transition GPU profile requires capacity validation before increased parallelism.

The full regression suite passed **243 tests** (174.19 s). Machine-readable native-weight parity, reset clearance and mixed-wrapper evidence is in [cat-correction-validation-20260915.json](assets/cat-correction-validation-20260915.json).

Subsequent startup-failure and recovery refinements passed their **72-test focused suite** (5.98 s). Preparation, logger initialization and partial checkpoint-store failures now retain the experiment identity. An explicit startup retry requires evidence of zero progress; missing or corrupt previously written recovery state cannot trigger fresh initialization.

Implementation commit `5f77a4a` was pushed to `feature/whole-body-furniture-traversal` and synchronized to dep-0. All 39 scene slots and their file/source hashes were verified again on that machine. No corrected training experiment was started.

## GPU check

The actual default **2048-environment** setup initialized on dep-0, with **2405 MiB peak GPU usage during initialization**. No PPO update was executed, no W&B run was opened, and no model/recovery checkpoint was written by this diagnostic. [Recorded setup measurements](assets/cat-correction-gpu-setup-20260915.json).

**Full PPO peak memory is still unverified.** The compile-only inspection hit a JAX public `lower()` incompatibility when reconstructing Brax's `UInt64` from argument metadata. This was reproduced independently of CAT. A flat-array inspection wrapper compiled locally, but repeated SSH timeouts prevented its transfer for the final GPU check. The prior diagnostic exited, and the last successful remote check found no GPU compute processes. This is an incomplete memory measurement, not evidence that the learner crashed during training.

The initialization measurement does not include the large rollout/optimizer workload. It also recorded the current JAX allocator limit as 25,277,005,824 bytes (about 23.54 GiB); GPU allocator limits and full-update memory should be checked together before raising the resource profile or restarting training.
