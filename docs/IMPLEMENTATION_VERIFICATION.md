# Implementation verification — 2026-09-14

This file records implementation checks, not trained traversal results.

## September 15 corrections

- Replaced CAT's rigid rubber hands with fixed Unitree Dex3-1 three-finger
  geometry and source inertials. Kept all 29 body joints/actions. Native MuJoCo
  checks all 81,432 compiled vertices per hand against the actual collision
  boxes; the measured minimum padding is 4.999998 mm. Thumbs are included.
- Regression checks cover nominal/raised/tucked arms and three wrist
  configurations, source mass and joint contracts, matching corner probes,
  actual contact in the thumb region missed by the previous generic box, and
  non-contact just outside the corrected box.
- The training audit found that the generic cached autoreset retained task
  info after termination. A furniture-specific wrapper now restores contact
  and fall flags, motor targets, map history, episode clock and clearance state
  for each terminated vector slot. Synthetic multi-episode tests cover
  asynchronous terminations, time limits and telemetry reductions.
- Training logs now include completed-episode task metrics and explicit episode
  counts. No completion means no reported outcome rates. Reward terms are
  summed; outcome/progress/minimum-clearance values retain their episode meaning.
- The full CPU suite passed **132 tests in 102.31 s** after both corrections.
  The 13 warnings are pinned JAX/JAXopt deprecations and MuJoCo sentinel casts;
  finite-state checks pass. This includes native CAT warm-start/export parity.
- The corrected model and training wrapper passed a **128-transition GPU PPO
  integration check** on dep-0, with four parallel environments and two
  checkpoint candidates. Initialization preserved the original actor/critic
  outputs exactly in the named-mapping parity check. Parameters updated by up
  to 0.00661; the selected step-128 ONNX matched native inference within
  4.63e-6. Training telemetry reported 15 completed episodes, and one checkpoint
  generation remained. This checks updates, resets, logging and export; the
  tiny run is not evidence of useful learned traversal. It used commit
  `3f21a4d`; [the saved report](assets/dex3-gpu-smoke.json) records provenance.
- A final optional-validation metric adjustment emits clearance/progress/mean
  summaries once at episode termination for Brax's additive evaluator. Its
  four focused wrapper regressions passed in 1.13 s. Training telemetry reads
  the same continuously updated info aggregates as in the GPU check. No
  performance evaluation was run.

The earlier short GPU checks below predate these fixes and establish only their
stated limited optimizer/compilation results. They did not establish reliable
repeated-episode learning. See [the training plan](TRAINING_READINESS.md) for the
bounded learning pilot and remaining performance questions.

## September 14 checks

- dep-0: RTX 5090, 32 GB; isolated Python 3.12.9 environment. Pinned JAX 0.4.38 CUDA initialization and numerical check passed. Environment size measured at 7.2 GB, native source weights about 5.4 MB.
- Native CAT checkpoint/ONNX parity, named 12→29 action/observation warm-start mapping, atomic selected-checkpoint retention and ONNX export regression checks passed (25 focused tests at that point).
- A preliminary 64-transition GPU PPO smoke completed parameter-update iterations and selected step 32. Its initial export comparison exposed reduced-precision GPU matmul differences. The precision setting was corrected without relaxing the 2e-5 parity threshold. Re-exporting the saved checkpoint passed, maximum native GPU/ONNX difference 8.19e-6.
- The dense scene's native MuJoCo model has 29 actuators and 379 geoms, including 298 obstacle boxes representing nine tables, 36 chairs and room constraints. The nominal start has no forbidden contact.
- GPU dense-scene reset and one zero-action step passed: finite 406/494 observations, no termination/contact. Including compilation, the bounded check took 67.9 s; a single step executed in 0.0426 s. These single-environment smoke timings are not PPO throughput estimates. See `assets/dense-mjx-smoke.json` for exact source hashes and measured memory scope.
- Original-family adapters passed 24 focused legacy/scene tests, including lossless voxel merging, source-coordinate alignment and nonzero vertical crouch guidance. Native reset checks passed for original forward, hurdle, crouch and random scenes.
- Strict evaluation/cache/provenance regression checks passed (43 focused tests at that point). No benchmark episodes or performance evaluation have been run.
- The combined CPU suite passed **114 tests in 82.04 s**, including the locally fetched pinned native CAT release fixture. One final generic-geometry regression was added afterward; the final changed-file suite passed **37 tests in 1.99 s**, covering that addition and the final recovery/evaluator changes. All **115 collected tests** have passed. The 12 combined-suite warnings come from pinned JAX/JAXopt deprecations and MuJoCo sentinel float casts; tested states and observations remain finite. Focused counts overlap and should not be summed.
- Mocked curriculum tests cover successful handoff, process failure, incomplete-stage preservation, corrupt payloads, invalid source lineage, interrupted state publication and interrupted model retirement.
- Final GPU integration passed: **two stages × 128 transitions = 256 transitions**, four environments per stage. Stage 1 used adapted original CAT forward geometry; stage 2 used generic pilot clutter. Both restored the actor/critic/normalizer, updated parameters and exported the selected step-64 native checkpoint. Maximum actor parameter changes were 0.00630 and 0.00637; maximum ONNX/native errors were 3.57e-6 and 5.27e-6. The added upper-body mean head also changed from its zero initialization. These are optimizer/integration checks, not useful trained policies or traversal scores.
- The successful handoff retired the first stage's model and left **one selected checkpoint generation** across the sequence. The final code revalidated both receipts, source lineage and retirement state without running further training. Training used commit `dd0c79dc10a0dae94ff22ab5258d26a1a06162c0`; final reconciliation used `18647b9`. Exact source hashes and summaries are in `assets/mixed-training-smoke.json`. No evaluation episodes or W&B sessions were started by the smoke run.
- The complete default test specification generated successfully: 120 rooms, 360 scene/start-goal cases, 1,080 planned episodes per controller across three training seeds. Manifest SHA256: `5e93453db3c32d220c931b065db3019798f24a458db44e4b99fe6f8e5a8c18a7`. This generated specifications only; no evaluation episodes ran.

`assets/dense-room.png` is an actual static MuJoCo render. No learned policy or rollout was used. Its JSON sidecar records geometry and source provenance.

Long training, dense traversal success, retention of original CAT skills, perception robustness and hardware deployment remain unmeasured. No convergence or collision-reduction claim is made.
