# Implementation verification — 2026-09-14

This file records implementation checks, not trained traversal results.

- dep-0: RTX 5090, 32 GB; isolated Python 3.12.9 environment. Pinned JAX 0.4.38 CUDA initialization and numerical check passed. Environment size measured at 7.2 GB, native source weights about 5.4 MB.
- Native CAT checkpoint/ONNX parity, named 12→29 action/observation warm-start mapping, atomic selected-checkpoint retention and ONNX export regression checks passed (25 focused tests at that point).
- A preliminary 64-transition GPU PPO smoke completed parameter-update iterations and selected step 32. Its initial export comparison exposed reduced-precision GPU matmul differences. The precision setting was corrected without relaxing the 2e-5 parity threshold. Re-exporting the saved checkpoint passed, maximum native GPU/ONNX difference 8.19e-6. A final integrated run is pending after the latest hand-reward/curriculum changes.
- The dense scene's native MuJoCo model has 29 actuators and 379 geoms, including 298 obstacle boxes representing nine tables, 36 chairs and room constraints. The nominal start has no forbidden contact.
- GPU dense-scene reset and one zero-action step passed: finite 406/494 observations, no termination/contact. Including compilation, the bounded check took 67.9 s; a single step executed in 0.0426 s. These single-environment smoke timings are not PPO throughput estimates. See `assets/dense-mjx-smoke.json` for exact source hashes and measured memory scope.
- Original-family adapters passed 24 focused legacy/scene tests, including lossless voxel merging, source-coordinate alignment and nonzero vertical crouch guidance. Native reset checks passed for original forward, hurdle, crouch and random scenes.
- Strict evaluation/cache/provenance regression checks passed (43 focused tests at that point). No benchmark episodes or performance evaluation have been run.
- Full combined CPU suite and final mixed-stage GPU integration are being completed in subsequent commits. The individual focused counts overlap and should not be summed.

`assets/dense-room.png` is an actual static MuJoCo render. No learned policy or rollout was used. Its JSON sidecar records geometry and source provenance.

Long training, dense traversal success, retention of original CAT skills, perception robustness and hardware deployment remain unmeasured. No convergence or collision-reduction claim is made.
