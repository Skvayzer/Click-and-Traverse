**Recommendation: neither a rigid pelvis-relative box nor a larger absolute-Z extent fixes the measured problem.** The pelvis-relative prototype was actually executed in all 12 protected scenes. It shortened initial compliance to 6–23 steps, produced 0/12 core crossings, and failed the lateral safety requirement. The absolute-Z-only explanation is not supported: heading rotation and horizontal error are major problems in these rollouts.

This is a CPU-only, measurement-only experiment. No training, bank changes, flags, production integration, live checkpoint reads, GPU calls, or tmux operations were performed. The existing 5.2 MiB frozen reference snapshot was reused in place. New artifacts are approximately 1.5 MiB. The simulation batches took 83.23 s; this excludes postprocessing. The shared pool still had approximately 3.6 GiB free after measurement.

**Frame and paired experiment.** The original box follows root X/Y but uses route-fixed axes and absolute Z. The alternative is rigidly attached to the free-joint pelvis/root, including yaw, pitch, roll, and vertical translation. At reset it exactly coincides with the original box, avoiding a target-position advantage. Its center and axes are then transported by the change in root pose. The root is a reproducible physical anchor independent of arm joints; waist motion remains a separate tracking demand. This is a full pelvis frame, not a yaw-locked or height-only hybrid. Those hybrids were not dynamically tested. Definition: `scripts/prototype_body_region_cpu.py:33`.

Both modes use half-extents (0.018, 0.010, 0.018) m, 0.05 m Euclidean distance-to-box tolerance, the same certified crouched reset, policy mean actions for legs/waist, and zero arm residual. The arm solver targets the appropriate box center at each 20 ms control step with the same 12-evaluation budget and bounds. Production slew, PD, randomization, feedforward, obstacle checks, and observations are retained. Only prototype reference generation and external compliance scoring change; production rewards still use their existing frame and are not used for optimization. Sources: `scripts/prototype_body_region_cpu.py:58`, `scripts/prototype_body_region_cpu.py:170`.

Frozen checkpoint SHA256: `042d1dacb2ad57ac0d665e6cf8057b166e7573b8f2b3af0ef8fc57677aa97419`; evaluation manifest SHA256: `f59aa8dcabccb0f8d14a43400a57ba75658af0360117f416f9d1a15e702a253e`. The historical four-scene absolute baseline reproduced **exactly** for hold, longest streak, compliant fraction, termination step, final error/progress, crossing, obstacle, and fall flags (`outputs/body_region_cpu/validation.json:2`). Other scenes use two additional four-environment batches with the same seed, paired between modes. The 12 scenes contain four geometries repeated at three curriculum rungs; they are not 12 independent geometries or statistical robustness trials.

**Error decomposition.** A binary “body versus arm” percentage would hide compensating vectors and arm-reference changes. Instead, at each endpoint, exact MuJoCo FK gives the identity

`actual hand − absolute target = arm tracking + pelvis motion + waist motion + requested-reference residual at reset pelvis/waist`.

Arm tracking compares the actual hand with the hand at the requested IK arm angles, at the same measured endpoint root/waist. It therefore includes arm slew/PD tracking against the request. Pelvis motion compares that requested arm pose at current versus reset root height/orientation, retaining current root X/Y. Waist is separately restored to its reset angles. The last term retains the requested arm configuration; it is not assumed to be zero. The identity is numerically asserted for every sample (`scripts/prototype_body_region_cpu.py:100`).

For additive **distance-to-box** attribution, subtract the nearest point in the absolute box from the last term and project all four vectors onto the measured error direction. These signed terms sum to the actual distance. Further split pelvis motion into height plus pitch/roll, followed by yaw, using a stated ZYX rotation order. This is an exact kinematic counterfactual decomposition, not an order-independent causal allocation or a new dynamics experiment (`scripts/analyze_body_region_cpu.py:47`, `scripts/analyze_body_region_cpu.py:74`).

At **1.00 s**, worst hand per scene; all quantities below are **cm**:

| Scene | Original sampled error | Fresh FK error | Height + pitch/roll contribution | Yaw contribution | Arm tracking contribution | Waist contribution | Reference/box remainder |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2351 | 4.72 | 4.65 | +0.58 | +7.35 | +0.55 | +0.09 | −3.92 |
| 2357 | 12.22 | 12.16 | −1.72 | +21.52 | +6.45 | +0.13 | −14.21 |
| 2363 | 9.51 | 9.42 | −0.95 | +14.12 | +2.20 | −0.58 | −5.37 |
| 2369 | 7.91 | 7.85 | +0.92 | +6.15 | +0.25 | −1.77 | +2.30 |

The height/pitch/roll contribution is **−1.72 to +0.92 cm**, not most of the 4.7–12.2 cm measured error. Its gross displacement magnitude is 3.17, 2.07, 1.27, and 4.30 cm respectively. Root-height changes alone are +11.93, +8.04, +6.80, and +8.43 cm, but pitch/roll largely offset their hand displacement. Gross yaw displacement is 26.78, 38.07, 32.94, and 30.20 cm. Gross arm tracking displacement is 2.01, 8.02, 4.55, and 2.26 cm. These magnitudes must not be added as scalar error shares. Raw samples and per-scene RMS values: `outputs/body_region_cpu/analysis.json:16`.

Another direct counterfactual removes only height and pitch/roll changes from the *actual* measured arm posture, preserving current yaw. The resulting absolute errors are **6.66, 12.72, 10.50, and 8.05 cm**, all worse than the corresponding fresh-FK baseline errors. This does not establish what a reoptimized controller would do, but it rejects interpreting those particular failures as simple uncompensated vertical bob.

At 1 s, horizontal-only box errors are **4.21, 12.16, 9.00, and 7.51 cm**. Requested arm angles evaluated at the measured endpoint still have region errors **4.49, 7.41, 8.25, and 7.93 cm**, respectively; perfect instantaneous tracking of that request would not fix three of four. This is distinct from proving global IK infeasibility. The absolute rollouts span route-relative yaw −0.05° to 83.37°, pitch −3.07° to 28.42°, roll −12.03° to 11.49°, and root height 0.541–0.739 m. These are substantial crouch/heading changes, not merely small steady-gait bob.

The original production hand sites are from the last pre-integration MuJoCo state, whereas qpos is post-integration. Comparable rollout compliance deliberately preserves the old sampling. Decomposition uses refreshed endpoint FK for an exact vector identity. Across all runs, maximum hand-position difference between these sampling conventions is 3.83 mm; at the four 1 s samples the maximum-error difference is under 1 mm. No claim of exact causal percentages is made across these conventions.

**Actual walking comparison.** Initial continuous compliance ends at the first violation. Percentage is newly traversed compliant forward distance divided by the full 1.35 m core, as in the baseline; backtracking cannot double-count. Start progress is 0.52 m, leaving 1.33 m / approximately 111 steps at commanded 0.6 m/s. A “full compliant crossing” requires reaching the core end without a sampled violation. Episodes stop on production termination or core end, with no post-reset samples included.

| Scene | Absolute continuous steps / seconds | Body continuous steps / seconds | Absolute compliant core | Body compliant core | Body termination step |
|---|---:|---:|---:|---:|---:|
| 2351 | 18 / 0.36 | 10 / 0.20 | 10.75% | 3.15% | 25 |
| 2353 | 19 / 0.38 | 8 / 0.16 | 8.38% | 0.00% | 19 |
| 2355 | 32 / 0.64 | 10 / 0.20 | 3.16% | 5.60% | 33 |
| 2357 | 10 / 0.20 | 6 / 0.12 | 9.57% | 5.15% | 26 |
| 2359 | 30 / 0.60 | 23 / 0.46 | 8.37% | 4.19% | 23 |
| 2361 | 24 / 0.48 | 7 / 0.14 | 5.10% | 2.87% | 19 |
| 2363 | 14 / 0.28 | 13 / 0.26 | 7.86% | 2.45% | 23 |
| 2365 | 13 / 0.26 | 12 / 0.24 | 7.71% | 1.12% | 23 |
| 2367 | 12 / 0.24 | 12 / 0.24 | 8.17% | 0.69% | 21 |
| 2369 | 32 / 0.64 | 12 / 0.24 | 9.40% | 1.98% | 20 |
| 2371 | 14 / 0.28 | 12 / 0.24 | 8.47% | 0.50% | 24 |
| 2373 | 12 / 0.24 | 12 / 0.24 | 2.99% | 1.03% | 24 |

Original four-scene subset is 2351/2357/2363/2369: **18/10/14/32 → 10/6/13/12 steps**, **10.75/9.57/7.86/9.40% → 3.15/5.15/2.45/1.98%**. Full compliant crossings are **0/4 in both modes** and **0/12 overall in both modes**. Two absolute episodes reach the core end noncompliantly; none of the body episodes do. All 12 body episodes terminate on obstacle checks after 19–33 steps (0.38–0.66 s). No falls occur before these stopping points. Do not reinterpret obstacle termination as a measured finger-mesh collision. Evidence: `outputs/body_region_cpu/measurements.json:4`; raw endpoint arrays in six small NPZ files alongside it.

**Safety fails.** Evaluated 12,490 observed root poses at 2 ms resolution: 9,690 absolute-rollout poses and 2,800 body-rollout poses, including terminating control steps and excluding auto-reset states. Every scene is represented. For each pose, the full oriented box's support along route-normal is expanded by the full 50 mm tolerance ball and the orientation-independent 103.446173 mm finger/thumb sphere. The analytic lateral bound applies when swept along every cabinet pair in the scene, covering all three protected zones without longitudinal sampling gaps. It includes either zero root cross-track (direct comparison with the old certificate) or the actual observed cross-track. Formula and implementation: `scripts/prototype_body_region_cpu.py:130`.

All entries below are **mm**. A negative value means the full allowed finger envelope does not fit between cabinets; it is not necessarily the penetration of the realized hand.

| Scene | Original box, centered | Body box, centered, body-run poses | Body box, actual cross-track, body-run poses | Body box, actual cross-track, poses from either run |
|---|---:|---:|---:|---:|
| 2351 | +80.36 | −98.80 | −77.14 | −200.26 |
| 2353 | +80.49 | −8.43 | −47.95 | −175.52 |
| 2355 | +78.99 | −145.07 | −106.13 | −228.28 |
| 2357 | +81.59 | −84.99 | −66.44 | −263.39 |
| 2359 | +80.36 | −59.27 | −50.39 | −248.56 |
| 2361 | +80.49 | −26.40 | −61.98 | −193.54 |
| 2363 | +78.99 | −135.32 | −116.26 | −189.78 |
| 2365 | +81.59 | −118.05 | −114.88 | −195.63 |
| 2367 | +80.36 | −65.36 | −54.65 | −191.20 |
| 2369 | +80.49 | −101.46 | −80.77 | −186.31 |
| 2371 | +78.99 | −130.41 | −101.11 | −191.48 |
| 2373 | +81.59 | −136.38 | −100.81 | −198.06 |

The important paired safety comparison is **+78.99 mm original centered minimum versus −145.07 mm body centered minimum** on body-run poses. Including their measured root drift gives a body minimum of **−116.26 mm**. Applying the proposed body box to the longer absolute-run pose range worsens this to **−263.39 mm**. The latter is a geometric counterfactual, not a body rollout that survived that long. Own body-run yaw reaches 34.07° before termination, already enough to destroy the lateral margin. Its metric-expanded finger envelope remains at least 366.79 mm above the floor: the failure is lateral, not ground clearance.

The old +79 mm was explicitly a centered-route certificate. With actual drift in the *absolute* runs, even the old box's lateral bound falls to −64.03 mm; with the shorter *body* runs' root drift the old box remains at least +29.87 mm. Thus neither the original certificate nor the present experiment supports unrestricted root drift. This qualification does not rescue the body box: it fails even with the root centered.

To avoid treating a loose sphere bound as proof of mesh contact, a separate exact-scene check constructs allowed-site witnesses from box corners plus up to 50 mm route-normal displacement. At the offending observed root pose, translate the real fixed Dex3 hand mesh to each allowed site, using that episode's measured terminal wrist orientation. **11/12 scenes have an actual hand-mesh vertex inside an actual finite cabinet**, with worst signed vertex distance **−39.00 mm** (scene 2357). Each witness passes the body's distance-to-box <=50 mm check. The remaining scene, 2373, has no negative vertex found by this limited witness search; its negative sphere/lateral bound still fails the clearance certificate. Implementation: `scripts/verify_body_region_safety_cpu.py:45`; coordinates, distances, and per-scene results: `outputs/body_region_cpu/safety_witnesses.json:2`.

These are geometric counterexamples to “body-region compliance implies finger protection,” not claims that those translated wrist poses are IK-reachable or were physically visited. Actual terminal-pose mesh vertices did not show cabinet penetration in this check; positive vertex distances alone do not certify whole-mesh clearance. No body rollout completed a traversal, so unobserved late-traversal poses cannot be certified. The counterexamples already occur within the observed early pose range; no extrapolation is needed to reject the proposed geometric safety guarantee. Since physical safety already fails, no new stored-field safety certificate is claimed.

**Absolute-Z alternative.** Across all 969 absolute-rollout endpoints, measured hand Z spans **0.546216–0.713421 m**. Keeping center **0.614059567 m**, the smallest symmetric half-extent that covers this *vertical-only* variation with the full 50 mm tolerance allocated to Z is **0.0493619 m**, or **98.724 mm full height**, versus the current 36 mm. This is an observed envelope, not a future-gait bound.

Once the Euclidean XY/Z coupling is included, **588/969 frames (60.68%) already violate 50 mm in XY alone**. No finite Z extent makes those frames compliant. On the remaining XY-feasible frames the required common Z half-extent is 41.976 mm. Computation: `scripts/analyze_body_region_cpu.py:42`; totals: `outputs/body_region_cpu/analysis.json:9`.

Rescoring the unchanged absolute trajectories with **50 mm Z half-extent**, and separately with **no Z constraint at all**, yields identical compliance outcomes on this sample. For the original four scenes: initial holds **29/30/14/32 steps**, compliant core **14.27/9.66/7.86/9.40%**, still **0/4 full compliant crossings**; across all scenes still **0/12**. This is an explicit metric rescore, not another dynamics run. No wider-box bank was created. Lateral box dimensions would remain unchanged, but that alone cannot repair the measured horizontal violations and route drift.

**Decision and limits.** Reject the tested full pelvis frame: it is worse dynamically and does not preserve lateral protection. Reject wider absolute Z as a sufficient fix: even unlimited Z fails this recorded crossing test. Body motion does create tracking demand, especially heading rotation, but the measurements do not identify absolute world Z as the dominant residual-error cause. A yaw-locked hybrid or coordinated heading/root/arm controller remains unverified; this report does not recommend an integration or a training run for either. One frozen policy, seeded starts, and one random realization per scene do not rule out every possible body-relative controller.

Reproduce from repository root, CPU only:

```bash
PYTHONDONTWRITEBYTECODE=1 nice -n 10 .venv-mjlab/bin/python scripts/prototype_body_region_cpu.py
PYTHONDONTWRITEBYTECODE=1 .venv-mjlab/bin/python scripts/analyze_body_region_cpu.py
PYTHONDONTWRITEBYTECODE=1 .venv-mjlab/bin/python scripts/verify_body_region_safety_cpu.py
```

The scripts reuse the retained snapshot and write only `outputs/body_region_cpu/`. The rollout refuses to start or continue a batch below 2 GiB free space. Raw NPZ fields retain endpoint qpos, requested arm targets, FK hand positions, exact vector components, progress, legacy-sampled hands/errors, and 2 ms root poses. Assertions check vector reconstruction, signed error reconstruction, original metric agreement, support-function agreement with all eight corners, witness compliance, and absence of Torch CUDA initialization. No full training/integration tests were needed or run.
