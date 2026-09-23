> **Historical record — superseded.** Native training now uses actor 222 / critic 310, no authored box features or rewards, and starts from scratch. Old expansion utilities are retired; no checkpoint trim is supported or requested. Commands and old test paths below describe the historical experiment, not the current workflow. See [current removal report](NATIVE_CLEARANCE_REMOVAL_20260922.md).

# Clearance-primary hand objective — measured CPU baseline and implementation

The current checkpoint **already reacts to an approaching obstacle**, but does not provide universal hand avoidance. Standing front/side responses are substantial; approaches from below still collide. During walking, much of the front/side displacement comes from changing locomotion rather than moving the arm relative to the root. The new objective is implemented; no training was launched.

**Actual diagnostic.** `scripts/diagnose_reactive_clearance_cpu.py:39` copies and loads `outputs/cat_recover_23_30720_20260922/best.pt`. SHA-256 of source and copy: `972cd94fd3ab6b6571dfd6eee202f368e23ff9f83a06af46f63d29c8d1f2dc84`. CPU MuJoCo, frozen deterministic policy, 24 conditions plus 24 matched no-obstacle controls, one seed, left hand, four directions, three speeds. Source PD/gravity compensation is retained; sensor noise, pushes and actuator randomization are disabled. Controls and treatments agree exactly before the stimulus (maximum position difference 0).

After 2 s settling, a radius-6 cm sphere approaches from an initial 55 cm hand-surface gap, at 5/10/20 cm/s, until its prescribed path reaches −2 cm nominal clearance or the first contact/termination. Zero command tests standing; walking command is 0.6 m/s. For walking, the sphere is advected by the **control's** root XY displacement, never by the treated hand; a robot can slow, turn or sidestep away. Reported walking mean forward speeds range 0.614–0.811 m/s. This is an open-loop approach probe, not a pursuing obstacle.

The sphere is instantiated as an analytic SDF/normal/guidance obstacle sampled by all production body/hand observation queries. As with production bank obstacles (`cat_mjlab/collision.py:1`, `cat_mjlab/model.py:15`), it applies no contact impulses. Thus retreat cannot be a physically pushed hand. These are ideal analytic fields, not a dynamic voxel-field rebuild; no robot-obstacle rigid-contact-force or full-body sphere collision validation is claimed. This probe measures hand-sphere clearance and native task termination. The checkpoint, banks and reset files are unchanged.

Onset is the first of five consecutive 20 ms samples with >1 cm **world** retreat relative to the paired control (`scripts/diagnose_reactive_clearance_cpu.py:96`). Root-relative retreat subtracts root translation; it does not remove torso rotation. All observation sites see the sphere, so onset outside 20 cm is possible and **does not prove clearance-reward causality**. Front/above standing onset is already near the initial placement; the true farther-away threshold is not identified. Root uprightness requires height >0.50 m and pelvis-up dot world-up >0.70 throughout the trial. This is stricter than waiting for native fall termination.

[Standing image strip](../outputs/reactive_clearance_cpu/standing_strip.png), [walking image strip](../outputs/reactive_clearance_cpu/walking_strip.png), [response traces](../outputs/reactive_clearance_cpu/response_traces.png), [full machine-readable measurements](../outputs/reactive_clearance_cpu/results.json). Strips reconstruct actual MuJoCo configurations on CPU; they are schematic skeletons, not photorealistic renders. Green is the treated hand and gray is its matched control. Full traces and copied checkpoint occupy about 8.3 MiB.

| Mode | From | Speed cm/s | Peak retreat cm | Root-relative peak cm | Onset gap cm | Min gap cm | <20 cm % | <4 cm % | Upright | Contact |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| standing | side | 5 | 7.5 | 3.8 | 47.1 | 4.21 | 34.2 | 0.0 | yes | no |
| standing | side | 10 | 9.0 | 5.6 | 46.3 | 3.10 | 35.8 | 1.4 | yes | no |
| standing | side | 20 | 8.2 | 4.3 | 45.2 | 4.69 | 35.7 | 0.0 | yes | no |
| standing | front | 5 | 23.4 | 20.4 | 55.9 | 12.59 | 12.6 | 0.0 | yes | no |
| standing | front | 10 | 18.4 | 15.8 | 53.7 | 10.37 | 18.9 | 0.0 | yes | no |
| standing | front | 20 | 23.6 | 21.9 | 50.8 | 9.36 | 24.5 | 0.0 | yes | no |
| standing | below | 5 | 3.7 | 2.3 | 42.3 | -0.03 | 32.8 | 4.6 | yes | yes |
| standing | below | 10 | 3.8 | 2.1 | 40.1 | -0.01 | 31.5 | 5.7 | yes | yes |
| standing | below | 20 | 3.2 | 2.1 | 35.6 | -0.23 | 34.3 | 9.1 | yes | yes |
| standing | above | 5 | 7.4 | 2.4 | 55.9 | 2.25 | 32.8 | 3.5 | yes | no |
| standing | above | 10 | 6.7 | 2.9 | 55.4 | 3.12 | 34.0 | 3.5 | yes | no |
| standing | above | 20 | 5.8 | 2.4 | 54.4 | 2.74 | 34.3 | 3.5 | yes | no |
| walking | side | 5 | 21.6 | 1.4 | 51.2 | 24.07 | 0.0 | 0.0 | yes | no |
| walking | side | 10 | 2.4 | 0.0 | 8.8 | 7.70 | 44.2 | 0.0 | yes | no |
| walking | side | 20 | -0.0 | -0.0 | none | 4.53 | 42.7 | 0.0 | yes | no |
| walking | front | 5 | 46.1 | 7.3 | 47.2 | 41.40 | 0.0 | 0.0 | yes | no |
| walking | front | 10 | 43.1 | 6.8 | 46.5 | 40.40 | 0.0 | 0.0 | yes | no |
| walking | front | 20 | 26.7 | 5.7 | 41.2 | 27.02 | 0.0 | 0.0 | yes | no |
| walking | below | 5 | -0.0 | -0.0 | none | 12.75 | 23.3 | 0.0 | NO | no |
| walking | below | 10 | -0.0 | -0.0 | none | 21.21 | 0.0 | 0.0 | yes | no |
| walking | below | 20 | -0.0 | -0.0 | none | 12.21 | 20.3 | 0.0 | yes | no |
| walking | above | 5 | 17.0 | 11.6 | 51.6 | 35.24 | 0.0 | 0.0 | yes | no |
| walking | above | 10 | 14.0 | 10.4 | 47.0 | 34.87 | 0.0 | 0.0 | yes | no |
| walking | above | 20 | 15.3 | 11.1 | 47.5 | 23.93 | 0.0 | 0.0 | yes | no |

Standing: 12/12 remained upright; below approaches contacted the hand in 3/3 conditions. Side peak root-relative retreat was 3.8–5.6 cm; front 15.8–21.9 cm; below only 2.1–2.3 cm; above 2.4–2.9 cm. Walking: front root-relative retreat was 5.7–7.3 cm and above 10.4–11.6 cm. Side was 0–1.4 cm; below had **no away retreat**. No walking hand contacts occurred in this finite probe, but the 5 cm/s below trial failed the uprightness criterion (root minimum 0.585 m; excessive tilt, without native termination). Clear space gained by locomotion is not evidence of successful hand tucking. These are single-seed baseline measurements, not reliability estimates or passage-acceptance results.

**Implemented controls.** `train_cat_mjlab.py:33`, `cat_mjlab/config.py:26`, and `cat_mjlab/clearance_objective.py:5` add:

- `--disable-hand-contrast`: defaults absent/off. In the new mode both region and heading rewards are zero, independently of their stored weights. The flat posture-box bonus is also zero. Box metrics are still calculated, but box inputs are zeroed with the existing observation dimensions retained; box compliance no longer selects checkpoints or drives protected/transition sampling success. Box-based tolerance/speed ladders are rejected in this mode.
- `--hand-clearance-weight`: signed, finite, ≤0; default −0.5 with hand protection.
- `--hand-clearance-target`: default 0.04 m.
- `--hand-clearance-anticipation`: default 0.20 m; must exceed target >0.
- `--hand-clearance-near-weight`: default 0.8; constrained to [0,1].

Explicit overrides are saved in the environment contract and inherited on subsequent warm starts. Legacy defaults and legacy resume state keys are preserved. `cat_mjlab/task.py:541` removes disabled rewards and applies the clearance penalty **after** the existing reward floor, only in the new mode. Otherwise large negative sums could make increasing clearance pressure ineffective. The inherited box-conditioned soft floor is explicitly disabled in the recommended command. Existing physical collision termination and native locomotion rewards remain active.

**Starting magnitudes, not a trained optimum.** Recommend weight **−20**, target **0.04**, anticipation **0.20**, near weight **0.5**. The earlier component receives half the pressure budget instead of 20%, while retaining the steep near-contact component. For one threatened hand and the other clear, at 50 Hz:

| Surface distance | Legacy reward/step (−0.5, near=.8) | Recommended reward/step (−20, near=.5) |
|---|---:|---:|
| 20 cm | 0 | 0 |
| 10 cm | −0.00025 | −0.025 |
| 4 cm | −0.00064 | −0.064 |
| Contact | −0.005 | −0.200 |

The copied policy's actual, censored treated trajectories give mean clearance contributions **−0.00004573/step** under the old settings and **−0.00385129/step** under the proposed settings (same trajectories, not newly learned behavior), an 84.2× magnitude increase. The recommendation at one-hand contact matches the former flat bonus's maximum +0.20/step (scale 10 × 0.02 s), which is now disabled. The existing `outputs/protected_objective_audit_cpu/audit.json` ledger also reported a raised-minus-nominal clearance difference of only +0.0093576 versus +1.3399477 for native `handsdf` and +0.2331327 for region reward; that is a prior audit under its own configuration, not this moving-sphere probe. The diagnostic's full raw ledger is retained and explicitly labeled as including controls and post-terminal resets; the censored clearance comparison above is the appropriate quantitative comparison. Native field terms still contribute: clearance-primary does not mean clearance-only.

**Acceptance.** `cat_mjlab/acceptance.py:216` adds `ClearancePassageGate`. Trials remain assigned before action, with pending and failed trials in the denominator. It preserves the original S gate's ordered entry/exit geometry, swept bypass checks, ≥0.2 m/s zone-crossing budget, deadline, fall-first and body/hand collision failure checks. Measured clearance replaces prescribed pose evidence. Any missing/nonfinite sample or either hand <0.04 m fails clearance acceptance; being within 0.20 m is reported, not automatically failed, because valid narrow passages may require it. Thresholds are fixed independently of reward tuning.

Training telemetry and `evaluate_assigned_trials(..., clearance_primary=True)` (requiring `initial_hand_clearance` on each assignment) report minimum hand-obstacle clearance, exposure time/fraction within 0.20 m and below 0.04 m, and hand-collision incidence per **assigned** trial. Exposure is dt weighted from reset to first outcome, not restricted to successful survivors or entered passages. Missing exposure and pending counts remain explicit. The training observer additionally reports mean per-trial minima and lifetime cohort minima, captured before autoreset. Distances are sampled at 20 ms; production collision signals remain latched over physics substeps. Seeded/inside-start diagnostics remain separate from main upstream trials.

`cat_mjlab/clearance_objective.py:39` replaces box-based best-checkpoint selection in the new mode: 0.6 assigned clearance S + 0.2 CAT clean-goal retention + 0.2 flat walking fraction. Missing evidence defers selection. S is cumulative changing-policy training telemetry; CAT/flat use the existing rolling window. This is explicitly **not frozen-checkpoint acceptance**. Release acceptance should use a fixed upstream cohort, require complete evidence, improve clearance S and contact incidence over the same baseline cohort, and preserve CAT/clutter success and crossing time. The moving-sphere results above do not establish those passage results.

**Pushback and mitigation.** Pure repulsion can reward refusing a passage, slowing indefinitely, rotating the whole body, or falling/ending an episode to avoid future penalties. The walking probe already demonstrates whole-body avoidance, and zero hand contacts alone hides the slow-below loss of uprightness. S rejects refusal, crawling, bypass and falling, but an evaluation gate alone does not remove the local PPO incentive to avoid entry. Preserve route tracking, deadlines and collision costs; retain CAT/clutter/flat locomotion cohorts; begin with the bounded pilot below and reject it if entry rate or S degrades, even if mean clearance improves. Monitor early termination rates as well as contacts. If refusal emerges, add a required-progress curriculum or progress/entry reward calibrated against the integrated avoidance cost; do not merely increase repulsion again. Stationary randomized moving-obstacle training episodes would be needed to deliberately train universal reactivity; **they are not added to the immutable banks or implemented as training here**. Generalization from static passage training remains unverified.

**Manipulation mode — design only, not implemented.** Append an explicit task-provided `manipulation_mode` observation scalar (0 avoidance, 1 intended reaching) to both actor and critic and gate hand-clearance pressure by `(1-mode)` before averaging/scaling. Default zero preserves avoidance. Actor 254→255 and critic 342→343 require a versioned observation contract/checkpoint upgrade, including normalizer entries; initialize new first-layer columns to zero. For this checkpoint's 512/1024 first layers, that adds 1,536 weights (~6 KiB FP32; ~12 KiB Adam moments), plus roughly 7.5 MiB for both added observation columns in a 30,720×32 rollout. Runtime multiplication is negligible; estimates exclude extra next-observation/storage copies. No new model input or gate has been implemented now.

To actually permit touch, reward gating alone is insufficient: native hand collision termination and other hand-distance penalties must distinguish the designated manipulation target from forbidden obstacles. Pair the mode with an explicit target/contact allowlist and exempt only intended hand-target contacts while preserving other obstacle/body collision rules. A single global bit must not silently authorize arbitrary contacts. That collision/target plumbing is additional work beyond the small observation/penalty change.

**Verification.** 117 distinct applicable CPU tests passed across the 115-test regression run and the final 49-test run (20 focused objective tests plus 29 acceptance tests; overlapping earlier coverage). `tests/test_clearance_primary.py:61` tests disabled rewards with live nonzero box diagnostics and floor survival. `tests/test_clearance_primary.py:97` tests adversarial clearance traces. `tests/test_clearance_primary.py:139` loads real immutable flat, generic clutter, CAT, narrow and protected fields, certified reset poses and the production collision checker: default versus explicit-default rewards, actor/critic observations and qpos are bitwise equal for five steps per world (0.10 s). This is regression equivalence, not a long-duration behavior study. Existing acceptance, runner, observation, floor and reward override tests also pass. Seven legacy JAX-reference cases were excluded after failing because this environment lacks `ml_collections`, `jaxlie`, and `mujoco_playground`; they were not silently counted as passes. [Regression log](../outputs/reactive_clearance_cpu/regression_tests.txt), [focused log](../outputs/reactive_clearance_cpu/focused_tests.txt).

**Prepared command, NOT launched.** The complete parser-tested command is in `configs/pilots/clearance_primary_50.sh:5`. It starts a fresh optimizer from the specified best checkpoint, uses the same v2 scene/collision/reset bank, runs a bounded 50-update pilot, and logs online to `CAT-wholebody / skvayzer`:

```bash
bash configs/pilots/clearance_primary_50.sh
```

Reproduce the diagnostic (CPU, no training) with:

```bash
.venv-mjlab/bin/python scripts/diagnose_reactive_clearance_cpu.py
MPLCONFIGDIR=/tmp/clearance-mpl .venv-mjlab/bin/python scripts/summarize_reactive_clearance.py
```

GPU probing failed because the NVIDIA driver was unavailable in this sandbox. All work used CPU; no other rendering job was interrupted. The shared disk remained at approximately 4.9 GiB available. No installation, bank generation, training launch, or source-checkpoint modification was performed.
