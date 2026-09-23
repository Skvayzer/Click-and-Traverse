# CPU residual-reference feasibility — 2026-09-22

**Result: NO for the tested arm-reference scheme with the current frozen locomotion policy.** Zero fully compliant protected-core crossings out of four zero-residual episodes; zero out of 20 episodes across the noise sweep. No training or GPU initialization.

## Method

Copied the requested best.pt once (5.2 MiB); checkpoint step 40304640, SHA256 `042d1dacb2ad57ac0d665e6cf8057b166e7573b8f2b3af0ef8fc57677aa97419`. The original run was not changed. Evaluation v6 manifest SHA256 `f59aa8dcabccb0f8d14a43400a57ba75658af0360117f416f9d1a15e702a253e`; checkpoint training-bank SHA256 `646c6fb2b1c1ea0112032cfee2d24e42f091878b9eaa61a8a230b0c64093d0cc` differs. Used the matching existing v6 collision bank, with its normal hash checks.

Four scenes, manifest indices [2351, 2357, 2363, 2369]. Start from cached certified compliant poses at progress 0.52 m, zero initial velocity. Current policy mean actions drive legs and waist. Runtime observation noise, PD randomization, force noise, production collision checks, 2 rad/s target slew and torque limits remain enabled. Gravity feedforward remains enabled; shoulder pitch authority is 1.0 rad/action as in the previous improved hold baseline (the copied checkpoint has 0.8). No root or joint teleportation after reset.

Each 20 ms control step fixes the measured root, legs and waist, then optimizes only 14 arm joints against active box centers in the route frame. Reuses commandable_limits; MuJoCo analytic hand Jacobians and warm-start bounded least squares, capped at 12 function evaluations. This is a local approximate solver, not a global reachability proof. Existing UpperGravity is evaluated each 2 ms physics substep.

Measured IK cost in the final sweep batch: mean 2.14 ms, p95 3.17 ms per robot/control step. Whole sweep wall time 121.5 s. CPU physics runs slower than real time; per-robot reference generation fits the 20 ms control period.

## Zero-residual measurements

Compliance means the maximum of the two hand distances to the active boxes is <= 0.05 m. Continuous duration below starts at reset and ends on the first violation; longest subsequent streak is also recorded in measurements.json. Passage fractions count newly traversed forward core distance whose endpoint is compliant, sampled every 20 ms; backtracking cannot double-count. No post-reset episode samples are counted.

| Scene index | Initial continuous hold | Longest hold | Full core traversed compliant | Terminal event | End step |
|---|---:|---:|---:|---|---:|
| 2351 | 18 steps / 0.36 s | 18 steps | 10.75% | Obstacle termination | 65 |
| 2357 | 10 steps / 0.20 s | 19 steps | 9.57% | Reached core end, noncompliant | 138 |
| 2363 | 14 steps / 0.28 s | 15 steps | 7.86% | Obstacle termination | 52 |
| 2369 | 32 steps / 0.64 s | 32 steps | 9.40% | Obstacle termination | 51 |

No falls detected in any of the 20 episodes before termination/core exit. This is not a claim about what would happen after collision termination. Three zero-residual episodes terminate on obstacle checks; the fourth reaches the core end at step 138 with 12.86 cm hand error.

The earlier 38–83-step result was zero-command hold, so this walking comparison is not a paired controller ablation. Here initial continuous hold is 10–32 steps, and even the longest zero-residual streak is only 32 steps: below the requested 52-step reference. Measured command is 0.6 m/s. These scenes have a 1.35 m protected core (0.50–1.85 m); from the seeded start, 1.33 m remains, requiring about 111 steps at commanded speed, rather than 52. The one core crossing actually takes 138 steps. Tests stop at core exit; no successful full-zone exit is claimed.

## Hand distance over time

Maximum left/right distance to active region, cm. Full 20 ms traces, individual hand distances, IK errors, commanded speed and root height are in `outputs/residual_reference_cpu/trace.csv`.

| Scene index | 0.20 s | 0.50 s | 1.00 s | Terminal/core-exit |
|---|---:|---:|---:|---:|
| 2351 | 0.89 | 3.09 | 4.72 | 8.43 |
| 2357 | 4.82 | 3.18 | 12.22 | 12.86 |
| 2363 | 1.66 | 1.89 | 9.51 | 9.30 |
| 2369 | 2.35 | 3.91 | 7.91 | 7.94 |

## Noise tolerance

Independent uniform residual per arm joint per control step, added in normalized action coordinates and clipped before normal production slew/PD. Each amplitude uses the same task seed and residual RNG seed; one rollout per scene/amplitude, not a statistical robustness bound. Small amplitudes can improve individual trajectories, so there is no monotonic threshold estimate.

| Residual half-range | Target perturbation half-range | Initial continuous steps, four scenes | Fully compliant crossings |
|---|---|---|---|
| ±0 | ±0–0 rad | 18, 10, 14, 32 | 0/4 |
| ±0.01 | ±0.008–0.01 rad | 27, 11, 12, 12 | 0/4 |
| ±0.03 | ±0.024–0.03 rad | 47, 28, 12, 12 | 0/4 |
| ±0.1 | ±0.08–0.1 rad | 16, 28, 11, 11 | 0/4 |
| ±0.3 | ±0.24–0.3 rad | 7, 6, 8, 9 | 0/4 |

**No passage-level residual tolerance is demonstrated: even zero fails.** At ±0.01 (±0.008–0.010 rad), hold is 11–27 steps; at ±0.3 it is 6–9 steps. This experiment cannot estimate a positive noise margin around a successful default because there is no successful default.

## Decision and limitations

Do not commit 1–2 weeks to this arm-only residual integration on this evidence. Early violations occur while the current-pose IK solution is still within tolerance, indicating dynamic tracking limitations. Later the local bounded IK itself exceeds tolerance as the locomotion policy changes body posture. This supports investigating coordinated pelvis/waist/leg posture and arm tracking, not assuming an arm reference alone guarantees the constraint. A different whole-body controller could still work; that is unverified. Four seeded scenes and one noise realization per amplitude do not prove that every residual-control design is impossible.

Only the isolated prototype script and tiny measurement/report artifacts were added. No CLI flags, training integration, production contract changes or GPU calls were added.

Reproduce from repository root with `.venv-mjlab/bin/python scripts/prototype_residual_reference_cpu.py`, using the retained frozen `outputs/residual_reference_cpu/policy_snapshot.pt`. This reruns evaluation only and overwrites this prototype's measurement files; it never refreshes or modifies the live checkpoint. Reference implementation: `scripts/prototype_residual_reference_cpu.py:25`; arm residual application: line 112; production actuator slew/PD: `cat_mjlab/task_math.py:52`; production gravity feedforward: `cat_mjlab/task.py:645`.

## Solver-budget validation

Replayed the zero-residual batch with diagnostic-only 200-evaluation refinements at the first pose in each scene where the 12-evaluation IK exceeded 5 cm. The diagnostic never changed the applied reference. Episode summary replay was True.

| Scene | Short IK error, cm | Refined error, cm | Jacobian maximum absolute error |
|---|---:|---:|---:|
| 2369 | 5.1797 | 5.1771 | 5.57e-10 |
| 2357 | 5.1446 | 5.1401 | 2.55e-10 |
| 2363 | 5.4704 | 5.4696 | 3.01e-10 |
| 2351 | 5.0016 | 4.9996 | 2.50e-10 |

Longer local solves change errors by less than 0.05 mm: three remain above 5 cm, while one borderline pose moves from 5.0016 to 4.9996 cm. Analytic Jacobians agree with central finite differences. These checks argue against the short solve budget or a Jacobian implementation error as the main explanation; they do not establish global IK infeasibility. Raw validation: `outputs/residual_reference_cpu_validation/measurements.json`.
