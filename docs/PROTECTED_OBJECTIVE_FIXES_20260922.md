# Protected-objective review fixes — 2026-09-22

No training was launched. All execution was CPU-only. Existing banks and real checkpoints were read-only; no checkpoint was loaded by the audit. This work does not modify observation construction, the observation contract, `cat_ppo/furniture/control.py`, `cat_mjlab/task_math.py`, or `cat_mjlab/config.py`. Those files already contain independent/concurrent changes; a dirty working tree must not be attributed wholly to this fix.

## Findings and implementation

1. **Confirmed: reset commandability.** `cat_mjlab/raised_reset.py` now intersects soft physical joint bounds with nominal ± `upper_action_scale` for every absolute upper-body target, propagates the configured scale through IK and certification, and records commandability in the certificate. Reset action history rejects out-of-range seeded actions and clamps only floating-point boundary roundoff. Leg actions are incremental (`previous + action * action_scale`); their reachable target set is the soft physical range, not nominal ± one step. Existing heuristic posture references already clip actions to [-1,1], and the flat audit arm IK already intersects nominal ±0.8. No existing bank was regenerated.

   **v5 is reachable with commandable targets.** All 12 scenes / 48 pool poses pass the CPU static certificate. The common route-frame palm positions are left **(0.426678, 0.124987, 0.614799) m**, right **(0.426678, −0.124977, 0.614799) m**. Both distances to the allowed region are **0 m**; maximum normalized upper action is 1.0. This solution reaches the action boundary, so it has no command margin on saturated joints. There is no static-geometric justification to move v5 again. A failed bounded solve now reports its closest numerical candidate and box distances, explicitly without claiming a global infeasibility proof.

2. **Not a direct competing-reward defect: flat gating already exists.** The flat target and protected target heights differ, but `posture_terms` returns exactly zero unless the bank row is flat. Protected records are not flat rows. A regression puts a protected row exactly at the flat target with forward velocity and bonus scale 10: its bonus remains zero; the flat row receives 10. Preserve flat walking compliance and its target. Use the requested flat bonus 1.0 in the recommended configuration. Learning different scene-conditioned poses still depends on the observation work owned by the other agent; this change does not claim to solve that representation problem.

3. **Confirmed: selection ignored protected compliance.** New default weights: **60% strict protected-core hand compliance, 20% flat walking compliance, 10% flat soft progress, 10% CAT goal retention**. Protected posture receives more weight than all other components combined; the remaining 40% retains pressure for useful flat locomotion and CAT retention. Configure with `--checkpoint-selection-weights PROTECTED FLAT_WALK FLAT_PROGRESS CAT`; weights must be finite, nonnegative and sum to one. The resolved weights are stored in the run contract. The score waits for evidence in every positive-weight component. Protected compliance is leader-only, strict 5 cm, counted over protected-core physical steps, including seeded and terminal steps; it is not a passage-completion score and can favor standing. Passage acceptance remains separately observable. The historical 50/25/25 score is logged as `training/legacy_flat_checkpoint_selection_score`.

4. **Confirmed: the old competing-cost audit was inadequate; its negative conclusion does not hold here.** The replacement uses the exact saved current configuration (`outputs/cat_hand_priority_30720_20260922/run.json`, SHA256 `f49b5c90a403e57f324c3680cba2f10b8a61deefab9172c333e6afa6398ac331`): v5, region −20, heading −5, flat bonus 1.0, foot balance −30. It calls production `CATTask._rewards`, samples the actual stored fields, and uses the production collision checker including all ten physics substeps. The static compliant pose has a positive reward advantage. Actual transient target-velocity and acceleration penalties are included in the slew experiment below. No cost weights were silently changed.

## Reward ledger

Every entry is **weighted reward per 0.02-second control step**, averaged equally across all 12 protected scenes at the first-zone midpoint. Positive deltas favor raised arms. The nominal comparison uses the **same crouched root and legs**, changing only upper joints, so it isolates arm posture rather than mixing crouching and raising. The initial nominal arms penetrate the furniture and trigger collision; this is a counterfactual, not a valid certified reset. The compliant initial poses all pass strict hand compliance.

Static columns use actual zero-velocity FK at time zero (zero actuator command). Dynamic columns use 20 actual MuJoCo PD steps (0.4 s), no synthetic translation, actual forces/contacts/sensors, finite-difference field velocities, and the midpoint 1.4 Hz native gait. Root roll/pitch are preserved in reward FK. Motor targets remain fixed for the hold; the slew case starts at nominal arms and approaches the compliant target at 2 rad/s. Upper history is synchronized for holds and explicitly advanced for the slew. Waist targets stay nominal. Domain-randomized pushes and motor noise are excluded from this paired deterministic audit. Guidance still supplies the actual route command; this is not a learned walking controller.

| Reward term | Static nominal | Static compliant | Static Δ | PD hold Δ | Slew raise Δ |
|---|---:|---:|---:|---:|---:|
| `tracking_orientation` | 0.039999999 | 0.039999999 | +0.000000000 | +0.007095248 | +0.001253297 |
| `tracking_root_field` | 0.004738555 | 0.004738555 | +0.000000000 | +0.000975984 | -0.000698160 |
| `body_motion` | 0.000000000 | 0.000000000 | +0.000000000 | +0.001929071 | -0.000032649 |
| `body_rotation` | 0.020000000 | 0.020000000 | +0.000000000 | -0.000002852 | -0.000266564 |
| `foot_contact` | -0.029999999 | -0.029999999 | +0.000000000 | +0.000000000 | +0.000000000 |
| `foot_clearance` | -0.000710703 | -0.000710703 | +0.000000000 | -0.000009308 | +0.000001854 |
| `foot_slip` | 0.000000000 | 0.000000000 | +0.000000000 | +0.000008740 | +0.000007112 |
| `foot_balance` | -0.022604715 | -0.002538425 | +0.020066290 | +0.038795160 | +0.013554414 |
| `straight_knee` | 0.000000000 | 0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| `foot_far` | 0.000000000 | 0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| `joint_limits` | 0.000000000 | 0.000000000 | +0.000000000 | +0.000117971 | +0.000363379 |
| `joint_torque` | 0.000000000 | 0.000000000 | +0.000000000 | -0.000261236 | -0.001074736 |
| `smoothness_joint` | 0.000000000 | 0.000000000 | +0.000000000 | +0.000005325 | -0.000087565 |
| `smoothness_action` | 0.000000000 | -0.000063540 | -0.000063540 | -0.000063540 | -0.000066717 |
| `headgf` | 0.000000000 | 0.000000000 | +0.000000000 | +0.048938050 | -0.000317175 |
| `feetgf` | 0.079999998 | 0.079999998 | +0.000000000 | -0.037163919 | -0.030672589 |
| `handsgf` | 0.000000000 | 0.000000000 | +0.000000000 | -0.004372473 | +0.068735254 |
| `headdf` | -0.000000274 | -0.000000274 | +0.000000000 | -0.000000029 | +0.000000000 |
| `feetdf` | -0.000188882 | -0.000188882 | +0.000000000 | +0.000000056 | +0.000000184 |
| `handsdf` | -1.650594622 | -0.011604039 | +1.638990583 | +1.339947718 | +0.422348499 |
| `kneesdf` | -0.000055589 | -0.000055589 | +0.000000000 | +0.000000006 | +0.000001629 |
| `shldsdf` | -0.000027887 | -0.000027887 | +0.000000000 | -0.000000967 | +0.000000146 |
| `wholebody_hand_clearance` | -0.010000000 | -0.000304408 | +0.009695592 | +0.009357631 | +0.003326920 |
| `wholebody_arm_clearance` | 0.000000000 | 0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| `wholebody_upper_target_velocity` | 0.000000000 | 0.000000000 | +0.000000000 | +0.000000000 | -0.000093442 |
| `wholebody_upper_target_acceleration` | 0.000000000 | 0.000000000 | +0.000000000 | +0.000000000 | -0.000040660 |
| `wholebody_upper_clear_posture` | 0.000000000 | 0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| `wholebody_hand_contrast_heading` | 0.000000000 | 0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| `wholebody_hand_contrast_region` | -0.366744161 | 0.000000000 | +0.366744161 | +0.233132690 | +0.108463757 |
| `flat_balance_posture_bonus` | 0.000000000 | 0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| `body_collision_event` | -1.000000000 | 0.000000000 | +1.000000000 | +0.362499997 | +0.124999996 |

| Net comparison | Static compliant − nominal | PD hold − nominal | Slew − nominal |
|---|---:|---:|---:|
| Before floor, excluding collision event | **+2.035433086** | **+1.638429330** | **+0.584706188** |
| Final reward, after floor then collision event | **+1.099244798** | **+0.384849260** | **+0.125196759** |

Do not sum the raw collision row into the pre-floor total and call that the final reward: the actual ordering is `clamp(sum * dt, 0, 10000) + collision_event`. The event is already in reward units and is not multiplied by dt again. Static upper velocity/acceleration/waist costs are zero because the targets are held with consistent history; the slew columns contain the nonzero motion costs.

Raising is **not net-negative** in any of these paired comparisons. No larger region weight is needed to turn these measured deltas positive. With trajectories fixed, `Δ(w)=Δ_without_region + w * (cost_nominal-cost_raised) * dt`; even w=0 leaves the measured advantage positive. This sensitivity is a fixed-trajectory statement, not a prediction of a policy trained at another weight. Much of the benefit comes from removing severe hand SDF penetration. Unlike the historical flat/synthetic case, the new pose also **improves** foot balance: static Δ +0.020066290/step.

## Conditional reward-floor clipping

A step is clipped iff its weighted **pre-collision** reward sum is negative. Denominator: forward-protected role AND `core_active`. Initial FK references are excluded from physical-step counts.

| Experiment | Full core trace clipped / steps | Fraction | Through first terminal step only |
|---|---:|---:|---:|
| nominal | 240/240 | 100.000% | 12/12 (100.000%) |
| commandable | 153/240 | 63.750% | 153/240 (63.750%) |
| slew_raise | 234/240 | 97.500% | 12/12 (100.000%) |

For the nominal/compliant pair, the episode-valid conditional measurement is **165/252 = 65.476190%**. The complete controlled hold counterfactual is **393/480 = 81.875%**. Nominal arms and the inside-zone slew experiment terminate on their first step because of body collision. Continued samples are retained only to expose the reward landscape, and must not be represented as valid training episodes. Compliant holds have no body collision in this horizon, but strict hand compliance lasts only **25%** of their steps. Thus commandability fixes the target-retreat bug, but does not establish dynamically sustained posture or passage success.

These are CPU probe-population rates, **not an estimate of the current policy's rollout distribution**. No trained-policy rollout was run. Production now records `training/protected_core_reward_floor_clipping_fraction` with its `_sample_count` from physical leader transitions, before collision events and autoresets; this will give the actual policy-conditioned number on a later authorized run. The 2.19% global rate cannot rule out the much larger protected-subset rates observed here.

## Recommended launch configuration — prepared, not executed

See `configs/pilots/hand_priority_fixed_50.sh`. Retain v5, region −20, heading −5, flat bonus 1.0 / flat shaping radius 0.40, 0.40 m hand approach, upper slew 2 rad/s, and PPO gamma 0.98. Use the 60/20/10/10 score. Seed 50% of protected resets, leaving half for unseeded approach/crossing evidence; do not judge success from seeded core compliance alone. Keep the pilot bounded to 50 updates in a new output directory, logging the old score, protected clipping, and passage acceptance. Preserve flat targets and the −30 foot-balance scale because the current paired ledger does not implicate them as a net-negative raising incentive.

The script requires `CAT_HAND_CHECKPOINT` to name an **observation-compatible native checkpoint** after the concurrent observation work/migration is complete, and requests fresh optimizer state. It intentionally does not silently expand the old actor or select an incompatible checkpoint. Shell syntax and argument parsing are tested without executing the launch. Given the measured posture drift and clipping, this is a diagnostic pilot configuration, not a claim that the objective is solved.

## Verification and artifacts

- CPU tests cover bounded IK, all 12 scene certificates, controller target hold, reset history rejection, flat/protected bonus isolation, configurable priority/legacy scores, physical leader/core clipping denominators, runner integration, and launch parsing.
- Final consolidated CPU suite: **69 passed, 4 skipped in 48.35 s**. All four skips are v3 integration fixtures because that local bank is unavailable; v5 certification and regression tests passed. `git diff --check` and launch `bash -n` passed.
- No banks, trained checkpoints, observation feature definitions or observation contracts were edited by this work. Unit-test checkpoints are temporary synthetic fixtures only. Existing/concurrent edits were preserved.
- Reproduce audit: `CUDA_VISIBLE_DEVICES='' .venv-mjlab/bin/python scripts/audit_protected_objective_cpu.py`.
- Full per-scene machine-readable evidence: `outputs/protected_objective_audit_cpu/audit.json` (small JSON only; no videos, trajectories or model copies).
