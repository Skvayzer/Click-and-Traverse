# Hand approach timing pilot — prepared, not launched

Live job, GPU, checkpoints and banks were not modified or loaded for this investigation. Run configuration and bank JSON/fields were read. New probes used CPU only.

## Findings

The original `/tmp/cat_signal_eval.py:80` emits min/p10/p50/p90/max, not p5/p25/p50/p75/p95. Its effective gradient is mean(hand-position cost gradient norm * positive-preclip mask * 3 * .02). It contains neither the tanh derivative nor a slew derivative, joint Jacobian, dynamics or PPO gradient. The 0.069278 value cannot establish slew-caused tenfold attenuation. Median tanh derivative .945135 rules out widespread tanh saturation in this sample. Median requested slew fraction .785714 establishes frequent target limiting, not its causal role in learning failure. PPO estimates a score-function gradient, not backpropagation through the target limiter.

The sampled policy is closer to tucked: median left/right box distances .207824/.372370 m, versus raised .767914/.501775 m. Even the minimum tucked distances .073903/.074456 m exceed the .05 m compliance tolerance. Both target alternatives are valid in all 36 protected zones. Raising is not mandatory. Heading compliance .015689 also matters because boxes are route-relative, not body-relative. Joint poses and motor targets were not saved; actual observed-pose minimum transition times cannot be reconstructed from marginal hand-distance quantiles.

## CPU kinematic probe

Reproduction: `/tmp/cat_timing_probe.py`; results: `/tmp/cat_timing_probe.json`. Uses the production model, default root/legs, aligned heading, actual .95 soft limits, nominal +/- .8 action bounds and .04 rad/step target limits. SLSQP minimizes paired raised-box distance within the reachable joint-target box for each integer step. These are earliest numerically found solutions, not certified global minima or physical settling-time measurements. MuJoCo FK only, no training or checkpoint loading.

| Start | Enter raised boxes | Within .05 m of both boxes | Exact raised preset |
|---|---:|---:|---:|
| Nominal | 17 steps / .34 s | 14 / .28 s | 20 / .40 s |
| Soft-limit-clipped tucked preset | 24 / .48 s | 18 / .36 s | 39 / .78 s |

The default preset elbow target clips at its soft joint limit; hence tucked-to-preset is 39 rather than the unconstrained 40 steps. These starts are controlled references, not the unrecorded live arm configurations.

## Timing and credit

All 12 protected scenes / 36 zones have .15 m incoming fade, 1.65 m zone length, and .625 m from zone start to hazard start. First zone starts at .35 m, reaches core at .50 m, hazard at .975 m. Command is .6 m/s on a visible, unblocked route, except goal taper; blocked command is zero. This gives 12.5 steps of incoming reward fade, 52.083 steps before the hazard, and 112.5 full-core steps per zone at commanded speed. Observed traversal speed is not recorded by the diagnostic. The .20 m protection anticipation knob controls clearance pressure/posture taper, not zone onset.

The reward fade is shorter than the found target transitions, but hazard lead time is longer. This supports an early-shaping experiment; it does not establish an impossible traversal objective, especially because tucked is also valid and the policy can act before reward onset. The diagnostic already records hundreds of active steps in most episodes.

Actual run gamma .98, lambda .95: gamma-lambda .931; e-fold 13.9868 steps; geometric mass 14.4928 steps. Gamma alone: e-fold 49.4983, mass 50. Delayed TD-residual weight at 19 steps .257066, at 40 steps .057278. These are decay scales, not hard credit cutoffs; unroll 32 bootstraps from the critic.

## Implemented experiment

`--hand-contrast-approach-distance .40` adds incoming continuous hand-region shaping only in role `forward_protected`. Clamp onset to route start and previous zone end. Keep original outgoing fade, original heading reward, original core/hand/heading compliance masks and metric tolerances. Relax the arm-neutral stability penalty during the new approach, as already done in the original active zone. Default zero preserves historical behavior; explicit values are validated, inherited on native initialization and recorded in the environment run contract. Existing strict resume checks remain in force.

New first-zone reward onset 0 m, core .50 m: 41.667 commanded steps. Later onset 2.15/4.35 m, core 2.70/4.90 m: 45.833 steps. Core and outgoing reward values remain the same. This is an earlier-reward hypothesis, not a proven learning cure. It also rewards progress toward tucked, since both alternatives are valid.

Ranking: (1) scoped earlier hand shaping; (2) broader region-distance shaping radius if distance-dependent saturation persists; (3) longer GAE horizon if delayed credit remains limiting; (4) faster slew. Changing the original clearance anticipation knob does not solve zone timing. Shared-policy interference remains possible even for scoped reward changes. A global horizon change affects flat/clutter/CAT advantages; faster slew affects dynamics, accelerations and fall risk directly.

Launch later, after live GPU work finishes:

```bash
bash configs/pilots/hand_approach_20260921.sh
```

This is a fresh weights-only native warm start with fresh optimizer, a new output directory, explicit existing reward weights and a 50-update limit. It does not resume or modify the live run. The source best checkpoint is resolved at launch time.

Falsification gate: compare protected-scene original-core hand compliance and original-core paired hand distance across updates 1–10 and 41–50, using the same scene mix. Reject the timing fix as sufficient if compliance gains < .05 absolute AND paired distance falls < 20%, with adequate coverage of all 12 protected scenes. Do not judge success by the reshaped mean reward or newly extended approach interval. Use core-restricted CPU evaluation if per-update core distances are unavailable. Earlier reward activation is testable immediately but is not evidence of learning. Abort this pilot's adoption if flat falls below .27, clutter below .63, CAT goal below .12, or falls appear; those are proposed retention gates relative to supplied .30/.68/.14/zero baselines, not statistical confidence bounds. A matched unchanged-reward fresh-optimizer control would strengthen causal attribution.

Validation: 23 targeted CPU tests passed, including legacy reward parity, unchanged scoring/heading, unaffected roles, target nonoverlap, config inheritance/validation, CPU full-graph reward capture and existing runner/pilot tests. Broader legacy task suite: 14 failures and 2 setup errors (missing jaxlie/ml_collections and stale width_levels fixtures); not a clean full-suite result. Bash syntax and diff whitespace checks passed. No GPU compiler execution or training performance validation.
