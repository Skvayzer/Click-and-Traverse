# Approved clearance overrides — prepared, not launched

The prepared run sets hand clearance **−20**, arm clearance **−8**, target **0.09 m**, anticipation **0.20 m**, near share **0.8**. These are explicit run overrides, leaving historical defaults and other launches untouched. Source: configs/pilots/clearance_weight_warmstart.sh:15–17. Only the missing arm override plumbing was added to production Python.

## Deliberate decisions

- Keep anticipation at 0.20 m. Increasing it taxes the correctly tucked pose unnecessarily. At 0.6 m/s and 20 ms control intervals, 0.20 m gives about 16.7 steps to contact (9.2 steps to the 0.09 m threshold). The threshold leaves about 5 cm below the stated approximately 14 cm maximum achievable clearance.
- Keep near share at 0.8. At 14.02 cm, lowering it to 0.5 increases cost 2.5 times, from 0.357604 to 0.894010 reward-rate units, without increasing the saturated contact penalty.
- Use the approved −8 arm weight, but do not call it calibrated or a whole-arm safety guarantee. No defensible better coefficient follows from the existing capsule measurements and point-probe reward. Geometry/formula changes are outside this task.

## Exact hand penalty

Production formula: cat_mjlab/task_math.py:128. For each hand, pressure is `0.8*clip((.09-d)/.09,0,1)^2 + 0.2*clip((.20-d)/.20,0,1)^2`; average the two hands, multiply by −20, then by dt=0.02. Hand cost is applied after the non-hand reward floor (cat_mjlab/task.py:536).

| Both hands at surface distance | Cost rate | Cost per control step |
|---|---:|---:|
| 14.02 cm | 0.357604 | 0.00715208 |
| 9 cm | 1.210000 | 0.02420000 |
| 5 cm | 5.410493827 | 0.108209877 |
| 0 cm | 20.000000 | 0.40000000 |

These are positive magnitudes; reward contributions are negative. With only one threatened hand and the other at least 20 cm clear, halve the entries. The supplied 0.358 “per step” is actually a rate. The good-pose cost is small in absolute terms but **35.7604% of the maximum +0.02 tracking reward per step**, so “near-free” should not mean negligible relative to tracking. Contact is severe: 20 times that tracking maximum. The 9 cm threshold starts the near component; anticipation already costs reward there. Better clearances remain strictly preferred over this table.

## Arm-weight limitation

The existing audit records ten hand-compliant endpoints below 78.99 mm elbow/forearm clearance, minimum 66.41 mm, and 64.01 mm across all arm capsules (docs/LATERAL_CORRIDOR_AUDIT_CPU_20260922.md:102). The old hands-only criterion did not guarantee 78.99 mm for the whole arm.

The actual reward uses elbow point SDF minus a 5 cm radius (cat_mjlab/task.py:268), and `−8*mean(max(0,.08-d_elbow)^2)` (cat_mjlab/task.py:497). CPU formula evaluations with BOTH elbow probes at each distance give:

| Elbow probe clearance | Cost per step before flooring |
|---|---:|
| 78.99 mm | 0.000000163216 |
| 66.41 mm | 0.000029550096 |
| 50 mm | 0.000144 |
| 0 mm | 0.001024 |

One affected probe gives half these values. These evaluations are not new measurements of the ten poses: capsule gaps and elbow point gaps are different observables. The arm cost can also be hidden by the non-hand reward floor. −8 quadruples the old arm penalty but remains weak; proportional numerical scaling is not calibration. No alternative weight is claimed measured or validated.

## Plumbing and verification

CLI: train_cat_mjlab.py:35–41. Runner forwards overrides at cat_mjlab/runner.py:465–470. Arm validation and inherited override: cat_mjlab/config.py:80–85; hand settings: cat_mjlab/clearance_objective.py:5–26. create_task returns the resolved environment configuration; run receives it at cat_mjlab/runner.py:632 and records it in contract.environment_config at :651. Native config reconstruction preserves both override sets.

Command run for CPU verification only:

```
PYTHONDONTWRITEBYTECODE=1 .venv-mjlab/bin/python -m pytest -q -s -p no:cacheprovider tests/test_approved_clearance_weights.py tests/test_clearance_primary.py
```

Result: **31 passed in 8.96 s**. Checks cover parsed launch arguments through create_task with mocked simulation/bank construction, JSON contract serialization, config reconstruction, unchanged defaults and unrelated config values, invalid arm signs/nonfinite values, exact penalty magnitudes and ordering, and identical unrelated reward ledger entries for flat/clutter/CAT/narrow CPU fixtures. Existing clearance suite also covers unchanged real-bank default transitions. New-weight ledger tests use deterministic synthetic CPU fixtures; these are not training or retention forecasts. Final rewards retain the existing floor behavior, so arm deltas can be clipped. Bash syntax validation passed. No optimizer updates, training launch, checkpoint changes, scene-bank/generator edits or analytic moving-object edits were performed.

## Prepared launch

Run only when the owner decides:

```
bash configs/pilots/clearance_weight_warmstart.sh
```

The script warm-starts outputs/cat_flat_balance_ppo_37632_20260920/resume.pt with a fresh optimizer, native 222/310 observations, W&B online / CAT-wholebody / skvayzer. Checkpoint file existence was verified without loading or modifying it. Owner-provided checkpoint provenance: 290M steps; clutter 0.7137 / narrow 0.5582 / CAT 0.1495. Those success rates were not remeasured. All other existing launch options are preserved.
