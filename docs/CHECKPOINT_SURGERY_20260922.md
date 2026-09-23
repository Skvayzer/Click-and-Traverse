> **Historical record — superseded.** Native training now uses actor 222 / critic 310, no authored box features or rewards, and starts from scratch. Old expansion utilities are retired; no checkpoint trim is supported or requested. Commands and old test paths below describe the historical experiment, not the current workflow. See [current removal report](NATIVE_CLEARANCE_REMOVAL_20260922.md).

# Exact observation checkpoint transfer

No training is performed by either tool:

```bash
CUDA_VISIBLE_DEVICES='' .venv-mjlab/bin/python scripts/upgrade_mjlab_checkpoint.py SOURCE.pt DESTINATION.pt
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv-mjlab/bin/python scripts/verify_checkpoint_upgrade_cpu.py
```

The second command compares frozen donors, selects one from the measured CPU
results, creates the upgrade, checks real-observation parity and rolls out the
upgraded policy. It refuses to overwrite an existing upgraded checkpoint; use a
new `--output-dir` for another verification. The first command converts only;
use the second command for physical and numerical validation.

`cat_mjlab/checkpoint_upgrade.py` checks the named legacy feature order against
the current append-only contract. It copies every old parameter without any
rescaling, approximation, distillation or optimization. Actor and critic first
layers receive 32 exactly zero columns immediately after their old observation
columns, before SAPG embedding columns when present. Every bias, downstream
weight, actor mean/scale head and value head is unchanged. This preserves the
function for all finite new feature values, not only zeros. Different matrix
widths can change float32 accumulation rounding; numerical parity therefore uses
floating-point tolerance rather than demanding bitwise forward outputs.

All Adam state is deliberately discarded. Fresh Adam lazily allocates zero
first/second moments for every parameter, including the new input columns.
`--fresh-optimizer` is advisable for the changed reward/bank and required by the
existing native warm-start path. Neither simulator state, sampler, RNG, training
counters nor W&B lineage is resumed. The source step is retained as checkpoint
provenance; the new learner starts its own counters.

The output stores the old complete run contract and source checkpoint SHA-256
under `upgrade`. Its active learner and environment observation dimensions and
named feature contract are updated. Source-code identity and immediate input
checkpoint hash are updated, and contract/observation hashes are added and
checked on native load. Original bank/collision/reset hashes remain truthful to
the inherited donor environment. The v6 warm-start launch constructs a new run
contract with the actual v6 hashes and resolved overrides. This is an explicit
weights-only transfer artifact, not a fabricated exact runtime resume.

Prepared launch, **not executed**:

```bash
bash configs/pilots/hand_reward_floor_v6_transfer_50.sh
```

That script starts a new 50-update PPO run from the upgraded checkpoint, uses the
v6 bank, soft reward floor 0.2, region -3, heading -5, approach 0.4 m, raised
seeding 0.5, selection weights 0.60/0.20/0.10/0.10, uncapped action standard
deviation, compiled task, and online W&B `CAT-wholebody` / `skvayzer`.
Bounded commandable IK is the existing production startup implementation, not a
new bank or observation change. CPU certification results are in
`outputs/checkpoint_upgrade_v6_20260922/bounded_ik_certificate.json`.

The artifact preserves the donor's 0.15 sigma cap to keep conversion identity.
The requested launcher explicitly removes that cap with `--max-action-std 0`.
That changes stochastic exploration at pilot startup; it does not change the
learned tensors or deterministic action means. Deterministic smoke results do
not establish the fall rate of that uncapped stochastic training distribution.

Measured results are recorded in `donor_comparison.json`, `verification.json`
and `contract_checks.json` under `outputs/checkpoint_upgrade_v6_20260922`.
The comparison uses identical reset seeds, ten starts across flat walking,
original CAT, procedural CAT, clutter and narrow passages, up to 400 control
steps (8 seconds). It ranks survival/goals, falls, goal count, steps and travelled
distance; historical best-selection scores are ignored. Travel is root XY path
length, not net forward progress. This small benchmark measures retention and
motion; it does not establish statistical superiority, protected compliance or
full-episode success. The supplied 98.9% fresh-policy training fall rate is a
different population and horizon, not a matched baseline.

## Measured result

Selected `outputs/cat_hand_feasible_v5_seeded_pilot50_20260921/best.pt`.

| Donor | Falls / starts | CAT goals / 4 | Mean root XY path length |
|---|---:|---:|---:|
| cat_hand_feasible_v5_seeded_pilot50_20260921/best.pt | 0/10 | 4/4 | 4.8951 m |
| cat_hand_priority_30720_20260922/best.pt | 0/10 | 4/4 | 4.6512 m |
| cat_flat_balance_ppo_37632_20260920/best.pt | 0/10 | 3/4 | 4.8940 m |
| cat_flat_balance_ppo_37632_20260920/resume.pt | 0/10 | 3/4 | 4.7702 m |

Actor-logit and critic-value max absolute differences were **0.0** on **1600 real observations**, both with zero and real added features. All original tensors were preserved exactly and every added weight was zero.

Upgraded rollout: 0/10 falls, 10/10 survived 400 steps, 4/4 CAT goals. Flat paths were 5.7110 and 5.8619 m over 8 seconds. Minimum flat root heights were 0.7395 and 0.7234 m.

Artifact: `outputs/checkpoint_upgrade_v6_20260922/upgraded.pt`, 5,475,111 bytes, SHA-256 `e28b939f8cc04d1e86ff04a8110ec8a2890c35fc169186f17cd4866f23a8836a`. Native loading, named feature and size checks, the existing recorder's bank/source/version contract check, launch argument parsing, and strict model load with launch overrides passed. **32 focused tests passed.** CPU evaluation took 416.7 seconds. No training, bank modifications, observation-contract modifications, or deletions.
