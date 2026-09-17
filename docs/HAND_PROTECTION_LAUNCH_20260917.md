# Hand-protection launch evidence — 17 September 2026

The new learner on `konstantinsmirnov@ws008090` uses frozen source commit
`8f0e09a864634fa2de014026964abe17685db029`, 16,384 MJX environments, the
2,362-scene bank, and the protected best at 104,857,600 transitions. Actor and
critic parameters are copied exactly; the optimizer is fresh. The policy/critic
observation sizes remain 222/310 and the action size remains 29.

[Single W&B run](https://wandb.ai/skvayzer/CAT-wholebody/runs/46a8a2af)
· [Setup, equations and scene preview](HAND_PROTECTION_CURRICULUM.md)

At **06:23 UTC on 17 September**, the detached learner had completed 12 PPO
updates (**6,291,456 transitions**), with finite losses and all logged nonfinite
flags zero. The W&B API confirmed the same global step and a running experiment.
The full recovery snapshot was 2,716,701,530 bytes. GPU use at that sample was
38,009 / 49,140 MiB (37.1 / 48.0 GiB), with 100% utilization. There is no fixed
training-step limit and no pending stop request. This verifies live learning and
checkpointing, not that protective hand movements have already improved.

The machine-readable audit is
`outputs/hand_protection_implementation_20260917/launch_evidence.json` on both
ws008090 and the Mac. It also independently checks the old scene-record prefix,
all old reset rows, protected archive checksum, and corrected W&B configuration.

## Startup retention measurement

These are **before-learning** first-clean-goal results on fixed training-bank
layouts, with 16 paired seeds per layout. They are regression checks, not a
held-out generalization estimate or evidence of newly learned arm motions.

| Population | Episodes per mode | Saved best deterministic | New deterministic | Saved best stochastic | New stochastic |
|---|---:|---:|---:|---:|---:|
| CAT regression layouts | 192 | 37.50% | 34.90% | 32.81% | 32.29% |
| Original four clutter layouts | 64 | 78.13% | 84.38% | 70.31% | 70.31% |
| New hand passages, all levels | 96 | — | 52.08% | — | 34.38% |
| New easy hand passages | 32 | — | 81.25% | — | 62.50% |
| New medium hand passages | 32 | — | 65.63% | — | 37.50% |
| New hard hand passages | 32 | — | 9.38% | — | 3.13% |

Both action modes pass the unchanged absolute source-best gates: at most a
5-percentage-point CAT drop and a 12.5-point ordinary-clutter drop. Deterministic
CAT success is 2.60 points lower; this is not an assertion of identical behavior.
Per-scene CAT gates during training also compare against the new baseline.

An initial implementation kept white-noise innovation magnitude while adding
0.95 temporal persistence. That inflated stationary variance by 10.26 times and
reduced stochastic ordinary-clutter success to 46.88%. The startup gate rejected
it at zero updates. Scaling innovations by `sqrt(1-0.95^2)` fixed that error while
retaining temporal/spatial coordination. The failed startup metadata was archived
under `outputs/startup_attempts/cat_hand_protection_20260917_unscaled`; its unused
W&B ID was reused only after checking no runtime, training steps or logged metric
events existed. The W&B API confirms the corrected scale `0.31224989991991997`.

## Data and checkpoint verification

All 2,338 old scene records/fields remain unchanged. The 24 appended scenes
underwent static 35-shape body checks, approach/exit arm-transition checks, and
the actual conservative 4 cm CAT hand-field queries. Sideways alternatives exist;
the scene certificates disclose them.

All **74,816 old reset poses** were preserved in their original order. The new
pool contains 32 poses for each of 2,362 layouts, with zero dropped scenes, and
every pose was checked against the new geometry bank. Model relocation was
verified by comparing all XML attributes and the contents of all 49 referenced
mesh files; only absolute asset paths differ.

Protected backup:

- Server: `archives/cat_best_before_hand_protection_20260917`.
- Mac: `~/Downloads/CAT-Checkpoint-Backups/cat_best_before_hand_protection_20260917.tar.gz`.
- Archive SHA256: `3fe8a9e1a0155560893e71745b78151a6255e08110655b831110e980d7119e40`.

Native loading, actor/critic parity, finite stochastic inference, and V2 native
checkpoint round trips were checked against the actual saved best. Raw actor
and critic parity errors are exactly zero.

## Software checks

The broad regression run passed 603 tests and found two stale test doubles that
lacked the newly explicit validation-scene argument. After a test-only fix, the
affected file passed all four tests, including both failures. That broad run
spanned the implementation and variance-fix commits; the final increment was
separately checked with 118 focused tests covering variance, checkpoint loading,
logging reuse, telemetry bounds and launcher behavior. No runtime change was
needed for the stale test doubles.

Runtime files live under `outputs/cat_hand_protection_20260917` on ws008090.
The process log is
`outputs/hand_protection_implementation_20260917/training_variance_normalized.log`.
Training is continuous until a manual stop request, with one selected best and
one overwritten full learner-resume state. A stop command is:

```bash
touch /home/konstantinsmirnov/robotics/Click-and-Traverse-WholeBody/outputs/cat_hand_protection_20260917/STOP
```
