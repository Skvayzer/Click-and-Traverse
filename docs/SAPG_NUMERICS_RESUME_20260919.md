# SAPG numerical repair and full-state continuation — 19 September 2026

The hand-protection specialist on `tl-server-0` stopped in Slurm job 680 with
nonfinite policy, value, entropy and total losses. The last complete learner
snapshot contains **326,762,496 physical transitions**; the failing update was
reported at **327,942,144**. This was a numerical failure, not a GPU-memory or
scheduler timeout. The separate PPO run on `dep-1` is outside this repair.

## Loss calculation

The previous off-policy actor calculation separately exponentiated the frozen
importance ratio and the current/old target-policy ratio:

```
log_mu = old_target_log_probability - behavior_log_probability
log_q  = new_target_log_probability - old_target_log_probability
mu * min(exp(log_q) * A, clip(exp(log_q), 1-epsilon, 1+epsilon) * A)
```

Both probabilities are joint likelihoods over 29 action dimensions. Separate
exponentials can become `0` and `inf`, making their product or gradient `NaN`
even when the combined expression is representable. The repaired expression
keeps importance weights in log space, computes the unclipped branch directly
as `new_target_log_probability - behavior_log_probability`, and selects the
appropriate clipped branch before exponentiation. It includes `log(abs(A))`
and mean normalization before that final exponential, and masks zero advantages
before exponentiation. Boundary-gradient tests preserve the original rounded
PPO clipping bounds.

This is a numerical reformulation of the existing SAPG objective. It introduces
no importance cap, action-scale cap, replacement of nonfinite gradients, skipped
updates, recovery rollback or optimization hyperparameter change. Truly
unrepresentable objectives remain errors. Mathematical regression tests alone
do not establish that this was the observed run's failure; saved-state replay
provides the separate integration check.

## Reproduced failure

Diagnostic GPU replay **Slurm 685** restored the complete archived learner and
reproduced the failure in its next native-size update. At Adam update index
**71,100**, `log_q` reached **91.657**, above float32's exponential overflow
threshold of approximately 88.7. One sample had `mu == 0` and `q == inf`.
The directly combined log ratio remained finite (batch range approximately
**-204.406 to 3.368**).

The clipped forward loss was still finite: total **0.0685319**, actor
**-0.000284216**, value **0.0494076**. However, differentiation produced
**298,634 nonfinite gradient entries**, and the resulting optimizer update
contaminated all **1,322,715 parameter entries**. This confirms that the overflow
can hide behind a finite reported loss; clipping the forward ratio was not
sufficient. Later minibatches then reported nonfinite losses as in the failed
production run. The isolated diagnostic run additionally hit Brax's final
replication assertion because its parameters contained NaNs.

Diagnostics add callbacks/reductions and can change compiler fusion; this replay
reproduces the numerical mechanism from the saved state rather than claiming
bitwise equivalence with the uninstrumented production executable.
Detailed evidence remains on the server in
`outputs/sapg_replay_20260919/original.json`.

Corrected diagnostic replay **Slurm 686** completed **two full updates**, through
**329,121,792 transitions**, with finite losses, gradients, optimizer and
parameters. The starting snapshot remained byte-identical. In its first update,
the old comparison expression overflowed in three minibatches; actual corrected
gradients remained finite in all 256 minibatches. At Adam index 71,100, the
corrected trajectory reached `log_q = 135.541` while retaining zero nonfinite
gradient/parameter entries (gradient norm approximately **0.59946**).
The result is recorded in `outputs/sapg_replay_20260919/fixed.json` with
`numerical_pass: true`. These isolated updates are validation only: production
continues from the original durable 326,762,496-transition learner snapshot.

Local validation: **69 focused tests passed** (42 SAPG loss, 15 replay and
12 migration tests). The Slurm continuation recipe also passed Bash syntax and
exact launch-argument reconstruction checks. Runtime code is pinned to commit
`351e336a079a68c93b39a40c8087c8da1fed56e4`; its numerical implementation is in
commit `4d36846`.

## Preservation and provenance

CPU Slurm job 684 archived the entire stopped run to
`outputs/archives/cat_hand_specialist_sapg_680_before_numerics_20260919`.
Its original 4,712,034,886-byte `resume.msgpack` has SHA-256
`69add45620a7d5dacb748e3b8123c605896bf6f4a8bcf1a6c9405c42299e2276`.
The archive includes the previous best leader checkpoint and run metadata.

`scripts/replay_sapg_update.py` restores this archive for isolated native-size
training updates. It neither logs to W&B nor writes learner checkpoints or runs
evaluation. Optional numerical diagnostics may change compiler fusion and are
reported explicitly. Each replay checks that its input snapshot remains intact.

`scripts/migrate_sapg_numerics_resume.py` permits this one reviewed runtime-source
change in `cat_ppo/learning/policy/sapg/losses.py`. It holds the learner lock and
checks the archived evidence, configuration, source hashes and W&B identity.
It streams the snapshot, replacing only the contract's source fingerprint and
copying every byte outside that metadata object verbatim. Actor, critic,
embeddings, optimizer moments, random state, environments, observation
normalization, curriculum and metric counters are preserved. A publication
journal supports recovery if the two metadata/file replacements are interrupted.
The ordinary strict resume checks are unchanged.

The continuation uses the existing run directory
`outputs/cat_hand_specialist_sapg_680` and
[W&B run f017f302](https://wandb.ai/skvayzer/CAT-wholebody/runs/f017f302).
There are still 36,864 environments, six policy groups, learning rate `3e-4`,
clipping `0.2`, entropy coefficient `0.003`, rollout length 32 and four optimizer
passes. The same 24 hand-protection scenes and curriculum remain in use.
No original/generated CAT scenes or ordinary clutter rooms are added to this
specialist. Its original released-checkpoint initialization remains recorded;
continuation restores the complete trained learner, rather than initializing a
new model.

## Continuous continuation

Production resumed as **Slurm job 689**, using
`scripts/slurm/resume_cat_sapg.sbatch` and the pinned checkout
`outputs/sources/cat_sapg_numerics_351e336a079a`. The learner has no step cap;
the scheduler allocation is seven days. Initialization confirmed the same
online W&B run, `f017f302`.

The metadata migration completed in CPU job 688. Its wrapper subsequently
stopped on an overly strict W&B/local-JSON equality check, before launching
training or modifying W&B. W&B omits an empty configuration dictionary when
serializing metadata; a second exact comparison also found one-ULP float64
round-trip differences in two hand-geometry values (about `1.39e-17` metres).
These are telemetry serialization differences, not changed observations or
geometry. The strict on-disk/runtime identity checks remain unchanged.
A separate provenance sync, job 695, changed only `code`
and `code_migrations` and verified exact before/after equality of every other
remote configuration field. The run history and warm-start provenance were
preserved. The pending production job was then released with its corrected
dependency; no duplicate training run was created.

The migrated runtime SHA-256 is
`87caa78b6c77a0ffdeded80c7a470e102d4644c58ff4aafb1c072aa6d24a536c`.
The copied bytes outside the contract object have SHA-256
`8ce886358344dd80bc16f9578dbe1a3777145cf9650dea84935dc716962dd8ef`.
The complete journal is in the run's
`code-migrations/sapg-numerics-326762496-351e336a079a/migration.json`.

The uninstrumented production job completed and saved its first resumed update
at **327,942,144 transitions**, the step where the original implementation
failed. Both local metadata and the live W&B API confirmed `running`,
`health/nonfinite = 0`, and the corrected source identity. Actual losses were
total **0.0639412**, actor **-0.00469532**, and value **0.0492162**.
An overlapping diagnostic inside the same Slurm allocation measured
**33,344 MiB** GPU memory for learner PID 710467. This verifies the restart and
numerical repair; it does not by itself establish improved learned skill.
