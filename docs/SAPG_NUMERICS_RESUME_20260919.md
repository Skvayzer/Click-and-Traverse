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
