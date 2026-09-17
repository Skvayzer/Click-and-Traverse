# Recovering hand-protection fine-tuning

The hand-protection learner improved initially, then lost room traversal. On the
unchanged validation layouts, deterministic ordinary-clutter success declined
from 79.7% at the selected 26,214,400-transition checkpoint to 29.7% at 445,644,800;
hand-passage success declined from 61.5% to 0%. Original-scene success stayed near
40%. Leg sampling standard deviation increased from approximately 0.24 to 0.52.
This is evidence of selective regression, not proof that exploration is its sole
cause. The easy hand curriculum never unlocked the harder levels.

## Recovery profile

`--finetuning hand_recovery` requires a verified protected best archive. The
17 September source is
`archives/cat_hand_best26M_before_recovery_20260917`, copied and hash-checked before
stopping the old run. Actor, critic and observation statistics are restored;
Adam starts fresh. The old learner stopped cleanly at 500,695,040 transitions.

- Learning rate: **1e-5**, previously 3e-5. PPO clipping remains 0.1.
- Entropy coefficient: **0**, previously 0.003. Sampling remains stochastic.
- For each observation, the 12 leg Gaussian scales come from the frozen source
  actor. The current policy still learns leg means, the waist and arms. Freezing
  only output weights would not preserve scales because shared features change.
- The fixed-noise actor is embedded in the native checkpoint configuration with
  architecture, float32 leaves and a SHA256 digest. Sampling, PPO likelihoods and
  entropy use the same scales; no external noise is added. Native policy reload
  is self-contained. Exact learner resume also verifies the protected archive.
- Reference-policy KL remains **0.05**, leg actions only, and now covers **all
  scene families**, including ordinary clutter and hand passages. Its teacher is
  the protected whole-body policy; the arms are not constrained to teacher poses.

The average logged leg standard deviation may change as the visited observations
change. The invariant is equality with the frozen source at the **same** input,
not a universal constant such as 0.24.

## Regression response

The unchanged 22 layouts and 16 seeds per layout are evaluated in deterministic
and stochastic modes every **10 updates (5,242,880 transitions)**. Original-scene
success may drop at most 5 percentage points; ordinary-clutter and hand-passage
success may drop at most 10 points. Both the immutable source floor and the
selected safe best are protected. Existing per-original-scene checks also apply.
These thresholds are operational safeguards, not statistical significance tests.

Two consecutive failed evaluations trigger actual learner recovery. The actor,
critic and normalizer return to the selected safe best (the source if no better
model has been selected), Adam is cleared, and the learning rate is halved with a
1e-6 floor. In-flight episodes, navigation history, adaptive-sampling statistics
and episode metric windows reset. The hand curriculum restarts at easy, with new
counters measuring the restored policy. Training continues in the same process
and W&B run; total sampled transitions remain monotonic and include discarded
learning. Guard state and the reduced Adam rate survive exact runtime resume.

Numerical failure detected by validation requests immediate recovery. Other
nonfinite optimizer/logging errors retain the existing fail-fast behavior; they
are not silently retried. Benchmark or checkpoint integrity errors also fail.

`recovery_status.json` and `recovery_events.jsonl` record decisions locally;
`resume.msgpack` is the authoritative durable learner/guard state. One selected
best and one overwritten full resume state are retained, alongside the protected
source archive. The failure guard does not guarantee rising success or learned
hand raising: fixed-scene performance and videos must establish that.

## Preserved experiment and logging

The bank still contains 2,362 layouts, including all 24 appended hand passages.
Geometry, route following, body-collision checks, rewards, reset probabilities,
222 actor observations, 310 critic observations and 29 actions are unchanged.
The launch uses 16,384 MJX environments and 524,288 transitions per PPO update.
Coherent arm exploration and its normalized variance are unchanged.

One new recovery experiment separates this changed setup from the stopped run;
all scene families and any subsequent automatic recoveries share it. Existing
metric keys are preserved. Use the [compact saved W&B view](WANDB_SUCCESS_VIEW.md)
to keep success rates in their own expanded section and diagnostics collapsed.

```bash
.venv/bin/python train_cat_wholebody.py run --finetuning hand_recovery \
  --num-envs 16384 --batch-size 256 \
  --bank-manifest /absolute/path/to/cat_hand_protection_v1_20260917/manifest.json \
  --body-collision-bank /absolute/path/to/body_collision_hand_v1_20260917/manifest.json \
  --body-collision-resets /absolute/path/to/body_collision_hand_resets_v1_20260917/manifest.json \
  --warmstart-best /absolute/path/to/archives/cat_hand_best26M_before_recovery_20260917 \
  --run-dir /absolute/path/to/outputs/cat_hand_recovery_20260917 --wandb-mode online
```
