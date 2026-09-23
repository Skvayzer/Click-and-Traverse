# Raised-state seeding and competing-cost audit

CPU only. No training launched, no checkpoint opened, no existing bank written.

## A. Measured reward ledger

The fast prescribed raise costs **0.0020127703** reward units in total and gains
**6.0032082899** region reward units relative to holding nominal over the same
20-motion-step + 100-hold-step interval. The hold alone contributes 5.5672670603.
The motion cost is 0.0015890926 velocity + 0.0004236777 acceleration + 0 waist/posture.
At 1 rad/s, the 39-step raise costs 0.0009094029 and gains 6.3556514136 including
100 hold steps. The fast trajectory includes a final float32 settling tick.
With gamma .98, fast cost/benefit are 0.0017224557 / 1.9371890864; slow values
are 0.0006674192 / 1.5389404670. The acceleration sum includes stopping.

These are pre-clipping reward units using the actual Torch upper-cost and
region-reward kernels, dt .02, the v3 pilot run.json weights, constant nominal
waist, the nominal DEFAULT_QPOS arms, and the 95% raised preset used by the bank.
The root is held fixed in an active zone at full phase weight. FK computes the
hand positions at every filtered target. The nominal region cost is .9278778434;
the endpoint cost is zero and compliance is one. This is a kinematic
counterfactual, **not an observed policy rollout or a dynamic execution test**.
The pilot metrics do not record observed joint trajectories; distance/cost alone
cannot reconstruct them. Thus the precise observed-pose request cannot be
identified from these logs without additional state data. No checkpoint was read.

`upper_acceleration_cost_scale=20` divides acceleration before squaring. It is
not a reward weight. The actual signed weights are -.02 acceleration and -.05
velocity, and the weighted-mean denominator is 26 (three waist weights of four
plus fourteen arm weights). Contrast-active arm gates remove the nominal
posture cost only. They do not remove velocity/acceleration costs. Waist motion
is zero in this controlled raise; no separate additive waist penalty exists.

**Verdict:** the specified regularizers do not make this raise net-negative.
The scale-20-versus-weight-3 argument is false. This does not establish the net
return of actual locomotion, including balance, torque, collisions, reward
clipping, and target-tracking errors. From the reported .8832 region cost,
100 full-weight compliant hold steps would recover 5.2992 reward units alone.

Reproduce the ledger (no learner or checkpoint access):

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 \
  .venv-mjlab/bin/python scripts/audit_raised_motion_cpu.py \
  --run-json outputs/cat_hand_raised_v3_pilot50_20260921/run.json \
  --bank-manifest data/furniture/cat_flat_hand_balance_v3_20260921/manifest.json
```

## B. Implementation and certification

`--hand-raised-reset-fraction` inherits the source config when omitted, falling
back to 0.0. Explicit 0 disables it. Finite values in [0,1] are accepted;
positive values require hand contrast. The fraction and generated certificate
are saved in the environment run contract; the normal strict resume comparison
therefore covers them. Default-zero resets consume no extra random draws.

Only forward_protected scenes are eligible. Four poses per scene start on
segment zero at progress .52, .56, .60, .64 m, aligned to the route with nominal
legs/waist and 95% of control.py's raised preset. Selection is a Bernoulli draw
per eligible episode, using the task's checkpointed RNG. The initial qvel is
zero. Motor targets, two previous upper targets and action history match the
pose, preventing fictitious initial acceleration and nominal target memory.

**No new bank is needed.** CPU startup generates a small in-memory pool from the
existing bank and freshly certifies each exact float32 pose. Checks include:
scene hash, soft joint limits, actual MuJoCo FK, ordered root-route validity,
active core zone, both-hand zero-cost compliance, every native SDF query and
hand/elbow margin, and the actual approved full-body collision bank/checker.
An unexpected runtime collision fails closed; it cannot silently substitute
an ordinary fallback while claiming a raised seed. Existing certified reset
poses and banks remain unchanged. This is static certification, not dynamic
feasibility certification.

Measured after full manifest/file verification: **48/48** certified poses in
**12/12** protected scenes, **0** body collisions, **1.0** compliance, **0.0**
region cost, **0.1076094955 m** minimum native field clearance. The generated
qpos-array SHA256 is recorded in `assets/raised-seeded-20260921/measurements.json`.

Validation: **20 targeted CPU tests passed**, including invalid fractions,
inheritance, unreachable-target rejection, controller history consistency,
only eligible scenes seeded, deterministic fractional selection, zero-default
behavior, and fail-closed runtime collision handling. The broader legacy task
suite is not green in this environment: missing jaxlie/ml_collections/
mujoco_playground dependencies and two stale fixtures missing width_levels.
Shell syntax and parser-only checks passed for the exact pilot script; the
runner was not imported by dry parsing.

The prepared launch command is:

```bash
bash configs/pilots/hand_raised_seeded_20260921.sh
```

That script pins 50 updates, fraction .5, online W&B project CAT-wholebody,
entity skvayzer, all three v3 manifests, and the same pre-v3 source checkpoint
listed in the v3 pilot launch. It avoids deliberately warm-starting from the
reported below-guard v3 endpoint. Source checkpoint contents/retention were
not re-evaluated or pinned by reading its bytes. Checkpoint interval is 50.

## C. Ranked recommendation

1. Run this one bounded seeded pilot. Discovery failure is plausible; it is
   not proven. Static reachability does not prove that the current policy can
   maintain the pose while walking. This experiment now separates those cases.
2. If maintenance improves but unseeded acquisition stays zero, stop blind
   joint-PPO reward/bank tuning. Prefer a successful hand specialist or
   demonstration curriculum, then scene-conditioned distillation with flat
   and CAT retention examples. Do not distill an unevaluated teacher.
3. Inspect the existing specialist before training another. Its sibling
   directory exists. Metadata says stopped at 660602880 steps; its last logged
   hand-goal result is 70/102 = .6862745098 (metrics.jsonl:1681). This is training
   telemetry from a different bank, not evidence of v3 raised compliance.
   Checkpoint weights, compatibility and held-out performance remain unverified.

## D. Prospective falsification and decision gates

New metrics split `training/policy_forward_protected_{seeded,unseeded,seeded_after100}_hand_contrast_hand_region_compliance_fraction`,
with a corresponding `_sample_count` for each. They use actual post-step
states. Reset-time compliance alone is not a successful pilot. after100 means
native episode age at least 100 control steps (2 seconds), not updates.

Primary falsification: with substantial seeded exposure, zero seeded-after100
compliance throughout 50 updates rejects the claim that supplying discovery
alone is sufficient. Repeated death before 100 steps is a maintenance failure,
not evidence of successful exploration. Missing protected-scene exposure is
an instrumentation/sampling failure and must not be called a negative result.

Predeclared practical gate for updates 41–50: at least 10,000 eligible
seeded-after100 samples, at least 10% compliance there, at least 1% unseeded
compliance with at least 10,000 eligible unseeded samples, nonzero protected
clean-goal success, and flat walking compliance >= .27. These are proposed
pilot decision thresholds, not empirically established universal cutoffs.
If only seeded maintenance passes, B diagnoses acquisition but is not a
complete solution; switch to specialist/demo acquisition rather than another
unseeded PPO tuning cycle. Flat below .27 fails retention regardless of hands.

Dense metrics use cumulative counts over the 100-update selection window.
For this fresh 50-update run, derive the last-ten-update ratio as
`(rate50*count50 - rate40*count40)/(count50-count40)` instead of interpreting
update 50's cumulative fraction as an instantaneous value.
