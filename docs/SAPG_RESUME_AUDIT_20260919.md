# SAPG host-resume and performance audit — 19 September 2026

The user asked whether switching PCs had changed the training setup or damaged
checkpoint performance. This audit is read-only with respect to training:
the existing continuous run remains active, and no evaluation, new metrics,
hyperparameter change or restart was introduced.

## Finding and scope

No changed training setup or lost learner state was found in the current
`tl-server-0` to `dep-1` migration. The complete stopped runtime at
451,805,184 transitions was byte-identical after transfer and metadata migration:
SHA-256 `77dcdc2985b619c507c6776f8165f09c5ccf1de51cb90547ac7d5c201eadfabd`.
It was not replaced by the selected best leader or by original pretrained weights.
This establishes preservation of the saved state, not bitwise identity of future
GPU trajectories or proof that policy quality cannot regress during learning.
It does not retrospectively certify every historical PPO restart.

Three independent code/configuration audits found:

- All 142 installed Python distributions match exactly. Training code remains
  pinned to `351e336a079a68c93b39a40c8087c8da1fed56e4`, with fingerprint
  `bd762e7cea773d042bd4aeaf4443cf6f75be7b0b53f801bc715e6cac19b1d301`.
- Recursive launch/run comparisons differ only in the four operational paths
  for fields, collisions, resets and STOP, their launch hash, and host migration
  metadata. Observation/action contracts, rewards, optimization and provenance
  match.
- Source shared robot assets have no Git differences against the pinned commit;
  destination uses the canonical assets from that commit (91 robot asset files).
  Environment preparation verifies hashes of actual field arrays, scene geometry,
  collision arrays and reset poses against the unchanged manifests.
- Full restoration includes actor, critic, policy embeddings, learned action
  scales, Adam moments and count, normalizer state, step count, simulator state,
  per-environment episode state, history, random keys, sampling weights,
  curriculum, navigation outcome counters and metric windows. Observation
  normalization remains disabled.
- Checkpoints publish at completed-update boundaries after all optimizer passes
  and metric callbacks. Rollouts and frozen targets are consumed within that
  update, so no partial rollout buffer needs to survive this boundary.
- The numerical overflow repair is present on both hosts. It changes the
  evaluation of the SAPG surrogate into log space, not its intended objective
  or coefficients. See the numerical repair report for the earlier same-host
  continuation at 326,762,496 transitions.

Actual hardware/runtime differences are NVIDIA driver **580.178.04 →
550.163.01**, Slurm CPU affinity versus destination CPUs 0–55, new compilation,
and filesystem paths. Both devices are RTX 6000 Ada; both launches set 16
OMP/MKL threads, no JAX preallocation, and memory fraction 0.90. These differences
can affect speed and floating-point reproducibility. No controlled cross-host
trajectory comparison was run, and identical future trajectories are not claimed.

## Performance across the boundary

These are actual training outcomes, not evaluation results:

| Host | Physical transitions | Pooled training success | Leader transition reward |
|---|---:|---:|---:|
| Source | 447,086,592 | 40.42% | 0.32813 |
| Source | 449,445,888 | 23.65% | 0.31191 |
| Source | 450,625,536 | 32.68% | 0.30147 |
| Source, final | 451,805,184 | 42.47% | 0.29259 |
| Destination, first | 452,984,832 | 47.81% | 0.28635 |
| Destination, second | 454,164,480 | 53.36% | 0.28356 |

The first two destination updates completed with finite losses and durable
checkpoints. W&B independently confirmed step 454,164,480, `running`, and
`health/nonfinite=0` in the same run `f017f302`.

There is no immediate success drop at this transfer boundary. Leader reward
continued a decline already underway on the source; it had been 0.34191 at
442,368,000 transitions. This trend deserves attention, but it does not identify
a host migration fault. Two destination updates are insufficient to establish
improved skill or to rule out ongoing policy regression.

The curriculum first unlocked medium at **291,373,056** and hard at
**312,606,720** transitions, before both the numerical-repair resume and this
host transfer. All 24 scenes are now eligible. The sampler continues adapting
toward less-successful scenes within each kind; identical assets do not mean a
constant difficulty mixture. Curriculum counters and eligibility were preserved.

Success is a first-outcome ratio for each update, so both its denominator and
scene mix vary. For example, the first two destination updates contained 4,770
resolved attempts and 2,416 clean goals, or 50.65% combined. Interval comparisons
should sum successes and attempts, rather than average percentages. Curriculum
per-level rates are cumulative, not current fixed-scene evaluation scores.

## Exact current SAPG setup

| Component | Active setting |
|---|---|
| Parallel simulation | MJX, 36,864 environments |
| Policy groups | Six groups of 6,144 environments; leader 0, followers 1–5 |
| Networks | Shared actor and shared critic, six learned 16-dimensional embeddings |
| Robot interface | 222 actor observations, 310 critic observations, 29 joint actions |
| Scene bank | 12 hand-height table-edge aisles and 12 shelf passages |
| Scene kinds | 50% table / 50% shelf probability at reset; adaptive weights within each kind |
| Curriculum | Easy → medium → hard; 64 resolved current-level attempts and 60% clean-goal success to unlock |
| Rollout | 32 steps per environment = 1,179,648 physical transitions per update |
| Experience sharing | One uniformly chosen follower block (196,608 transitions) copied for the leader |
| Optimization data | 1,376,256 samples, 64 minibatches, four passes |
| Adam / gradient norm limit | Constant learning rate 0.0003 / 1.0 |
| Clipping / entropy | 0.2 / 0.003 for every policy |
| Discount / GAE | 0.98 / 0.95 |
| Exploration | Trainable, state-dependent independent Gaussian, then tanh |
| Evaluation / retention / reference KL / rollback | Disabled |

The original launch expanded the released CAT generalist actor and critic and
initialized Adam once. Added embedding input weights started at zero, preserving
all six initial policy outputs. Resumes continue the learned state without that
initialization. Policy groups are not separate task specialists: every group
samples the same hand-protection bank. There are no ordinary-clutter or original
CAT scene episodes in this specialist, and no current DAgger distillation.

Each group contributes its own on-policy rollout. One follower's rollout is
also used to improve the leader, with importance correction for the difference
between follower and leader action probabilities. Old probabilities and targets
stay frozen over all optimizer passes. On-policy data use GAE; copied data use
a one-step leader value target. Copied samples do not increase physical-step or
success counters. The off-policy critic loss is not importance weighted.

## Interpretation limits discovered

1. **Success pools all six stochastic policies.** It is not a measurement of
   the exported leader alone. First clean goal arrival counts as success; a
   fault before arrival counts as failure. Later post-goal collisions still
   penalize and terminate physics but do not erase the recorded navigation goal.
2. **Best means highest leader training reward, not highest success.** The
   selected leader at audit time is from 68,419,584 transitions, with proxy
   0.3550876081. Training resumes the latest full learner at 451,805,184, not
   that older best export. Also, selection reward comes from the pre-update
   rollout while the saved parameters are post-update. This is a selection
   limitation, not a migration change.
3. **All policies have identical entropy incentives.** The current adaptation
   retains the requested CAT coefficient of 0.003 for every group, rather than
   the differing entropy coefficients in the inspected SAPG reference setup.
   Different trajectories and learned embeddings allow differentiation, but
   meaningful policy diversity and an efficiency improvement are not established.
4. **Generalist retention is unmeasured here.** Hand-only training can change
   behavior on excluded original or ordinary-clutter scenes; these logs cannot
   establish preservation of those skills.
5. No leader-only success, policy-diversity or importance-ratio diagnostics are
   currently logged. No extra metrics were added during this audit.

Implementation references: `cat_ppo/learning/policy/ppo/train.py` (full-state
restore, leader reward and snapshots), `cat_ppo/learning/policy/sapg/losses.py`
(sharing and targets), `cat_ppo/learning/policy/sapg/config.py` (active algorithm
settings), `cat_ppo/furniture/hand_curriculum.py` (success and curriculum), and
`train_cat_wholebody.py` (leader export and best selection).
