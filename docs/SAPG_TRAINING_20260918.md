# Optional SAPG training for CAT

SAPG is selected with `--algorithm sapg`. Omitting that argument retains the
existing PPO implementation. This integration changes the learner, while keeping
the whole-body task, scene banks, robot observations, and action interface.
This document and the batch script do not start training.

## References and scope

- [Split and Aggregate: Scaling and Improving Deep Reinforcement Learning](https://proceedings.mlr.press/v235/singla24a.html), ICML 2024;
  [equations and implementation discussion](https://arxiv.org/html/2407.20230v1).
- [Official SAPG implementation](https://github.com/jayeshs999/sapg/tree/e96efaef69755e82e8e921febf2184d622fd73fe), pinned to
  `e96efaef69755e82e8e921febf2184d622fd73fe`.
- The user's existing reference on `tl-server-0`:
  `/data1/users/konstantin.smirnov/play2perfect`, commit
  `70e79b5e53f912ef04af294ff8f61ac1c7f42160`.
  Relevant files are `isaacsimenvs/cfg/train/G1Dex3PlaySAPG.yaml`,
  `rl_games/rl_games/common/a2c_common.py`,
  `rl_games/rl_games/common/common_losses.py`, and
  `rl_games/rl_games/algos_torch/network_builder.py`.

There is a reference-code discrepancy: both inspected repositories define the
correct `decoupled_actor_loss`, but their active continuous learner calls ordinary
PPO clipping. CAT implements the paper's correction described below, using the
[decoupled helper](https://github.com/jayeshs999/sapg/blob/e96efaef69755e82e8e921febf2184d622fd73fe/rl_games/rl_games/common/common_losses.py#L51)
as a cross-check. It does not reproduce that active clipping discrepancy.

## Policy groups and CAT settings

The default is six equal groups of environments, with policy 0 as the leader.
One shared actor and one shared critic retain CAT's separate MLP trunks. A single
learned table contains six 16-dimensional embeddings; both trunks use the same
selected row. Policy IDs are internal learner context, fixed by environment
slot. They are independent of scene IDs and survive episode resets.

The robot interface remains **222 actor inputs, 310 critic inputs, and 29
actions**. Embeddings are appended after observation preprocessing inside the
network; no sensor observation is added. The native state-dependent, independent
Gaussian followed by `tanh` is retained. There is no arm autoregression, frozen
noise, or added action-standard-deviation clipping.

| Setting | This CAT integration |
|---|---|
| Adam learning rate | `3e-4`, constant |
| PPO clipping epsilon | `0.2` |
| Entropy coefficient | `0.003` for every policy, including the leader |
| Discount / GAE lambda | `0.98` / `0.95` |
| Critic loss | `0.25 * mean((target - value)^2)` |
| Rollout length | 32 steps |
| Optimizer passes / minibatches | 4 / 64 |
| Observation normalization | Disabled, as in the released CAT configuration |
| Hardware support | One device on one host |

These are deliberate CAT adaptations. The inspected Play configuration instead
uses dimension 32, an LSTM, different optimizer settings, and per-policy learned
state-independent log standard deviations. Its entropy scale `0.002` produces
coefficients `[0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0]`, with the leader last.
CAT retains uniform entropy `0.003`; this removes the reference's heterogeneous
entropy incentive. Independent trajectories and learned conditioning still
allow policies to differentiate, but improved learning efficiency is not assumed.

Initialization loads the **original released CAT checkpoint**, expands its actor
and critic for the whole-body task, and starts fresh Adam state. Newly added
action scales initialize to `0.05`. Added embedding input weights initialize to
zero, preserving all six policies' initial outputs; embeddings initialize with
standard deviation `0.01`. The launcher checks actor/critic parity after this
expansion. A fine-tuned best checkpoint cannot replace original initialization
under `cat_train_only`.

## One learner update

1. Collect stochastic rollouts from every policy group, storing raw actions,
   behavior log probabilities, and policy IDs.
2. Compute each group's on-policy GAE using the old critic. Select one follower
   uniformly from policies 1–5 and copy its entire rollout block for the leader.
3. On copied transitions, evaluate the old leader's likelihood and current/next
   state values. Freeze these quantities and every target before optimizer work.
4. Normalize advantages once across the complete augmented rollout. Shuffle and
   optimize that same frozen dataset for all four passes.

For a copied follower action, let `b` be its stored behavior log probability,
`o` its frozen old-leader log probability, and `n` its current leader log
probability. With `mu = exp(o - b)` and `q = exp(n - o)`, the actor loss is:

```text
-mean(mu * min(q * advantage,
               clip(q, 1 - epsilon, 1 + epsilon) * advantage))
```

Equivalently, `exp(n - b)` is clipped around `mu`, with bounds
`mu * (1 +/- epsilon)`. The original behavior likelihood is never overwritten
when changing the target policy ID. On-policy rows have `mu = 1`.

The copied leader target is the one-step return
`reward + gamma * (1 - termination) * old_leader_value(next_state)`; its advantage
subtracts `old_leader_value(state)`. Physical termination removes bootstrapping.
Time truncations follow CAT/Brax's baseline-target convention and are masked
from the actor advantage, including after normalization. The off-policy critic
is **not importance weighted**. Each of the six original blocks and one copied
block has equal weight in the joint mean loss.

At 24,576 environments, each policy owns 4,096 environments:

| Per update | Count |
|---|---:|
| Actual environment transitions | `24,576 * 32 = 786,432` |
| Copied follower transitions | `4,096 * 32 = 131,072` |
| Learner samples before reuse across four passes | `917,504` |
| Learner samples per minibatch | `14,336` |

Copied samples do not advance environment-step counters or success counts.

## Scenes, logging, and checkpoints

Use the existing complete scene and collision contracts:

```text
data/furniture/cat_hand_protection_v1_20260917/manifest.json
data/furniture/body_collision_hand_v1_20260917/manifest.json
data/furniture/body_collision_hand_resets_v1_20260917/manifest.json
```

SAPG retains the `cat_train_only` task rewards, hand curriculum, geometry, and
collision/reset behavior. It performs no evaluation, retention validation,
reference KL regularization, or automatic rollback. Training has no learner-step
cap; a manual STOP request or the scheduler's allocation limit ends the job.

One W&B run reports the same four first-outcome training success rates: overall,
CAT, ordinary clutter, and hand protection, with attempt counts. These rates
pool all policy groups. They are not a separate evaluation of the leader.

Best-model selection uses the **leader's mean transition reward from the training
rollout**. It is a training proxy, not held-out success. One best model folds the
leader embedding into both first-layer biases and saves ordinary CAT actor/critic
parameters. Existing native viewers need no policy-ID input. One overwritten
`resume.msgpack` instead preserves the full six-policy learner, embeddings, Adam,
normalizer, environment, PRNG, and counters. A folded best model is not an exact
SAPG continuation snapshot. Resume checks the original configuration and source
identity; do not change the source worktree or relabel a run as PPO.

## Launch preparation on tl-server-0

The batch recipe is [train_cat_sapg.sbatch](../scripts/slurm/train_cat_sapg.sbatch).
It requests one RTX 6000 Ada, 16 CPUs, 96 GiB system memory, and two days in the
`batch` partition. It requires a full commit hash selected **at submission**,
creates a detached source worktree, and uses the project's existing `.venv`.
The worktree's `data` symlink points to the verified shared asset tree. The
tracked data skeleton is retained alongside it as `data.git-checkout`.

The default resource profile is the conservative, divisible **1,536 environments
with batch size 192**. The explicit larger candidate is **24,576 / 384**.
**24,576-environment SAPG VRAM usage is unvalidated**: successful PPO training at
that size does not establish SAPG capacity. Its larger learner batches and target
preparation need measurement. The script defaults to a 0.90 JAX memory fraction
with preallocation disabled; it leaves Slurm's GPU assignment intact.

After reviewing the committed version, the submission command is:

```bash
cd /data1/users/konstantin.smirnov/Click-and-Traverse-WholeBody
bash -n scripts/slurm/train_cat_sapg.sbatch
export CAT_SOURCE_COMMIT="$(git rev-parse HEAD)"
sbatch --export=ALL scripts/slurm/train_cat_sapg.sbatch
```

For a deliberately chosen larger resource test, add
`CAT_NUM_ENVS=24576 CAT_BATCH_SIZE=384` to the exported environment before
submission. This document does not submit either configuration. Run GPU
correctness and memory checks under Slurm, not an SSH shell.

Each submission owns `outputs/cat_sapg_JOBID`. To stop that specific run
cooperatively, create its `STOP` file, or signal its batch shell with
`scancel --signal=USR1 --batch JOBID`. The script also receives USR1 five minutes
before the allocation ends. It requests a completed-update stop without
signalling other jobs. Finishing the current update and snapshot still needs
time; the scheduler can enforce its deadline before a slow update finishes.
The previously completed atomic resume snapshot remains the recovery boundary.

## Validation status

Source review covers grouping, frozen targets, importance clipping, leader
export, and resume identity. Automated tests cover network parity, numerical
loss cases, actual tiny learner updates, and exact resume. Execution results
must be reported separately; **a GPU correctness check is pending in this
document, and no learning-quality improvement is claimed**. A tiny correctness
run can establish valid updates and checkpoint behavior, not convergence,
overnight stability, or 24,576-environment memory capacity.
