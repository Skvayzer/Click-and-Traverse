# Pretrained whole-body training plan

The starting policy is the released **CAT generalist-v1 native checkpoint
`005033164800`**, from `Axian12138/Click-and-Traverse` at Hugging Face revision
`46ce4b57ba0639168d51741b661ff62f7ce6f045`. The repository manifest pins every
downloaded file by SHA-256. These weights are already cached on dep-0.

The loader restores the actor, critic and observation normalizer. Named feature
and action mappings preserve the pretrained leg outputs at initialization; new
observation columns and upper-body action means begin at zero, with small
exploration for the new actions. PPO fine-tunes the entire actor and critic with
a fresh optimizer. There is no frozen leg controller or imitation/distillation
stage in this launcher. The public release has no optimizer/RNG continuation
state, so this is fine-tuning, not exact continuation of the original run.

The robot uses Unitree's **three-finger Dex3-1** hands. Each hand has seven finger
joints (three in the thumb); these are fixed for traversal. The policy has 29
body actions: 12 legs, 3 waist and 14 arms/wrists. It can raise, tuck and rotate
the hands without learning finger manipulation. Collision boxes include the
entire fixed hand mesh and an additional 5 mm geometric padding. The separate
hand-clearance reward encourages advance avoidance before physical contact.

## Continuous overnight training

The requested run uses dep-0's RTX 5090 and online W&B logging, with **no global
step, stage or time limit**. It continues until an explicit manual stop or an
error. Scene changes are automatic; reaching a per-scene cadence does not end
the training job. A small implementation smoke is not evidence of learning.

Use 16-step rollouts, four minibatches, four PPO updates per batch and learning
rate `3e-4`. The larger restart uses 8,192 environments on original typical CAT
and simple generic clutter, 4,096 on simple furniture, 2,048 on randomized CAT,
2,048 on dense generic clutter and 1,024 on dense furniture. These counts target
high GPU memory use; measured allocation is recorded in the run notes and W&B.
Native dense furniture MJX state alone occupies about 9.44 MiB per environment,
compared with 0.293 MiB on original forward. Training also needs cached resets,
rollout buffers and compiler temporaries, so scene families need different counts.

```bash
.venv/bin/python -m cat_ppo.furniture.continuous run \
  --run-dir outputs/continuous_cat_dex3_20260915_large \
  --restart-from-run /home/konstantin.smirnov/robotics/Click-and-Traverse-WholeBody/outputs/continuous_cat_dex3_20260915_live \
  --num-envs 1024 --legacy-num-envs 8192 --pilot-num-envs 8192 \
  --pilot-furniture-num-envs 4096 --random-num-envs 2048 \
  --generic-num-envs 2048 --furniture-num-envs 1024 \
  --steps-per-stage 8388608 --checkpoint-epochs 16 \
  --seed 0 --wandb-mode online
```

The first four stages are original CAT forward, simple generic clutter, original
CAT hurdle, and simple furniture clutter. Each stage starts from the preceding
selected checkpoint; only the first starts from the published CAT weights.
Scenes switch every 8,388,608 transitions, with checkpoint candidates every
524,288 transitions. This preserves 64 rollout batches per light scene while
amortizing compilation over more work. These intervals organize an ongoing run; they do not impose
a total training budget. Each stage is a W&B run in the same group, with an
explicit cumulative `global_step` axis. `wandb.json` contains its exact URL.

Request a manual stop from the same checkout, using the actual absolute run path
when the training source is in an isolated worktree:

```bash
.venv/bin/python -m cat_ppo.furniture.continuous stop \
  --run-dir outputs/continuous_cat_dex3_20260915_large
```

The active trainer finishes its current checkpoint epoch and exports the
selected model before exiting. SIGINT/SIGTERM to the runner makes the same
cooperative request. The runner never silently retries a failed or partially
completed stage with invented optimizer state.

An explicit `--restart-from-run` requires a stopped, verified source run. It
copies the selected actor, critic and normalizer into the new run and verifies
every file hash. A stopped partial scene is replayed with the new batch size;
the W&B offset includes all previously executed transitions. The optimizer is
fresh. The original source run is preserved, and the owned imported copy is
retired only after the first new stage completes and its handoff is verified.

Task metrics come from completed training episodes: success, forbidden/hand/arm
contact, falls, numerical failures, timeouts, completion time, route progress and
minimum hand clearance. PPO losses and throughput are logged alongside these in
local JSONL and W&B's `CAT-furniture` project. GPU allocator live/peak bytes are
read after completed epochs and logged alongside throughput; allocation is not
increased by preallocating unused memory. Training episodes are not held-out
evaluation. A window with no completed episodes must not be presented as zero
contact or zero fall rate.

## Longer curriculum

| Round | New environments | Original CAT rehearsal |
| --- | --- | --- |
| 1 | Simple generic/furniture encounters | Forward, hurdles, narrow gaps, crouching, random and combined obstacles |
| 2 | Dense generic clutter and tables/chairs | Same original families, with increasing randomized difficulty where supported |
| 3 | Dense clutter with delayed/noisy/partly unknown maps | Original families continue |

The mixture by equal stage budgets is 50% original CAT, 25% generic clutter and
25% furniture. One scene is compiled per stage, with a fresh optimizer at each
handoff. This is sequential rehearsal, rather than many different room layouts
in the same PPO batch. Rehearsal is intended to reduce forgetting; retention
still needs measurement.

At the larger cadence, three full rounds comprise 36 stages / 301,989,888 transitions, but the
continuous runner keeps going afterward with new scene seeds and capped
difficulty. Three independent seeds belong to the eventual study; the requested
overnight run follows one evolving seed. Scene switching is based on cadence,
not automatically gated by skill mastery. A final demo policy needs separate
held-out checks on both original CAT environments and dense clutter; no
evaluation is launched automatically.

## Checkpoints and readiness

Keep one selected native checkpoint plus its ONNX export per run. During a
curriculum, retire the predecessor only after a verified successful handoff.
Metric histories remain. With the default `num_evals=0`, selection uses a
within-stage training reward proxy; it is not globally best across all domains.

The September 15 audit identified incomplete task-info reset in the generic
autoreset wrapper. The furniture training wrapper restores the complete task
state between episodes and reduces terminal metrics according to their meaning
(for example, minimum clearance is not summed over time). Regression and GPU
implementation-check results are recorded in
[IMPLEMENTATION_VERIFICATION.md](IMPLEMENTATION_VERIFICATION.md).

The continuous launch above supersedes the earlier bounded-pilot recommendation.
Learned traversal and retained CAT skills must be judged from actual training
and later held-out results. No GitHub or W&B credentials are copied into the
repository.
