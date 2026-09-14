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

## First training pilot

Use dep-0's isolated `.venv` and one RTX 5090. Begin with four parallel environments,
16-step rollouts, four minibatches, four PPO updates per batch, learning rate
`3e-4`, and **32,768 transitions per stage**. Measure actual throughput and GPU
memory before increasing the batch toward the proposed 32 environments or
predicting wall-clock duration.

```bash
.venv/bin/python -m cat_ppo.furniture.curriculum prepare \
  --output outputs/plans/dex3_pilot_seed0.json --rounds 1 \
  --steps-per-stage 32768 --num-envs 4 --seed 0 --wandb-mode online

# Launch explicitly when ready; preparation above only writes the plan.
.venv/bin/python -m cat_ppo.furniture.curriculum run \
  --plan outputs/plans/dex3_pilot_seed0.json \
  --run-dir outputs/dex3_pilot_seed0 --max-stages 4
```

The first four stages are original CAT forward, generic pilot clutter, original
CAT hurdle, and furniture pilot clutter: **131,072 transitions total**. Each
stage starts from the preceding selected checkpoint; only the first starts from
the published CAT weights. This pilot is for checking learning behavior,
episode resets, hand-contact/fall trends and throughput. It is not a convergence
budget or a benchmark result.

Task metrics come from completed training episodes: success, forbidden/hand/arm
contact, falls, numerical failures, timeouts, completion time, route progress and
minimum hand clearance. PPO losses and throughput are logged alongside these in
local JSONL and W&B's `CAT-furniture` project. Training episodes are not held-out
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

A starting full budget is 36 stages × 1,048,576 transitions = **37,748,736
transitions per training seed**. Three independent seeds belong to the eventual
study, after the first seed establishes a useful training regime. Stage budgets
are fixed, not automatically gated by skill mastery. A final demo policy needs
separate held-out checks on both original CAT environments and dense clutter;
no evaluation is launched automatically.

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

The next learning run should be the bounded pilot above. The full training and
held-out performance study remain unrun. No GitHub or W&B credentials need to be
copied into the repository.
