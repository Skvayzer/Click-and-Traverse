# Training success in the W&B workspace

## Current training view — 2026-09-18

The new training run is [ca47143c](https://wandb.ai/skvayzer/CAT-wholebody/runs/ca47143c).
Its [filtered training-success view](https://wandb.ai/skvayzer/CAT-wholebody?nw=9b4d8422c6a)
shows only this experiment.
It uses training outcomes only: no separate evaluation rollouts, retention
evaluation charts, or automatic policy rollback are part of this profile.

The user's normal [personal workspace](https://wandb.ai/skvayzer/CAT-wholebody?nw=nwuserskvayzer)
places a pinned, expanded **Training success** section first, containing four
individual charts:

| Chart | Metric |
| --- | --- |
| All training environments | `training/goal_success_rate` |
| Original CAT environments | `training/cat_goal_success_rate` |
| Ordinary clutter | `training/ordinary_clutter_goal_success_rate` |
| Hand-protection passages | `training/hand_protection_goal_success_rate` |

The CAT group includes original, published, and procedurally generated CAT
tasks. Ordinary clutter excludes the dedicated hand-protection passages. The
total aggregates all three groups by their episode counts; it is not the
unweighted mean of the three group rates.

### Training success definition

Each physical training episode contributes at most one outcome. Its first clean
goal is a success; its first failure or timeout before that goal is a failure.
The outcome is recorded when it first resolves, even if CAT's physical episode
continues after reaching the goal. Subsequent transitions cannot count the same
episode again.

For each completed PPO rollout/update, the logged rate is
`goal_success_count / resolved_count`. The total uses
`training/goal_success_count` and `training/resolved_count`; group counts add
the `cat_`, `ordinary_clutter_`, or `hand_protection_` prefix. When a group has
no resolved outcomes in that rollout, its rate is omitted rather than reported
as zero. These curves describe the current stochastic training policy and the
current curriculum population. They are not held-out evaluation estimates.

Older runs used `training/goal_success_rate` for a different completed-episode
metric. Their history is preserved. Use the new run-filtered saved view for an
unambiguous view of the current definition; personal-workspace run selections
remain available for inspecting historical experiments.

### What the released CAT training measured

The released logger reports `rollout/timeout_rate`: among the episodes ending in
a rollout, the fraction marked as a time-limit truncation rather than an earlier
native failure. Its scene sampler computes a per-scene moving survival rate
from `done * truncation`, then samples low-survival scenes more often. This is
survival through the episode horizon, not an explicit goal-arrival success rate.
The current compact logger retains that timeout metric separately from our
navigation-success curves. Both are measured from training, without evaluation.

Sources at the pinned original commit:
[sampler](https://github.com/Skvayzer/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/cat_ppo/learning/train/pf_utils.py#L112),
[logger](https://github.com/Skvayzer/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/cat_ppo/learning/policy/ppo/train.py#L110).

### Layout and logging

`scripts/configure_training_success_workspace.py` updates the existing personal
workspace from its raw JSON. It preserves run filters, unknown browser settings,
and previous panels. The former seven-chart **Success rates** section is renamed
**Previous evaluation results** and collapsed. Other existing sections also
remain accessible and collapsed.

The same utility creates a **new saved view** filtered to exactly the supplied
run ID. That view contains only the four training charts, with automatic panel
generation disabled, 0–1 axes, no smoothing, and no outlier suppression. Existing
saved views and run histories remain unchanged. Personal filters are preserved;
the saved view provides isolation from historical success definitions.

Dry-run is the default. The required backup records the original personal view,
the proposed views, and existing views before any mutation. Apply checks for
concurrent edits, then verifies exact readback and unchanged legacy views. Use
a new backup filename for each invocation:

```sh
/tmp/cat-recovery-wandb-view312/bin/python \
  scripts/configure_training_success_workspace.py \
  --run-id ca47143c \
  --backup analysis/training-success-ca47143c-preview.json

/tmp/cat-recovery-wandb-view312/bin/python \
  scripts/configure_training_success_workspace.py \
  --run-id ca47143c \
  --backup analysis/training-success-ca47143c-apply.json --apply
```

The utility prints both workspace URLs. Verify the actual personal workspace in
the browser after applying; API readback alone does not establish what is visible
in an already open tab.

The new training profile enables `GeneralistLogger(compact_logging=True)`.
It retains the four success rates and associated counts, essential PPO losses,
entropy and action-standard-deviation metrics, throughput, curriculum progress,
reward components, failure aggregates, and numerical health. It omits per-scene
diagnostics and evaluation metrics. Existing logging defaults remain unchanged
for older profiles.

## Historical evaluation view — 2026-09-17

Before the training-only profile, the personal workspace placed a seven-chart
**Success rates** section first. Each of three validation groups had its own
deterministic and stochastic chart, followed by the earlier completed-episode
training success metric. These panels are retained as collapsed historical
results; they do not describe the new run's monitoring.

`scripts/fix_wandb_success_section.py` implemented that earlier personal layout.
It defaults to a read-only preview; `--apply` writes the change after saving the
original raw view and proposed layout to the required `--backup` JSON file.
It checks for concurrent edits and verifies the saved raw layout. It preserves
filters and all settings outside the section list. The public Workspace SDK
rejects personal views, so this utility uses its underlying GraphQL operations
without converting the original view through the SDK's narrower model.

### Historical saved view

`scripts/configure_wandb_workspace.py` creates a compact **saved workspace view**
for one training run. Success is the only expanded section. It shows the three
comparable validation groups (original environments, ordinary clutter, hand
protection), with separate deterministic, stochastic and training panels.
All fractions retain a 0–1 axis, without smoothing or outlier suppression.

Collisions/falls, hand behavior/exploration, curriculum/rewards and optimization
are separate, initially collapsed sections. The view does not automatically
create hundreds of panels. Detailed metrics remain in the run history and can
still be explored through the ordinary W&B interface. Existing metric names,
definitions, logged values and dashboards are preserved; no duplicate metrics
or separate runs are created.

The run's `validation/clutter_goal_success_rate` combines ordinary clutter and
hand passages. For comparison to clutter performance before hand passages were
introduced, this view uses `validation/ordinary_clutter_goal_success_rate`.
Training success counts completed episodes in the latest rollout, so it should
be compared to its own history, not treated as another fixed-scene validation.

Preview the exact layout with no network calls or optional dependencies:

```sh
python scripts/configure_wandb_workspace.py --run-id NEW_RUN_ID \
  --output /tmp/cat-success-view.json
```

To create the saved view, use the official optional W&B Workspaces SDK in a
separate utility environment. This does not upgrade the learner's W&B package:

```sh
python3 -m venv /tmp/cat-wandb-view
/tmp/cat-wandb-view/bin/pip install wandb-workspaces==0.4.11
/tmp/cat-wandb-view/bin/python scripts/configure_wandb_workspace.py \
  --run-id NEW_RUN_ID --save --output /tmp/cat-success-view.json
```

The SDK uses the machine's existing W&B authentication. `--save` creates a new
saved view each time, filtered by the exact run ID. Open the URL written to the
JSON output. It does not overwrite the current project workspace or modify
older runs. Creating the view is independent of starting the learner.

This saved view does **not** change the personal workspace or a single-run page.
Verifying its API representation alone does not establish what the user sees in
their open tab. Check the actual browser page after changing a layout.

API references: [official overview](https://docs.wandb.ai/models/ref/wandb_workspaces)
and [official SDK source](https://github.com/wandb/wandb-workspaces).
