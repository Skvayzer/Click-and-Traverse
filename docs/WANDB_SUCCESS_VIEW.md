# Success rates in a separate W&B section

## Personal workspace

The user's normal [personal workspace](https://wandb.ai/skvayzer/CAT-wholebody?nw=nwuserskvayzer)
has a pinned, expanded **Success rates** section at the top. Each of the three
validation groups has its own deterministic and stochastic chart, followed by
training success (seven charts total). Existing run selections remain available
for comparing experiments. Other sections are collapsed, with their existing
automatic charts preserved.

`scripts/fix_wandb_success_section.py` updates this existing personal workspace.
It defaults to a read-only preview; `--apply` writes the change after saving the
original raw view and proposed layout to the required `--backup` JSON file.
It checks for concurrent edits and verifies the saved raw layout. It preserves
filters and all settings outside the section list. The public Workspace SDK
rejects personal views, so this utility uses its underlying GraphQL operations
without converting the original view through the SDK's narrower model.

## Separate saved view

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
