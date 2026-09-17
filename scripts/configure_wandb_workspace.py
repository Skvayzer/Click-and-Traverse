"""Create a compact saved W&B view without changing metrics or run history.

The default prints a reviewable layout and performs no network calls. ``--save``
uses the optional official ``wandb-workspaces==0.4.11`` SDK to create a NEW saved
view filtered to exactly one run. It never updates an existing view or run.
Install that SDK in a separate utility environment, not the training environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


VALIDATION_GROUPS = (
    ("cat", "Original environments"),
    ("ordinary_clutter", "Ordinary clutter"),
    ("hand_protection", "Hand protection"),
)


def panel(title, metrics, *, rate=False):
    return dict(title=title, metrics=dict(metrics), rate=rate)


def validation_metrics(suffix, *, stochastic=False):
    prefix = "validation/stochastic/" if stochastic else "validation/"
    return [(f"{prefix}{group}_{suffix}", label) for group, label in VALIDATION_GROUPS]


def layout(entity, project, run_id, name=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Expected a bare W&B run ID, not a URL")
    return dict(
        entity=entity, project=project, run_id=run_id,
        name=name or f"Recovery training - {run_id}",
        x_axis="global_step", auto_generate_panels=False,
        sections=[
            dict(name="Success rates", is_open=True, panels=[
                panel("Fixed scenes - deterministic success", validation_metrics("goal_success_rate"), rate=True),
                panel("Fixed scenes - stochastic success", validation_metrics("goal_success_rate", stochastic=True), rate=True),
                panel("Training success - completed episodes in latest rollout", [
                    ("training/goal_success_rate", "Training episodes")], rate=True),
            ]),
            dict(name="Collisions and falls", is_open=False, panels=[
                panel("Fixed scenes - obstacle failures", validation_metrics("obstacle_violation_rate"), rate=True),
                panel("Fixed scenes - falls", validation_metrics("fall_rate"), rate=True),
            ]),
            dict(name="Hand behavior and exploration", is_open=False, panels=[
                panel("Hand passages - minimum clearance per episode (mean, m)", [
                    ("validation/hand_protection_mean_min_hand_clearance_m", "Hand clearance (m)")]),
                panel("Action distribution standard deviation", [
                    ("training/leg_std_mean", "Legs"),
                    ("training/upper_std_mean", "Upper body")]),
            ]),
            dict(name="Curriculum and rewards", is_open=False, panels=[
                panel("Hand curriculum stage (0 easy, 1 medium, 2 hard)", [
                    ("hand_curriculum/stage", "Stage")]),
                panel("Episode reward", [("episode/sum_reward", "Episode return")]),
            ]),
            dict(name="Optimization and throughput", is_open=False, panels=[
                panel("Leg reference divergence", [("training/reference_kl", "Reference KL")]),
                panel("Training throughput (transitions/s)", [("training/sps", "Transitions/s")]),
            ]),
        ],
    )


def build_workspace(spec):
    # Keep the dry-run path usable without the SDK or W&B credentials.
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr

    sections = []
    for section in spec["sections"]:
        panels = []
        for definition in section["panels"]:
            panels.append(wr.LinePlot(
                title=definition["title"], x=spec["x_axis"],
                y=list(definition["metrics"]), line_titles=definition["metrics"],
                range_y=(0, 1) if definition["rate"] else (None, None),
                title_y="Episode fraction" if definition["rate"] else None,
                smoothing_type="none", smoothing_factor=0,
                ignore_outliers=False, aggregate=False,
            ))
        sections.append(ws.Section(
            name=section["name"], panels=panels, is_open=section["is_open"],
        ))
    return ws.Workspace(
        entity=spec["entity"], project=spec["project"], name=spec["name"],
        sections=sections, auto_generate_panels=False,
        settings=ws.WorkspaceSettings(x_axis=spec["x_axis"], smoothing_type="none"),
        runset_settings=ws.RunsetSettings(filters=[expr.Metric("ID") == spec["run_id"]]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="skvayzer")
    parser.add_argument("--project", default="CAT-wholebody")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name")
    parser.add_argument("--output", type=Path, help="Write the layout and saved-view URL as JSON")
    parser.add_argument("--save", action="store_true", help="Create a NEW saved view on W&B")
    args = parser.parse_args()
    spec = layout(args.entity, args.project, args.run_id, args.name)
    if args.save:
        workspace = build_workspace(spec).save_as_new_view()
        spec["url"] = workspace.url
    serialized = json.dumps(spec, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
