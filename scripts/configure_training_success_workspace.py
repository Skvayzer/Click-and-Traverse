"""Show four training success charts in the personal W&B workspace.

Dry-run reads views and writes a required backup without changing W&B. Apply
creates a NEW saved view filtered to one run, then updates the authenticated
user's raw personal view. Existing saved views are never overwritten. The old
seven-chart success section stays available as collapsed evaluation history.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid


READ_QUERY = """
query TrainingSuccessWorkspace($entity: String!, $project: String!) {
  viewer { username }
  project(name: $project, entityName: $entity) {
    allViews(viewType: "project-view") {
      edges { node { id name displayName spec } }
    }
  }
}
"""
WRITE_QUERY = """
mutation TrainingSuccessWorkspace($id: ID, $entityName: String, $projectName: String,
  $type: String, $name: String, $displayName: String, $spec: String) {
  upsertView(input: {id: $id, entityName: $entityName, projectName: $projectName,
    type: $type, name: $name, displayName: $displayName, spec: $spec,
    createdUsing: WANDB_SDK}) { view { id name } inserted }
}
"""
SUCCESS_CHARTS = (
    ("training/goal_success_rate", "All training environments"),
    ("training/cat_goal_success_rate", "Original CAT environments"),
    ("training/ordinary_clutter_goal_success_rate", "Ordinary clutter"),
    ("training/hand_protection_goal_success_rate", "Hand-protection passages"),
)


def read_views(api, execute, entity, project):
    response = execute(api, READ_QUERY, {"entity": entity, "project": project})
    if response.get("project") is None:
        raise RuntimeError("Project unavailable; no changes made")
    return response["viewer"]["username"], {
        entry["node"]["name"]: entry["node"]
        for entry in response["project"]["allViews"]["edges"]
    }


def training_section():
    panels = []
    for metric, label in SUCCESS_CHARTS:
        panels.append({
            "__id__": uuid.uuid4().hex[:11], "isAuto": False,
            "layout": {"x": 0, "y": 0, "w": 8, "h": 6},
            "viewType": "Run History Line Plot",
            "config": {"chartTitle": f"Training success: {label}",
                       "xAxis": "global_step", "metrics": [metric],
                       "yAxisMin": 0., "yAxisMax": 1.,
                       "yAxisTitle": "Clean goals / resolved training episodes",
                       "ignoreOutliers": False, "smoothingWeight": 0.,
                       "smoothingType": "none", "aggregate": False,
                       "overrideSeriesTitles": {metric: label}},
        })
    return {
        "__id__": uuid.uuid4().hex[:11], "name": "Training success",
        "isOpen": True, "isPanelsAuto": False, "pinned": True, "type": "flow",
        "flowConfig": {"snapToColumns": True, "columnsPerPage": 2,
                       "rowsPerPage": 2, "gutterWidth": 16,
                       "boxWidth": 460, "boxHeight": 300},
        "sorted": 0, "panels": panels,
        "localPanelSettings": {"xAxis": "global_step", "xAxisActive": True,
                               "smoothingWeight": 0., "smoothingType": "none",
                               "smoothingActive": False, "ignoreOutliers": False,
                               "useRunsTableGroupingInPanels": True},
    }


def prepare_personal(personal):
    """Preserve raw unknown fields, run filters, and all historical panels."""
    original = json.loads(personal["spec"])
    proposed = copy.deepcopy(original)
    sections = proposed["section"]["panelBankConfig"]["sections"]
    existing = [s for s in sections if s.get("name") == "Training success"]
    if len(existing) > 1:
        raise RuntimeError("Ambiguous existing training sections; no changes made")
    if existing:
        first = existing[0]
        metrics = [p.get("config", {}).get("metrics") for p in first.get("panels", [])]
        if metrics != [[metric] for metric, _ in SUCCESS_CHARTS]:
            raise RuntimeError("Existing Training success section has different charts")
        sections.remove(first)
        first.update(isOpen=True, isPanelsAuto=False, pinned=True)
    else:
        first = training_section()
    for section in sections:
        if section.get("name") == "Success rates":
            panels = section.get("panels", [])
            validation = [m for p in panels for m in p.get("config", {}).get("metrics", [])
                          if m.startswith("validation/")]
            if len(panels) != 7 or len(validation) != 6:
                raise RuntimeError("Expected the seven existing success charts before renaming")
            section["name"] = "Previous evaluation results"
        section["isOpen"] = False
    sections.insert(0, first)
    return original, proposed


def prepare_saved(personal_spec, run_id, *, hand_specialist=False):
    """Build a training-only saved view while leaving every old view untouched."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Expected a bare run ID")
    saved = copy.deepcopy(personal_spec)
    section = saved["section"]
    first = copy.deepcopy(section["panelBankConfig"]["sections"][0])
    if hand_specialist:
        first["panels"] = [panel for panel in first["panels"] if panel.get("config", {}).get("metrics")
                           == ["training/hand_protection_goal_success_rate"]]
        if len(first["panels"]) != 1:
            raise ValueError("Expected exactly one hand-protection success chart")
        first["name"] = "Hand-protection success"
        first["flowConfig"].update(columnsPerPage=1, rowsPerPage=1)
    section["panelBankConfig"]["sections"] = [first]
    section.setdefault("settings", {}).update(
        shouldAutoGeneratePanels=False, xAxis="global_step", xAxisActive=True,
        smoothingType="none", smoothingWeight=0, smoothingActive=False)
    # Personal workspaces may carry additional report panels outside sections.
    if "panelBankSectionConfig" in section:
        section["panelBankSectionConfig"]["panels"] = []
        section["panelBankSectionConfig"]["isOpen"] = False
    section["runSets"] = [{
        "id": uuid.uuid4().hex[:11], "name": "Training run", "enabled": True,
        "search": {"query": ""}, "grouping": [],
        "runFeed": {"version": 2, "columnVisible": {}, "columnPinned": {},
                    "columnWidths": {}, "columnOrder": [], "pageSize": 10,
                    "onlyShowSelected": False},
        "filters": {"filterFormat": "filterV2", "filters": [
            {"key": {"section": "run", "name": "name"}, "op": "=",
             "value": run_id, "disabled": False}]},
        "sort": {"keys": [{"key": {"section": "run", "name": "createdAt"},
                            "ascending": False}]},
        "selections": {"root": 1, "bounds": [], "tree": []},
        "expandedRowAddresses": [],
    }]
    section["openRunSet"] = 0
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="skvayzer")
    parser.add_argument("--project", default="CAT-wholebody")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--personal-view", default="nw-nwuserskvayzer-w")
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--saved-only", action="store_true", help="Create a run-specific view without changing the personal workspace")
    parser.add_argument("--hand-specialist", action="store_true", help="Show the specialist's hand success rate without empty CAT/clutter panels")
    args = parser.parse_args()
    if args.hand_specialist and not args.saved_only:
        parser.error("--hand-specialist requires --saved-only")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id must be a bare run ID")
    import wandb
    from wandb_workspaces._graphql import execute_graphql

    api = wandb.Api()
    username, views = read_views(api, execute_graphql, args.entity, args.project)
    if args.personal_view != f"nw-nwuser{username}-w":
        raise RuntimeError("Target must be the authenticated user's personal workspace")
    personal = views[args.personal_view]
    original, proposed = prepare_personal(personal)
    saved = prepare_saved(proposed, args.run_id, hand_specialist=args.hand_specialist)
    saved_name = "nw-" + uuid.uuid4().hex[:11] + "-v"
    if saved_name in views:
        raise RuntimeError("New saved-view ID collision; retry")
    review = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "apply_requested": args.apply,
        "entity": args.entity, "project": args.project, "run_id": args.run_id,
        "original_personal_view": personal, "proposed_personal_spec": proposed,
        "proposed_saved_spec": saved, "saved_view_name": saved_name,
        "existing_views": views,
    }
    args.backup.parent.mkdir(parents=True, exist_ok=True)
    with args.backup.open("x") as handle:
        json.dump(review, handle, indent=2)
        handle.write("\n")
    print(json.dumps({
        "action": "apply" if args.apply else "dry-run", "backup": str(args.backup),
        "first_section": "Hand-protection success" if args.hand_specialist else "Training success",
        "training_charts": 1 if args.hand_specialist else 4,
        "personal_run_filter": "unchanged", "saved_view_run_filter": args.run_id,
        "previous_evaluation_charts": "preserved and collapsed",
        "personal_url": f"https://wandb.ai/{args.entity}/{args.project}?nw=nwuser{username}",
        "saved_url": f"https://wandb.ai/{args.entity}/{args.project}?nw={saved_name[3:-2]}",
    }, indent=2))
    if not args.apply:
        return

    _, latest = read_views(api, execute_graphql, args.entity, args.project)
    if latest[args.personal_view] != personal:
        raise RuntimeError("Personal workspace changed; retry with a new backup")
    inserted = execute_graphql(api, WRITE_QUERY, {
        "id": None, "entityName": args.entity, "projectName": args.project,
        "type": "project-view", "name": saved_name,
        "displayName": f"Training success - {args.run_id}", "spec": json.dumps(saved),
    })
    if not inserted["upsertView"].get("inserted"):
        raise RuntimeError("Expected a new saved view; inspect the response")
    if args.saved_only:
        _, actual = read_views(api, execute_graphql, args.entity, args.project)
        if json.loads(actual[saved_name]["spec"]) != saved:
            raise RuntimeError("Saved workspace readback differs from the proposed raw spec")
        if any(actual.get(name) != before for name, before in views.items()):
            raise RuntimeError("A pre-existing view changed concurrently")
        print("Verified new run-filtered saved view and unchanged existing workspaces.")
        return
    _, latest = read_views(api, execute_graphql, args.entity, args.project)
    if latest[args.personal_view] != personal:
        raise RuntimeError("Personal workspace changed; saved view created, personal view untouched")
    changed = execute_graphql(api, WRITE_QUERY, {
        "id": personal["id"], "entityName": args.entity, "projectName": args.project,
        "type": "project-view", "name": personal["name"],
        "displayName": personal["displayName"], "spec": json.dumps(proposed),
    })
    if changed["upsertView"].get("inserted"):
        raise RuntimeError("Unexpected personal-view insertion; inspect the response")
    _, actual = read_views(api, execute_graphql, args.entity, args.project)
    if json.loads(actual[args.personal_view]["spec"]) != proposed:
        raise RuntimeError("Personal workspace readback differs from the proposed raw spec")
    if json.loads(actual[saved_name]["spec"]) != saved:
        raise RuntimeError("Saved workspace readback differs from the proposed raw spec")
    for name, before in views.items():
        if name != args.personal_view and actual.get(name) != before:
            raise RuntimeError(f"Pre-existing view changed concurrently: {name}")
    print("Verified personal training section, new run-filtered saved view, and unchanged legacy views.")


if __name__ == "__main__":
    main()
