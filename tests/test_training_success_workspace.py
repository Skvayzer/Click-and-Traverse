import copy
import json

import pytest

from scripts.configure_training_success_workspace import (
    SUCCESS_CHARTS, prepare_personal, prepare_saved,
)


@pytest.fixture
def personal():
    panels = [
        {"__id__": f"old-{i}", "config": {"metrics": [f"validation/group_{i}"],
                                          "privateBrowserOption": i}}
        for i in range(6)
    ] + [{"__id__": "old-training", "config": {"metrics": ["training/goal_success_rate"]}}]
    return {"id": "existing-personal", "name": "nw-nwuserexample-w", "spec": json.dumps({
        "unknownTopLevel": {"preserve": True},
        "section": {
            "runSets": [{"filters": {"custom": "keep"}, "selections": ["old-run"]}],
            "workspaceSettings": {"unknownBrowserSetting": [1, 2]},
            "panelBankSectionConfig": {"panels": [{"old": "report"}], "isOpen": True},
            "panelBankConfig": {"settings": {"unknown": "preserve"}, "sections": [
                {"name": "Success rates", "isOpen": True, "panels": panels},
                {"name": "Auto metrics", "isOpen": True, "isPanelsAuto": True, "panels": []},
            ]},
        },
    })}


def test_personal_retains_raw_fields_filters_and_all_seven_legacy_charts(personal):
    untouched = copy.deepcopy(personal)
    original, proposed = prepare_personal(personal)
    assert personal == untouched
    new_sections = proposed["section"]["panelBankConfig"]["sections"]
    first, history, automatic = new_sections
    assert first["name"] == "Training success" and first["isOpen"] and first["pinned"]
    assert [p["config"]["metrics"] for p in first["panels"]] == [[m] for m, _ in SUCCESS_CHARTS]
    assert all(p["config"]["yAxisMin"] == 0 and p["config"]["yAxisMax"] == 1 for p in first["panels"])
    assert history["name"] == "Previous evaluation results" and not history["isOpen"]
    assert history["panels"] == original["section"]["panelBankConfig"]["sections"][0]["panels"]
    assert automatic["isPanelsAuto"] is True and not automatic["isOpen"]
    restored = copy.deepcopy(proposed)
    restored["section"]["panelBankConfig"]["sections"] = original["section"]["panelBankConfig"]["sections"]
    assert restored == original


def test_saved_view_has_only_four_training_charts_and_exact_new_run_filter(personal):
    _, proposed = prepare_personal(personal)
    saved = prepare_saved(proposed, "new12345")
    assert saved["section"]["settings"]["shouldAutoGeneratePanels"] is False
    sections = saved["section"]["panelBankConfig"]["sections"]
    assert len(sections) == 1 and len(sections[0]["panels"]) == 4
    assert all(m.startswith("training/") for p in sections[0]["panels"] for m in p["config"]["metrics"])
    assert saved["section"]["panelBankSectionConfig"]["panels"] == []
    assert saved["section"]["runSets"][0]["filters"]["filters"] == [{
        "key": {"section": "run", "name": "name"}, "op": "=",
        "value": "new12345", "disabled": False,
    }]
    assert proposed["section"]["runSets"] == json.loads(personal["spec"])["section"]["runSets"]


def test_personal_repeated_application_is_idempotent(personal):
    _, first = prepare_personal(personal)
    _, second = prepare_personal({**personal, "spec": json.dumps(first)})
    assert second == first


def test_unexpected_existing_success_section_is_not_replaced(personal):
    malformed = json.loads(personal["spec"])
    malformed["section"]["panelBankConfig"]["sections"][0]["panels"].pop()
    with pytest.raises(RuntimeError, match="seven existing"):
        prepare_personal({**personal, "spec": json.dumps(malformed)})
