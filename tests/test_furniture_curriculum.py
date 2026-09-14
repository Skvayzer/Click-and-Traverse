import json
import hashlib
from pathlib import Path

import pytest

from cat_ppo.furniture.curriculum import build_plan, load_plan, _retire_owned_stage
from cat_ppo.furniture.scenes import _digest


def test_plan_rehearses_cat_through_dense_and_corrupted_stages(tmp_path):
    plan = build_plan(steps_per_stage=1024, rounds=3)
    assert len(plan["stages"]) == 36
    assert plan["total_steps"] == 36864
    for first, second in zip(plan["stages"][::2], plan["stages"][1::2]):
        assert first["scene"]["domain"] == "legacy"
        assert second["scene"]["domain"] in ("generic", "furniture")
        assert first["environment_config"]["action_dofs"] == second["environment_config"]["action_dofs"] == 29
    assert plan["stages"][-1]["scene"]["difficulty"] == "dense"
    assert plan["stages"][-1]["environment_config"]["map_latency_steps"] == 3
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert load_plan(path) == plan
    plan["stages"][0]["scene"]["split"] = "test"
    plan["sha256"] = _digest({key: value for key, value in plan.items() if key != "sha256"})
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Only training"):
        load_plan(path)


def test_retirement_preserves_run_records_and_refuses_symlink(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "run.json").write_text("{}")
    (stage / "summary.json").write_text("{}")
    receipt = {"stage_name": stage.name,
               "run_sha256": hashlib.sha256((stage / "run.json").read_bytes()).hexdigest(),
               "summary_sha256": hashlib.sha256((stage / "summary.json").read_bytes()).hexdigest()}
    (stage / "curriculum_receipt.json").write_text(json.dumps(receipt))
    receipt_hash = hashlib.sha256((stage / "curriculum_receipt.json").read_bytes()).hexdigest()
    for name in (".checkpoint-generations", "checkpoints", "export"):
        (stage / name).mkdir()
        (stage / name / "weights").write_bytes(b"model")
    _retire_owned_stage(stage, receipt_sha256=receipt_hash)
    assert (stage / "summary.json").exists()
    assert not (stage / ".checkpoint-generations").exists()
    assert (stage / "model_retired.json").exists()
    symlink = tmp_path / "link"
    symlink.symlink_to(stage)
    with pytest.raises(ValueError, match="unrecognized"):
        _retire_owned_stage(symlink, receipt_sha256=receipt_hash)


def test_budget_is_never_rounded_up():
    with pytest.raises(ValueError, match="divisible"):
        build_plan(steps_per_stage=1025)
