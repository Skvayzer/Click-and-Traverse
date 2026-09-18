"""Capacity checker resource accounting and assigned-GPU diagnostics, without CUDA."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/verify_sapg_training.py"
SPEC = importlib.util.spec_from_file_location("verify_sapg_training", PATH)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.mark.parametrize("flags,expected,transitions", [
    ([], (12, 6, 4, 2, 2), 48),
    (["--production-batch", "--num-envs", "24576", "--batch-size", "384"],
     (24576, 384, 32, 64, 4), 786432),
    (["--production-batch", "--num-envs", "36864", "--batch-size", "576"],
     (36864, 576, 32, 64, 4), 1179648),
])
def test_actual_physical_batch_geometry(flags, expected, transitions):
    args = checker.parser().parse_args(["--report", "report.json", *flags])
    settings = checker.resource_settings(args)
    assert tuple(settings[key] for key in ("num_envs", "batch_size", "unroll_length",
                                           "num_minibatches", "num_updates_per_batch")) == expected
    assert settings["batch_size"] * settings["unroll_length"] * settings["num_minibatches"] == transitions
    assert args.updates == 2 and args.min_free_vram_mib == 0.


@pytest.mark.parametrize("flags", [
    ["--num-envs", "24576", "--batch-size", "384"],
    ["--num-envs", "13"], ["--batch-size", "5"],
    ["--min-free-vram-mib", "nan"], ["--min-free-vram-mib", "-1"], ["--updates", "1"],
])
def test_rejects_invalid_or_mismatched_capacity_geometry(flags):
    args = checker.parser().parse_args(["--report", "report.json", *flags])
    with pytest.raises(ValueError):
        checker.resource_settings(args)


def test_all_three_specialist_manifest_overrides_are_retained():
    args = checker.parser().parse_args([
        "--report", "report.json", "--bank-manifest", "/task/fields/manifest.json",
        "--body-collision-bank", "/task/collision/manifest.json",
        "--body-collision-resets", "/task/resets/manifest.json",
    ])
    assert args.bank_manifest == Path("/task/fields/manifest.json")
    assert args.body_collision_bank == Path("/task/collision/manifest.json")
    assert args.body_collision_resets == Path("/task/resets/manifest.json")


def test_gpu_uuid_is_resolved_from_own_pid_and_never_default_or_visible_device_index(monkeypatch):
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    def query(command, **kwargs):
        calls.append(command)
        assert kwargs == dict(text=True, timeout=15)
        if len(calls) == 1:
            return "999, GPU-not-ours, 30000\n123, GPU-assigned-seven, 28000\n"
        return "GPU-assigned-seven, NVIDIA RTX 6000 Ada Generation, 49140, 29000, 20140, 97\n"

    monkeypatch.setattr(checker.subprocess, "check_output", query)
    sample = checker.assigned_gpu_sample(pid=123)
    assert sample["uuid"] == "GPU-assigned-seven"
    assert sample["used_mib"] == 29000 and sample["free_mib"] == 20140
    assert sample["process_memory_records"] == [{"uuid": "GPU-assigned-seven", "process_used_mib": 28000.}]
    assert "--id=GPU-assigned-seven" in calls[1]
    assert checker.os.environ["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.parametrize("apps", ["999, GPU-other, 1000\n", "123, GPU-a, 1000\n123, GPU-b, 1000\n"])
def test_ambiguous_or_missing_own_process_gpu_is_not_silently_replaced(monkeypatch, apps):
    monkeypatch.setattr(checker.subprocess, "check_output", lambda *args, **kwargs: apps)
    with pytest.raises(RuntimeError, match="one assigned GPU"):
        checker.assigned_gpu_sample(pid=123)


def test_rejects_device_query_with_different_uuid(monkeypatch):
    output = iter(["123, GPU-a, 1000\n", "GPU-b, Device, 2000, 1000, 1000, 90\n"])
    monkeypatch.setattr(checker.subprocess, "check_output", lambda *args, **kwargs: next(output))
    with pytest.raises(RuntimeError, match="UUID could not be verified"):
        checker.assigned_gpu_sample(pid=123)


def test_completed_update_memory_and_threshold_use_only_completed_callbacks(monkeypatch):
    calls = []
    monkeypatch.setattr(checker, "assigned_gpu_sample",
                        lambda: calls.append(True) or {"uuid": "GPU-assigned", "free_mib": 1800.})
    device = SimpleNamespace(memory_stats=lambda: {"bytes_in_use": 1000, "peak_bytes_in_use": 2000})
    assert checker.completed_update_record(48, {"training/goal_success_rate": .3}, device,
        started=0., min_free_vram_mib=2000.) is None
    assert calls == []
    record = checker.completed_update_record(786432,
        {"training/sps": 40000., "training/rollout_reward_mean": .7}, device,
        started=0., min_free_vram_mib=2000.)
    assert record["metrics_finite"] and record["reward_finite"] and record["sps_positive"]
    assert not record["free_vram_passes"] and record["gpu"]["free_mib"] == 1800.
    assert record["jax_memory"] == {"bytes_in_use": 1000, "peak_bytes_in_use": 2000}
    assert record["step"] == 786432 and calls == [True]


def test_nonfinite_progress_is_reportable_strict_json_without_false_pass(monkeypatch):
    monkeypatch.setattr(checker, "assigned_gpu_sample", lambda: {"free_mib": 3000.})
    device = SimpleNamespace(memory_stats=lambda: None)
    record = checker.completed_update_record(48,
        {"training/sps": float("inf"), "training/rollout_reward_mean": float("nan")}, device,
        started=0., min_free_vram_mib=2000.)
    assert not record["metrics_finite"] and not record["reward_finite"] and not record["sps_positive"]
    assert record["nonfinite_metrics"] == {"training/sps": "inf", "training/rollout_reward_mean": "nan"}
    assert json.loads(json.dumps(record, allow_nan=False))["metrics"]["training/rollout_reward_mean"] is None


def test_failure_report_is_saved_without_claiming_capacity_pass(monkeypatch, tmp_path):
    target = tmp_path / "report.json"
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(checker.sys, "argv", [str(PATH), "--report", str(target)])

    def failure(args, report, settings, start):
        report["updates"].append({"step": 48, "free_vram_passes": False})
        raise RuntimeError("Assigned GPU has insufficient free VRAM")

    monkeypatch.setattr(checker, "run_check", failure)
    with pytest.raises(RuntimeError, match="insufficient free VRAM"):
        checker.main()
    report = json.loads(target.read_text())
    assert report["status"] == "failed" and report["error_type"] == "RuntimeError"
    assert report["updates"] == [{"step": 48, "free_vram_passes": False}]
    assert report["wandb_initialized"] is False and report["evaluation_enabled"] is False
