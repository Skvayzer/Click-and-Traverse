"""CPU lifecycle fixtures; no policy training, GPU allocation, or W&B network."""

import json
from pathlib import Path

import pytest

import train_furniture as training
from cat_ppo.furniture.checkpoint import BestCheckpointStore


class FakeWandb:
    def __init__(self, *, finish_error=None):
        self.summary = {}
        self.exit_codes = []
        self.finish_error = finish_error

    def finish(self, *, exit_code):
        self.exit_codes.append(exit_code)
        if self.finish_error is not None:
            raise self.finish_error


@pytest.fixture
def args(tmp_path):
    return training.parser().parse_args([
        "--scene-dir", str(tmp_path / "scene"), "--run-dir", str(tmp_path / "run"),
        "--steps", "16", "--num-envs", "4", "--unroll-length", "2", "--checkpoint-epochs", "2",
        "--global-step-offset", "100"])


@pytest.mark.parametrize("message,gpu,expected", [
    ("CUDA out of memory. Tried to allocate 8.00 GiB", False, "gpu_oom"),
    ("CUDA_ERROR_OUT_OF_MEMORY", False, "gpu_oom"),
    ("GPU_0_bfc allocator ran out of memory allocating a buffer", False, "gpu_oom"),
    ("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 27078127104 bytes", True, "gpu_oom"),
    ("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 27078127104 bytes", False, "other"),
    ("CPU allocator ran out of memory while trying to allocate 1024 bytes", True, "other"),
    ("RESOURCE_EXHAUSTED: CPU out of memory while trying to allocate 1024 bytes", True, "other"),
    ("RESOURCE_EXHAUSTED: request quota exceeded", True, "other"),
    ("out of memory is not a valid configuration choice", True, "other"),
    ("GPU kernel shape mismatch", True, "other"),
])
def test_oom_classification_requires_device_allocator_evidence(message, gpu, expected):
    assert training._failure_classification(RuntimeError(message), gpu_execution=gpu) == expected


def test_host_memoryerror_and_wrapped_gpu_cause_are_distinguished():
    error = MemoryError("RESOURCE_EXHAUSTED: out of memory trying to allocate host storage")
    assert training._failure_classification(error, gpu_execution=True) == "other"
    wrapped = RuntimeError("JIT compilation failed")
    wrapped.__cause__ = RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")
    assert training._failure_classification(wrapped) == "gpu_oom"


def test_initial_gpu_oom_writes_bound_failure_before_exit_and_marks_wandb_failed(args, monkeypatch):
    wandb = FakeWandb()
    original = RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4096 bytes")
    def fail(args, stop, attempt):
        attempt.phase = "training"
        attempt.gpu_execution = True
        attempt.wandb_run = wandb
        raise original
    monkeypatch.setattr(training, "_train", fail)
    with pytest.raises(RuntimeError) as captured:
        training.train(args)
    assert captured.value is original
    record = json.loads((args.run_dir / "failure.json").read_text())
    assert record["schema"] == "cat-furniture-training-failure-v1"
    assert record["classification"] == "gpu_oom"
    assert record["stage"] == args.run_dir.name and record["run_dir"] == str(args.run_dir)
    assert record["completed_steps"] == 0 and record["completed_update_evidence"] is False
    assert record["metrics_present"] is False and record["metrics_last_step"] is None
    assert record["selected_checkpoint"] is None and record["checkpoint_inspection_error"] is None
    assert record["requested_steps"] == 16 and record["global_step_offset"] == 100
    assert wandb.exit_codes == [1]
    assert wandb.summary["failure/classification"] == "gpu_oom"


def test_setup_failure_without_wandb_still_gets_machine_readable_metadata(args, monkeypatch):
    def fail(args, stop, attempt):
        attempt.phase = "environment_setup"
        raise ValueError("invalid collision geometry")
    monkeypatch.setattr(training, "_train", fail)
    with pytest.raises(ValueError, match="geometry"):
        training.train(args)
    record = json.loads((args.run_dir / "failure.json").read_text())
    assert record["classification"] == "other" and record["phase"] == "environment_setup"


def _checkpoint(directory, *, step=8):
    store = BestCheckpointStore(directory)
    def write(path):
        path.mkdir()
        (path / "weights.bin").write_bytes(b"selected fixture actor critic normalizer")
    store.consider(step=step, metrics={"proxy_score": 1.0}, source="training_proxy",
                   write_checkpoint=write, contract={"fixture": True})
    return store.selected()


def test_progress_and_checkpoint_prevent_zero_update_failure_claim(args):
    args.run_dir.mkdir()
    selected = _checkpoint(args.run_dir, step=8)
    metrics = args.run_dir / "metrics.jsonl"
    metrics.write_text('{"step": 12, "metrics": {"reward": 1}}\n{"step": 4, "metrics": {"reward": 0}}\n')
    attempt = training._TrainingAttempt(args)
    attempt.scored_step = 8
    record = attempt.failure_record(RuntimeError("CUDA out of memory"))
    assert record["completed_steps"] == 12 and record["completed_update_evidence"]
    assert record["metrics_last_step"] == 12 and record["latest_metrics"] == {"reward": 1}
    assert record["selected_checkpoint"] == selected


def test_inspection_uncertainty_is_reported_and_blocks_zero_update_claim(args):
    args.run_dir.mkdir()
    selected = _checkpoint(args.run_dir)
    (Path(selected["path"]) / "weights.bin").write_bytes(b"corrupted")
    (args.run_dir / "metrics.jsonl").write_text('{"step":')
    record = training._TrainingAttempt(args).failure_record(RuntimeError("CUDA out of memory"))
    assert record["completed_update_evidence"]
    assert record["selected_checkpoint"] is None and record["checkpoint_inspection_error"]
    assert record["metrics_present"] and record["metrics_inspection_error"]


@pytest.mark.parametrize("phase", ["training", "export", "summary"])
def test_wandb_finish_cleanup_never_masks_original_error(args, monkeypatch, phase):
    wandb = FakeWandb(finish_error=ConnectionError("logging service unavailable"))
    original = RuntimeError(f"{phase} failed")
    def fail(args, stop, attempt):
        attempt.phase = phase
        attempt.final_steps = 16 if phase != "training" else 0
        attempt.wandb_run = wandb
        raise original
    monkeypatch.setattr(training, "_train", fail)
    with pytest.raises(RuntimeError) as captured:
        training.train(args)
    assert captured.value is original and wandb.exit_codes == [1]
    record = json.loads((args.run_dir / "failure.json").read_text())
    assert record["phase"] == phase and record["classification"] == "other"


def test_failure_metadata_write_error_does_not_mask_training_error(args, monkeypatch):
    wandb, original = FakeWandb(), RuntimeError("original training failure")
    def fail(args, stop, attempt):
        attempt.wandb_run = wandb
        raise original
    def disk_full(path, record):
        raise OSError("disk full")
    monkeypatch.setattr(training, "_train", fail)
    monkeypatch.setattr(training, "_atomic_failure_json", disk_full)
    with pytest.raises(RuntimeError) as captured:
        training.train(args)
    assert captured.value is original and wandb.exit_codes == [1]


@pytest.mark.parametrize("manual_stop", [False, True])
def test_valid_saved_completion_and_manual_stop_finish_successfully(args, monkeypatch, manual_stop):
    wandb = FakeWandb()
    expected = {"actual_steps": 8 if manual_stop else 16, "stopped_by_request": manual_stop}
    def complete(args, stop, attempt):
        attempt.wandb_run = wandb
        (args.run_dir / "summary.json").write_text(json.dumps(expected))
        attempt.phase = "complete"
        return expected
    monkeypatch.setattr(training, "_train", complete)
    assert training.train(args) == expected
    assert wandb.exit_codes == [0] and not (args.run_dir / "failure.json").exists()


def test_return_before_summary_completion_is_failure(args, monkeypatch):
    wandb = FakeWandb()
    def incomplete(args, stop, attempt):
        attempt.wandb_run = wandb
        attempt.phase = "export"
        return {}
    monkeypatch.setattr(training, "_train", incomplete)
    with pytest.raises(RuntimeError, match="before export and summary"):
        training.train(args)
    assert wandb.exit_codes == [1]


def test_existing_run_is_not_modified_by_rejected_attempt(args, monkeypatch):
    args.run_dir.mkdir()
    existing = args.run_dir / "run.json"
    existing.write_text('{"historical": true}')
    monkeypatch.setattr(training, "_train", lambda *values: pytest.fail("must not start"))
    with pytest.raises(ValueError, match="never overwritten"):
        training.train(args)
    assert existing.read_text() == '{"historical": true}'
    assert not (args.run_dir / "failure.json").exists()
