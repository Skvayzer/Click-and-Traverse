import json
import os
from pathlib import Path

import pytest

from cat_ppo.furniture.checkpoint import BestCheckpointStore, selection_score

CONTRACT = {"action_names": ["a"], "actor_features": ["x"], "critic_features": ["v"]}


def writer(payload):
    def write(path):
        path.mkdir()
        (path/"weights.bin").write_bytes(payload)
        (path/"ppo_network_config.json").write_text("{}")
    return write


def submit(store, step, reward, payload=b"weights"):
    return store.consider(step=step, metrics={"proxy_score": reward}, source="training_proxy",
                          write_checkpoint=writer(payload), contract=CONTRACT)


def test_best_checkpoint_preserves_ties_and_worse_final(tmp_path):
    store = BestCheckpointStore(tmp_path)
    assert submit(store, 10, 2, b"first")
    assert not submit(store, 20, 2, b"tie")
    assert not submit(store, 30, 1, b"final worse")
    assert store.selected()["step"] == 10
    assert submit(store, 40, 3, b"improved")
    selected = store.selected()
    assert (Path(selected["path"])/"weights.bin").read_bytes() == b"improved"
    assert len(list(store.generations.glob("candidate-*"))) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_score_cannot_replace_best(tmp_path, value):
    store = BestCheckpointStore(tmp_path)
    submit(store, 10, 1)
    with pytest.raises(ValueError, match="finite"):
        submit(store, 20, value)
    assert store.selected()["step"] == 10


def test_validation_safety_gates_precede_success_and_time():
    unsafe, _ = selection_score(dict(fall_rate=.01, contact_rate=0, strict_success_rate=.99, completion_time=1), "validation")
    safe, rules = selection_score(dict(fall_rate=0, contact_rate=0, strict_success_rate=.2, completion_time=5), "validation")
    faster, _ = selection_score(dict(fall_rate=0, contact_rate=0, strict_success_rate=.2, completion_time=4), "validation")
    assert unsafe < safe < faster and rules["eligible"]


def test_partial_write_keeps_previous_and_cleans_candidate(tmp_path):
    store = BestCheckpointStore(tmp_path);submit(store, 10, 1, b"good")
    def broken(path):
        path.mkdir();(path/"weights.bin").write_bytes(b"partial")
        raise OSError("interrupted write")
    with pytest.raises(OSError):
        store.consider(step=20, metrics={"proxy_score": 2}, source="training_proxy", write_checkpoint=broken, contract=CONTRACT)
    assert store.selected()["step"] == 10
    assert len(list(store.generations.glob("candidate-*"))) == 1


def test_pointer_failure_keeps_model_and_metadata_together(tmp_path, monkeypatch):
    store = BestCheckpointStore(tmp_path);submit(store, 10, 1)
    def fail(*args):
        raise OSError("rename failed")
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        submit(store, 20, 2)
    assert store.selected()["step"] == 10


def test_signal_immediately_after_publication_preserves_new_generation(tmp_path, monkeypatch):
    store = BestCheckpointStore(tmp_path);submit(store, 10, 1)
    original = os.replace
    def interrupted(src, dst):
        original(src, dst)
        raise KeyboardInterrupt()
    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        submit(store, 20, 2, b"new")
    assert store.selected()["step"] == 20
    reopened = BestCheckpointStore(tmp_path)
    assert reopened.selected()["step"] == 20
    assert len(list(reopened.generations.glob("candidate-*"))) == 1


def test_existing_path_and_external_symlink_are_refused(tmp_path):
    checkpoint_dir = tmp_path/"checkpoints";checkpoint_dir.mkdir()
    (checkpoint_dir/"best").write_text("user file")
    with pytest.raises(ValueError, match="not this store"):
        BestCheckpointStore(tmp_path)
    assert (checkpoint_dir/"best").read_text() == "user file"
    (checkpoint_dir/"best").unlink();(checkpoint_dir/"best").symlink_to(tmp_path/"outside")
    with pytest.raises((ValueError, FileNotFoundError)):
        BestCheckpointStore(tmp_path)


def test_hash_mismatch_is_detected_and_never_relabelled(tmp_path):
    store = BestCheckpointStore(tmp_path);submit(store, 10, 1)
    native = Path(store.selected()["path"])
    (native/"weights.bin").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="hash manifest"):
        store.selected()
    with pytest.raises(ValueError, match="hash manifest"):
        submit(store, 20, 2)


def test_validation_and_training_proxy_cannot_be_mixed(tmp_path):
    store = BestCheckpointStore(tmp_path);submit(store, 10, 1)
    with pytest.raises(ValueError, match="Cannot compare"):
        store.consider(step=20, metrics=dict(fall_rate=0, contact_rate=0, strict_success_rate=1, completion_time=3),
                       source="validation", write_checkpoint=writer(b"new"), contract=CONTRACT)


def test_unknown_generation_is_preserved(tmp_path):
    store = BestCheckpointStore(tmp_path);submit(store, 10, 1)
    other = store.generations/"candidate-user-data";other.mkdir()
    (other/"owner.json").write_text(json.dumps({"owner": "not this run"}))
    (other/"valuable.txt").write_text("keep")
    submit(store, 20, 2)
    assert (other/"valuable.txt").read_text() == "keep"
