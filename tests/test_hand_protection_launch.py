import hashlib
import json

import pytest

from cat_ppo.furniture.control import wholebody_observation_contract
from cat_ppo.furniture.generalist_config import training_config
from cat_ppo.furniture.protected_warmstart import verified_best_archive
from cat_ppo.furniture.retention_validation import FIXED_SCENE_IDS, validation_scene_ids


def test_hand_profile_preserves_cat_optimizer_and_leg_regularizer():
    old = training_config(finetuning="stabilized", num_envs=16384)
    new = training_config(finetuning="hand_protection", num_envs=16384)
    assert old["policy_config"] == new["policy_config"]
    assert old["env_config"] == new["env_config"]
    assert new["fine_tuning"]["reference_kl"]["action_indices"] == list(range(12))
    distribution = new["fine_tuning"]["action_distribution"]
    assert distribution["arm_last_action_indices"] == list(range(79, 93))
    assert distribution["arm_persistence"] == .95
    assert new["fine_tuning"]["effective_batch_geometry"]["parallel_environments"] == 16384


def archive_fixture(root):
    native = root / "checkpoint/native"
    native.mkdir(parents=True)
    contract = wholebody_observation_contract()
    config = training_config()["policy_config"]["network_factory"]
    (native / "observation_contract.json").write_text(json.dumps(contract))
    (native / "ppo_network_config.json").write_text(json.dumps(
        dict(normalize_observations=False, network_factory_kwargs=config)))
    (native / "weights").write_bytes(b"checkpoint data")
    files = {p.name: dict(size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in native.iterdir()}
    backup = dict(schema="cat-protected-best-backup-v1", source_run="old-run",
                  source_checkpoint_step=104857600, checkpoint_files=files)
    selection = dict(files=files, selection_source="retention_validation", step=104857600,
                     metrics=dict(selection=dict(eligible=True)))
    (root / "backup.json").write_text(json.dumps(backup))
    (root / "checkpoint/selection.json").write_text(json.dumps(selection))
    return contract, config


def test_selected_best_integrity_and_exact_feature_order(tmp_path):
    contract, config = archive_fixture(tmp_path)
    _, _, record = verified_best_archive(tmp_path, target_contract=contract, network_config=config)
    assert record["source_step"] == 104857600
    contract["actor_features"][0:2] = reversed(contract["actor_features"][0:2])
    with pytest.raises(ValueError, match="feature layout"):
        verified_best_archive(tmp_path, target_contract=contract)
    (tmp_path / "checkpoint/native/weights").write_bytes(b"corrupted data!")
    with pytest.raises(ValueError, match="integrity"):
        verified_best_archive(tmp_path)


def test_ineligible_checkpoint_cannot_initialize_hand_training(tmp_path):
    archive_fixture(tmp_path)
    selection_path = tmp_path / "checkpoint/selection.json"
    selection = json.loads(selection_path.read_text())
    selection["metrics"]["selection"]["eligible"] = False
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="eligible"):
        verified_best_archive(tmp_path)


def test_hand_validation_appends_without_changing_old_draw_ordinals():
    scenes = []
    for kind in ("hand_table_aisle", "hand_shelf_passage"):
        for level in range(3):
            for variant in range(2):
                scenes.append(dict(scene_id=f"{kind}-{level}-{variant}", source=dict(
                    hand_protection=dict(kind=kind, level=level))))
    ids = validation_scene_ids(dict(scenes=scenes), hand_protection=True)
    assert ids[:16] == FIXED_SCENE_IDS
    assert len(ids) == 22
    assert all(scene.endswith("-0") for scene in ids[16:])
    assert validation_scene_ids(dict(scenes=[])) == FIXED_SCENE_IDS
    with pytest.raises(ValueError, match="Missing hand"):
        validation_scene_ids(dict(scenes=scenes[:5]), hand_protection=True)
