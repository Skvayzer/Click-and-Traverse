"""The successor must change only the intended alternative and retain strict pins."""
import copy
import json
from pathlib import Path
import pytest
from cat_ppo.furniture.balanced_region_bank import validate
from cat_ppo.furniture.generalist_fields import load_generalist_manifest
from cat_ppo.furniture.balance_bank import sampling_plan

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT/'data/furniture/cat_flat_hand_balance_v5_20260921/manifest.json'
OLD = ROOT/'data/furniture/cat_flat_hand_balance_v3_20260921/manifest.json'


def test_v5_preserves_sampling_and_rejects_unrelated_manifest_changes():
    old = load_generalist_manifest(OLD, verify_files=False)
    new = load_generalist_manifest(NEW, verify_files=False)
    for a,b in zip(sampling_plan(old), sampling_plan(new)):
        assert (a == b).all()
    for mutate in (lambda m: m['scenes'].reverse(),
                   lambda m: m['flat_balance']['settings'].__setitem__('tolerance', .1),
                   lambda m: m['scenes'][0].__setitem__('scene_sha256', 'bad')):
        bad = copy.deepcopy(new)
        mutate(bad)
        with pytest.raises(ValueError):
            validate(bad, path=NEW)
