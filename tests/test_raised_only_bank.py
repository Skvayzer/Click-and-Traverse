"""The successor must change only the intended alternative and retain strict pins."""
import copy
import json
from pathlib import Path
import pytest
from cat_ppo.furniture.raised_only_bank import raised_only, validate
from cat_ppo.furniture.generalist_fields import load_generalist_manifest
from cat_ppo.furniture.balance_bank import sampling_plan

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT/'data/furniture/cat_flat_hand_balance_v3_20260921/manifest.json'
OLD = ROOT/'data/furniture/cat_flat_hand_balance_v2_20260921/manifest.json'


def test_height_selection_survives_reordered_regions():
    m = json.loads(OLD.read_text())
    r = next(r for r in m['scenes'] if r.get('source', {}).get('hand_contrast', {}).get('role') == 'forward_protected')
    scene = json.loads((OLD.parent/r['path']/'scene.json').read_text())
    scene['hand_contrast']['mode_names'].reverse()
    for z in scene['hand_contrast']['zones']:
        for key in ('hand_regions_min', 'hand_regions_max'):
            z[key].reverse()
    changed = raised_only(scene)
    assert all(z['region_valid'] == [False, True] for z in changed['hand_contrast']['zones'])
    assert all(z['region_valid'] == [True, True] for z in scene['hand_contrast']['zones'])


def test_v3_preserves_sampling_and_rejects_unrelated_manifest_changes():
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
