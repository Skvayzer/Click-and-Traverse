"""CPU regression checks for objective coverage and immutable scene composition."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from cat_ppo.furniture.generalist_fields import load_generalist_manifest
from cat_ppo.furniture.balance_bank import validate_balance_manifest, sampling_plan
from cat_ppo.furniture.contrast_preflight import contrast_preflight

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'data/furniture/cat_flat_balance_v1_20260920/manifest.json'
NEW = ROOT/'data/furniture/cat_flat_hand_balance_v2_20260921/manifest.json'


def test_zero_coverage_fails_explicit_and_automatic_requests():
    for request in (True, None):
        with pytest.raises(ValueError, match='coverage is zero'):
            contrast_preflight(OLD, require_hand_contrast=request)
    from cat_mjlab.runner import create_task
    # No device or simulation arguments: the guard must fail before touching them.
    with pytest.raises(ValueError, match='coverage is zero'):
        create_task(SimpleNamespace(bank_manifest=OLD, require_hand_contrast=True))


def test_coverage_and_actual_sampler_preserve_category_masses():
    result = contrast_preflight(NEW, require_hand_contrast=True)
    assert result['active_hand_objective_zones'] == result['positive_heading_zones'] == 60
    from cat_mjlab.scene_bank import SceneBank
    manifest = load_generalist_manifest(NEW, verify_files=False)
    ids, masses = sampling_plan(manifest)
    bank = SceneBank.__new__(SceneBank)
    bank.roles = torch.zeros(len(ids), dtype=torch.long)
    bank.width_levels = None
    bank.sampling_ids = torch.tensor(ids)
    bank.sampling_masses = torch.tensor(masses)
    bank.weights = torch.linspace(.001, 1, len(ids))
    probabilities = bank.probabilities()
    measured = torch.zeros_like(bank.sampling_masses).scatter_add_(0, bank.sampling_ids, probabilities)
    torch.testing.assert_close(measured, bank.sampling_masses)
    np.testing.assert_array_equal(np.bincount(ids), [64, 2226, 24, 24, 12, 1, 12, 12])
    assert probabilities.sum().item() == pytest.approx(1)


def test_mutations_cannot_silently_replace_composition():
    manifest = load_generalist_manifest(NEW, verify_files=False)
    for mutate in (lambda m: m['scenes'].pop(),
                   lambda m: m['flat_balance']['masses'].__setitem__(6, 0),
                   lambda m: m['scenes'][-1]['source']['hand_contrast'].__setitem__('role', 'narrow')):
        bad = copy.deepcopy(manifest); mutate(bad)
        with pytest.raises(ValueError):
            validate_balance_manifest(bad, path=NEW)
