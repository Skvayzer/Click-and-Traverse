"""Role sampling and first-outcome posture accounting under JIT."""
from copy import deepcopy

import jax
import jax.numpy as jp
import numpy as np
import pytest

from cat_ppo.furniture.contrastive_bank import role_balanced_logits
from cat_ppo.furniture.contrastive_metrics import (
    COUNTS, STEPS, initial_contrast_outcomes, update_contrast_outcomes,
    contrast_count_snapshot, contrast_rollout_metrics,
)
from cat_ppo.furniture.generalist_logging import compact_metric_allowed


def test_adaptive_sampling_keeps_each_role_at_one_quarter():
    roles = jp.array([0, 1, 1, 2, 2, 2, 3])
    weights = jp.array([1., .01, 1., .001, .2, .9, .3])
    probabilities = np.asarray(jax.nn.softmax(jax.jit(role_balanced_logits)(weights, roles)))
    np.testing.assert_allclose(np.bincount(roles, weights=probabilities), .25, atol=1e-7)
    assert probabilities[2] > probabilities[1]


def _info():
    info = initial_contrast_outcomes((3,))
    info['wholebody_navigation_outcome_counted'] = jp.ones(3, bool)
    required = jp.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]], bool)
    info['hand_contrast'] = dict(core_active=jp.ones(3, bool), zone_index=jp.full(3, 2),
        required_forward_zones=required, required_hand_zones=required, role=jp.array([1, 2, 3]))
    info['wholebody_telemetry'] = dict(hand_contrast_heading_good=jp.ones(3), hand_contrast_hand_good=jp.ones(3))
    info[STEPS] = jp.zeros((3, 6, 3), jp.int32).at[:, :3].set(10)
    return info


def _update(info, resolved, successful):
    update_contrast_outcomes(info, resolved, successful)
    return info


def test_success_requires_posture_coverage_not_only_goal_and_counts_once():
    info = _info()
    # First protected episode skips one entire hazard; transition episode turns
    # sideways in its forward zone. Narrow task has no heading requirement.
    info[STEPS] = info[STEPS].at[0, 0].set(0).at[2, 0, 1].set(0)
    result = jax.jit(_update)(info, jp.ones(3, bool), jp.ones(3, bool))
    np.testing.assert_array_equal(contrast_count_snapshot(result), [[1, 0], [1, 1], [1, 0]])
    again = jax.jit(_update)(result, jp.zeros(3, bool), jp.zeros(3, bool))
    np.testing.assert_array_equal(again[COUNTS], result[COUNTS])
    np.testing.assert_array_equal(again[STEPS], result[STEPS])


def test_clean_posture_success_and_failures_have_same_denominator():
    result = jax.jit(_update)(_info(), jp.ones(3, bool), jp.array([True, False, True]))
    np.testing.assert_array_equal(contrast_count_snapshot(result), [[1, 1], [1, 0], [1, 1]])
    rates = contrast_rollout_metrics(np.zeros((3, 2), np.int32), contrast_count_snapshot(result))
    assert rates['success/forward_protected_success_rate'] == 1.
    assert rates['success/narrow_passage_success_rate'] == 0.
    assert all(compact_metric_allowed(name) for name in rates)
    empty = contrast_rollout_metrics(np.zeros((3, 2), np.int32), np.zeros((3, 2), np.int32))
    assert not any(name.endswith('_rate') for name in empty)


def test_qualification_tolerates_at_most_ten_percent_bad_posture():
    info = _info()
    info['hand_contrast']['core_active'] = jp.zeros(3, bool)
    info[STEPS] = info[STEPS].at[0, 0, 2].set(9).at[2, 0, 2].set(8)
    result = jax.jit(_update)(info, jp.ones(3, bool), jp.ones(3, bool))
    np.testing.assert_array_equal(contrast_count_snapshot(result), [[1, 1], [1, 1], [1, 0]])


def test_contrastive_workspace_has_four_explicit_training_charts():
    from scripts.configure_training_success_workspace import prepare_saved, CONTRAST_CHARTS, training_section
    personal = {'section': {'panelBankConfig': {'sections': [training_section()]}}}
    saved = prepare_saved(personal, 'testrun', contrastive=True)
    section = saved['section']['panelBankConfig']['sections'][0]
    assert section['name'] == 'Contrastive training success'
    assert [p['config']['metrics'] for p in section['panels']] == [[name] for name, _ in CONTRAST_CHARTS]
    assert personal['section']['panelBankConfig']['sections'][0]['name'] == 'Training success'


@pytest.fixture(scope='module')
def contrast_manifest():
    from cat_ppo.furniture.contrastive_bank import MARKER, ROLES
    from cat_ppo.furniture.contrastive_passages import generate_contrastive_group
    scenes = generate_contrastive_group(20260919)
    return dict(schema='cat-generalist-field-bank-v2', original_count=0,
        byte_verified_original_count=0, reconstructed_original_count=0,
        contrastive_specialist=dict(schema=MARKER, role_reset_masses=dict.fromkeys(ROLES, .25), group_count=1),
        scenes=[dict(family='generic_clutter', dx=.04, source={'hand_contrast': s['hand_contrast']}) for s in scenes])


def test_contrastive_bank_is_explicit_and_requires_complete_groups(contrast_manifest):
    from cat_ppo.furniture.contrastive_bank import contrastive_roles, validate_contrastive_manifest
    np.testing.assert_array_equal(contrastive_roles(contrast_manifest), [0, 1, 2, 3])
    for mutate in (lambda m: m['scenes'].pop(),
                   lambda m: m['scenes'].append(m['scenes'][0]),
                   lambda m: m.update(original_count=37),
                   lambda m: m.update(specialist={}),
                   lambda m: m['scenes'][0]['source']['hand_contrast'].pop('certificate')):
        malformed = deepcopy(contrast_manifest)
        mutate(malformed)
        with pytest.raises(ValueError):
            validate_contrastive_manifest(malformed)
    assert contrastive_roles({'scenes': []}) is None
