"""Adding table pairs must preserve cabinet groups and existing scene roles."""
from copy import deepcopy

import jax
import jax.numpy as jp
import numpy as np
import pytest

from cat_ppo.furniture.contrastive_bank import (
    TABLE_MARKER, ROLES, contrastive_roles, role_balanced_logits, validate_contrastive_manifest,
)
from cat_ppo.furniture.generalist_fields import _json_hash


@pytest.fixture(scope="module")
def manifest():
    from cat_ppo.furniture.contrastive_passages import generate_contrastive_group
    from cat_ppo.furniture.table_edge_passages import generate_table_edge_pair
    cabinets = generate_contrastive_group(20260919)
    tables = generate_table_edge_pair(20260919)
    records = [dict(family="generic_clutter", dx=.04, source={"hand_contrast": scene["hand_contrast"]})
               for scene in cabinets + tables]
    return dict(schema="cat-generalist-field-bank-v2", original_count=0,
        byte_verified_original_count=0, reconstructed_original_count=0, scenes=records,
        contrastive_specialist=dict(schema=TABLE_MARKER, group_count=2,
            role_reset_masses=dict.fromkeys(ROLES, .25),
            geometry_group_counts=dict(cabinet=1, table_edges=1),
            preserved_cabinet_scene_count=4, preserved_scene_records_sha256=_json_hash(records[:4])))


def test_add_table_pairs_without_changing_narrow_or_transition_role_mass(manifest):
    roles = contrastive_roles(manifest)
    np.testing.assert_array_equal(roles, [0, 1, 2, 3, 0, 1])
    probabilities = np.asarray(jax.nn.softmax(role_balanced_logits(jp.ones(6), jp.asarray(roles))))
    np.testing.assert_allclose(np.bincount(roles, weights=probabilities), .25, atol=1e-7)
    np.testing.assert_allclose(probabilities[4:].sum(), .25, atol=1e-7)


@pytest.mark.parametrize("mutation", [
    lambda m: m["scenes"].pop(),
    lambda m: m["scenes"][5]["source"]["hand_contrast"]["zones"][0].update(region_valid=[True, True]),
    lambda m: m["scenes"][5]["source"]["hand_contrast"]["certificate"].update(raised_hand_above_table_min_m=0.),
    lambda m: m["scenes"][5]["source"]["hand_contrast"]["certificate"].update(tabletop_geometry_validated=False),
    lambda m: m["contrastive_specialist"].update(preserved_scene_records_sha256="a" * 64),
    lambda m: m["contrastive_specialist"].update(geometry_group_counts=dict(cabinet=0, table_edges=2)),
    lambda m: m["scenes"][5]["source"]["hand_contrast"].update(geometry_family="unknown"),
])
def test_invalid_table_extensions_fail_closed(manifest, mutation):
    candidate = deepcopy(manifest)
    mutation(candidate)
    with pytest.raises(ValueError):
        validate_contrastive_manifest(candidate)
