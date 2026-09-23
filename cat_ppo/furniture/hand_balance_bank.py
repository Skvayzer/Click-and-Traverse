"""Explicit successor mixture; original balance bank semantics stay immutable."""
from pathlib import Path
import numpy as np

SCHEMA = 'cat-flat-hand-balance-v2'
MASSES = (.18, .36, .05625, .03375, .045, .225, .07, .03)
COMPOSITION = dict(retention=2338, narrow=12, flat=1, forward_protected=12, transition=12)


def objective_indices(manifest):
    return [i for i, r in enumerate(manifest['scenes'])
            if r.get('source', {}).get('hand_contrast', {}).get('role') in ('forward_protected', 'transition')
            and r['source']['hand_contrast']['curriculum_rung'] <= 2]


def validate(manifest, *, path, verify_files=False):
    from .generalist_fields import load_generalist_manifest, sha256
    from .balance_bank import SETTINGS
    marker = manifest['flat_balance']
    if (marker['schema'] != SCHEMA or marker['masses'] != list(MASSES)
            or marker['settings'] != SETTINGS or marker['composition'] != COMPOSITION
            or marker.get('require_hand_contrast') is not True):
        raise ValueError('Invalid v2 hand-balance mixture/settings')
    if any(k in manifest for k in ('width_curriculum', 'hand_protection_curriculum',
            'hand_posture_upgrade', 'contrastive_specialist', 'specialist', 'external_retention_bank')):
        raise ValueError('Hand balance cannot inherit another sampler')
    parents = []
    for key in ('base_source', 'objective_source'):
        pin = marker[key]; p = Path(pin['manifest'])
        if sha256(p) != pin['sha256']:
            raise ValueError('Hand balance source pin differs')
        parents.append(load_generalist_manifest(p, verify_files=False))
    base, objective = parents
    if base['flat_balance']['schema'] != 'cat-flat-balance-v1':
        raise ValueError('Expected original fixed balance bank')
    added = [objective['scenes'][i] for i in objective_indices(objective)]
    if len(added) != 24 or manifest['scenes'] != base['scenes'] + added:
        raise ValueError('Hand balance must preserve all base records and the 24 pinned objectives')
    if marker['retention_storage_root'] != base['flat_balance']['retention_storage_root']:
        raise ValueError('Retention storage root differs')
    if len({r['scene_id'] for r in manifest['scenes']}) != len(manifest['scenes']):
        raise ValueError('Duplicate scene identity')
    return marker


def sampling_plan(manifest):
    from .generalist_fields import SAMPLING_GROUPS
    records = manifest['scenes']
    ids = [SAMPLING_GROUPS.index(r['sampling_group']) for r in records[:2338]] + [4]*12 + [5]
    ids += [6 if r['source']['hand_contrast']['role'] == 'forward_protected' else 7 for r in records[2351:]]
    return np.asarray(ids, np.int64), np.asarray(MASSES, np.float32)
