"""Mixture-only v6 successor; all v5 scene and reward records stay exact."""
import copy
from pathlib import Path
import numpy as np

SCHEMA = 'cat-protected-heavy-v6'
# Preserve relative non-hand bucket masses, giving 45% to protected/transition.
MASSES = tuple(x * .55 / .90 for x in (.18, .36, .05625, .03375, .045, .225)) + (.30, .15)


def validate(manifest, *, path):
    from .generalist_fields import load_generalist_manifest, sha256
    pin = manifest['flat_balance']['mixture_source']
    parent_path = Path(pin['manifest'])
    if sha256(parent_path) != pin['sha256']:
        raise ValueError('Mixture source pin differs')
    parent = load_generalist_manifest(parent_path, verify_files=False)
    if parent['flat_balance']['schema'] != 'cat-protected-region-v5':
        raise ValueError('Protected-heavy bank requires v5')
    expected = copy.deepcopy(parent)
    expected['flat_balance'].update(schema=SCHEMA, masses=list(MASSES), mixture_source=pin)
    expected.pop('manifest_sha256')
    actual = dict(manifest); actual.pop('manifest_sha256', None)
    if actual != expected:
        raise ValueError('Only reset mixture may differ from v5')
    return manifest['flat_balance']


def sampling_plan(manifest):
    from .hand_balance_bank import sampling_plan as parent_plan
    ids, _ = parent_plan(manifest)
    return ids, np.asarray(MASSES, np.float32)
