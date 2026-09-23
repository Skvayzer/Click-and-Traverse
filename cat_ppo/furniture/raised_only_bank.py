"""Strict validity-only successor to the pinned v2 hand-balance bank."""
import copy
import json
from pathlib import Path

SCHEMA = 'cat-flat-hand-balance-v3'


def raised_only(scene):
    """Select by both hands' z heights, never by an assumed region ordering."""
    result = copy.deepcopy(scene)
    metadata = result.get('hand_contrast', {})
    if metadata.get('role') != 'forward_protected':
        return result
    for zone in metadata['zones']:
        lo, hi = zone['hand_regions_min'], zone['hand_regions_max']
        if len(lo) != 2 or zone['region_valid'] != [True, True]:
            raise ValueError('Expected two valid v2 alternatives')
        heights = [[(a[2] + b[2]) / 2 for a, b in zip(l, h)] for l, h in zip(lo, hi)]
        raised = max(range(2), key=lambda i: sum(heights[i]))
        if not all(a > b for a, b in zip(heights[raised], heights[1-raised])):
            raise ValueError('Ambiguous raised region heights')
        if metadata['mode_names'][raised] != 'raised':
            raise ValueError('Height selection disagrees with posture label')
        zone['region_valid'] = [i == raised for i in range(2)]
    return result


def validate(manifest, *, path):
    from .generalist_fields import load_generalist_manifest, scene_directory, sha256
    marker = manifest['flat_balance']
    pin = marker['validity_source']
    parent_path = Path(pin['manifest'])
    if sha256(parent_path) != pin['sha256']:
        raise ValueError('Validity source pin differs')
    parent = load_generalist_manifest(parent_path, verify_files=False)
    if parent['flat_balance']['schema'] != 'cat-flat-hand-balance-v2':
        raise ValueError('Validity successor requires v2')
    expected = copy.deepcopy(parent)
    expected['flat_balance'].update(schema=SCHEMA, validity_source=pin)
    for old, record in zip(parent['scenes'], expected['scenes']):
        if old.get('source', {}).get('hand_contrast', {}).get('role') != 'forward_protected':
            continue
        source = scene_directory(parent, parent_path, old) / 'scene.json'
        target = scene_directory(manifest, path, record) / 'scene.json'
        if sha256(source) != old['scene_sha256']:
            raise ValueError('Validity source scene changed')
        if json.loads(target.read_text()) != raised_only(json.loads(source.read_text())):
            raise ValueError('Only protected-region validity may change')
        record['scene_sha256'] = sha256(target)
    # All other manifest fields, scene ordering, weights and pins remain exact.
    expected.pop('manifest_sha256')
    actual = dict(manifest)
    actual.pop('manifest_sha256', None)
    if actual != expected:
        raise ValueError('Validity-only manifest differs from pinned v2')
    return marker
