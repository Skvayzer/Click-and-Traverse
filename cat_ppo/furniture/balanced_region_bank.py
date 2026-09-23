"""Strict geometry-only successor to the pinned v3 hand-balance bank."""
import copy
import json
from pathlib import Path

SCHEMA = 'cat-protected-region-v5'


def feasible_region(scene):
    result = copy.deepcopy(scene)
    if result.get('hand_contrast', {}).get('role') != 'forward_protected':
        return result
    for zone in result['hand_contrast']['zones']:
        if zone['region_valid'] != [True, False]:
            raise ValueError('Expected v3 single valid region')
        for hand, center in enumerate(((.429404140,.124795869,0.614059567451477),(.429404199,-.124786183,0.614059567451477))):
            for axis, half in enumerate((.018,.010,.018)):
                zone['hand_regions_min'][0][hand][axis] = center[axis]-half
                zone['hand_regions_max'][0][hand][axis] = center[axis]+half
    return result


def validate(manifest, *, path):
    from .generalist_fields import load_generalist_manifest, scene_directory, sha256
    marker = manifest['flat_balance']
    pin = marker['region_geometry_source']
    parent_path = Path(pin['manifest'])
    if sha256(parent_path) != pin['sha256']:
        raise ValueError('Geometry source pin differs')
    parent = load_generalist_manifest(parent_path, verify_files=False)
    if parent['flat_balance']['schema'] != 'cat-flat-hand-balance-v3':
        raise ValueError('Geometry successor requires v3')
    expected = copy.deepcopy(parent)
    expected['flat_balance'].update(schema=SCHEMA, region_geometry_source=pin)
    for old, record in zip(parent['scenes'], expected['scenes']):
        if old.get('source', {}).get('hand_contrast', {}).get('role') != 'forward_protected':
            continue
        source = scene_directory(parent, parent_path, old) / 'scene.json'
        target = scene_directory(manifest, path, record) / 'scene.json'
        if sha256(source) != old['scene_sha256']:
            raise ValueError('Geometry source scene changed')
        if json.loads(target.read_text()) != feasible_region(json.loads(source.read_text())):
            raise ValueError('Only protected-region geometry may change')
        record['scene_sha256'] = sha256(target)
    # All other manifest fields, scene ordering, weights and pins remain exact.
    expected.pop('manifest_sha256')
    actual = dict(manifest)
    actual.pop('manifest_sha256', None)
    if actual != expected:
        raise ValueError('Geometry-only manifest differs from pinned v3')
    return marker
