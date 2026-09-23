"""CPU-only coverage audit of the actual, hash-pinned scene metadata."""
import json
from collections import Counter
from pathlib import Path


def contrast_preflight(path, *, require_hand_contrast=None):
    from .generalist_fields import load_generalist_manifest, scene_directory, sha256
    path = Path(path)
    manifest = load_generalist_manifest(path, verify_files=False)
    roles = Counter()
    zones = hand = active = heading = 0
    for record in manifest['scenes']:
        if not record.get('source', {}).get('hand_contrast'):
            continue
        scene_path = scene_directory(manifest, path, record) / 'scene.json'
        if sha256(scene_path) != record['scene_sha256']:
            raise ValueError(f'Contrast scene bytes changed: {scene_path}')
        metadata = json.loads(scene_path.read_text())['hand_contrast']
        roles[metadata['role']] += 1
        for zone in metadata['zones']:
            zones += 1
            hand += bool(any(zone['hand_active']))
            active += bool(any(zone['hand_active']) and any(zone['region_valid']))
            heading += zone['forward_weight'] > 0
    result = dict(scene_count=len(manifest['scenes']), roles=dict(roles), zones=zones,
                  hand_active_zones=hand, active_hand_objective_zones=active,
                  positive_heading_zones=heading)
    print('Hand-contrast preflight: ' + json.dumps(result, sort_keys=True), flush=True)
    # The runner automatically requests contrast rewards whenever contrast scenes exist.
    requested = bool(roles) if require_hand_contrast is None else require_hand_contrast
    if requested and (active == 0 or heading == 0):
        raise ValueError('Hand-contrast training requested but active hand/heading coverage is zero')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--require-hand-contrast', action='store_true')
    args = parser.parse_args()
    contrast_preflight(args.manifest, require_hand_contrast=True if args.require_hand_contrast else None)
