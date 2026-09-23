#!/usr/bin/env python3
"""CPU-only, independent v3 copies and an auditable validity-only transformation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1')
import copy
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cat_ppo.furniture.generalist_fields import sha256, load_generalist_manifest
from cat_ppo.furniture.raised_only_bank import SCHEMA, raised_only
from cat_ppo.furniture.contrast_preflight import contrast_preflight
from cat_ppo.furniture.body_collision_bank import load_body_collision_bank
from scripts.build_flat_balance_bank import write, publish


def snapshot(root):
    return {str(p.relative_to(root)): dict(sha256=sha256(p), size=p.stat().st_size,
            mtime_ns=p.stat().st_mtime_ns, inode=p.stat().st_ino)
            for p in sorted(root.rglob('*')) if p.is_file()}


def build(*, live_log=None):
    old = ROOT/'data/furniture/cat_flat_hand_balance_v2_20260921'
    new = old.with_name('cat_flat_hand_balance_v3_20260921')
    suffixes = ('', '_collision', '_resets')
    source = [old.with_name(old.name+s) for s in suffixes]
    target = [new.with_name(new.name+s) for s in suffixes]
    if any(p.exists() for p in target):
        raise FileExistsError('Refusing to overwrite an existing bank')
    before = [snapshot(p) for p in source]
    live = Path(live_log) if live_log is not None else None
    def live_snapshot():
        return (dict(bytes=live.stat().st_size, lines=len(live.read_text().splitlines()))
                if live is not None else None)
    live_before = live_snapshot()
    parent = load_generalist_manifest(old/'manifest.json', verify_files=False)
    for src, dst in zip(source, target):
        subprocess.run(['cp', '-a', '--reflink=auto', str(src), str(dst)], check=True)
    manifest = copy.deepcopy(parent)
    changes = []
    for record in manifest['scenes']:
        if record.get('source', {}).get('hand_contrast', {}).get('role') != 'forward_protected':
            continue
        path = new/record['path']/'scene.json'
        original = json.loads(path.read_text())
        revised = raised_only(original)
        write(path, revised)
        record['scene_sha256'] = sha256(path)
        for i, zone in enumerate(revised['hand_contrast']['zones']):
            centers = [[(a[2]+b[2])/2 for a,b in zip(lo,hi)] for lo,hi in zip(zone['hand_regions_min'],zone['hand_regions_max'])]
            changes.append(dict(scene=record['scene_id'], zone=i, z_centers=centers,
                                valid=zone['region_valid']))
    manifest['flat_balance'].update(schema=SCHEMA, validity_source=dict(
        manifest=str(old/'manifest.json'), sha256=sha256(old/'manifest.json')))
    publish(new/'manifest.json', manifest)
    # Geometry and reset arrays remain exact; update their source pins only.
    for name in ('manifest.json', 'geometry-inventory.json', 'build-plan.json'):
        path = target[1]/name
        obj = json.loads(path.read_text())
        if 'field_manifest_sha256' in obj:
            obj.update(field_manifest_sha256=sha256(new/'manifest.json'),
                       field_manifest_content_sha256=manifest['manifest_sha256'])
        for geometry in obj.get('scenes', []):
            r = manifest['scenes'][geometry['index']]
            if r.get('source', {}).get('hand_contrast', {}).get('role') == 'forward_protected':
                geometry['provenance']['source_sha256'] = r['scene_sha256']
        (publish if name == 'manifest.json' else write)(path, obj)
    rp = target[2]/'manifest.json'
    resets = json.loads(rp.read_text())
    resets.update(field_manifest=str(new/'manifest.json'), field_manifest_sha256=sha256(new/'manifest.json'),
                  collision_bank=str(target[1]/'manifest.json'), collision_bank_sha256=sha256(target[1]/'manifest.json'))
    write(rp, resets)
    load_generalist_manifest(new/'manifest.json', verify_files=True)
    load_body_collision_bank(target[1]/'manifest.json', expected_field_manifest=new/'manifest.json')
    assert sha256(target[2]/resets['file']) == resets['sha256']
    results = [contrast_preflight(p/'manifest.json', require_hand_contrast=True) for p in (old,new)]
    # Prove every nonmetadata file, including geometry and poses, is identical.
    after = [snapshot(p) for p in source]
    assert before == after, 'Source bytes or inode/mtime changed'
    differences = []
    for src, dst, snap in zip(source, target, before):
        current = snapshot(dst)
        assert set(snap) == set(current)
        changed = [p for p in snap if snap[p]['sha256'] != current[p]['sha256']]
        assert all(p.endswith('.json') for p in changed)
        assert all(snap[p]['inode'] != current[p]['inode'] for p in snap), 'Unexpected shared inode'
        differences.append(changed)
    report = dict(preflight_v2=results[0], preflight_v3=results[1], changed_zones=changes,
        source_unchanged=True, source_file_counts=[len(s) for s in before], changed_files=differences,
        live_before=live_before, live_after=live_snapshot(),
        source_manifest_sha256=sha256(old/'manifest.json'), v3_manifest_sha256=sha256(new/'manifest.json'))
    write(ROOT/'docs/assets/raised-only-v3-20260921/verification.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('changed_zones','changed_files')}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-log', type=Path, help='Optional existing metrics log to snapshot before/after')
    build(live_log=parser.parse_args().live_log)
