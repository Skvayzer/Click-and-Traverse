#!/usr/bin/env python3
"""Independent disk audit: scene coverage, exact source reset rows, geometry pins."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
import json
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def audit(name):
    path = ROOT/'data/furniture'/name/'manifest.json'
    m = json.loads(path.read_text()); roles = Counter(); counts = Counter()
    for r in m['scenes']:
        if not r['source'].get('hand_contrast'): continue
        sp = path.parent/r['path']/'scene.json'
        assert digest(sp) == r['scene_sha256']
        s = json.loads(sp.read_text()); c = s['hand_contrast']; roles[c['role']] += 1
        for z in c['zones']:
            counts['zones'] += 1
            counts['hand_active'] += int(any(z['hand_active']))
            counts['active_hand_objective'] += int(any(z['hand_active']) and any(z['region_valid']))
            counts['positive_heading'] += int(z['forward_weight'] > 0)
            if c['role'] == 'narrow':
                assert not any(z['hand_active']) and not any(z['region_valid']) and z['forward_weight'] == 0
    result = dict(scenes=len(m['scenes']), roles={r:roles[r] for r in ('open','forward_protected','narrow','transition')}, **counts)
    if name.startswith('cat_flat_hand'):
        cp = path.parent.with_name(name+'_collision')/'manifest.json'; cm = json.loads(cp.read_text())
        rp = path.parent.with_name(name+'_resets')/'manifest.json'; rm = json.loads(rp.read_text())
        assert cm['field_manifest_sha256'] == rm['field_manifest_sha256'] == digest(path)
        assert rm['collision_bank_sha256'] == digest(cp)
        assert digest(rp.parent/rm['file']) == rm['sha256']
        pool = np.load(rp.parent/rm['file']); offset = 0
        for pin in rm['sources']:
            source = Path(pin['manifest']); assert digest(source) == pin['sha256']
            sm = json.loads(source.read_text()); assert digest(source.parent/sm['file']) == sm['sha256']
            expected = np.load(source.parent/sm['file'])[pin['indices']]
            np.testing.assert_array_equal(pool[offset:offset+len(expected)], expected)
            scp = Path(sm['collision_bank']); assert digest(scp) == sm['collision_bank_sha256']
            scm = json.loads(scp.read_text())
            for j, i in enumerate(pin['indices']):
                k = offset+j
                assert sm['scenes'][i]['scene_id'] == m['scenes'][k]['scene_id'] == cm['scenes'][k]['scene_id']
                assert scm['scenes'][i]['geometry_sha256'] == cm['scenes'][k]['geometry_sha256']
                assert digest(cp.parent/cm['scenes'][k]['geometry_file']) == cm['scenes'][k]['geometry_sha256']
            offset += len(expected)
        assert offset == len(m['scenes'])
        result['reset_shape'] = list(pool.shape); result['exact_source_pose_rows'] = int(pool.shape[0]*pool.shape[1])
        files = [p for folder in (path.parent, cp.parent, rp.parent) for p in folder.rglob('*') if p.is_file()]
        result['logical_bytes'] = sum(p.stat().st_size for p in files)
        result['allocated_bytes'] = sum(p.stat().st_blocks*512 for p in files)
        result['exclusive_file_bytes'] = sum(p.stat().st_blocks*512 for p in files if p.stat().st_nlink == 1)
    return result

if __name__ == '__main__':
    print(json.dumps({n:audit(n) for n in ('cat_flat_balance_v1_20260920', 'cat_flat_hand_balance_v2_20260921')}, indent=2))
