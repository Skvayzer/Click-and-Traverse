#!/usr/bin/env python3
"""CPU-only independent v6 copy, mixture-only change, pinned companions and audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1')
import copy
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cat_ppo.furniture.generalist_fields import sha256, load_generalist_manifest
from cat_ppo.furniture.protected_heavy_bank import SCHEMA, MASSES
from cat_ppo.furniture.body_collision_bank import load_body_collision_bank
from cat_ppo.furniture.contrast_preflight import contrast_preflight
from scripts.build_flat_balance_bank import write, publish


def snapshot(paths):
    # Hash shared immutable data once per inode, but verify every pathname.
    cache={}; result={}
    for root in paths:
        for p in sorted(root.rglob('*')):
            if not p.is_file():continue
            stat=p.stat(); key=(stat.st_dev,stat.st_ino)
            if key not in cache:cache[key]=sha256(p)
            result[str(p.relative_to(ROOT))]=[stat.st_size,stat.st_mtime_ns,stat.st_ino,cache[key]]
    return result


def build():
    parent=ROOT/'data/furniture';old=parent/'cat_flat_hand_balance_v5_20260921'
    new=parent/'cat_flat_hand_balance_v6_20260922'
    sources=[old.with_name(old.name+s) for s in ('','_collision','_resets')]
    targets=[new.with_name(new.name+s) for s in ('','_collision','_resets')]
    if any(p.exists() for p in targets):raise FileExistsError('Refusing to overwrite existing bank')
    size=sum(p.stat().st_size for root in sources for p in root.rglob('*') if p.is_file())
    free=shutil.disk_usage(parent).free
    # Budget full logical copies, metadata/audit overhead and a 128 MiB cushion.
    if free-size-128*2**20 < 2*2**30:
        raise RuntimeError(f'Disk reserve blocks build: free={free}, required={size}; use sampling override')
    existing=list(parent.iterdir())
    existing=[p for p in existing if p.is_dir()]
    print(f'Hashing existing bank files; free={free}, copy bytes={size}',flush=True)
    before=snapshot(existing)
    live=ROOT/'outputs/cat_hand_priority_30720_20260922/metrics.jsonl'
    live_before=dict(size=live.stat().st_size,mtime_ns=live.stat().st_mtime_ns)
    for src,dst in zip(sources,targets):
        if shutil.disk_usage(parent).free-sum(p.stat().st_size for p in src.rglob('*') if p.is_file()) < 2*2**30:
            raise RuntimeError('Disk reserve reached during build')
        shutil.copytree(src,dst,copy_function=shutil.copy2)
    m=copy.deepcopy(load_generalist_manifest(old/'manifest.json',verify_files=False))
    m['flat_balance'].update(schema=SCHEMA,masses=list(MASSES),
        mixture_source=dict(manifest=str(old/'manifest.json'),sha256=sha256(old/'manifest.json')))
    publish(new/'manifest.json',m)
    # Geometry and certified reset payloads are invariant under reset weighting.
    # Regenerate companion metadata/hash pins, retaining exact certified arrays.
    for name in ('manifest.json','geometry-inventory.json','build-plan.json'):
        p=targets[1]/name;obj=json.loads(p.read_text())
        if 'field_manifest_sha256' in obj:
            obj.update(field_manifest_sha256=sha256(new/'manifest.json'),field_manifest_content_sha256=m['manifest_sha256'])
        (publish if name=='manifest.json' else write)(p,obj)
    p=targets[2]/'manifest.json';r=json.loads(p.read_text())
    r.update(field_manifest=str(new/'manifest.json'),field_manifest_sha256=sha256(new/'manifest.json'),
        collision_bank=str(targets[1]/'manifest.json'),collision_bank_sha256=sha256(targets[1]/'manifest.json'))
    write(p,r)
    load_generalist_manifest(new/'manifest.json',verify_files=True)
    load_body_collision_bank(targets[1]/'manifest.json',expected_field_manifest=new/'manifest.json')
    assert sha256(targets[2]/r['file'])==r['sha256']
    preflight=contrast_preflight(new/'manifest.json',require_hand_contrast=True)
    after=snapshot(existing)
    assert before==after,'Existing bank bytes/metadata changed'
    changed=[]
    for src,dst in zip(sources,targets):
        for p in src.rglob('*'):
            if not p.is_file():continue
            q=dst/p.relative_to(src)
            assert p.stat().st_ino!=q.stat().st_ino
            if sha256(p)!=sha256(q):changed.append(str(q.relative_to(new.parent)))
    assert all(Path(p).name in ('manifest.json','build-plan.json') for p in changed)
    report=dict(preflight=preflight,existing_files_verified=len(before),existing_bytes_identical=True,
        existing_snapshot_sha256=__import__('hashlib').sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
        changed_files=changed,reset_masses=list(MASSES),free_before_bytes=free,free_after_bytes=shutil.disk_usage(parent).free,
        bank_logical_bytes={p.name:sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in targets},
        live_before=live_before,live_after=dict(size=live.stat().st_size,mtime_ns=live.stat().st_mtime_ns))
    write(ROOT/'docs/assets/hand-tolerance-v6-20260922/verification.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':build()
