#!/usr/bin/env python3
"""CPU-only v2 composition of identical certified scenes; never overwrite a bank."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import copy
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.build_flat_balance_bank import link, write, publish
from cat_ppo.furniture.generalist_fields import load_generalist_manifest, scene_directory, sha256
from cat_ppo.furniture.hand_balance_bank import SCHEMA, MASSES, COMPOSITION, objective_indices
from cat_ppo.furniture.body_collision_bank import build_body_collision_bank, load_body_collision_bank
from cat_ppo.furniture.contrast_preflight import contrast_preflight


def build():
    out = ROOT/'data/furniture/cat_flat_hand_balance_v2_20260921'
    cout = out.with_name(out.name+'_collision'); rout = out.with_name(out.name+'_resets')
    if any(p.exists() for p in (out, cout, rout)):
        raise FileExistsError('Use a fresh bank path; published or partial outputs are never overwritten')
    sources = [ROOT/'data/furniture/cat_flat_balance_v1_20260920/manifest.json',
               ROOT/'data/furniture/cat_width_retention_v2/manifest.json']
    manifests = [load_generalist_manifest(p, verify_files=False) for p in sources]
    selections = [list(range(len(manifests[0]['scenes']))), objective_indices(manifests[1])]
    records = []; geometry = []; pools = []; provenance = []
    proposal = ROOT/'docs/assets/collision-proxy-proposal-20260916/proposal.json'
    for source, manifest, indices in zip(sources, manifests, selections):
        cp = source.parent.with_name(source.parent.name+'_collision')/'manifest.json'
        rp = source.parent.with_name(source.parent.name+'_resets')/'manifest.json'
        _, cm = load_body_collision_bank(cp, expected_field_manifest=source, expected_proxy_sha256=sha256(proposal))
        rm = json.loads(rp.read_text())
        if (rm['status'] != 'complete' or rm['field_manifest_sha256'] != sha256(source)
                or rm['collision_bank_sha256'] != sha256(cp) or rm['proxy_sha256'] != sha256(proposal)
                or sha256(rp.parent/rm['file']) != rm['sha256']):
            raise ValueError('Source certification pins differ')
        pool = np.load(rp.parent/rm['file'], allow_pickle=False)
        if list(pool.shape) != rm['shape'] or not np.isfinite(pool).all():
            raise ValueError('Invalid source reset pool')
        pools.append(pool[indices])
        provenance.append(dict(manifest=str(rp), sha256=sha256(rp), indices=indices))
        for i in indices:
            record = copy.deepcopy(manifest['scenes'][i]); index = len(records)
            if rm['scenes'][i]['scene_id'] != record['scene_id'] or cm['scenes'][i]['scene_id'] != record['scene_id']:
                raise ValueError('Source reset/geometry scene identity mismatch')
            folder = scene_directory(manifest, source, record)
            # Retention remains referenced; link all contrast and flat scene files unchanged.
            if record['source'].get('hand_contrast') or record['source'].get('flat_balance'):
                for name in ('sdf.npy', 'bf.npy', 'gf.npy', 'source.json', 'scene.json'):
                    link(folder/name, out/record['path']/name)
            g = copy.deepcopy(cm['scenes'][i]); target = f'geometry/{index:05d}.npz'
            if sha256(cp.parent/g['geometry_file']) != g['geometry_sha256']:
                raise ValueError('Source geometry bytes differ')
            link(cp.parent/g['geometry_file'], cout/target)
            g.update(index=index, geometry_file=target)
            records.append(record); geometry.append(g)
    m = copy.deepcopy(manifests[0]); marker = m['flat_balance']
    marker.update(schema=SCHEMA, masses=list(MASSES), composition=COMPOSITION, require_hand_contrast=True,
                  base_source=dict(manifest=str(sources[0]), sha256=sha256(sources[0])),
                  objective_source=dict(manifest=str(sources[1]), sha256=sha256(sources[1])))
    m.update(scenes=records, scene_count=len(records),
             fields_bytes=sum(f['size_bytes'] for r in records for f in r['fields'].values()),
             storage='Hash-pinned retention; unchanged hardlinked contrast/flat fields and geometry',
             sampling_group_masses=dict(original_cat=.18, procedural_cat=.36, furniture=.05625, generic_clutter=.40375))
    dest = out/'manifest.json'; publish(dest, m)
    contrast_preflight(dest, require_hand_contrast=True)
    write(cout/'geometry-inventory.json', dict(scenes=geometry))
    build_body_collision_bank(dest, proposal, cout, workers=1, progress=lambda p: print(p, flush=True))
    pool = np.concatenate(pools); rout.mkdir(); np.save(rout/'reset_qpos.npy', pool)
    write(rout/'manifest.json', dict(schema='cat-body-collision-resets-v1', status='complete',
          backend='composed-identical-scene-certified-pools', field_manifest=str(dest), field_manifest_sha256=sha256(dest),
          collision_bank=str(cout/'manifest.json'), collision_bank_sha256=sha256(cout/'manifest.json'),
          proxy_sha256=sha256(proposal), file='reset_qpos.npy', sha256=sha256(rout/'reset_qpos.npy'),
          scene_count=len(records), shape=list(pool.shape), dtype=str(pool.dtype), poses_per_scene=pool.shape[1],
          sources=provenance, scenes=[dict(index=i, scene_id=r['scene_id']) for i,r in enumerate(records)],
          composition='Exact source rows for identical pinned scenes and geometry; no fabricated or cross-scene poses'))
    load_generalist_manifest(dest, verify_files=True)
    load_body_collision_bank(cout/'manifest.json', expected_field_manifest=dest)
    print('VALIDATED', dest, flush=True)


if __name__ == '__main__':
    build()
