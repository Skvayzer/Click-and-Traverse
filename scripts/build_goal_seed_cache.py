#!/usr/bin/env python3
"""Build the ground-projected-goal (column seed) guidance cache for a packed field bank.

  python scripts/build_goal_seed_cache.py --bank data/furniture/tablegap_v1 --out data/furniture/tablegap_v1_goal_column \
      --workers 32

Writes OUT/<scene_id>/gf.npy (+ gf_valid.npy) in the bank's packed encoding and OUT/cache.json pinning the bank
manifest hash, the seed parameters and every file hash (cat_mjlab/goal_seed.py). The bank is not modified. A scene
whose goal column has no free voxel keeps its original guidance (listed under "kept_ball").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _one(job):
    from cat_mjlab.goal_seed import scene_column_field, _sha256
    directory, record, out = job
    target = Path(out) / record['scene_id']; target.mkdir(parents=True, exist_ok=True)
    try:
        codes, valid = scene_column_field(directory, record)
        kept = False
    except ValueError:
        import shutil
        shutil.copyfile(Path(directory) / 'gf.npy', target / 'gf.npy')
        if (Path(directory) / 'gf_valid.npy').exists():
            shutil.copyfile(Path(directory) / 'gf_valid.npy', target / 'gf_valid.npy')
        kept = True
    else:
        np.save(target / 'gf.npy', codes); np.save(target / 'gf_valid.npy', valid)
    return record['scene_id'], dict(gf_sha256=_sha256(target / 'gf.npy'), shape=list(record['shape'])), kept


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank', required=True, help='bank prefix (uses <bank>_packed/manifest.json)')
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args(argv)
    from cat_ppo.furniture.generalist_fields import load_generalist_manifest, scene_directory
    from cat_mjlab.goal_seed import SCHEMA, SEED_RADIUS, COLUMN_Z, R_PROJ_CELLS
    path = Path(str(args.bank) + '_packed/manifest.json').resolve()
    manifest = load_generalist_manifest(path, verify_files=False)
    records = manifest['scenes']; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(scene_directory(manifest, path, r)), r, str(out)) for r in records]
    start = time.time(); scenes = {}; kept = []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, (scene_id, entry, was_kept) in enumerate(pool.map(_one, jobs, chunksize=4)):
            scenes[scene_id] = entry
            if was_kept:
                kept.append(scene_id)
            if k % 250 == 0:
                print(f'{k + 1}/{len(jobs)} scenes, {time.time() - start:.0f}s', flush=True)
    meta = dict(schema=SCHEMA, goal_seed='column', bank_manifest=str(path), bank_manifest_sha256=manifest['manifest_sha256'],
                seed_radius_m=SEED_RADIUS, column_z_m=list(COLUMN_Z), r_proj_cells=R_PROJ_CELLS,
                generator='cat_mjlab/goal_seed.py column_guidance (pf_modular progressive blend, column seed)',
                kept_ball=kept, scenes=scenes)
    meta['cache_sha256'] = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (out / 'cache.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(dict(out=str(out), scenes=len(scenes), kept_ball=len(kept), seconds=round(time.time() - start),
                          cache_sha256=meta['cache_sha256'])))


if __name__ == '__main__':
    main()
