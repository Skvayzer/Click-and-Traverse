"""Explicit flat-balance isolation mixture; no changes to legacy bank semantics."""
import json
from pathlib import Path
import numpy as np

SCHEMA = 'cat-flat-balance-v1'
MASSES = (.20, .40, .0625, .0375, .05, .25)
SETTINGS = dict(target_centers=[[.429,.125,.870],[.429,-.125,.870]],
                half_size=[.018,.010,.018], tolerance=.05, region_scale=.15,
                bonus_scale=3., target_speed=.6, walking_speed=.2, episode_steps=500)


def validate_balance_manifest(manifest, *, path, verify_files=False):
    from .generalist_fields import sha256, load_generalist_manifest
    marker=manifest['flat_balance']
    if marker.get('schema') == 'cat-protected-heavy-v6':
        from .protected_heavy_bank import validate
        return validate(manifest, path=path)
    if marker.get('schema') == 'cat-protected-region-v5':
        from .balanced_region_bank import validate
        return validate(manifest, path=path)
    if marker.get('schema') == 'cat-protected-region-v4':
        from .feasible_region_bank import validate
        return validate(manifest, path=path)
    if marker.get('schema') == 'cat-flat-hand-balance-v3':
        from .raised_only_bank import validate
        return validate(manifest, path=path)
    if marker.get('schema') == 'cat-flat-hand-balance-v2':
        from .hand_balance_bank import validate
        return validate(manifest, path=path, verify_files=verify_files)
    if marker.get('schema') == 'cat-extended-balance-v1':
        from .extended_balance_bank import validate
        return validate(manifest, path=path, verify_files=verify_files)
    if marker.get('schema') == 'cat-packed-balance-v1':
        from .packed_balance_bank import validate
        return validate(manifest, path=path, verify_files=verify_files)
    if marker.get('schema')!=SCHEMA or marker.get('settings')!=SETTINGS or marker.get('masses')!=list(MASSES):
        raise ValueError('Unrecognized flat-balance experiment settings/masses')
    if any(k in manifest for k in ('width_curriculum','hand_protection_curriculum','hand_posture_upgrade','contrastive_specialist','specialist','external_retention_bank')):
        raise ValueError('Flat balance cannot inherit another sampler or curriculum')
    parents=[]
    for key in ('retention_source','narrow_source'):
        pin=marker[key];p=Path(pin['manifest'])
        if sha256(p)!=pin['sha256']:raise ValueError('Balance source manifest pin differs')
        parents.append(load_generalist_manifest(p,verify_files=False))
    parent_root=Path(parents[0].get('external_retention_bank',{}).get('manifest',marker['retention_source']['manifest'])).resolve().parent
    if Path(marker['retention_storage_root']).resolve()!=parent_root:raise ValueError('Retention storage root differs from pinned parent')
    records=manifest['scenes'];retained=parents[0]['scenes'][:2338]
    narrow=[s for s in parents[1]['scenes'] if s.get('source',{}).get('hand_contrast',{}).get('role')=='narrow' and s['source']['hand_contrast']['curriculum_rung']<=2]
    if len(narrow)!=12 or len(records)!=2351 or records[:2338]!=retained or records[2338:2350]!=narrow:
        raise ValueError('Balance scene composition differs from pinned 2338+12+1 records')
    if any(s.get('source',{}).get('hand_protection') for s in records):
        raise ValueError('Hand table/shelf scenes forbidden during balance isolation')
    flat=records[-1]
    if (flat['scene_id']!='flat-balance-walk-v1' or flat['source'].get('flat_balance')!=SCHEMA
            or flat['task_kind']!='room' or flat['family']!='generic_clutter' or flat['reset_xy_scale']!=[.08,.08]
            or flat['reset_yaw']!=0. or flat['start']!=[0.,0.,.8] or flat['goal']!=[100.,0.,.8]
            or flat['source'].get('hand_contrast') or 'hand_contrast' in flat):
        raise ValueError('Invalid dedicated flat-balance task')
    if verify_files:
        folder=Path(path).parent/flat['path'];scene=json.loads((folder/'scene.json').read_text())
        if scene['boxes'] or scene['route']!=[[0.,0.],[100.,0.]] or scene.get('hand_contrast') or scene.get('hand_protection'):
            raise ValueError('Flat task must be genuinely empty and unqualified')
        for name in ('sdf','gf','bf'):
            a=np.load(folder/(name+'.npy'),allow_pickle=False)
            expected=10. if name=='sdf' else np.array([.6,0.,0.],np.float32) if name=='gf' else 0.
            if not np.allclose(a,expected,rtol=0,atol=1e-7):raise ValueError('Flat fields must be analytic empty-space fields')
    return marker


def _extended_plan(manifest):
    """Parent prefix keeps its own plan; appended scenes join their declared group."""
    import json as _json
    from .generalist_fields import SAMPLING_GROUPS
    parent = _json.loads(Path(manifest['flat_balance']['parent']['manifest']).read_text())
    ids, masses = sampling_plan(parent)
    extra = [SAMPLING_GROUPS.index(s['sampling_group'])
             for s in manifest['scenes'][len(parent['scenes']):]]
    return np.concatenate([ids, np.asarray(extra, np.int64)]), masses


def sampling_plan(manifest):
    if manifest['flat_balance']['schema'] == 'cat-protected-heavy-v6':
        from .protected_heavy_bank import sampling_plan as heavy_plan
        return heavy_plan(manifest)
    if manifest['flat_balance']['schema'] in ('cat-flat-hand-balance-v2', 'cat-flat-hand-balance-v3', 'cat-protected-region-v4', 'cat-protected-region-v5'):
        from .hand_balance_bank import sampling_plan as revised_plan
        return revised_plan(manifest)
    if manifest['flat_balance']['schema'] == 'cat-packed-balance-v1':
        if 'parent' in manifest['flat_balance'] and 'additions' in manifest['flat_balance']:
            # A packed EXTENDED bank keeps the parent/additions pins, so it samples like the
            # extended bank it was packed from (the legacy plan below hard-codes 2375 scenes).
            return _extended_plan(manifest)
        # A packed bank holds exactly the source's scenes, so it samples exactly as the
        # source did. Without this it fell through to the v1 branch below, whose scene
        # counts are hard-coded and produced a 2351-long plan for a 2375-scene bank.
        from .hand_balance_bank import sampling_plan as revised_plan
        return revised_plan(manifest)
    if manifest['flat_balance']['schema'] == 'cat-extended-balance-v1':
        return _extended_plan(manifest)
    from .generalist_fields import SAMPLING_GROUPS
    ids=[SAMPLING_GROUPS.index(s['sampling_group']) for s in manifest['scenes'][:2338]]+[4]*12+[5]
    return np.asarray(ids,np.int64),np.asarray(MASSES,np.float32)
