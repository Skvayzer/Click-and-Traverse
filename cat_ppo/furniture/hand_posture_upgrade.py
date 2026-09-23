"""Validation for a metadata-only posture upgrade of the existing hand bank."""
from __future__ import annotations

SCHEMA = 'cat-hand-posture-upgrade-v1'


def validate_hand_posture_upgrade(manifest, parent, *, path=None, verify_files=False):
    import json
    from pathlib import Path
    from .generalist_fields import _json_hash
    from .contrastive_rewards import pack_hand_contrast
    from .room_navigation import scene_navigation_radius
    marker=manifest['hand_posture_upgrade']
    if (marker.get('schema')!=SCHEMA or any(k in manifest for k in ('width_curriculum','contrastive_specialist','specialist'))
            or not manifest.get('hand_protection_curriculum')):
        raise ValueError('Posture upgrade must retain only the hand curriculum')
    n=marker['retained_scene_count'];old=parent['scenes'];new=manifest['scenes']
    if n!=2338 or len(old)!=2362 or len(new)!=len(old) or new[:n]!=old[:n]:
        raise ValueError('Posture upgrade must preserve the exact 2338-scene retention prefix')
    if marker.get('retained_scene_records_sha256')!=_json_hash(old[:n]):
        raise ValueError('Posture retention prefix hash differs')
    if manifest['sampling_group_masses']!=dict(original_cat=.2,procedural_cat=.4,furniture=.25,generic_clutter=.15):
        raise ValueError('Posture upgrade must preserve documented sampling masses')
    if marker.get('selection_roles')!=[1]:raise ValueError('Posture upgrade selection requires protected role only')
    counts={'hand_table_aisle':0,'hand_shelf_passage':0}
    for before,after in zip(old[n:],new[n:]):
        # Only canonical scene/source metadata hashes and added source contrast metadata differ.
        a=dict(after);a['source']=dict(a['source']);a['source'].pop('hand_contrast',None)
        a['source']['metadata_sha256']=before['source']['metadata_sha256'];a['scene_sha256']=before['scene_sha256']
        if a!=before:raise ValueError('Posture upgrade changed fields, order, reset law or legacy scene metadata')
        kind=before['source']['hand_protection']['kind'];counts[kind]+=1
        c=after['source'].get('hand_contrast',{})
        if c.get('role')!='forward_protected' or c.get('navigation_radius_m')!=.23:
            raise ValueError('Hand posture scenes must retain the legacy navigation radius and protected role')
        pack_hand_contrast([{'hand_contrast':c}])
        if kind=='hand_table_aisle' and any(z['region_valid']!=[True,False] for z in c['zones']):
            raise ValueError('Tables require raised-only targets')
        if verify_files:
            scene=json.loads((Path(path).parent/after['path']/'scene.json').read_text())
            source=json.loads((Path(path).parent/after['path']/'source.json').read_text())
            original=json.loads((Path(manifest['external_retention_bank']['manifest']).parent/before['path']/'scene.json').read_text())
            stripped=dict(scene);stripped.pop('hand_contrast',None)
            if stripped!=original or scene['hand_contrast']!=c or source['hand_contrast']!=c:
                raise ValueError('Posture upgrade changed canonical geometry or mismatched metadata')
            scene_navigation_radius(scene)
    if counts!={'hand_table_aisle':12,'hand_shelf_passage':12}:raise ValueError('Missing hand scene family')
