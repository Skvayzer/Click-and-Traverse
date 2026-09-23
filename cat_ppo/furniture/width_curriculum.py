"""Native width ladder with fixed retention/role masses and honest scaffold certificates."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

SCHEMA = 'cat-width-curriculum-v1'
ROLE_NAMES = ('open', 'forward_protected', 'narrow', 'transition')
RETENTION_ROLE = -1
DEFAULT_MASSES = dict(retention=.5, open=.025, forward_protected=.10, narrow=.30, transition=.075)
RETENTION_GROUP_MASSES = dict(original_cat=.2, procedural_cat=.4, furniture=.25, generic_clutter=.15)


def validate_width_manifest(manifest, *, path=None, verify_files=False):
    from .generalist_fields import _json_hash
    from .contrastive_rewards import pack_hand_contrast
    from .room_navigation import scene_navigation_radius
    marker = manifest['width_curriculum']
    if marker.get('schema') != SCHEMA or manifest.get('schema') != 'cat-generalist-field-bank-v2':
        raise ValueError('Invalid width curriculum schema')
    if any(key in manifest for key in ('contrastive_specialist', 'specialist', 'hand_protection_curriculum')):
        raise ValueError('Width curriculum cannot silently replace another sampler')
    widths = marker['rung_widths_m']
    if (not isinstance(widths, list) or not widths or any(not isinstance(w,(int,float)) or not math.isfinite(w) or not .4<=w<=.74 for w in widths)
            or any(a<=b for a,b in zip(widths,widths[1:]))):
        raise ValueError('Width rungs must decrease strictly within [.4,.74]')
    masses = marker['role_reset_masses']
    if set(masses) != {'retention',*ROLE_NAMES} or any(not math.isfinite(v) or v<0 for v in masses.values()) or not math.isclose(sum(masses.values()),1.,abs_tol=1e-8):
        raise ValueError('Invalid configured retention/role masses')
    retention = marker['retention_group_masses']
    if set(retention)!=set(RETENTION_GROUP_MASSES) or any(not math.isfinite(v) or v<0 for v in retention.values()) or not math.isclose(sum(retention.values()),1.,abs_tol=1e-8):
        raise ValueError('Invalid retention family masses')
    n = marker['retained_scene_count']
    if type(n) is not int or not 0<=n<len(manifest['scenes']):
        raise ValueError('Invalid retained prefix count')
    if (n==0) != (masses['retention']==0):
        raise ValueError('Retention mass must match retained scene presence')
    if _json_hash(manifest['scenes'][:n]) != marker['retained_scene_records_sha256']:
        raise ValueError('Retention field records differ from immutable prefix')
    if any(s.get('source',{}).get('hand_contrast') for s in manifest['scenes'][:n]):
        raise ValueError('Retention prefix must not contain contrastive scenes')
    if type(marker['min_completed']) is not int or marker['min_completed']<1 or not 0<marker['success_threshold']<=1:
        raise ValueError('Invalid width progression threshold')
    expected={k:masses['retention']*retention[k] for k in retention}
    expected['generic_clutter']+=1-masses['retention']
    if any(not math.isclose(manifest['sampling_group_masses'][k],v,abs_tol=1e-8) for k,v in expected.items()):
        raise ValueError('Family masses disagree with retention/contrastive split')
    groups={}; seen=set()
    for scene in manifest['scenes'][n:]:
        c=scene.get('source',{}).get('hand_contrast',{});role=c.get('role');level=c.get('curriculum_rung')
        if role not in ROLE_NAMES or type(level) is not int or not 0<=level<len(widths):
            raise ValueError('Invalid width scene role/rung')
        if c.get('certificate_semantics')!='width-curriculum-scaffold-v1':
            raise ValueError('Width scenes need explicit scaffold semantics')
        if c.get('narrow_width_range_m') != [widths[level],widths[level]]:
            raise ValueError('Scene width differs from declared rung')
        pack_hand_contrast([{'hand_contrast':c}])
        cert=c.get('certificate',{})
        if (cert.get('route_transition_validated') is not True or cert.get('sampled_kinematic_stances_validated') is not True
                or cert.get('exterior_lanes_sealed') is not True or cert.get('primitive_count')!=35):
            raise ValueError('Missing width scene safety certificate')
        group=c['group_id'];key=(group,role)
        if key in seen:raise ValueError('Duplicate matched width group role')
        seen.add(key);groups.setdefault(group,[]).append((role,level))
        for module,zone,audit in zip(c['modules'],c['zones'],cert['module_audit']):
            if module['role']=='narrow':
                value=audit.get('achievable_hand_field_min_m',0)
                if (module['width_m']!=widths[level] or value<=.025 or audit.get('achievable_body_min_m',0)<=.008
                        or zone.get('certified_achievable_clearance_m')!=value or not 0<=zone.get('sdf_reward_knee',-1)<=.05):
                    raise ValueError('Uncertified rung clearance or reward metadata')
        if verify_files:
            directory=(Path(path).parent/scene['path']).resolve()
            if not directory.is_relative_to(Path(path).parent.resolve()):raise ValueError('Scene escapes bank')
            raw=json.loads((directory/'scene.json').read_text())
            if raw.get('hand_contrast')!=c or raw.get('scene_id')!=scene['scene_id']:raise ValueError('Canonical width metadata differs')
            scene_navigation_radius(raw)
    if any(set(role for role,level in rows)!=set(ROLE_NAMES) or len({level for role,level in rows})!=1 for rows in groups.values()):
        raise ValueError('Width groups must contain all four matched roles at one rung')
    if len(groups)!=marker['group_count'] or {rows[0][1] for rows in groups.values()}!=set(range(len(widths))):
        raise ValueError('Missing width group/rung')


def sampling_plan(manifest):
    """Eight fixed buckets: four retention families and four contrast roles."""
    from .generalist_fields import SAMPLING_GROUPS
    validate_width_manifest(manifest)
    marker=manifest['width_curriculum'];ids=[];levels=[]
    for i,s in enumerate(manifest['scenes']):
        if i<marker['retained_scene_count']:
            ids.append(SAMPLING_GROUPS.index(s['sampling_group']));levels.append(-1)
        else:
            c=s['source']['hand_contrast'];role=ROLE_NAMES.index(c['role'])
            ids.append(4+role);levels.append(c['curriculum_rung'] if role in (2,3) else -1)
    masses=[marker['role_reset_masses']['retention']*marker['retention_group_masses'][k] for k in SAMPLING_GROUPS]
    masses += [marker['role_reset_masses'][k] for k in ROLE_NAMES]
    return np.array(ids,np.int64),np.array(masses,np.float32),np.array(levels,np.int64)
