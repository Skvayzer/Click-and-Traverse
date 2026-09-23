#!/usr/bin/env python3
"""Build certified width scaffolds and an immutable, hardlinked retention mixture (CPU only)."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cat_ppo.furniture.generalist_fields import load_generalist_manifest,make_clutter_fields,_json_hash,sha256,_field_records
from cat_ppo.furniture.width_curriculum import SCHEMA,DEFAULT_MASSES,RETENTION_GROUP_MASSES,validate_width_manifest


def link_scene(source,target):
    target.mkdir(parents=True,exist_ok=True)
    for name in ('sdf.npy','bf.npy','gf.npy','obs.npy','source.json','scene.json'):
        old,new=source/name,target/name
        if not old.exists():continue
        if new.exists():
            if not os.path.samefile(old,new) and sha256(old)!=sha256(new):raise ValueError(f'Existing file differs: {new}')
        else:
            # Do not silently spend 13+ GB copying a retention bank.
            os.link(old,new)


def publish(manifest,path):
    manifest.pop('manifest_sha256',None);manifest['manifest_sha256']=_json_hash(manifest)
    pending=path.with_suffix('.pending.json');pending.write_text(json.dumps(manifest,indent=2)+'\n')
    load_generalist_manifest(pending,verify_files=True)
    pending.replace(path)


def clean_header(base, *, retention):
    """Keep source generation statistics explicitly scoped to the retained bank."""
    result=copy.deepcopy(base)
    keys=('requested_generated_count','duplicate_generated_count','rejected_generated_count',
          'requested_family_counts','retained_family_counts','procedural_generation','navigation_migration')
    provenance={key:result.pop(key) for key in keys if key in result}
    if retention:result['retention_source_provenance']=provenance
    return result


def build(base_manifest,output,merged_output,*,widths=(.70,.64,.58,.52,.46,.40),seeds=(20260919,20260920,20260921,20260922), role_masses=None, retention_group_masses=None, min_completed=64, success_threshold=.6):
    from cat_ppo.furniture.contrastive_passages import generate_contrastive_group
    import math
    role_masses=dict(DEFAULT_MASSES if role_masses is None else role_masses)
    retention_group_masses=dict(RETENTION_GROUP_MASSES if retention_group_masses is None else retention_group_masses)
    for actual,expected in ((role_masses,DEFAULT_MASSES),(retention_group_masses,RETENTION_GROUP_MASSES)):
        if set(actual)!=set(expected) or any(not math.isfinite(v) or v<0 for v in actual.values()) or not math.isclose(sum(actual.values()),1.,abs_tol=1e-8):
            raise ValueError('Configured sampling masses must name every bucket and sum to one')
    if not 0<role_masses['retention']<1 or type(min_completed) is not int or min_completed<1 or not 0<success_threshold<=1:
        raise ValueError('Mixture needs positive retention and contrastive mass and valid progression thresholds')
    standalone_masses={k:(v/(1-role_masses['retention']) if k!='retention' else 0.) for k,v in role_masses.items()}
    base_manifest=Path(base_manifest).resolve();output=Path(output).resolve();merged_output=Path(merged_output).resolve()
    if output==merged_output or any(p==base_manifest.parent or p.is_relative_to(base_manifest.parent) for p in (output,merged_output)):
        raise ValueError('Outputs must be distinct and outside the source bank')
    base=load_generalist_manifest(base_manifest,verify_files=True)
    settings=dict(base_manifest_sha256=sha256(base_manifest),widths=list(widths),seeds=list(seeds))
    output.mkdir(parents=True,exist_ok=True);merged_output.mkdir(parents=True,exist_ok=True)
    plan=output/'build-plan.json'
    if plan.exists() and json.loads(plan.read_text())!=settings:raise ValueError('Partial build arguments differ')
    plan.write_text(json.dumps(settings,indent=2)+'\n')
    manifest_path=output/'manifest.json';merged_path=merged_output/'manifest.json'
    if manifest_path.exists():
        specialist=load_generalist_manifest(manifest_path);records=specialist['scenes']
        old=specialist['width_curriculum']
        if old['role_reset_masses']!=standalone_masses or old['retention_group_masses']!=retention_group_masses or old['min_completed']!=min_completed or old['success_threshold']!=success_threshold:
            raise ValueError('Existing immutable curriculum uses different sampling/progression settings')
    else:
        records=[]
        for rung,width in enumerate(widths):
            for seed in seeds:
                cache=output/f'group-r{rung}-{seed}.json'
                if cache.exists():scenes=json.loads(cache.read_text())
                else:
                    print(f'Certifying rung {rung}: {width:.2f}m seed {seed}',flush=True)
                    scenes=generate_contrastive_group(seed,narrow_width_range=(width,width),curriculum_rung=rung)
                    cache.write_text(json.dumps(scenes)+'\n')
                for scene in scenes:
                    relative=Path('scenes')/scene['scene_id'];directory=output/relative;directory.mkdir(parents=True,exist_ok=True)
                    record_path=directory/'field-record.json'
                    if record_path.exists():
                        record=json.loads(record_path.read_text())
                        if record['fields']!=_field_records(directory) or record['scene_sha256']!=sha256(directory/'scene.json'):
                            raise ValueError('Cached width field changed')
                        if record['source']['hand_contrast']!=scene['hand_contrast'] or record['source']['metadata_sha256']!=sha256(directory/'source.json'):
                            raise ValueError('Cached width scene metadata changed')
                    else:
                        record=make_clutter_fields(scene,directory,dx=.04)
                        source=dict(record['source'],kind='width-curriculum-scaffold',hand_contrast=scene['hand_contrast'])
                        (directory/'source.json').write_text(json.dumps(source,indent=2)+'\n');source['metadata_sha256']=sha256(directory/'source.json')
                        record.update(path=str(relative),source=source,task_kind='room',reset_mode='room',crossed_mode='goal_radius',episode_length=4000,sampling_group='generic_clutter')
                        record_path.write_text(json.dumps(record,indent=2)+'\n')
                    records.append(record);print(f'Fields {len(records)}/{len(widths)*len(seeds)*4}: {scene["scene_id"]}',flush=True)
        specialist=clean_header(base,retention=False)
        for key in ('hand_protection_curriculum','specialist','contrastive_specialist'):specialist.pop(key,None)
        specialist.update(scenes=records,scene_count=len(records),original_count=0,byte_verified_original_count=0,reconstructed_original_count=0,
            fields_bytes=sum(f['size_bytes'] for r in records for f in r['fields'].values()),sampling_group_masses=dict(original_cat=0.,procedural_cat=0.,furniture=0.,generic_clutter=1.))
        marker=dict(schema=SCHEMA,rung_widths_m=list(widths),group_count=len(widths)*len(seeds),retained_scene_count=0,
            retained_scene_records_sha256=_json_hash([]),role_reset_masses=standalone_masses,
            retention_group_masses=retention_group_masses,min_completed=min_completed,success_threshold=success_threshold,
            progression='leader first clean goals on current narrow rung; earlier rungs remain eligible',
            build=settings,dynamic_feasibility_validated=False)
        specialist['width_curriculum']=marker
        specialist['retained_family_counts']={'generic_clutter':len(records)}
        publish(specialist,manifest_path)
    merged=clean_header(base,retention=True)
    for key in ('hand_protection_curriculum','specialist','contrastive_specialist'):merged.pop(key,None)
    merged['scenes']=copy.deepcopy(base['scenes'])+copy.deepcopy(records)
    # Cross-filesystem hardlinks may be unavailable. Pin the old bank directly
    # rather than silently duplicating 13+ GiB or allowing unchecked symlinks.
    probe=merged_output/'.retention-link-probe'
    try:
        os.link(base_manifest.parent/base['scenes'][0]['path']/'sdf.npy',probe)
        same_device=True
    except OSError:
        same_device=False
    finally:
        probe.unlink(missing_ok=True)
    if not same_device:
        merged['external_retention_bank']=dict(manifest=str(base_manifest),sha256=sha256(base_manifest),storage='immutable external prefix; no field copies')
    sources=[(manifest_path,records)]
    if same_device:sources.insert(0,(base_manifest,base['scenes']))
    for source_manifest,entries in sources:
        for record in entries:link_scene(source_manifest.parent/record['path'],merged_output/record['path'])
    marker=copy.deepcopy(specialist['width_curriculum']);marker.update(retained_scene_count=len(base['scenes']),
        retained_scene_records_sha256=_json_hash(base['scenes']),role_reset_masses=role_masses)
    group_masses={k:role_masses['retention']*v for k,v in retention_group_masses.items()}
    group_masses['generic_clutter']+=1-role_masses['retention']
    merged.update(width_curriculum=marker,scene_count=len(merged['scenes']),
        fields_bytes=sum(f['size_bytes'] for r in merged['scenes'] for f in r['fields'].values()),
        sampling_group_masses=group_masses)
    from collections import Counter
    merged["retained_family_counts"]=dict(Counter(r["family"] for r in merged["scenes"]))
    if merged_path.exists():
        old=load_generalist_manifest(merged_path);candidate=copy.deepcopy(merged);candidate.pop('manifest_sha256',None)
        if old['manifest_sha256']!=_json_hash(candidate):raise ValueError('Existing merged bank differs')
    else:publish(merged,merged_path)
    print(json.dumps(dict(curriculum=str(manifest_path),merged=str(merged_path),scenes=merged['scene_count'])),flush=True)
    return manifest_path,merged_path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-manifest',type=Path,default=ROOT/'data/furniture/cat_diversity_v3_navigation_20260916/manifest.json')
    p.add_argument('--output',type=Path,default=ROOT/'data/furniture/contrastive_width_v2')
    p.add_argument('--merged-output',type=Path,default=ROOT/'data/furniture/cat_width_retention_v2')
    p.add_argument('--widths',type=float,nargs='+',default=[.70,.64,.58,.52,.46,.40])
    p.add_argument('--seeds',type=int,nargs='+',default=[20260919,20260920,20260921,20260922])
    p.add_argument('--role-masses',type=json.loads,help='JSON object: retention/open/forward_protected/narrow/transition masses')
    p.add_argument('--retention-group-masses',type=json.loads,help='JSON object: four conditional retention-family masses')
    p.add_argument('--min-completed',type=int,default=64)
    p.add_argument('--success-threshold',type=float,default=.6)
    build(**vars(p.parse_args()))
