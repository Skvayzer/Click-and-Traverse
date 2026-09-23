#!/usr/bin/env python3
"""CPU-only immutable composition: hardlinked retention/narrow fields + tiny empty field."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu';os.environ['OPENBLAS_NUM_THREADS']='1'
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from cat_ppo.furniture.generalist_fields import load_generalist_manifest,scene_directory,sha256,_json_hash,_field_records
from cat_ppo.furniture.balance_bank import SCHEMA,MASSES,SETTINGS
from cat_ppo.furniture.body_collision_bank import build_body_collision_bank

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def publish(p,v):v.pop('manifest_sha256',None);v['manifest_sha256']=_json_hash(v);write(p,v)
def link(a,b):
 b.parent.mkdir(parents=True,exist_ok=True)
 if b.exists():
  if sha256(a)!=sha256(b):raise ValueError(f'Existing file differs {b}')
 else:
  try:os.link(a,b)
  except OSError as error:
   if error.errno!=18 or a.suffix!='.npz':raise
   import shutil
   shutil.copy2(a,b)

def build():
 out=ROOT/'data/furniture/cat_flat_balance_v1_20260920';dest=out/'manifest.json'
 if dest.exists():raise FileExistsError(dest)
 sources=[ROOT/'data/furniture/cat_hand_posture_v1_20260919/manifest.json',ROOT/'data/furniture/cat_width_retention_v2/manifest.json']
 manifests=[load_generalist_manifest(p,verify_files=False) for p in sources]
 selections=[list(range(2338)),[i for i,r in enumerate(manifests[1]['scenes']) if r.get('source',{}).get('hand_contrast',{}).get('role')=='narrow' and r['source']['hand_contrast']['curriculum_rung']<=2]]
 assert len(selections[1])==12
 records=[];geometries=[];pools=[];cout=out.with_name(out.name+'_collision');rout=out.with_name(out.name+'_resets')
 for source,manifest,indices in zip(sources,manifests,selections):
  collision=source.parent.with_name(source.parent.name+'_collision')/'manifest.json';reset=source.parent.with_name(source.parent.name+'_resets')/'manifest.json'
  cm=json.loads(collision.read_text());rm=json.loads(reset.read_text())
  assert cm['field_manifest_sha256']==sha256(source) and rm['field_manifest_sha256']==sha256(source) and rm['collision_bank_sha256']==sha256(collision)
  assert sha256(reset.parent/rm['file'])==rm['sha256']
  pool=np.load(reset.parent/rm['file']);pools.append(pool[indices])
  for i in indices:
   r=copy.deepcopy(manifest['scenes'][i]);folder=scene_directory(manifest,source,r)
   for name in ([] if source==sources[0] else ['sdf.npy','bf.npy','gf.npy','source.json','scene.json']):link(folder/name,out/r['path']/name)
   g=copy.deepcopy(cm['scenes'][i]);newidx=len(records);target=f'geometry/{newidx:05d}.npz'
   assert sha256(collision.parent/g['geometry_file'])==g['geometry_sha256']
   link(collision.parent/g['geometry_file'],cout/target);g.update(index=newidx,geometry_file=target)
   records.append(r);geometries.append(g)
  print('Linked',len(records),'scenes',flush=True)
 folder=out/'flat/balance';folder.mkdir(parents=True,exist_ok=True)
 scene=dict(scene_id='flat-balance-walk-v1',boxes=[],route=[[0.,0.],[100.,0.]],start=[0.,0.,.8],goal=[100.,0.,.8],room_dimensions=[200.,200.,3.])
 source=dict(kind='analytic-empty-flat-ground-v1',flat_balance=SCHEMA,occupancy='conservative-voxel-cell-OBB-intersection-v1',room_navigation='ordered-certified-route-v1')
 write(folder/'scene.json',scene);write(folder/'source.json',source)
 for name in ('sdf','gf','bf'):
  a=np.full((3,3,3),10.,np.float32) if name=='sdf' else np.zeros((3,3,3,3),np.float32)
  if name=='gf':a[...,0]=.6
  np.save(folder/(name+'.npy'),a)
 flat=dict(scene_id=scene['scene_id'],family='generic_clutter',path='flat/balance',shape=[3,3,3],origin=[-100.,-100.,0.],dx=100.,start=scene['start'],goal=scene['goal'],reset_xy_scale=[.08,.08],reset_yaw=0.,sampling_weight=1.,fields=_field_records(folder),source=dict(source,metadata_sha256=sha256(folder/'source.json')),scene_sha256=sha256(folder/'scene.json'),task_kind='room',reset_mode='room',crossed_mode='goal_radius',episode_length=4000,sampling_group='generic_clutter')
 records.append(flat)
 m=copy.deepcopy(manifests[0])
 for key in ('external_retention_bank','hand_posture_upgrade','hand_protection_curriculum'):m.pop(key,None)
 from collections import Counter
 m['retained_family_counts']=dict(Counter(r['family'] for r in records[:2338]))
 for key in ('requested_generated_count','duplicate_generated_count','rejected_generated_count','procedural_generation'):m.pop(key,None)
 m.update(scenes=records,scene_count=len(records),fields_bytes=sum(f['size_bytes'] for r in records for f in r['fields'].values()),storage='Hash-pinned referenced retention; hardlinked narrow fields; analytic empty flat field',sampling_group_masses=dict(original_cat=.2,procedural_cat=.4,furniture=.0625,generic_clutter=.3375))
 m['flat_balance']=dict(retention_storage_root=str(Path(manifests[0]['external_retention_bank']['manifest']).parent),schema=SCHEMA,settings=SETTINGS,masses=list(MASSES),retention_source=dict(manifest=str(sources[0]),sha256=sha256(sources[0])),narrow_source=dict(manifest=str(sources[1]),sha256=sha256(sources[1])),composition=dict(retention=2338,narrow=12,flat=1,hand_tables=0,hand_shelves=0))
 publish(dest,m)
 # Reuse proven geometry caches, rebuilding only the compact shared index arrays.
 geometry=cout/f'geometry/{len(records)-1:05d}.npz';np.savez(geometry,centers=np.empty((0,3)),half_sizes=np.empty((0,3)),rotations=np.empty((0,3,3)))
 geometries.append(dict(index=len(records)-1,scene_id=flat['scene_id'],family=flat['family'],boxes=0,geometry_file=str(geometry.relative_to(cout)),geometry_sha256=sha256(geometry),provenance=dict(kind='canonical-room-OBBs',source_sha256=flat['scene_sha256'],exact_canonical_box_count=0)))
 write(cout/'geometry-inventory.json',dict(scenes=geometries))
 proposal=ROOT/'docs/assets/collision-proxy-proposal-20260916/proposal.json'
 build_body_collision_bank(dest,proposal,cout,workers=1,progress=lambda p:print(p,flush=True))
 from cat_mjlab.constants import DEFAULT_QPOS
 assert pools[0].shape[1:]==pools[1].shape[1:]
 flatpool=np.repeat(np.asarray(DEFAULT_QPOS,np.float32)[None,None],pools[0].shape[1],axis=1)
 pool=np.concatenate([*pools,flatpool]);rout.mkdir(exist_ok=True);np.save(rout/'reset_qpos.npy',pool)
 rm=dict(schema='cat-body-collision-resets-v1',status='complete',backend='composed-certified-cpu-pools',
     field_manifest=str(dest),field_manifest_sha256=sha256(dest),collision_bank=str(cout/'manifest.json'),
     collision_bank_sha256=sha256(cout/'manifest.json'),proxy_sha256=sha256(proposal),
     file='reset_qpos.npy',sha256=sha256(rout/'reset_qpos.npy'),scene_count=len(records),shape=list(pool.shape),
     dtype=str(pool.dtype),poses_per_scene=int(pool.shape[1]),reused_pose_count=int(sum(len(x) for x in pools)*pool.shape[1]),
     empty_ground_nominal_pose_count=int(pool.shape[1]),
     sources=[dict(manifest=str(p.parent.with_name(p.parent.name+'_resets')/'manifest.json'),
                   sha256=sha256(p.parent.with_name(p.parent.name+'_resets')/'manifest.json'),indices=indices)
              for p,indices in zip(sources,selections)],
     scenes=[dict(index=i,scene_id=r['scene_id']) for i,r in enumerate(records)],
     composition='Unchanged pinned source poses and geometry; flat nominal pool is collision-free because its obstacle set is empty')
 write(rout/'manifest.json',rm)
 load_generalist_manifest(dest,verify_files=True)
 from cat_ppo.furniture.contrast_preflight import contrast_preflight
 contrast_preflight(dest,require_hand_contrast=False)  # Legacy balance isolation only.
 print('VALIDATED',dest,flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--hand-objectives',action='store_true',help='Build the explicit v2 hand-objective mixture')
 args=parser.parse_args()
 if args.hand_objectives:
  from scripts.build_hand_balance_bank import build as build_revised
  build_revised()
 else:
  build()
