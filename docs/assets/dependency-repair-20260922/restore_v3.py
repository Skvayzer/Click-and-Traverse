import os, copy, json, shutil
from pathlib import Path
from cat_ppo.furniture.generalist_fields import sha256, _json_hash
from cat_ppo.furniture.raised_only_bank import raised_only, SCHEMA
root=Path.cwd(); base=root/'data/furniture'; old=base/'cat_flat_hand_balance_v2_20260921'; new=base/'cat_flat_hand_balance_v3_20260921'
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')
def publish(p,v):
 v.pop('manifest_sha256',None); v['manifest_sha256']=_json_hash(v); write(p,v)
m=json.loads((old/'manifest.json').read_text()); changes={}
for r in m['scenes']:
 if r.get('source',{}).get('hand_contrast',{}).get('role')=='forward_protected':
  rel=r['path']+'/scene.json'; b=(json.dumps(raised_only(json.loads((old/rel).read_text())),indent=2)+'\n').encode(); changes[rel]=b
  import hashlib
  r['scene_sha256']=hashlib.sha256(b).hexdigest()
m['flat_balance'].update(schema=SCHEMA,validity_source=dict(manifest=str(old/'manifest.json'),sha256=sha256(old/'manifest.json')))
m.pop('manifest_sha256'); m['manifest_sha256']=_json_hash(m)
raw=(json.dumps(m,indent=2)+'\n').encode(); expected=json.loads((base/'cat_flat_hand_balance_v5_20260921/manifest.json').read_text())['flat_balance']['region_geometry_source']['sha256']
assert hashlib.sha256(raw).hexdigest()==expected, 'Reconstruction does not match existing pin'
for suffix in ('','_collision','_resets'):
 src=old.with_name(old.name+suffix); dst=new.with_name(new.name+suffix)
 assert not dst.exists()
 shutil.copytree(src,dst,copy_function=lambda a,b: shutil.copy2(a,b) if a.endswith('.json') else os.link(a,b))
for rel,b in changes.items(): (new/rel).write_bytes(b)
(new/'manifest.json').write_bytes(raw)
collision=new.with_name(new.name+'_collision')
for name in ('manifest.json','geometry-inventory.json','build-plan.json'):
 p=collision/name; obj=json.loads(p.read_text())
 if 'field_manifest_sha256' in obj: obj.update(field_manifest_sha256=sha256(new/'manifest.json'),field_manifest_content_sha256=m['manifest_sha256'])
 for g in obj.get('scenes',[]):
  r=m['scenes'][g['index']]
  if r.get('source',{}).get('hand_contrast',{}).get('role')=='forward_protected': g['provenance']['source_sha256']=r['scene_sha256']
 (publish if name=='manifest.json' else write)(p,obj)
p=new.with_name(new.name+'_resets')/'manifest.json'; obj=json.loads(p.read_text()); obj.update(field_manifest=str(new/'manifest.json'),field_manifest_sha256=sha256(new/'manifest.json'),collision_bank=str(collision/'manifest.json'),collision_bank_sha256=sha256(collision/'manifest.json')); write(p,obj)
print(json.dumps(dict(restored_manifest_sha256=expected,changed_scenes=len(changes),binary_storage='hardlinks to v2; JSON independent'),indent=2))
