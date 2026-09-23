import json, shlex
from pathlib import Path
from cat_ppo.furniture.generalist_fields import load_generalist_manifest, sha256
from cat_ppo.furniture.body_collision_bank import load_body_collision_bank
from cat_ppo.furniture.contrast_preflight import contrast_preflight
from train_cat_mjlab import parser
root=Path.cwd(); results={}
for version,date in [('v3','20260921'),('v5','20260921'),('v6','20260922')]:
 base=root/f'data/furniture/cat_flat_hand_balance_{version}_{date}'
 m=load_generalist_manifest(base/'manifest.json',verify_files=True)
 c=base.with_name(base.name+'_collision')/'manifest.json'
 load_body_collision_bank(c,expected_field_manifest=base/'manifest.json')
 r=base.with_name(base.name+'_resets')/'manifest.json'; reset=json.loads(r.read_text())
 assert reset['field_manifest_sha256']==sha256(base/'manifest.json')
 assert reset['collision_bank_sha256']==sha256(c)
 assert reset['sha256']==sha256(r.parent/reset['file'])
 results[version]=dict(full_file_verification=True,collision_verified=True,reset_hashes_verified=True,preflight=contrast_preflight(base/'manifest.json',require_hand_contrast=True))
 print(version,'verified',flush=True)
pilot=root/'configs/pilots/hand_reward_floor_v6_50.sh'
words=shlex.split(pilot.read_text().replace('\\\n',' ')); start=words.index('train_cat_mjlab.py')+1
args=parser().parse_args(words[start:]); assert args.command=='run' and args.from_scratch and args.require_hand_contrast
assert args.bank_manifest.resolve()==base/'manifest.json'
results['pilot']=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},method='Parsed exact shell command with real parser; verified field/collision/reset banks and preflight separately. Never imported runner, called main/run, constructed simulation, or stepped an environment.',training_started=False,gpu_used=False)
(root/'docs/assets/dependency-repair-20260922/validation.json').write_text(json.dumps(results,indent=2)+'\n')
