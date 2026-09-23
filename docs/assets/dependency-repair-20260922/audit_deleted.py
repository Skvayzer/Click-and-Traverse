import subprocess,json
from pathlib import Path
names=[f'cat_flat_hand_balance_{v}_20260921{s}' for v in ('v3','v4') for s in ('','_collision','_resets')]+['cat_hand_cont_30720_20260921','cat_hand_approach_r3_h2_30720_20260921','cat_hand_raised_v3_pilot50_20260921','cat_hand_interim_30720_20260921','cat_hand_interim2_30720_20260921']
args=['rg','--hidden','--no-ignore','-n','-F','-g','!.git/**','-g','!.venv*/**','-g','!**/__pycache__/**','-g','!docs/assets/dependency-repair-20260922/**']
out=Path('docs/assets/dependency-repair-20260922'); counts={}
with (out/'deleted-path-hits.txt').open('w') as f:
 for name in names:
  result=subprocess.run(args+['--',name,'.'],capture_output=True,text=True)
  assert result.returncode in (0,1),result.stderr
  lines=result.stdout.splitlines();counts[name]=len(lines)
  f.write(f'### {name}: {len(lines)} matching lines\n'+result.stdout+'\n')
(out/'grep-counts.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts,indent=2))
