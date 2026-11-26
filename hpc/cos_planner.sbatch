#!/bin/bash
# C2: planner on COS (docs/cos_planner_reflex.md). Mixes COS planner data with planner v1's data (symlinked shards),
# warm-starts from planner v1, trains STEPS, then the kinematic C2 evaluation of planner v1 vs the new planner.
#   CLONE=~/cat_worktrees/cos_<sha> COSDATA=/l/.../cos/planner_data_v1 OUT=/l/.../cos/c2/planner_cos_v1 sbatch --export=ALL hpc/cos_planner.sbatch
#SBATCH --job-name=cos_planner
#SBATCH --output=/l/users/%u/cos/logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=54000M
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov; V1=$L/runs/wbp/planner_v1
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root OMP_NUM_THREADS=8
cd "${CLONE:?}"; P=.venv-mjlab/bin/python
echo "== cos_planner $(git rev-parse --short HEAD) on $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
M=${OUT:?}/mix; mkdir -p $M/train $M/test
if [[ ! -f $M/info.json ]]; then
  ncos=$(ls ${COSDATA:?}/train/*_state.npy | wc -l); nv1=$(ls $V1/all/train/*_state.npy | wc -l)
  keep=$(( ncos < nv1 ? ncos : nv1 )); keep=$(( keep > 15 ? 15 : keep ))       # <= 300k v1 examples (20k per shard)
  for f in $(ls $V1/all/train/*_state.npy | shuf --random-source=<(yes) | head -$keep); do k=$(basename $f _state.npy)
    for x in grid state chunk goal meta pose; do ln -sf $V1/all/train/${k}_$x.npy $M/train/v${k}_$x.npy; done; done
  for f in $COSDATA/train/*_state.npy; do k=$(basename $f _state.npy)
    for x in grid state chunk goal meta pose; do ln -sf $COSDATA/train/${k}_$x.npy $M/train/c${k}_$x.npy; done; done
  for d in $V1/all/test $COSDATA/test; do t=$([[ $d == $V1* ]] && echo v || echo c)
    for f in $d/*_state.npy; do k=$(basename $f _state.npy)
      for x in grid state chunk goal meta pose; do ln -sf $d/${k}_$x.npy $M/test/$t${k}_$x.npy; done; done; done
  $P -c "import json;a=json.load(open('$V1/all/info.json'));b=json.load(open('$COSDATA/info.json'));a['scene_types']=list(a['scene_types'])+[None]*(100-len(a['scene_types']))+b['scene_types'];a['schema']='wbp-planner-data-v2-mixed-cos';a['cos']=b;json.dump(a,open('$M/info.json','w'),indent=1)"
  echo "mixed: v1 shards $keep, cos shards $ncos"
fi
$P -c "import torch;c=torch.load('$V1/train/planner.pt',map_location='cpu',weights_only=True);print('v1 args',c['args'])"
[[ -f $OUT/train/planner.pt ]] || $P -u scripts/planner_train.py --data $M --out $OUT/train --steps ${STEPS:-20000} --device cuda:0 \
    --init $V1/train/planner.pt --warmup ${WARMUP:-500} --lr ${LR:-1.5e-4} ${TRAIN_EXTRA:-}
$P -u scripts/cos_planner_eval.py --cos-data $COSDATA --bench-data $V1/all --checkpoint v1=$V1/train/planner.pt \
    --checkpoint cos=$OUT/train/planner.pt --out $OUT/eval --device cuda:0
echo "== done $(date -Is)"
