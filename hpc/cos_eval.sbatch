#!/bin/bash
# C4 (docs/cos_planner_reflex.md): rollouts with identical starts (--start-replay) on the COS test-200, the 384 narrow
# scenes or the 72 planning layouts. CK=<checkpoint> MODE=plan|none|reactive|fullpath SET=cos|384|E4 OUT=<dir>
# [PLANNER=<planner.pt> for MODE=plan; default planner v1]
#SBATCH --job-name=cos_eval
#SBATCH --output=/l/users/%u/cos/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov; FP=$L/fullpath; C4=$L/cos/c4
export WANDB_MODE=disabled MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"; P=.venv-mjlab/bin/python
echo "== cos_eval $(git rev-parse --short HEAD) $(hostname) CK=$CK MODE=$MODE SET=$SET PLANNER=${PLANNER:-v1} $(date -Is)"
if [[ "$SET" = cos && ! -f $C4/starts/DONE ]]; then
  mkdir -p $C4/starts
  $P -c "
import glob,json,os,numpy as np
for f in glob.glob('$C4/plans_test/*.npz'):
    d=np.load(f); sid=json.loads(str(d['meta']))['scene']
    for k in (0,1):
        np.savez_compressed('$C4/starts/%s_%d.npz'%(sid,k),**{x:d[x] for x in d.files})
open('$C4/starts/DONE','w').write('ok')"
fi
case "$SET" in
  cos) n=$(wc -l < $C4/bank_test_packed/scene_ids.txt); S=$C4/starts
       B=(--bank $C4/bank_test --scenes @$C4/bank_test_packed/scene_ids.txt --goal-seed-cache $C4/bank_test_goal --num-envs $((n*4)) --max-steps 4000) ;;
  384) S=$FP/eval384; B=(--bank data/furniture/tablegap_v1 --scenes @$FP/scenes384.txt --goal-seed-cache data/furniture/tablegap_v1_goal_column --num-envs 1536 --max-steps 21000) ;;
  E4) E4=$HOME/cat_worktrees/energy_overnight_38b62ce/inputs/planning_v1; S=$FP/evalE4; B=(--bank $E4 --scenes @${E4}_packed/scene_ids.txt --goal-seed-cache ${E4}_goal --planning-benchmark --num-envs 288 --max-steps 14500) ;;
esac
case "$MODE" in
  fullpath) M=(--tracker-mix plan=1 --fullpath-plans $S) ;;
  plan) M=(--tracker-mix plan=1 --planner-checkpoint ${PLANNER:-$L/runs/wbp/planner_v1/train/planner.pt}) ;;
  none) M=(--tracker-mix none=1) ;;
  reactive) M=() ;;
esac
mkdir -p "$OUT"
$P -u scripts/eval_per_scene.py "$CK" --nominal --subset-fields --episodes 2 --operator-speeds .15 .3 \
  --start-replay $S --traversal-metrics "${B[@]}" "${M[@]}" --out-dir "$OUT" 2>&1 | tee "$OUT/evaluation.log" | grep -v "^step " 
echo "== done $(date -Is)"
