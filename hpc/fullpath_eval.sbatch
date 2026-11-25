#!/bin/bash
#SBATCH --job-name=fp_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=54G
#SBATCH --time=04:00:00
# S3 evaluation (docs/fullpath_plan.md) with identical starts for every arm (--start-replay).
# CK=<checkpoint> MODE=fullpath|plan|none SET=384|E4 OUT=<dir>
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov; FP=$L/fullpath
export WANDB_MODE=disabled MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
used=$(nvidia-smi --id=${CUDA_VISIBLE_DEVICES%%,*} --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
echo "== $(hostname) GPU ${CUDA_VISIBLE_DEVICES}: ${used} MiB in use"
case "$SET" in
  384) S=$FP/eval384; B=(--bank data/furniture/tablegap_v1 --scenes @$FP/scenes384.txt --goal-seed-cache data/furniture/tablegap_v1_goal_column --num-envs 1536 --max-steps 21000) ;;
  E4) E4=$HOME/cat_worktrees/energy_overnight_38b62ce/inputs/planning_v1; S=$FP/evalE4; B=(--bank $E4 --scenes @${E4}_packed/scene_ids.txt --goal-seed-cache ${E4}_goal --planning-benchmark --num-envs 288 --max-steps 14500) ;;
esac
case "$MODE" in
  fullpath) M=(--tracker-mix plan=1 --fullpath-plans $S) ;;
  plan) M=(--tracker-mix plan=1 --planner-checkpoint $L/runs/wbp/planner_v1/train/planner.pt) ;;
  none) M=(--tracker-mix none=1) ;;
  reactive) M=() ;;
esac
mkdir -p "$OUT"
.venv-mjlab/bin/python -u scripts/eval_per_scene.py "$CK" --nominal --subset-fields --episodes 2 --operator-speeds .15 .3 \
  --start-replay $S "${B[@]}" "${M[@]}" --out-dir "$OUT" 2>&1 | tee "$OUT/evaluation.log"
