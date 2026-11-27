#!/bin/bash
# BlinkMimic training + evaluation (docs/cos_blinkmimic.md). RUN=<log dir on Lustre> [MOTION_DIR=<dir> BM_MOTION_GLOB=*.npz]
# [EXTRA="train.py flags"] [EVAL=1 EVAL_ARGS="--group 18 --reps 4"] sbatch --export=ALL hpc/bm_train.sbatch
#SBATCH --job-name=bm_train
#SBATCH --output=/l/users/%u/cos/tracker/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
export PATH=/usr/local/bin:/usr/bin:/bin
set -euo pipefail
T=/l/users/konstantin.smirnov/cos/tracker; export PYTHON=$T/venv/bin/python OMP_NUM_THREADS=4 MUJOCO_GL=egl
cd ~/blinkmimic
echo "== bm $(git rev-parse --short HEAD) on $(hostname) $(date -Is) RUN=$RUN MOTION_DIR=${MOTION_DIR:-lafan1} EXTRA=${EXTRA:-}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
mkdir -p "$RUN"
if [[ -z "${EVAL_ONLY:-}" ]]; then
  bash scripts/train.sh "$RUN" ${MOTION_DIR:+--motion_dir $MOTION_DIR} ${EXTRA:-} | grep -v "^\s*$" | tail -n 40
fi
if [[ "${EVAL:-1}" != 0 ]]; then
  PYTHONPATH=src $PYTHON -u tools/eval_batch.py --run_dir "$RUN" ${EVAL_ARGS:---group 18 --reps 4} 2>&1 | tail -n 15
  PYTHONPATH=src $PYTHON -u tools/time_to_target.py "$RUN" 2>&1 | tail -n 15
fi
if [[ -n "${METRICS:-}" ]]; then        # accuracy on the training set and on held-out motions (last checkpoint)
  CK=$(ls $RUN/ckpt_*.pt | sort | tail -1)
  for M in ${MOTION_DIR:-} ${HELDOUT:-}; do
    PYTHONPATH=src BM_MOTION_GLOB=${BM_MOTION_GLOB:-*.csv} $PYTHON -u tools/eval_metrics.py $CK --motion_dir $M --out $RUN/metrics_$(basename $M) 2>&1 | tail -n 3
  done
fi
echo "== done $(date -Is)"
