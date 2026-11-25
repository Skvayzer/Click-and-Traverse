#!/bin/bash
#SBATCH --job-name=fp_s3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=54G
#SBATCH --time=12:00:00
# Tracker v2 (S3 of docs/fullpath_plan.md): COMBO_FIX seed $SEED warm start + --tracker-fullpath $FULLPATH.
# SMOKE=1: 3 updates at NUM_ENVS (default 4096) into runs/smoke_hands_COMBO_FIX_<job>.
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
seed=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
if [[ "$seed" = 0 ]]; then parent=$L/runs/stationary_hands_20261113_COMBO_FIX/run
else parent=$L/runs/stationary_hands_20261113_R4_COMBO_FIX_seed${seed}_318295/run; fi
grep -qE '"updates":[[:space:]]*150,' "$parent/status.json"
export STUDENT=$parent/resume.pt ANCHOR=$L/checkpoints/stationary_hands_E2_u300.pt
test -s "$STUDENT"; test -s "$ANCHOR"; test -s "${FULLPATH:?}"
unset ENERGY_RESUME_FROM ENERGY_MODEL ENERGY_ALPHA ARM_MULT LEG_MULT NO_ENERGY_PRICE ROUTE_ENERGY ROUTE_ENERGY_P0
export VARIANT=COMBO_FIX FRESH=--fresh-optimizer FULLPATH
if [[ -n "${SMOKE:-}" ]]; then
  export NUM_ENVS=${NUM_ENVS:-4096} SMOKE_UPDATES=${SMOKE_UPDATES:-3}
else
  export NUM_ENVS=${NUM_ENVS:-12288} EXTRA="--max-updates ${UPDATES:-150} --seed $seed"
  export ENERGY_RUN_TAG=${TAG:-fullpath_S3}_seed${seed}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
fi
echo "== S3 seed $seed from $STUDENT, $(git rev-parse HEAD), FULLPATH=$FULLPATH SMOKE=${SMOKE:-}"
bash hpc/stationary_hands.sbatch
