#!/bin/bash
# C3 (docs/cos_planner_reflex.md): tracker + reflex fine-tune on COS. Warm start = S3 tracker v2 seed ${PARENT_SEED:-1}
# (u150), bank = tablegap_v1 + COS train scenes (BANK_OVERRIDE), full-path plans = S3 oracle bank + COS tracked references
# (FULLPATH spec), online planner = PLANNER (the C2 planner when it exists). SMOKE=1: 3 updates at 4096 worlds.
#   CLONE=... FULLPATH=<spec.json> BANK_OVERRIDE=/l/.../cos/c3/bank_train GOAL_SEED_CACHE=/l/.../cos/c3/bank_train_goal \
#   PLANNER=<planner.pt> SEED=0 sbatch --export=ALL hpc/cos_c3.sbatch
#SBATCH --job-name=cos_c3
#SBATCH --output=/l/users/%u/cos/logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=54000M
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
seed=${SEED:-0}; ps=${PARENT_SEED:-1}
parent=$(ls -d $L/runs/stationary_hands_20261113_fullpath_S3_seed${ps}_*/run | head -1)
grep -qE '"updates":[[:space:]]*150,' "$parent/status.json"
export STUDENT=$parent/resume.pt ANCHOR=$L/checkpoints/stationary_hands_E2_u300.pt
test -s "$STUDENT"; test -s "$ANCHOR"; test -s "${FULLPATH:?}"; test -s "${BANK_OVERRIDE:?}_packed/manifest.json"
unset ENERGY_RESUME_FROM ENERGY_MODEL ENERGY_ALPHA ARM_MULT LEG_MULT NO_ENERGY_PRICE ROUTE_ENERGY ROUTE_ENERGY_P0
export VARIANT=COMBO_FIX FRESH=--fresh-optimizer FULLPATH BANK_OVERRIDE GOAL_SEED_CACHE=${GOAL_SEED_CACHE:?} MAX_FOREIGN_MIB=${MAX_FOREIGN_MIB:-4000}
[[ -n "${PLANNER:-}" ]] && export PLANNER
if [[ -n "${SMOKE:-}" ]]; then
  export NUM_ENVS=${NUM_ENVS:-4096} SMOKE_UPDATES=${SMOKE_UPDATES:-3}
else
  export NUM_ENVS=${NUM_ENVS:-12288} EXTRA="--max-updates ${UPDATES:-150} --seed $seed"
  export ENERGY_RUN_TAG=${TAG:-cos_C3}_seed${seed}_${SLURM_JOB_ID}
fi
echo "== C3 seed $seed from $STUDENT, $(git rev-parse HEAD), FULLPATH=$FULLPATH BANK=$BANK_OVERRIDE PLANNER=${PLANNER:-v1} SMOKE=${SMOKE:-}"
bash hpc/stationary_hands.sbatch
