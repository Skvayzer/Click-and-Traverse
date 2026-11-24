#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
seed=${SLURM_ARRAY_TASK_ID:?}
if [[ "$seed" = 0 ]]; then
  parent=$L/runs/stationary_hands_20261113_COMBO_FIX/run
else
  parent=$L/runs/stationary_hands_20261113_R4_COMBO_FIX_seed${seed}_318295/run
fi
grep -q completed_requested_updates "$parent/status.json"
grep -qE '"updates":[[:space:]]*150,' "$parent/status.json"
export STUDENT=$parent/resume.pt
export ANCHOR=$L/checkpoints/stationary_hands_E2_u300.pt
test -s "$STUDENT"
test -s "$ANCHOR"
unset ENERGY_RESUME_FROM SMOKE
# Keep each parent's price model, including the legacy default of seeds 1–2.
unset ENERGY_MODEL ENERGY_ALPHA ARM_MULT LEG_MULT NO_ENERGY_PRICE ROUTE_ENERGY ROUTE_ENERGY_P0
export VARIANT=COMBO_FIX FRESH=--fresh-optimizer NUM_ENVS=12288
export EXTRA="--max-updates 150 --seed $seed --refusal-training"
export ENERGY_RUN_TAG=overnight_E2_seed${seed}_$SLURM_ARRAY_JOB_ID
run=$L/runs/stationary_hands_20261113_$ENERGY_RUN_TAG/run
out=$L/runs/overnight_E2_seed${seed}_$SLURM_ARRAY_JOB_ID
[[ ! -e "$out" && ! -e "$run" ]]
echo "== E2 seed $seed, 150 updates from $STUDENT, $(git rev-parse HEAD), $out"
bash hpc/stationary_hands.sbatch
grep -q completed_requested_updates "$run/status.json"
grep -qE '"updates":[[:space:]]*150,' "$run/status.json"
test -s "$run/resume.pt"
mkdir "$out"
.venv-mjlab/bin/python -u scripts/eval_per_scene.py "$run/resume.pt" --bank data/furniture/tablegap_v1   --scenes "@$HOME/jobs/retention_scenes_20261114.txt" --episodes 2 --num-envs 1536 --max-steps 21000   --nominal --operator-speeds .15 .3 --tracker-mix plan=1 --stall-audit --refused-rewards --out-dir "$out"
