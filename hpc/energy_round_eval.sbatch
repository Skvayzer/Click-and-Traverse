#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1
L=/l/users/konstantin.smirnov
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
PY=.venv-mjlab/bin/python
O=$L/runs/energy_round2_${ENERGY_CASE:?}_$SLURM_JOB_ID
mkdir "$O"
echo "== job $SLURM_JOB_ID on $(hostname), $ENERGY_CASE, $(git rev-parse HEAD), output $O"
case "$ENERGY_CASE" in
  standing)
    "$PY" scripts/eval_standing_hands.py --checkpoint       "$L/checkpoints/generalist_robust_delay_v1_u150.pt"       "$L/runs/stationary_hands_20261113_S1b_seed1/run/resume.pt"       "$L/checkpoints/stationary_hands_S2_seed1_u180.pt"       "$L/runs/stationary_hands_20261113_S2_seed2/run/resume.pt"       --rows 0:2000 --steps 800 --out "$O/standing.json"
    ;;
  combo1b)
    "$PY" scripts/eval_per_scene.py "$L/checkpoints/plan_reflex_COMBO1b_u500.pt"       --bank data/furniture/tablegap_v1 --scenes "@$HOME/jobs/retention_scenes_20261114.txt"       --episodes 2 --num-envs 1536 --max-steps 21000 --nominal --operator-speeds 0.15 0.3       --tracker-mix plan=1 --out-dir "$O"
    ;;
  trained)
    "$PY" scripts/eval_standing_hands.py --checkpoint       "$L/runs/stationary_hands_20261113_${ENERGY_RUN_TAG:?}/run/resume.pt"       --rows 0:2000 --steps 800 --out "$O/standing.json"
    ;;
  *) echo "Unknown energy case $ENERGY_CASE" >&2; exit 2 ;;
esac
echo "== completed $ENERGY_CASE at $(date -Is)"
