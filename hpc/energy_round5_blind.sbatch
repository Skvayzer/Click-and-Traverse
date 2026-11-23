#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
variants=(COMBO_NOREFLEX COMBO_FIX COMBO_NOREFLEX COMBO_FIX COMBO_NOREFLEX)
seeds=(0 1 1 2 2)
variant=${variants[SLURM_ARRAY_TASK_ID]}
seed=${seeds[SLURM_ARRAY_TASK_ID]}
if [[ "$seed" = 0 ]]; then
  run=$L/runs/stationary_hands_20261113_$variant/run
else
  run=$L/runs/stationary_hands_20261113_R4_${variant}_seed${seed}_318295/run
fi
grep -qE '"phase":[[:space:]]*"completed_requested_updates"' "$run/status.json"
grep -qE '"updates":[[:space:]]*150,' "$run/status.json"
test -s "$run/resume.pt"
O=$L/runs/energy_round5_blind_${variant}_seed${seed}_$SLURM_ARRAY_JOB_ID
mkdir "$O"
PY=.venv-mjlab/bin/python
for mode in static moving; do
  for speed in 0.15 0.3; do
    name=${mode}_${speed}
    args=()
    if [[ "$seed" = 0 ]]; then args=(--blind-record "$O/trace_$name.npz"); fi
    "$PY" -u scripts/eval_per_scene.py "$run/resume.pt" --bank data/furniture/tablegap_v1 \
      --scenes "@${GUIDES:?}/scenes.txt" --episodes 2 --num-envs 768 --max-steps 21000 --nominal \
      --operator-speed "$speed" --tracker-mix plan=1 --goal-seed-cache data/furniture/tablegap_v1_goal_column \
      --blind-variant "$mode" --blind-guide "$GUIDES/$name.npz" --out "$O/$name.json" "${args[@]}"
  done
done
