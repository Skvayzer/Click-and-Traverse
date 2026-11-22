#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1
L=/l/users/konstantin.smirnov
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
case "${ENERGY_CASE:?}" in
  COMBO_FIX|COMBO_NOREFLEX|Flip_arm_high_u150|Flip_arm100)
    run=$L/runs/stationary_hands_20261113_$ENERGY_CASE/run ;;
  *) echo "Unknown evaluation case $ENERGY_CASE" >&2; exit 2 ;;
esac
grep -qE '"phase":[[:space:]]*"completed_requested_updates"' "$run/status.json"
grep -qE '"updates":[[:space:]]*150,' "$run/status.json"
O=$L/runs/energy_round3_${ENERGY_CASE}_$SLURM_JOB_ID
mkdir "$O"
echo "== job $SLURM_JOB_ID on $(hostname), $ENERGY_CASE, $(git rev-parse HEAD), output $O"
PY=.venv-mjlab/bin/python
if [[ "$ENERGY_CASE" = COMBO_* ]]; then
  args=(--bank data/furniture/tablegap_v1 --scenes "@$HOME/jobs/retention_scenes_20261114.txt"
        --episodes 2 --num-envs 1536 --max-steps 21000 --nominal --operator-speeds 0.15 0.3
        --tracker-mix plan=1 --refused-rewards --out-dir "$O")
  "$PY" scripts/eval_per_scene.py "$run/resume.pt" "${args[@]}"
else
  "$PY" scripts/eval_standing_hands.py --checkpoint "$run/resume.pt" --rows 0:2000 --steps 800 --out "$O/standing.json"
fi
