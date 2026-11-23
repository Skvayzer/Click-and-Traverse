#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
key=${SLURM_ARRAY_TASK_ID}
case "${PHASE:?}" in
  E0) seed=$key; variant=baseline; reps=1; extra=(--stall-audit) ;;
  E1) modes=(fallback margin both); seed=$((key % 3)); variant=${modes[key / 3]}; reps=2
      extra=(--refusal-fix "$variant" --stall-audit) ;;
  *) exit 2 ;;
esac
if [[ "$seed" = 0 ]]; then run=$L/runs/stationary_hands_20261113_COMBO_FIX/run
else run=$L/runs/stationary_hands_20261113_R4_COMBO_FIX_seed${seed}_318295/run; fi
grep -q completed_requested_updates "$run/status.json"
grep -qE '"updates":[[:space:]]*150,' "$run/status.json"
test -s "$run/resume.pt"
O=$L/runs/overnight_${PHASE}_${variant}_seed${seed}_$SLURM_ARRAY_JOB_ID
mkdir "$O"
echo "== $PHASE $variant seed $seed, $(git rev-parse HEAD), $run/resume.pt, $O"
.venv-mjlab/bin/python -u scripts/eval_per_scene.py "$run/resume.pt" --bank data/furniture/tablegap_v1 \
  --scenes "@$HOME/jobs/retention_scenes_20261114.txt" --episodes "$reps" --num-envs "$((reps * 768))" \
  --max-steps 21000 --nominal --operator-speeds .15 .3 --tracker-mix plan=1 --refused-rewards --out-dir "$O" "${extra[@]}"
