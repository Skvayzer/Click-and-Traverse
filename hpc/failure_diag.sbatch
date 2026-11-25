#!/bin/bash
# Failure diagnosis on the 144 side-gap and course scenes: one policy per array task. Env: CLONE GUIDES
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
V7R=$L/runs/stationary_hands_20261113_V7r/run/snapshots/update_0400.pt
FIX=$L/runs/stationary_hands_20261113_COMBO_FIX/run/resume.pt
CAT=$HOME/cat_worktrees/energy_overnight_38b62ce/inputs/original-cat-expanded.npz
names=(cat v7r_u400 e2_u300 planner_on planner_off arbitrate)
n=${names[SLURM_ARRAY_TASK_ID]}
extra=()
case $n in
  cat) ck=$CAT; extra=(--env-from "$V7R") ;;
  v7r_u400) ck=$V7R ;;
  e2_u300) ck=$L/checkpoints/stationary_hands_E2_u300.pt ;;
  planner_on) ck=$FIX; extra=(--tracker-mix plan=1) ;;
  planner_off) ck=$FIX; extra=(--tracker-mix none=1) ;;
  arbitrate) ck=$FIX; extra=(--tracker-mix plan=1 --plan-select value_safe --plan-arbitrate -3.719) ;;
esac
test -s "$ck"
R=$L/runs/diag_fail_${n}_${SLURM_ARRAY_JOB_ID}
mkdir "$R"
echo "== diag $n $(git rev-parse HEAD) $ck $R"
.venv-mjlab/bin/python -u scripts/eval_per_scene.py "$ck" "${extra[@]}" --bank data/furniture/tablegap_v1 --scenes "@${GUIDES:?}/scenes144.txt" \
  --nominal --episodes 2 --num-envs 576 --max-steps 21000 --operator-speeds .15 .3 --subset-fields --diagnose "$R/diag.json" --out-dir "$R"
