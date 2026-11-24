#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
B=${BANK:?}
EVAL=(".venv-mjlab/bin/python" -u scripts/eval_per_scene.py)
CHECKPOINT=${CHECKPOINT:-}
MODEL=${MODEL:?}
AID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
case "$MODEL" in
  planner|plan_only|field_no_reflex)
    seed=${SLURM_ARRAY_TASK_ID:?}
    family=COMBO_FIX
    if [[ "$MODEL" == field_no_reflex || "$MODEL" == plan_only ]]; then family=COMBO_NOREFLEX; fi
    if [[ "$seed" = 0 ]]; then run="$L/runs/stationary_hands_20261113_$family/run"
    else run="$L/runs/stationary_hands_20261113_R4_${family}_seed${seed}_318295/run"; fi
    grep -q completed_requested_updates "$run/status.json"
    grep -qE '"updates":[[:space:]]*150,' "$run/status.json"
    CHECKPOINT="$run/resume.pt"
    policy=(--tracker-mix plan=1)
    if [[ "$MODEL" == plan_only ]]; then policy+=(--plan-only); fi
    if [[ "$MODEL" == planner && -n "${VARIANT:-}" ]]; then policy+=(--refusal-fix "$VARIANT"); fi
    ;;
  refusal_trained)
    seed=${SLURM_ARRAY_TASK_ID:?}
    run="$L/runs/stationary_hands_20261113_overnight_E2_seed${seed}_${REFUSAL_JOB:?}/run"
    grep -q completed_requested_updates "$run/status.json"
    grep -qE '"updates":[[:space:]]*150,' "$run/status.json"
    CHECKPOINT="$run/resume.pt"
    policy=(--tracker-mix plan=1)
    ;;
  reactive) seed=0; policy=() ;;
  released_cat) seed=0; policy=(--env-from "${ENV_CHECKPOINT:?}") ;;
  no_price|price_reference) seed=0; policy=(--tracker-mix plan=1) ;;
  *) exit 2 ;;
esac
test -s "$CHECKPOINT"
R="$L/runs/overnight_${BENCH:?}_${MODEL}${VARIANT:+_$VARIANT}_seed${seed}_$AID"
mkdir "$R"
echo "== $BENCH $MODEL seed $seed, $(git rev-parse HEAD), $CHECKPOINT, $R"
if [[ "$BENCH" == E4 ]]; then
  "${EVAL[@]}" "$CHECKPOINT" --bank "$B" --scenes "@${B}_packed/scene_ids.txt"     --goal-seed-cache "${B}_goal" --planning-benchmark --episodes 2 --num-envs 432 --nominal     --max-steps 14500 --operator-speeds .15 .3 .5 --out-dir "$R" "${policy[@]}"
elif [[ "$BENCH" == E5 ]]; then
  if [[ "$MODEL" != plan_only ]]; then policy+=(--blind-replay-starts); fi
  for mode in static moving; do
    for speed in 0.15 0.3; do
      stem=${mode}_$speed
      "${EVAL[@]}" "$CHECKPOINT" --bank "$B" --scenes "@${GUIDES:?}/scenes.txt"         --blind-guide "$GUIDES/$stem.npz" --blind-variant "$mode" --episodes 2 --num-envs 768         --max-steps 21000 --nominal --operator-speed "$speed" --out "$R/$stem.json" "${policy[@]}"
    done
  done
else
  exit 2
fi
