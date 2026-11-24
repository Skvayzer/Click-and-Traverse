#!/bin/bash
# A1 (throw), A2 (robust), A3 (plan off) evaluations. Env: CLONE BENCH MODEL [SEED via SLURM_ARRAY_TASK_ID] GUIDES BANK E4
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
EVAL=(.venv-mjlab/bin/python -u scripts/eval_per_scene.py)
MODEL=${MODEL:?}; BENCH=${BENCH:?}
seed=${SLURM_ARRAY_TASK_ID:-0}
policy=()
case "$MODEL" in
  planner|noreflex)
    family=COMBO_FIX; [[ "$MODEL" == noreflex ]] && family=COMBO_NOREFLEX
    if [[ "$seed" = 0 ]]; then run="$L/runs/stationary_hands_20261113_$family/run"
    else run="$L/runs/stationary_hands_20261113_R4_${family}_seed${seed}_318295/run"; fi
    grep -q completed_requested_updates "$run/status.json"
    CK="$run/resume.pt"; policy=(--tracker-mix plan=1) ;;
  planner_noplan)   # the same planner-reflex network with its plan input switched off
    run="$L/runs/stationary_hands_20261113_R4_COMBO_FIX_seed${seed}_318295/run"
    [[ "$seed" = 0 ]] && run="$L/runs/stationary_hands_20261113_COMBO_FIX/run"
    grep -q completed_requested_updates "$run/status.json"
    CK="$run/resume.pt"; policy=(--tracker-mix none=1) ;;
  reactive) seed=0; CK=$L/runs/stationary_hands_20261113_V7r/run/snapshots/update_0600.pt ;;
  *) exit 2 ;;
esac
test -s "$CK"
R="$L/runs/overnight_${BENCH}_${MODEL}_seed${seed}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}${TAG:-}"
mkdir "$R"
echo "== $BENCH $MODEL seed $seed, $(git rev-parse HEAD), $CK, $R"
case "$BENCH" in
  A1)
    for speed in 0.15 0.3; do
      "${EVAL[@]}" "$CK" --bank "${BANK:?}" --scenes "@${GUIDES:?}/scenes.txt" --blind-guide "$GUIDES/throw_$speed.npz" \
        --blind-variant throw --blind-replay-starts --episodes 2 --num-envs 768 --max-steps 21000 --nominal \
        --operator-speed "$speed" --out "$R/throw_$speed.json" "${policy[@]}"
    done ;;
  A2)
    for push in nopush push; do
      extra=(); [[ "$push" == push ]] && extra=(--pushes)
      mkdir "$R/$push"
      "${EVAL[@]}" "$CK" --bank "${BANK:?}" --scenes "@${GUIDES:?}/scenes.txt" --episodes 2 --num-envs 1536 --max-steps 21000 \
        --operator-speeds .15 .3 --out-dir "$R/$push" "${extra[@]}" "${policy[@]}"
    done ;;
  A3)
    B=${E4:?}
    "${EVAL[@]}" "$CK" --bank "$B" --scenes "@${B}_packed/scene_ids.txt" --goal-seed-cache "${B}_goal" --planning-benchmark \
      --episodes 2 --num-envs 432 --nominal --max-steps 14500 --operator-speeds .15 .3 .5 --out-dir "$R" "${policy[@]}" ;;
  *) exit 2 ;;
esac
