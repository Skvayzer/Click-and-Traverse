#!/bin/bash
# G2: plain SONIC (zero residual) driven by its generator along the field route; PHASE=smoke|384|E4.
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
A=$HOME/cat_worktrees/overnight-brace-inputs
export PYTHONPATH="$A/ort-cu118${PYTHONPATH:+:$PYTHONPATH}"
libs="$A/cudnn8/nvidia/cudnn/lib"
for p in "$CONDA_PREFIX"/lib/python3.12/site-packages/nvidia/*/lib; do libs="$libs:$p"; done
export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
P=.venv-mjlab/bin/python
export SONIC_DIR=$L/sonic_release
ON=$L/runs/stationary_hands_20261113_COMBO_FIX/run/resume.pt
test -s "$ON"
O=$L/runs/sonic_G2_${PHASE:?}_$SLURM_JOB_ID
mkdir "$O"
COMMON=(zero --sonic --nominal --tracker-mix field=1 --env-from "$ON" --sonic-generator-sessions 5)
echo "== SONIC G2 $PHASE, $(git rev-parse HEAD), $O, $(date -Is)"
case "$PHASE" in
 smoke)
   head -n 3 "$HOME/jobs/retention_scenes_20261114.txt" >| "$O/scenes.txt"
   "$P" -u scripts/eval_per_scene.py "${COMMON[@]}" --bank data/furniture/tablegap_v1 --scenes "@$O/scenes.txt" \
     --episodes 1 --num-envs 3 --max-steps 500 --operator-speed .3 --out "$O/smoke.json"
   test -s "$O/smoke.json" ;;
 384)
   "$P" -u scripts/eval_per_scene.py "${COMMON[@]}" --bank data/furniture/tablegap_v1 \
     --scenes "@$HOME/jobs/retention_scenes_20261114.txt" --episodes 2 --num-envs 1536 --max-steps 21000 \
     --operator-speeds .15 .3 --out-dir "$O" ;;
 E4)
   B=$HOME/cat_worktrees/energy_overnight_38b62ce/inputs/planning_v1
   "$P" -u scripts/eval_per_scene.py "${COMMON[@]}" --bank "$B" --scenes "@${B}_packed/scene_ids.txt" \
     --goal-seed-cache "${B}_goal" --planning-benchmark --episodes 2 --num-envs 432 --max-steps 14500 \
     --operator-speeds .15 .3 .5 --out-dir "$O" ;;
 *) exit 2 ;;
esac
echo "== done $(date -Is)"
