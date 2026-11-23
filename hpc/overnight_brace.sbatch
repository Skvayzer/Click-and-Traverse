#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
A=${BRACE_INPUTS:?}
export PYTHONPATH="$A/ort-cu118${PYTHONPATH:+:$PYTHONPATH}"
libs="$A/cudnn8/nvidia/cudnn/lib"
for p in "$CONDA_PREFIX"/lib/python3.12/site-packages/nvidia/*/lib; do libs="$libs:$p"; done
export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
P=.venv-mjlab/bin/python
ON=$L/runs/stationary_hands_20261113_COMBO_FIX/run/resume.pt
test -s "$ON"
O=$L/runs/overnight_E6_BRACE_${PHASE:?}_$SLURM_JOB_ID
mkdir "$O"
COMMON=(--nominal --tracker-mix plan=1 --env-from "$ON"
        --brace-config "$A/model/unified_pipeline.yaml" --brace-robot "$A/robot/mjcf/g1_holo_compat.xml")
cd "$CLONE"
echo "== E6 BRACE ${PHASE}, $(git rev-parse HEAD), $O"
case "$PHASE" in
 smoke)
   head -n 3 "$HOME/jobs/retention_scenes_20261114.txt" >| "$O/scenes.txt"
   "$P" -u scripts/eval_per_scene.py "$A/model/unified_pipeline.onnx" "${COMMON[@]}"       --bank data/furniture/tablegap_v1 --scenes "@$O/scenes.txt" --episodes 1 --num-envs 3       --max-steps 500 --operator-speed .3 --out "$O/smoke.json"
   test -s "$O/smoke.json"
   ;;
 384)
   "$P" -u scripts/eval_per_scene.py "$A/model/unified_pipeline.onnx" "${COMMON[@]}"       --bank data/furniture/tablegap_v1 --scenes "@$HOME/jobs/retention_scenes_20261114.txt"       --episodes 2 --num-envs 1536 --max-steps 21000 --operator-speeds .15 .3 --out-dir "$O"
   ;;
 E4)
   B=${E4_BANK:?}
   "$P" -u scripts/eval_per_scene.py "$A/model/unified_pipeline.onnx" "${COMMON[@]}"       --bank "$B" --scenes "@${B}_packed/scene_ids.txt" --goal-seed-cache "${B}_goal"       --planning-benchmark --episodes 2 --num-envs 432 --max-steps 14500 --operator-speeds .15 .3 .5 --out-dir "$O"
   ;;
 *) exit 2 ;;
esac
