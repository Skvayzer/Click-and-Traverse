#!/bin/bash
# run-batch-dyn.sh SHA NAME PLANNER_SCRIPT : waits for the GPU lock, then decides its jobs with PLANNER_SCRIPT and runs them in
# parallel inside the one lock hold. Job lines "tag|ckpt|args" (384 scenes) or "E4:tag|ckpt|args" (72 held-out layouts).
set -uo pipefail
SHA=$1; NAME=$2; PLAN=$3
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export C=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab G=/tmp/cat-plan-blind P=/tmp/cat-energy-test-env/bin/python
export R=/home/konstantinsmirnov/robotics/cat-energy-local-runs E4B=/tmp/cat-overnight/planning_v1 SHA NAME PLAN
echo "queued $NAME $(date -Is)"
flock -x /tmp/cat-energy-dep1-gpu.lock bash -c '
  cd /tmp/cat-improve/code-$SHA
  $P $PLAN > /tmp/cat-improve/$NAME.jobs 2> /tmp/cat-improve/$NAME.notes || { echo "planning failed"; cat /tmp/cat-improve/$NAME.notes; exit 1; }
  cat /tmp/cat-improve/$NAME.notes; cat /tmp/cat-improve/$NAME.jobs
  pids=()
  while IFS="|" read -r tag ckpt extra; do
    [[ -z "$tag" ]] && continue
    if [[ "$tag" == E4:* ]]; then
      tag=E4_${tag#E4:}; O=$R/improve_${tag}_$SHA; mkdir -p "$O"
      $P -u scripts/eval_per_scene.py "$ckpt" --bank $E4B --scenes @${E4B}_packed/scene_ids.txt --goal-seed-cache ${E4B}_goal \
        --planner-checkpoint $G/planner.pt --planning-benchmark --episodes 2 --num-envs 432 --nominal --max-steps 14500 \
        --operator-speeds .15 .3 .5 $extra --out-dir "$O" > "$O/evaluation.log" 2>&1 &
    else
      O=$R/improve_${tag}_$SHA; mkdir -p "$O"
      $P -u scripts/eval_per_scene.py "$ckpt" --nominal --subset-fields --bank $C/data/furniture/tablegap_v1 --scenes @$G/guides/scenes.txt \
        --goal-seed-cache $G/goal-cache --planner-checkpoint $G/planner.pt $extra \
        --episodes 2 --num-envs 1536 --max-steps 21000 --operator-speeds .15 .3 --out-dir "$O" > "$O/evaluation.log" 2>&1 &
    fi
    echo "start $tag $(date -Is) -> $O"
    pids+=($!)
  done < /tmp/cat-improve/$NAME.jobs
  for p in "${pids[@]}"; do wait $p; echo "pid $p exit $?"; done'
echo "finished $NAME $(date -Is)"
