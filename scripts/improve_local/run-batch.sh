#!/bin/bash
# usage: run-batch.sh SHA NAME "variant:checkpoint:extra args" ...   (all run in parallel under one GPU lock hold)
set -uo pipefail
SHA=$1; NAME=$2; shift 2
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab; G=/tmp/cat-plan-blind; P=/tmp/cat-energy-test-env/bin/python
R=/home/konstantinsmirnov/robotics/cat-energy-local-runs
cd /tmp/cat-improve/code-$SHA
cat > /tmp/cat-improve/$NAME.jobs <<J
$(printf '%s\n' "$@")
J
echo "queued $NAME $(date -Is)"
flock -x /tmp/cat-energy-dep1-gpu.lock bash -c '
  pids=()
  while IFS="|" read -r tag ckpt extra; do
    [[ -z "$tag" ]] && continue
    O='"$R"'/improve_${tag}_'"$SHA"'
    mkdir -p "$O"
    echo "start $tag $(date -Is) -> $O"
    '"$P"' -u scripts/eval_per_scene.py "$ckpt" --nominal --subset-fields --bank '"$C"'/data/furniture/tablegap_v1 --scenes @'"$G"'/guides/scenes.txt \
      --goal-seed-cache '"$G"'/goal-cache --planner-checkpoint '"$G"'/planner.pt $extra \
      --episodes 2 --num-envs 1536 --max-steps 21000 --operator-speeds .15 .3 --out-dir "$O" > "$O/evaluation.log" 2>&1 &
    pids+=($!)
  done < /tmp/cat-improve/'"$NAME"'.jobs
  for p in "${pids[@]}"; do wait $p; echo "pid $p exit $?"; done'
echo "finished $NAME $(date -Is)"
