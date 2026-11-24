#!/bin/bash
# Improve-overnight eval-only variants: several 384-scene evaluations of one COMBO_FIX seed in parallel on one GPU.
# sbatch --array=<seeds> --export=ALL,CLONE=<clone>,TAG=<batch>,VARIANTS='name|args;name|args' hpc/improve_eval.sbatch
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
seed=${SLURM_ARRAY_TASK_ID:?}
if [[ -n "${CKPT:-}" ]]; then ckpt=${CKPT//\{seed\}/$seed}
elif [[ "$seed" = 0 ]]; then ckpt=$L/runs/stationary_hands_20261113_COMBO_FIX/run/resume.pt
else ckpt=$L/runs/stationary_hands_20261113_R4_COMBO_FIX_seed${seed}_318295/run/resume.pt; fi
test -s "$ckpt"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "== improve ${TAG:?} seed $seed, $(git rev-parse HEAD), $ckpt"
pids=()
IFS=';' read -ra items <<< "${VARIANTS:?}"
for item in "${items[@]}"; do
  name=${item%%|*}; extra=${item#*|}
  O=$L/runs/improve_${name}_seed${seed}_${SLURM_ARRAY_JOB_ID}
  mkdir "$O"
  echo "start $name -> $O ($extra)"
  # shellcheck disable=SC2086
  .venv-mjlab/bin/python -u scripts/eval_per_scene.py "$ckpt" --bank data/furniture/tablegap_v1 \
    --scenes "@$HOME/jobs/retention_scenes_20261114.txt" --episodes 2 --num-envs 1536 \
    --max-steps 21000 --nominal --operator-speeds .15 .3 $extra --out-dir "$O" > "$O/evaluation.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "== done status $status"
exit $status
