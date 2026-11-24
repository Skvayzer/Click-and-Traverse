#!/bin/bash
# Expert iteration for planner v1: collect contact-free 2 s windows from the planner–reflex tracker's own rollouts
# (plan source only, evaluation scenes excluded, narrow scene types up-weighted), merge them with planner v1's data,
# and fine-tune planner v1 on the union. Stages skip when their output exists.
#   sbatch --export=ALL,CLONE=<clone>,RUN=<name> hpc/improve_expert_iter.sbatch
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
D=$L/runs/wbp/${RUN:?}; V1=$L/runs/wbp/planner_v1
SRC=${SRC:-$L/runs/stationary_hands_20261113_COMBO_FIX/run/resume.pt}
mkdir -p "$D"
echo "== expert iteration $RUN on $(hostname) $(git rev-parse HEAD) from $SRC at $(date -Is)" | tee -a "$D/run_info.txt"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
if [[ ! -f $D/data_combo/info.json ]]; then
  rm -rf "$D/data_combo"
  .venv-mjlab/bin/python -u scripts/planner_collect.py --checkpoint "$SRC" --bank data/furniture/tablegap_v1 --out "$D/data_combo" \
    --device cuda:0 --num-envs ${ENVS:-4096} --steps 1500 --rounds ${ROUNDS:-3} --test-rounds 1 --max-train ${MAX_NEW:-150000} \
    --max-test 15000 --source-id 4 --tracker-mix plan=1 --exclude-scenes "$HOME/jobs/retention_scenes_20261114.txt" \
    --types-scale side_gap=2 narrow_passage=2 protected_passage=2 transition_passage=2 table_edges=2 --seed 7
fi
[[ -f $D/all/info.json ]] || .venv-mjlab/bin/python scripts/planner_merge.py --union-splits --out "$D/all" \
  "$V1/data_v7" "$V1/data_e2" "$V1/data_t2" "$D/data_combo"
if [[ ! -f $D/train/DONE ]]; then
  echo "== fine-tune at $(date -Is)"
  .venv-mjlab/bin/python -u scripts/planner_train.py --data "$D/all" --out "$D/train" --init "$V1/train/planner.pt" \
    --steps ${STEPS:-20000} --lr ${LR:-1e-4} --warmup 500 --eval-every 5000 --device cuda:0
  touch "$D/train/DONE"
fi
.venv-mjlab/bin/python scripts/planner_eval.py --data "$D/all" --checkpoint "$D/train/planner.pt" --out "$D/eval" --save-plans 0 || true
echo "== EXPERT ITERATION DONE at $(date -Is)" | tee -a "$D/run_info.txt"
