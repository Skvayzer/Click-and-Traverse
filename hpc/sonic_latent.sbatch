#!/bin/bash
# SONIC + latent adaptor on the cluster. PHASE=throughput (3 updates) | train (UPDATES, default 600).
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SONIC_DIR=$L/sonic_release
cd "${CLONE:?}"
P=.venv-mjlab/bin/python
ON=$L/runs/stationary_hands_20261113_COMBO_FIX/run/resume.pt
PL=$L/runs/wbp/planner_v1/train/planner.pt
test -s "$ON"; test -s "$PL"
B=data/furniture/tablegap_v1
NENV=${NENV:-4096}
B=${BANK:-$B}
case "${PHASE:?}" in
  throughput) RUN=$L/runs/sonic_latent_throughput_$SLURM_JOB_ID; UPD=3 ;;
  train) RUN=${RESUME_DIR:-$L/runs/sonic_latent_${TAG:-seed0}_$SLURM_JOB_ID}; UPD=${UPDATES:-600} ;;
  *) exit 2 ;;
esac
if [[ -n "${RESUME_DIR:-}" ]]; then
  SOURCE=(--from-scratch --resume)                                    # continue a run with its own state
elif [[ -n "${NATIVE:-}" ]]; then
  SOURCE=(--checkpoint-native "$NATIVE" --fresh-optimizer --init-action-std ${INIT_STD:-0.3})   # warm start
else
  SOURCE=(--from-scratch --fresh-optimizer --init-action-std ${INIT_STD:-0.3})
fi
echo "== SONIC latent $PHASE, $(git rev-parse HEAD), $RUN, $NENV worlds, ${SOURCE[*]}, $(date -Is)"
"$P" -u train_cat_mjlab.py run "${SOURCE[@]}" --algorithm ppo \
  --sonic-residual latent --sonic-env-from "$ON" --tracker-planner "$PL" \
  --bank-manifest ${B}_packed/manifest.json --body-collision-bank ${B}_collision/manifest.json \
  --body-collision-resets ${B}_resets/manifest.json --run-dir "$RUN" --num-envs $NENV \
  --max-updates $UPD --checkpoint-interval-updates 10 --snapshot-interval-updates ${SNAP:-50} --wandb-mode offline \
  --seed ${SEED:-0} --compile-task \
  --unroll-length ${UNROLL:-16} --batch-size $((NENV / 4)) --num-minibatches 4 ${EXTRA:-}
echo "== done $(date -Is)"
