#!/bin/bash
# Continue SONIC latent run 1 (job 322876) with its original code (910307a) and contract: 1400 more updates.
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SONIC_DIR=$L/sonic_release
cd $HOME/cat_worktrees/sonic_910307a
test "$(git rev-parse --short=7 HEAD)" = 910307a
B=data/furniture/tablegap_v1
.venv-mjlab/bin/python $HOME/jobs/patch_contract.py $L/runs/sonic_latent_seed0_322876
echo "== resume $(date -Is)"
.venv-mjlab/bin/python -u train_cat_mjlab.py run --from-scratch --resume --algorithm ppo \
  --sonic-residual latent --sonic-env-from $L/runs/stationary_hands_20261113_COMBO_FIX/run/resume.pt \
  --tracker-planner $L/runs/wbp/planner_v1/train/planner.pt \
  --bank-manifest ${B}_packed/manifest.json --body-collision-bank ${B}_collision/manifest.json \
  --body-collision-resets ${B}_resets/manifest.json --run-dir $L/runs/sonic_latent_seed0_322876 --num-envs 4096 \
  --max-updates 1400 --checkpoint-interval-updates 10 --snapshot-interval-updates 100 --wandb-mode offline \
  --init-action-std 0.3 --seed 0 --compile-task --unroll-length 16 --batch-size 1024 --num-minibatches 4
echo "== done $(date -Is)"
