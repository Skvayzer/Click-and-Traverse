#!/usr/bin/env bash
# PREPARED ONLY. Run manually after reviewing composition and freeing the GPU.
# No stop/kill commands: the existing hand-posture run remains under user control.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec .venv-mjlab/bin/python train_cat_mjlab.py run \
  --checkpoint-native outputs/cat_hand_posture_30720_20260919/resume.pt \
  --fresh-optimizer \
  --bank-manifest data/furniture/cat_flat_balance_v1_20260920/manifest.json \
  --body-collision-bank data/furniture/cat_flat_balance_v1_20260920_collision/manifest.json \
  --body-collision-resets data/furniture/cat_flat_balance_v1_20260920_resets/manifest.json \
  --run-dir outputs/cat_flat_balance_v1_30720_20260920 \
  --algorithm sapg --num-envs 30720 --batch-size 768 --num-minibatches 40 \
  --unroll-length 32 --max-action-std 0.15 \
  --nconmax 64 --njmax 256 --compile-task \
  --max-updates 300 --checkpoint-interval-updates 50 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer --seed 0
