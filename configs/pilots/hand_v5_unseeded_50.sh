#!/usr/bin/env bash
# Prepared and syntax checked only; launches training when run.
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs 30720 --batch-size 768 \
  --num-minibatches 40 --unroll-length 32 \
  --checkpoint-native outputs/cat_hand_interim3_30720_20260921/best.pt \
  --fresh-optimizer \
  --bank-manifest data/furniture/cat_flat_hand_balance_v5_20260921/manifest.json \
  --body-collision-bank data/furniture/cat_flat_hand_balance_v5_20260921_collision/manifest.json \
  --body-collision-resets data/furniture/cat_flat_hand_balance_v5_20260921_resets/manifest.json \
  --run-dir outputs/cat_hand_feasible_v5_pilot50_20260921 \
  --hand-contrast-region-weight -3 --hand-contrast-heading-weight -2 \
  --hand-contrast-approach-distance 0.40 --require-hand-contrast --hand-raised-reset-fraction 0.0 \
  --max-updates 50 --compile-task --device cuda:0 \
  --checkpoint-interval-updates 10 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
