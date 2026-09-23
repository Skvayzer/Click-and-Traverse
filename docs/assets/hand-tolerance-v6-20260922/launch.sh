#!/usr/bin/env bash
# Prepared only: do not run alongside the existing GPU job.
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
exec .venv-mjlab/bin/python train_cat_mjlab.py run \
  --checkpoint-native outputs/cat_hand_priority_30720_20260922/best.pt \
  --fresh-optimizer --algorithm ppo \
  --bank-manifest data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json \
  --body-collision-bank data/furniture/cat_flat_hand_balance_v6_20260922_collision/manifest.json \
  --body-collision-resets data/furniture/cat_flat_hand_balance_v6_20260922_resets/manifest.json \
  --run-dir outputs/cat_hand_tolerance_v6_30720_20260922 \
  --num-envs 30720 --batch-size 768 --num-minibatches 40 --unroll-length 32 \
  --hand-tolerance-curriculum --hand-tolerance-curriculum-window 256 \
  --hand-curriculum-success-threshold 0.60 \
  --hand-contrast-region-weight -20 --hand-contrast-heading-weight -5 \
  --hand-contrast-approach-distance 0.40 \
  --flat-bonus-scale 1.0 --flat-region-scale 0.40 \
  --hand-raised-reset-fraction 0.8 --max-action-std 0 \
  --require-hand-contrast --compile-task --checkpoint-interval-updates 10 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
