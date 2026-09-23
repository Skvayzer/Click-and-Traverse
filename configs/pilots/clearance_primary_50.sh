#!/usr/bin/env bash
# Native 222/310, from scratch. Prepared only; NOT launched.
# 50 updates is an early diagnostic budget, not a claim of learned walking.
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs 30720 --batch-size 768 \
  --num-minibatches 40 --unroll-length 32 \
  --from-scratch --max-action-std 0 \
  --bank-manifest data/furniture/cat_flat_hand_balance_v2_20260921/manifest.json \
  --body-collision-bank data/furniture/cat_flat_hand_balance_v2_20260921_collision/manifest.json \
  --body-collision-resets data/furniture/cat_flat_hand_balance_v2_20260921_resets/manifest.json \
  --run-dir outputs/cat_clearance_native_scratch_pilot50 \
  --disable-hand-contrast \
  --hand-clearance-weight -20 --hand-clearance-target 0.04 \
  --hand-clearance-anticipation 0.20 --hand-clearance-near-weight 0.5 \
  --hand-reward-soft-floor 0 --hand-raised-reset-fraction 0 \
  --upper-gravity-compensation --protected-hand-sdf-margin \
  --max-updates 50 --compile-task --device cuda:0 --seed 0 \
  --checkpoint-interval-updates 10 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
