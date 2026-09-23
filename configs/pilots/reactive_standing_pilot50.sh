#!/usr/bin/env bash
# Reactive standing approach-object pilot, 50 updates.
# Owner-approved scenes (docs/assets/reactive-approval-20260922, 30/30 certified).
# Weights identical to APPROVED_CLEARANCE_WEIGHTS_20260922; only --reactive-bank is new.
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs 30720 --batch-size 768 \
  --num-minibatches 40 --unroll-length 32 \
  --checkpoint-native outputs/cat_flat_balance_ppo_37632_20260920/resume.pt \
  --fresh-optimizer --max-action-std 0 \
  --bank-manifest data/furniture/cat_flat_hand_balance_v2_20260921/manifest.json \
  --body-collision-bank data/furniture/cat_flat_hand_balance_v2_20260921_collision/manifest.json \
  --body-collision-resets data/furniture/cat_flat_hand_balance_v2_20260921_resets/manifest.json \
  --reactive-bank data/furniture/reactive_standing_approval_20260922/manifest.json \
  --run-dir outputs/cat_reactive_standing_pilot50 \
  --disable-hand-contrast --hand-clearance-weight -20 --arm-clearance-weight -8 --tracking-root-field-weight 1 \
  --hand-clearance-target 0.09 --hand-clearance-anticipation 0.20 \
  --hand-clearance-near-weight 0.8 --hand-reward-soft-floor 0 \
  --hand-raised-reset-fraction 0 --upper-gravity-compensation \
  --compile-task --device cuda:0 --seed 0 \
  --max-updates 50 --checkpoint-interval-updates 10 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
