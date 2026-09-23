#!/usr/bin/env bash
# Prepared only. From scratch for actor 222 / critic 310; bounded IK is certified at startup.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec .venv-mjlab/bin/python train_cat_mjlab.py run \
  --from-scratch --algorithm ppo \
  --bank-manifest data/furniture/cat_flat_hand_balance_v6_20260922/manifest.json \
  --body-collision-bank data/furniture/cat_flat_hand_balance_v6_20260922_collision/manifest.json \
  --body-collision-resets data/furniture/cat_flat_hand_balance_v6_20260922_resets/manifest.json \
  --run-dir outputs/cat_hand_reward_floor_v6_pilot50_20260922 \
  --num-envs 30720 --batch-size 768 --num-minibatches 40 --unroll-length 32 \
  --hand-contrast-region-weight -3 --hand-contrast-heading-weight -5 \
  --hand-reward-soft-floor 0.2 --hand-contrast-approach-distance 0.40 \
  --hand-raised-reset-fraction 0.5 \
  --flat-bonus-scale 1.0 --flat-region-scale 0.40 \
  --checkpoint-selection-weights 0.60 0.20 0.10 0.10 \
  --max-action-std 0 --discounting 0.98 \
  --require-hand-contrast --device cuda:0 --compile-task \
  --max-updates 50 --checkpoint-interval-updates 10 --seed 0 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
