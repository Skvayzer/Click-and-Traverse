#!/usr/bin/env bash
# Prepared only; NOT launched. Supply the other agent's observation-compatible
# native checkpoint after its migration is complete. Uses a new run directory.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec .venv-mjlab/bin/python train_cat_mjlab.py run \
  --checkpoint-native "${CAT_HAND_CHECKPOINT:?Set an observation-compatible native checkpoint}" \
  --fresh-optimizer --algorithm ppo \
  --bank-manifest data/furniture/cat_flat_hand_balance_v5_20260921/manifest.json \
  --body-collision-bank data/furniture/cat_flat_hand_balance_v5_20260921_collision/manifest.json \
  --body-collision-resets data/furniture/cat_flat_hand_balance_v5_20260921_resets/manifest.json \
  --run-dir outputs/cat_hand_priority_fixed_pilot50_20260922 \
  --num-envs 30720 --batch-size 768 --num-minibatches 40 --unroll-length 32 \
  --flat-bonus-scale 1.0 --flat-region-scale 0.40 \
  --hand-contrast-region-weight -20 --hand-contrast-heading-weight -5 \
  --hand-contrast-approach-distance 0.40 --hand-raised-reset-fraction 0.5 \
  --checkpoint-selection-weights 0.60 0.20 0.10 0.10 \
  --max-action-std 0.15 --discounting 0.98 \
  --require-hand-contrast --device cuda:0 --compile-task \
  --max-updates 50 --checkpoint-interval-updates 10 --seed 0 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
