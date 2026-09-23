#!/usr/bin/env bash
# Full reactive-standing training run (not a pilot). Fresh run directory and fresh W&B lineage.
# Warm-started from the native 222/310 CAT checkpoint, exactly as the pilot was.
# Scenes: owner-approved approaching-object standing scenes (30/30 certified) mixed 25/75
# with the retained cat_flat_hand_balance_v2 distribution.
# Weights per docs/APPROVED_CLEARANCE_WEIGHTS_20260922; no --max-updates, runs until stopped.
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
  --run-dir outputs/cat_reactive_standing_20260923 \
  --disable-hand-contrast --hand-clearance-weight -20 --arm-clearance-weight -8 --tracking-root-field-weight 1 \
  --hand-clearance-target 0.09 --hand-clearance-anticipation 0.20 \
  --hand-clearance-near-weight 0.8 --hand-reward-soft-floor 0 \
  --hand-raised-reset-fraction 0 --upper-gravity-compensation \
  --compile-task --device cuda:0 --seed 0 \
  --checkpoint-interval-updates 10 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
