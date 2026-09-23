#!/usr/bin/env bash
# Training on the expanded bank, warm-started from the reactive-standing checkpoint.
#
#   2855 scene bank  = 2375 pinned parent + 200 narrow (continuous widths, 0.60-0.66 held out)
#                      + 280 procedural rooms (clutter 200 / furniture 80, no empty rooms)
#   2000 reactive standing scenes, continuously sampled, replacing the old 10x3 grid of 30
#
# Weights are unchanged from APPROVED_CLEARANCE_WEIGHTS_20260922. Only the scenes changed,
# so any metric movement is attributable to the scenes and not to a reward edit.
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs 30720 --batch-size 768 \
  --num-minibatches 40 --unroll-length 32 \
  --checkpoint-native outputs/cat_reactive_standing_20260923/resume.pt \
  --fresh-optimizer --max-action-std 0 \
  --bank-manifest data/furniture/cat_flat_hand_balance_v2_packed_20260923/manifest.json \
  --body-collision-bank data/furniture/cat_flat_hand_balance_v2_20260921_collision/manifest.json \
  --body-collision-resets data/furniture/cat_flat_hand_balance_v2_20260921_resets/manifest.json \
  --reactive-bank data/furniture/reactive_standing_v2_20260923/manifest.json \
  --run-dir outputs/cat_reactive2000_20260923 \
  --disable-hand-contrast --hand-clearance-weight -20 --arm-clearance-weight -8 --tracking-root-field-weight 1 \
  --hand-clearance-target 0.09 --hand-clearance-anticipation 0.20 \
  --hand-clearance-near-weight 0.8 --hand-reward-soft-floor 0 \
  --hand-raised-reset-fraction 0 --upper-gravity-compensation \
  --compile-task --device cuda:0 --seed 0 \
  --checkpoint-interval-updates 10 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
