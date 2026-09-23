#!/usr/bin/env bash
# Prepared and CPU-parsed only. This script launches training when the user runs it.
set -euo pipefail
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu MUJOCO_GL=egl PYTHONUNBUFFERED=1
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
exec .venv-mjlab/bin/python train_cat_mjlab.py run \
  --checkpoint-native outputs/cat_hand_cont_30720_20260921/best.pt --fresh-optimizer \
  --bank-manifest data/furniture/cat_flat_hand_balance_v3_20260921/manifest.json \
  --body-collision-bank data/furniture/cat_flat_hand_balance_v3_20260921_collision/manifest.json \
  --body-collision-resets data/furniture/cat_flat_hand_balance_v3_20260921_resets/manifest.json \
  --run-dir outputs/cat_hand_raised_seeded_pilot50_20260921 \
  --algorithm ppo --num-envs 30720 --batch-size 768 --num-minibatches 40 \
  --unroll-length 32 --discounting 0.98 --max-action-std 0.15 \
  --flat-bonus-scale 10.0 --flat-region-scale 0.40 \
  --hand-contrast-region-weight -3 --hand-contrast-heading-weight -2 \
  --hand-contrast-approach-distance 0.40 --hand-raised-reset-fraction 0.5 \
  --require-hand-contrast --nconmax 64 --njmax 256 --compile-task --device cuda:0 \
  --max-updates 50 --checkpoint-interval-updates 50 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer --seed 0
