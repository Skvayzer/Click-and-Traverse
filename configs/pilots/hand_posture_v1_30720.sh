#!/usr/bin/env bash
# Prepared launch only. The user must first stop the current GPU run cleanly.
# Fresh weights-only initialization from its durable snapshot; no optimizer or sampler transfer.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec .venv-mjlab/bin/python train_cat_mjlab.py run \
  --checkpoint-native outputs/cat_width_curriculum_compiled_30720/resume.pt \
  --fresh-optimizer \
  --bank-manifest data/furniture/cat_hand_posture_v1_20260919/manifest.json \
  --body-collision-bank data/furniture/cat_hand_posture_v1_20260919_collision/manifest.json \
  --body-collision-resets data/furniture/cat_hand_posture_v1_20260919_resets/manifest.json \
  --run-dir outputs/cat_hand_posture_30720_20260919 \
  --algorithm sapg --num-envs 30720 --batch-size 768 --num-minibatches 40 \
  --unroll-length 32 --max-action-std 0.15 \
  --hand-curriculum-success-threshold 0.35 \
  --nconmax 64 --njmax 256 --compile-task \
  --max-updates 300 --checkpoint-interval-updates 50 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer --seed 0
