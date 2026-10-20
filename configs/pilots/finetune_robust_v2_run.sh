#!/usr/bin/env bash
# Launch / resume wrapper for the real-deployment robustness fine-tune v2 (from robust v1 u226) (train_watchdog.sh).
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
RUN=outputs/finetune_robust_v2_20261020
export NUM_ENVS=32768 NUM_MINIBATCHES=32 BATCH_SIZE=1024 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.10 0.03" RUN_DIR=$RUN
if [ -f $RUN/status.json ]; then export EXTRA="--resume" FRESH=""; else export EXTRA=""; fi
exec bash configs/pilots/finetune_robust_v2_20261020.sh
