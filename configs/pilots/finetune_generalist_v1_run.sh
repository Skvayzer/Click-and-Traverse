#!/usr/bin/env bash
# Launch / resume wrapper for the RL fine-tune of the three-teacher student (2026-10-19), for train_watchdog.sh.
# 32768 envs (40960 does not fit next to three frozen teachers on the 49 GB GPU).
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
RUN=outputs/finetune_generalist_v1_20261019
export NUM_ENVS=32768 NUM_MINIBATCHES=32 BATCH_SIZE=1024 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.10 0.03" RUN_DIR=$RUN
if [ -f $RUN/status.json ]; then export EXTRA="--resume" FRESH=""; else export EXTRA=""; fi
exec bash configs/pilots/finetune_generalist_v1_20261019.sh
