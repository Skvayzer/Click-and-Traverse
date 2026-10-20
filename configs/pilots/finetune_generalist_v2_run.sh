#!/usr/bin/env bash
# RL FINE-TUNE v2 (2026-10-20), launch / resume wrapper (train_watchdog.sh). Same setup as v1
# (configs/pilots/finetune_generalist_v1_20261019.sh) from the DAgger student, with three changes after v1 lost
# 5-10 points on CAT / narrow skills over 596 updates (paired same-configuration SGS outcomes, docs section 7k):
#   * the teacher anchor stays: DAgger regulariser 1.0 -> 0.3 over 300 updates, then 0.3 (v1: -> 0 at update 300;
#     action distance to the teachers then grew 3-5x)
#   * exploration noise fixed at 0.10 (v1: PPO raised it 0.10 -> 0.148, the 0.15 ceiling, by update 150)
#   * SGS uniform share 30% per group (v1: 5%; mastered configurations were barely practised and eroded)
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
RUN=outputs/finetune_generalist_v2_20261020
export NUM_ENVS=32768 NUM_MINIBATCHES=32 BATCH_SIZE=1024 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.10 0.03" RUN_DIR=$RUN
export STUDENT=outputs/distill_three_teachers_v1_20261018/resume.pt REG="1.0 0.3 300" MAX_STD=0.10 SGS_MIX=0.30
if [ -f $RUN/status.json ]; then export EXTRA="--resume" FRESH=""; else export EXTRA=""; fi
exec bash configs/pilots/finetune_generalist_v1_20261019.sh
