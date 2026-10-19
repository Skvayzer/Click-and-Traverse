#!/usr/bin/env bash
# Launch / resume wrapper for the three-teacher DAgger run (2026-10-19). 32768 envs (40960 ran out of GPU memory in the
# first rollout: the traversal expert alone used 47.9 of 49 GB at 40960). Total 200 updates; a resume runs the rest.
# Routing v2: rooms scenes by the rooms eval (v9 90% clean on all 597 room scenes, passages included).
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
RUN=outputs/distill_three_teachers_v1_20261018; TOTAL=${TOTAL:-200}
export NUM_ENVS=32768 NUM_MINIBATCHES=32 BATCH_SIZE=1024 MASSES="0.15 0.35 0.08 0.14 0.12 0.03 0.10 0.03" RUN_DIR=$RUN ROUTING=outputs/distill_v1_prep/routing_v2.json
if [ -f $RUN/status.json ]; then
  DONE=$(python3 -c "import json;print(json.load(open('$RUN/status.json'))['updates'])")
  export EXTRA="--resume --max-updates $((TOTAL - DONE))" FRESH=""
else
  export EXTRA="--max-updates $TOTAL"
fi
exec bash configs/pilots/distill_three_teachers_v1_20261018.sh
