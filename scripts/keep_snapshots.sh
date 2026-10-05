#!/usr/bin/env bash
# Keep a copy of a run's resume.pt every N updates (resume.pt itself is overwritten at each save).
# usage: keep_snapshots.sh <run_dir> [every=20]; exits when the training process is gone.
# Copies whenever >= N updates passed since the last copy: status.json is written at checkpoint intervals
# counted from the start of THIS process, so after a resume (e.g. from update 48) it may never show a multiple of N.
RUN=$1; EVERY=${2:-20}; mkdir -p "$RUN/snapshots"
last=$(ls "$RUN/snapshots" 2>/dev/null | sed -n 's/^update_0*\([0-9][0-9]*\)\.pt$/\1/p' | sort -n | tail -1); last=${last:-0}
while pgrep -f "train_cat_mjlab.py run" > /dev/null; do
  u=$(python3 -c "import json;print(json.load(open('$RUN/status.json')).get('updates',0))" 2>/dev/null || echo 0)
  if [ "$u" -gt 0 ] && [ $((u - last)) -ge "$EVERY" ]; then
    sleep 20   # let the checkpoint write finish
    cp "$RUN/resume.pt" "$RUN/snapshots/update_$(printf %04d $u).pt" && echo "$(date +%H:%M) kept update $u" && last=$u
  fi
  sleep 60
done
