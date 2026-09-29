#!/usr/bin/env bash
# Keep a copy of a run's resume.pt every N updates (resume.pt itself is overwritten at each save).
# usage: keep_snapshots.sh <run_dir> [every=20]; exits when the training process is gone.
RUN=$1; EVERY=${2:-20}; mkdir -p "$RUN/snapshots"
while pgrep -f "train_cat_mjlab.py run" > /dev/null; do
  u=$(python3 -c "import json;print(json.load(open('$RUN/status.json')).get('updates',0))" 2>/dev/null || echo 0)
  if [ "$u" -gt 0 ] && [ $((u % EVERY)) -eq 0 ] && [ ! -e "$RUN/snapshots/update_$(printf %04d $u).pt" ]; then
    sleep 20   # let the checkpoint write finish
    cp "$RUN/resume.pt" "$RUN/snapshots/update_$(printf %04d $u).pt" && echo "$(date +%H:%M) kept update $u"
  fi
  sleep 60
done
