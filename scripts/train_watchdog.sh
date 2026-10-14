#!/usr/bin/env bash
# Resume a training run automatically if it dies WITHOUT a clean stop.
# usage: train_watchdog.sh RUN_DIR RESUME_CONFIG [MAX_RESTARTS=5]
# A clean stop (SIGINT -> status "stopped_by_request", or "completed_requested_updates") ends the watchdog; a dead
# trainer whose status still says "running" (crash / external kill) is resumed exactly from RUN_DIR/resume.pt.
RUN=$1; RESUME=$2; MAX=${3:-5}; LOG=$RUN.watchdog.log; cd "$(dirname "$0")/.."
trainer() { ps -eo pid,args | awk '$2 ~ /python$/ && $3 == "train_cat_mjlab.py" && $4 == "run" {print $1}' | head -1; }
restarts=0
while true; do
  while [ -n "$(trainer)" ]; do sleep 60; done
  sleep 30                                                     # let a clean shutdown finish writing status.json
  [ -n "$(trainer)" ] && continue                              # someone (e.g. the hurdle gate) already resumed it
  PHASE=$(python3 -c "import json;print(json.load(open('$RUN/status.json')).get('phase',''))" 2>/dev/null)
  UPD=$(python3 -c "import json;print(json.load(open('$RUN/status.json')).get('updates',''))" 2>/dev/null)
  if [ "$PHASE" != "running" ]; then echo "$(date +%F_%T) trainer gone, phase '$PHASE' at update $UPD: clean stop, watchdog exits" >> "$LOG"; exit 0; fi
  if [ "$restarts" -ge "$MAX" ]; then echo "$(date +%F_%T) trainer died again at update $UPD; $MAX restarts used, giving up" >> "$LOG"; exit 1; fi
  restarts=$((restarts+1))
  echo "$(date +%F_%T) trainer died at update $UPD (status still 'running'): resume #$restarts" >> "$LOG"
  nohup setsid bash "$RESUME" >> "$RUN.resume.log" 2>&1 < /dev/null &
  pgrep -f "keep_snapshots.sh $RUN" >/dev/null || { sleep 5; nohup setsid scripts/keep_snapshots.sh "$RUN" 20 >> "$RUN.snapshots.log" 2>&1 < /dev/null & }
  sleep 300
done
