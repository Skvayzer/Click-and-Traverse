#!/usr/bin/env bash
# Hurdle-retention gate for a running CAT-expert training.
# usage: hurdle_gate.sh RUN_DIR RESUME_CONFIG THRESHOLD UPDATE [UPDATE ...]
# At each UPDATE: wait for a kept snapshot >= UPDATE, stop the training (SIGINT, final save), run the
# deterministic published-hurdle diagnostic on that snapshot, and resume the run exactly (RESUME_CONFIG)
# only if hurdle success >= THRESHOLD. Otherwise leave it stopped. Log: RUN_DIR.gate.log
RUN=$1; RESUME=$2; THRESHOLD=$3; shift 3
LOG=$RUN.gate.log; cd "$(dirname "$0")/.."
for U in "$@"; do
  while [ -z "$(ls "$RUN/snapshots" 2>/dev/null | awk -F'[_.]' -v u="$U" '$2+0>=u')" ]; do
    pgrep -f "python train_cat_mjlab.py run" >/dev/null || { echo "$(date +%F_%T) training not running; gate exits" >> "$LOG"; exit 1; }
    sleep 60
  done
  SNAP=$(ls "$RUN/snapshots" | awk -F'[_.]' -v u="$U" '$2+0>=u' | head -1)
  P=$(pgrep -f "python train_cat_mjlab.py run" | head -1)
  kill -INT "$P"; while kill -0 "$P" 2>/dev/null; do sleep 5; done
  pkill -f "keep_snapshots.sh $RUN" 2>/dev/null
  RESULT=$(timeout 1800 .venv-mjlab/bin/python scripts/diagnose_hurdles.py "$RUN/snapshots/$SNAP" field 2>&1 | grep '^{')
  SUCCESS=$(echo "$RESULT" | python3 -c "import json,sys;print(json.load(sys.stdin)['success'])")
  echo "$(date +%F_%T) $SNAP hurdle success $SUCCESS (threshold $THRESHOLD) $RESULT" >> "$LOG"
  if python3 -c "import sys;sys.exit(0 if float('$SUCCESS')>=float('$THRESHOLD') else 1)"; then
    nohup setsid bash "$RESUME" >> "$RUN.resume.log" 2>&1 < /dev/null &
    sleep 5; nohup setsid scripts/keep_snapshots.sh "$RUN" 20 >> "$RUN.snapshots.log" 2>&1 < /dev/null &
    echo "$(date +%F_%T) PASS at $SNAP: resumed" >> "$LOG"
  else
    echo "$(date +%F_%T) FAIL at $SNAP: training left stopped" >> "$LOG"; exit 2
  fi
done
echo "$(date +%F_%T) all gates passed" >> "$LOG"
