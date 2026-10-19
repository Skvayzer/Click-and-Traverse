#!/usr/bin/env bash
# After the three-teacher DAgger run (2026-10-19): student per-scene evals + pass test, ONLY if the run really
# trained (status updates >= 190; the 01:28 attempt died at update 0 and the old pipeline evaluated the untrained init).
set -u
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
O=outputs/distill_v1_prep; RUN=outputs/distill_three_teachers_v1_20261018
log() { echo "$(date +%F_%T) $*"; }
trainer() { ps -eo pid,args | awk '$2 ~ /python$/ && $3 == "train_cat_mjlab.py" && $4 == "run" {print $1}' | head -1; }
watchdog() { ps -eo pid,args | awk '/train_watchdog.sh outputs\/distill/ && !/awk/ {print $1}' | head -1; }
while [ -n "$(trainer)" ] || [ -n "$(watchdog)" ]; do sleep 60; done
UPD=$(python3 -c "import json;print(json.load(open('$RUN/status.json'))['updates'])" 2>/dev/null || echo 0)
PHASE=$(python3 -c "import json;print(json.load(open('$RUN/status.json'))['phase'])" 2>/dev/null)
log "DAgger finished: phase $PHASE, updates $UPD"
[ "${UPD:-0}" -ge 190 ] || { log "DAgger did not train enough (updates $UPD): no student evals"; exit 1; }
run() { name=$1; shift; log "start eval $name"; .venv-mjlab/bin/python scripts/eval_per_scene.py "$@" --out $O/eval_$name.json > $O/eval_$name.log 2>&1; log "done eval $name exit $?: $(tail -1 $O/eval_$name.log | cut -c1-300)"; }
run student_v1_cat $RUN/resume.pt --scenes cat --episodes 12 --num-envs 32768
run student_v1_room $RUN/resume.pt --scenes room --episodes 12 --num-envs 8192
.venv-mjlab/bin/python scripts/compare_student_teachers.py --routing $O/routing_v2.json \
  --student $O/eval_student_v1_cat.json --student $O/eval_student_v1_room.json --out $O/compare_student_v1.json > $O/compare_student_v1.log 2>&1
log "compare exit $?"
log "post-DAgger done"
