#!/usr/bin/env bash
# Overnight pipeline (2026-10-19): traversal eval (already running) -> narrow eval -> routing -> three-teacher DAgger
# (<= 200 updates) -> student + rooms-teacher per-scene evals -> pass test. Each stage logs to pipeline.log.
set -u
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
O=outputs/distill_v1_prep
RUN=outputs/distill_three_teachers_v1_20261018
log() { echo "$(date +%F_%T) $*"; }
run() { name=$1; shift; log "start eval $name"; .venv-mjlab/bin/python scripts/eval_per_scene.py "$@" --out $O/eval_$name.json > $O/eval_$name.log 2>&1; log "done eval $name exit $?: $(tail -1 $O/eval_$name.log | cut -c1-300)"; }
while pgrep -f "eval_per_scene.py outputs/traversal_expert_v1_20261012/snapshots/update_1764.pt" > /dev/null; do sleep 30; done
log "traversal eval finished: $(tail -1 $O/eval_traversal_u1764.log | cut -c1-300)"
[ -f $O/eval_traversal_u1764.json ] || { log "traversal eval missing: abort"; exit 1; }
run narrow_u1400 outputs/saved_checkpoints/narrow_expert_v2_u1400.pt --scenes cat --episodes 12 --num-envs 32768
[ -f $O/eval_narrow_u1400.json ] || { log "narrow eval missing: abort"; exit 1; }
.venv-mjlab/bin/python scripts/build_distill_routing.py --eval traversal=$O/eval_traversal_u1764.json --eval narrow=$O/eval_narrow_u1400.json \
  --rooms-trained-types --out $O/routing_v1.json > $O/routing_v1.log 2>&1 || { log "routing failed: abort"; cat $O/routing_v1.log; exit 1; }
log "routing: $(tr -d '\n' < $O/routing_v1.log | cut -c1-600)"
log "start DAgger $RUN"
MASSES="0.15 0.36 0.08 0.15 0.13 0.03 0.10 0" EXTRA="--max-updates ${DAGGER_UPDATES:-200}" \
  bash configs/pilots/distill_three_teachers_v1_20261018.sh > $RUN.log 2>&1
log "DAgger exit $?: $(grep -E '\"updates\"|status' $RUN/status.json | tr -d '\n ')"
run student_v1_cat $RUN/resume.pt --scenes cat --episodes 12 --num-envs 32768
run student_v1_room $RUN/resume.pt --scenes room --episodes 12 --num-envs 8192
run rooms_v9 outputs/saved_checkpoints/cat_g1_rooms_keyboard_v9_u492_full.pt --scenes room --episodes 12 --num-envs 8192
.venv-mjlab/bin/python scripts/compare_student_teachers.py --routing $O/routing_v1.json --teacher-eval rooms=$O/eval_rooms_v9.json \
  --student $O/eval_student_v1_cat.json --student $O/eval_student_v1_room.json --out $O/compare_student_v1.json > $O/compare_student_v1.log 2>&1
log "compare exit $?"
log "pipeline done"
