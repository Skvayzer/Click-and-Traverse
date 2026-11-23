#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
PY=.venv-mjlab/bin/python
O=$L/runs/videos/current_5cm_band_$SLURM_JOB_ID
mkdir -p "$O" "$O/recordings"
for input in "${COMBO:?}" "${REACTIVE:?}" "${SCENES:?}"; do test -s "$input"; done
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  used=$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  echo "== assigned GPU memory already in use: $used MiB"
  if (( used > 1500 )); then exit 3; fi
fi
printf 'Code %s on %s\n' "$(git rev-parse HEAD)" "$(hostname)"
sha256sum "$COMBO" "$REACTIVE" "$SCENES"
for policy in planner_reflex reactive; do
  if [[ "$policy" = planner_reflex ]]; then checkpoint=$COMBO; mix=plan=1; else checkpoint=$REACTIVE; mix=none=1; fi
  "$PY" -u scripts/record_plan_reflex.py "$checkpoint" --scenes "@$SCENES" \
    --bank data/furniture/tablegap_v1 --mix "$mix" --speed 0.15 --seed 1 --steps 3000 --out "$O/recordings/$policy.npz"
done
for policy in planner_reflex reactive; do
  "$PY" -u scripts/render_plan_reflex.py --rec "$O/recordings/$policy.npz" --clip-anchor "$O/recordings/planner_reflex.npz" \
    --out "$O/$policy" --max-seconds 15 --fps 25 --width 1280 --height 720 --camera 3.4 -28 55 --ghost-ahead 0.3
done
for policy in planner_reflex reactive; do
  test "$(find "$O/$policy" -maxdepth 1 -name '*.mp4' -type f | wc -l)" -eq 11
  for video in "$O/$policy/"*.mp4; do
    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,duration -of csv=p=0 "$video"
  done
done
