#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
source_start=$L/checkpoints/generalist_robust_delay_v1_u150.pt
source_contact=$L/runs/energy_contact_CONTACT_ONLY_${CONTACT_JOB:?}/run/resume.pt
source_energy=$L/runs/energy_contact_ENERGY_ONLY_${ENERGY_JOB:?}/run/resume.pt
for c in "$source_start" "$source_contact" "$source_energy"; do test -s "$c"; done
rows=(0 3 1 4)
buckets=(danger danger anticipation anticipation)
row=${rows[SLURM_ARRAY_TASK_ID]}
bucket=${buckets[SLURM_ARRAY_TASK_ID]}
O=$L/runs/energy_contact_videos_$SLURM_ARRAY_JOB_ID
mkdir -p "$O"
B=data/furniture/sidegap_v2
RB=data/furniture/reactive_standing_v2_20260923/manifest.json
PY=.venv-mjlab/bin/python
scene=$("$PY" - "$B" "$RB" "$row" "$bucket" <<'PY'
import json
import sys
bank, reactive, row, bucket = sys.argv[1:]
reactive = json.load(open(reactive))
assert reactive["scenes"][int(row)]["bucket"] == bucket
print(json.load(open(bank + "_packed/manifest.json"))["scenes"][reactive["background_scene_index"]]["scene_id"])
PY
)
work=${TMPDIR:?}/energy-contact-video-${SLURM_JOB_ID}
mkdir "$work"
for label in start contact energy; do
  case "$label" in
    start) checkpoint=$source_start ;;
    contact) checkpoint=$source_contact ;;
    energy) checkpoint=$source_energy ;;
  esac
  "$PY" -u scripts/record_mjlab_rollout.py --checkpoint "$checkpoint" --bank-manifest "${B}_packed/manifest.json" \
    --body-collision-bank "${B}_collision/manifest.json" --body-collision-resets "${B}_resets/manifest.json" \
    --reactive-bank "$RB" --reactive-row "$row" --scene-id "$scene" --frames 400 --seed 0 \
    --allow-source-mismatch --allow-bank-mismatch --nominal --no-pushes --shadow-stationary \
    --upper-target-average --no-curriculum-starts --device cuda:0 --output-dir "$work/$label"
done
name=$(printf 'standing-controls-%s-row-%04d.mp4' "$bucket" "$row")
"$PY" -u scripts/render_energy_panels.py recordings --panel "Starting policy" "$work/start" \
  --panel "Contact only" "$work/contact" --panel "Energy + contact" "$work/energy" \
  --title "${bucket^} threat · row $row · seed 0" --output "$O/$name"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,duration "$O/$name"
