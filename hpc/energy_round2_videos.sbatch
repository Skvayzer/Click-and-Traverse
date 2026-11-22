#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
L=/l/users/konstantin.smirnov
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
labels=(u150 S2_seed1_u180 S2H_peak Flip_arm_low Flip_arm_center Flip_arm_high Flip_leg_mirror)
label=${labels[SLURM_ARRAY_TASK_ID]}
case "$label" in
  u150) checkpoint=$L/checkpoints/generalist_robust_delay_v1_u150.pt ;;
  S2_seed1_u180) checkpoint=$L/checkpoints/stationary_hands_S2_seed1_u180.pt ;;
  *)
    run=$L/runs/stationary_hands_20261113_$label/run
    grep -qE '"phase":[[:space:]]*"completed_requested_updates"' "$run/status.json"
    grep -qE '"updates":[[:space:]]*50,' "$run/status.json"
    checkpoint=$run/resume.pt ;;
esac
test -s "$checkpoint"
O=$L/runs/energy_videos/round2_${SLURM_ARRAY_JOB_ID}/$label
mkdir -p "$O"
B=data/furniture/sidegap_v2
RB=data/furniture/reactive_standing_v2_20260923/manifest.json
PY=.venv-mjlab/bin/python
scene=$("$PY" - "$B" "$RB" <<'PY'
import json
import sys
bank, reactive = sys.argv[1:]
background = json.load(open(reactive))["background_scene_index"]
print(json.load(open(bank + "_packed/manifest.json"))["scenes"][background]["scene_id"])
PY
)
for specification in "${VIDEO_DANGER:?}" "${VIDEO_ANTICIPATION:?}" "${VIDEO_NEGATIVE:?}"; do
  bucket=${specification%%:*}
  row=${specification#*:}
  out=$O/${bucket}_row${row}_seed0.mp4
  test ! -e "$out"
  rec=${TMPDIR:?}/energy-rec-$SLURM_JOB_ID-$row
  "$PY" scripts/record_mjlab_rollout.py --checkpoint "$checkpoint" --bank-manifest "${B}_packed/manifest.json" \
    --body-collision-bank "${B}_collision/manifest.json" --body-collision-resets "${B}_resets/manifest.json" \
    --reactive-bank "$RB" --reactive-row "$row" --scene-id "$scene" --frames 400 --seed 0 \
    --allow-source-mismatch --allow-bank-mismatch --nominal --no-pushes --shadow-stationary \
    --no-curriculum-starts --device cuda:0 --output-dir "$rec"
  "$PY" scripts/render_standing_hands.py --input-dir "$rec" --output "$out" --label "$label | $bucket | row $row"
  test -s "$out"
  echo "VIDEO=$out"
done
