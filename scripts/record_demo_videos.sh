#!/usr/bin/env bash
# Record demo videos for the reactive-standing checkpoint.
# RUN ONLY WHEN TRAINING IS STOPPED: SceneBank loads all 2375 scene fields (~6 GB) onto the GPU
# regardless of num_envs, and a live training run leaves under 4 GB free.
#
# Paired by construction: identical scene_id and identical --seed for both policies, so any
# visible difference is the policy, not scene luck.
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab

B=data/furniture/cat_flat_hand_balance_v2_20260921
TRAINED=outputs/cat_reactive_standing_20260923/resume.pt
BASELINE=outputs/cat_flat_balance_ppo_37632_20260920/resume.pt
OUT=outputs/demo_20260923
mkdir -p "$OUT"

# Scenes chosen before looking at any outcome: first scene of each retained family.
SCENES=(
  "clutter:random-generic_clutter-dense-train-005001-1eaef5f4115f"
  "furniture:random-furniture-dense-train-004001-5ef9779babcf"
  "cat:D8G0L1O0S3"
  "published:published-forward"
)

for policy in trained baseline; do
  ckpt=$TRAINED; [ "$policy" = baseline ] && ckpt=$BASELINE
  for entry in "${SCENES[@]}"; do
    label=${entry%%:*}; sid=${entry#*:}
    dir="$OUT/${policy}_${label}"
    [ -e "$dir" ] && { echo "skip $dir (exists)"; continue; }
    echo "=== $policy / $label ==="
    .venv-mjlab/bin/python scripts/record_mjlab_rollout.py \
      --checkpoint "$ckpt" \
      --bank-manifest $B/manifest.json \
      --body-collision-bank ${B}_collision/manifest.json \
      --body-collision-resets ${B}_resets/manifest.json \
      --scene-id "$sid" --frames 600 --seed 0 --device cuda:0 \
      --output-dir "$dir"
    .venv-mjlab/bin/python scripts/record_mjlab_rollout.py render \
      --input-dir "$dir" --output "$OUT/${policy}_${label}.mp4" --fps 25
    df -h / | tail -1
  done
done
echo "done: $(du -sh $OUT | cut -f1)"
