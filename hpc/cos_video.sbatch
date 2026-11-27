#!/bin/bash
# C4 videos: record the COS system on chosen test-200 scenes and render 720p H.264 (yuv420p, faststart) mp4s.
#   CLONE=... CK=<tracker> PLANNER=<planner.pt> SCENES="id1 id2 ..." OUT=<dir> [MIX=plan=1] [SPEED=0.15] sbatch --export=ALL hpc/cos_video.sbatch
#SBATCH --job-name=cos_video
#SBATCH --output=/l/users/%u/cos/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov; C4=$L/cos/c4
export WANDB_MODE=disabled MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"; P=.venv-mjlab/bin/python; mkdir -p "${OUT:?}"
echo "== cos_video $(git rev-parse --short HEAD) $(hostname) CK=$CK SCENES=$SCENES $(date -Is)"
$P -u scripts/record_plan_reflex.py "$CK" --scenes $SCENES --out $OUT/rec.npz --mix ${MIX:-plan=1} --speed ${SPEED:-0.15} --steps 1100 \
   --bank $C4/bank_test --goal-seed-cache $C4/bank_test_goal --start-replay $C4/starts ${PLANNER:+--planner-checkpoint $PLANNER} 2>&1 | grep -v "^step" | tail -20
$P -u scripts/render_plan_reflex.py --rec $OUT/rec.npz --out $OUT --bank $C4/bank_test --width 1280 --height 720 ${SCENE_MAP:+--scene-map $SCENE_MAP} 2>&1 | tail -10
echo "== done $(date -Is)"
