#!/bin/bash
# Execute planner plans with the trained full-body tracker and score the executed motion (scripts/planner_track_eval.py):
# held-out situations of a planner run, plus the inserted-column probe.
#   T=$HOME/cat_worktrees/<commit>; cd ~/jobs; TRACKER=<model_X.pt> CLONE=$T sbatch --export=ALL $T/hpc/planner_track_eval.sbatch
#SBATCH --job-name=track_eval
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
source $HOME/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1
set -euo pipefail
L=/l/users/$USER
CLONE=${CLONE:-$HOME/Click-and-Traverse-Mjlab}
export REPO=$TMPDIR/code
rsync -a --exclude .git $CLONE/ $REPO/
cd $REPO
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root WANDB_MODE=disabled
P=$L/runs/wbp/${PLANNER_RUN:-planner_v1}
OUT=${OUT:-$L/runs/mimic/${RUN:-mimic_v1}/track_eval}
: "${TRACKER:?set TRACKER=<mjlab checkpoint model_X.pt>}"
python scripts/planner_track_eval.py --data $P/all --planner $P/train/planner.pt --tracker $TRACKER --out $OUT --per-type ${PER_TYPE:-150}
python scripts/planner_track_eval.py --data $P/all --planner $P/train/planner.pt --tracker $TRACKER --out $OUT --per-type ${PER_TYPE:-150} --probe
echo "== TRACK EVAL DONE at $(date -Is)"
