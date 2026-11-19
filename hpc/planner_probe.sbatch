#!/bin/bash
# Inserted-obstacle probe (scripts/planner_probe.py) of a trained planner on its dataset's held-out test half, plus the
# standard evaluation with classifier-free guidance 2:
#   T=$HOME/cat_worktrees/<commit>; cd ~/jobs; RUN=planner_v1 CLONE=$T sbatch --job-name=planner_probe --export=ALL $T/hpc/planner_probe.sbatch
#SBATCH --job-name=planner_probe
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
source $HOME/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1
set -euo pipefail
L=/l/users/$USER
CLONE=${CLONE:-$HOME/Click-and-Traverse-Mjlab}
export REPO=$TMPDIR/code
rsync -a --exclude .git $CLONE/ $REPO/
cd $REPO
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
D=$L/runs/wbp/${RUN:-planner_v1}
python scripts/planner_probe.py --data $D/all --per-type ${PER_TYPE:-300} --checkpoint $D/train/planner.pt --out $D/probe --device cuda:0 --guidance 1 2 3
python scripts/planner_eval.py --data $D/all --checkpoint $D/train/planner.pt --out $D/eval_g2 --guidance 2
echo "== PROBE JOB DONE at $(date -Is)"
