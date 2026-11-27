#!/bin/bash
# B4 videos: RUN=<B3 out dir> PICK="ids" OUT=<dir> [NOBOX=1] [TAG=x]
#SBATCH --job-name=bm_render
#SBATCH --output=/l/users/%u/cos/tracker/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
export PATH=/usr/local/bin:/usr/bin:/bin MUJOCO_GL=egl
cd ~/cat_worktrees/cos_9c1edb0
.venv-mjlab/bin/python -u scripts/cos_bm_render.py --run "$RUN" --scenes /l/users/konstantin.smirnov/cos/splits_v2/test200.jsonl --pick $PICK --out "$OUT" ${NOBOX:+--no-boxes} ${TAG:+--tag $TAG} 2>&1 | tail -n 10
