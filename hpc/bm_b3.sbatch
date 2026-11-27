#!/bin/bash
# B3: BlinkMimic tracker through the COS test scenes. CK=<ckpt> OUT=<dir> [BM_ANCHOR_OBS BM_ROOT_POS_STD] [NOOBS=1]
#SBATCH --job-name=bm_b3
#SBATCH --output=/l/users/%u/cos/tracker/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000M
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
export PATH=/usr/local/bin:/usr/bin:/bin
set -euo pipefail
T=/l/users/konstantin.smirnov/cos/tracker; S=/l/users/konstantin.smirnov/cos/splits_v2/test200.jsonl
cd ~/blinkmimic
echo "== B3 $(git rev-parse --short HEAD) $(hostname) CK=$CK OUT=$OUT ANCHOR=${BM_ANCHOR_OBS:-0} ROOTSTD=${BM_ROOT_POS_STD:-0.3} $(date -Is)"
PYTHONPATH=src $T/venv/bin/python -u tools/cos_obstacle_eval.py "$CK" --scenes $S --out "$OUT" ${NOOBS:+--no_obstacles} 2>&1 | grep -v "^Module\|^Warp\|took" | tail -n 8
cd ~/cat_worktrees/${COSCLONE:-cos_9c1edb0}
.venv-mjlab/bin/python scripts/cos_bm_b3_clearance.py --run "$OUT" --scenes $S 2>&1 | tail -n 12
echo "== done $(date -Is)"
