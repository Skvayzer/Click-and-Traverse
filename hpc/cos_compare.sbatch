#!/bin/bash
# GMR (ours) vs ember-lab/fleaven G1 retargeting on the overlapping AMASS sequences (docs/cos_obstacles.md).
#   CODE=~/cat_worktrees/cos_<sha> sbatch --export=ALL $CODE/hpc/cos_compare.sbatch
#SBATCH --job-name=cos_compare
#SBATCH --output=/l/users/%u/cos/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000M
#SBATCH --time=02:00:00
source $HOME/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1
set -euo pipefail
C=/l/users/$USER/cos
cd "${CODE:?}"
export PYTHONPATH=$PWD OMP_NUM_THREADS=4
python -u scripts/cos_compare_retarget.py --a $C/library/amass_gmr --b $C/library/ember \
    --feas-a $C/scenes_v1/amass_gmr/feasibility.jsonl --feas-b $C/scenes_v1/ember/feasibility.jsonl \
    --out $C/scenes_v1/retarget_comparison.json --limit 300
