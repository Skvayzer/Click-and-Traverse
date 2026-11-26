#!/bin/bash
# Generic COS CPU job: CODE=~/cat_worktrees/cos_<sha> CMD="python -u scripts/..." sbatch --export=ALL hpc/cos_cpu.sbatch
#SBATCH --job-name=cos_cpu
#SBATCH --output=/l/users/%u/cos/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=12:00:00
source $HOME/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1
set -euo pipefail
cd "${CODE:?}"
export PYTHONPATH=$PWD:/l/users/$USER/cos/third_party/pydeps_ort OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "== $CMD on $(hostname) at $(date -Is), code $(git rev-parse --short HEAD)"
eval "$CMD"
echo "== exit $? at $(date -Is)"
