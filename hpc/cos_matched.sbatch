#!/bin/bash
# Matched SONIC feasibility of GMR vs ember-lab retargeting (same 6 s windows), docs/cos_obstacles.md.
#SBATCH --job-name=cos_matched
#SBATCH --output=/l/users/%u/cos/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=12
#SBATCH --mem=24000M
#SBATCH --time=02:00:00
source $HOME/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1
set -euo pipefail
C=/l/users/$USER/cos; O=$C/scenes_v1/retarget_matched
cd "${CODE:?}"
export PYTHONPATH=$PWD:$C/third_party/pydeps_ort OMP_NUM_THREADS=1 SONIC_ORT_THREADS=1
mkdir -p $O
python -u scripts/cos_matched_windows.py --a $C/library/amass_gmr --b $C/library/ember --out-a $O/win_gmr.jsonl --out-b $O/win_ember.jsonl
for s in gmr ember; do
  python -u scripts/cos_feasibility.py --windows $O/win_$s.jsonl --sonic $C/sonic --out $O/feas_$s.jsonl --workers 12
done
