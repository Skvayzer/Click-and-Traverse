#!/bin/bash
# COS scenes v1 for one motion source on Lustre (docs/cos_obstacles.md): 50 Hz library -> events/windows -> SONIC
# feasibility flags (lite harness) -> multi-fit obstacle scenes. Each stage is skipped when its output exists.
#   CODE=~/cat_worktrees/cos_<sha> NAME=amass_gmr KIND=gmr SRC=$C/amass_gmr sbatch --export=ALL $CODE/hpc/cos_process.sbatch
#SBATCH --job-name=cos_process
#SBATCH --output=/l/users/%u/cos/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=12:00:00
source $HOME/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1
set -euo pipefail
C=/l/users/$USER/cos
LIB=$C/library/${NAME:?}; OUT=$C/scenes_v1/$NAME
cd "${CODE:?}"
export PYTHONPATH=$PWD:$C/third_party/pydeps_ort OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SONIC_ORT_THREADS=1
echo "== $NAME $KIND $SRC on $(hostname) at $(date -Is), code $(git rev-parse --short HEAD)"
[[ -f $LIB/summary.json ]] || python -u scripts/cos_library.py --kind ${KIND:?} --src ${SRC:?} --out $LIB
[[ -f $LIB/events_summary.json ]] || python -u scripts/cos_process.py events --lib $LIB
mkdir -p $OUT
if [[ ! -f $OUT/FEAS_DONE ]]; then
  python -u scripts/cos_feasibility.py --windows $LIB/windows.jsonl --sonic $C/sonic --out $OUT/feasibility.jsonl \
      --workers 12 --save-dir $OUT/tracked && touch $OUT/FEAS_DONE
fi
if [[ ! -f $OUT/SYNTH_DONE ]]; then
  python -u scripts/cos_process.py synth --lib $LIB --gait-bank $C/gait_bank_v1.pkl --out $OUT \
      --feasibility $OUT/feasibility.jsonl --tracked-dir $OUT/tracked --jobs 12 && touch $OUT/SYNTH_DONE
fi
echo "== done $(date -Is)"
