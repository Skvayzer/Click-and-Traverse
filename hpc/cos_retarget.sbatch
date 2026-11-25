#!/bin/bash
# GMR retargeting of the raw AMASS subsets (CMU, KIT, BMLmovi, SFU) on Lustre, one shard per job (docs/cos_obstacles.md).
# The archives come from dep-1 (rsync to $C/amass_raw); shard 0 extracts them first, other shards wait for the marker.
#   CODE=~/cat_worktrees/cos_<sha> SHARD=0/2 sbatch --export=ALL $CODE/hpc/cos_retarget.sbatch
#SBATCH --job-name=cos_retarget
#SBATCH --output=/l/users/%u/cos/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=06:00:00
source $HOME/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1
set -euo pipefail
C=/l/users/$USER/cos; T=$C/third_party; X=$C/amass_raw/extracted
cd "${CODE:?}"
if [[ "${SHARD:-0/1}" == 0/* ]]; then
  mkdir -p $X
  for s in SFU BMLmovi CMU KIT; do
    [[ -f $X/.$s.done ]] && continue
    while [[ ! -f $C/amass_raw/$s.tar.bz2 ]]; do sleep 30; done      # rsync renames the file when complete
    echo "extract $s $(date +%T)"; tar -xjf $C/amass_raw/$s.tar.bz2 -C $X && touch $X/.$s.done
  done
else
  while [[ ! -f $X/.KIT.done ]]; do sleep 60; done
fi
export PYTHONPATH=$T/pydeps:$T/GMR OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/cos_retarget_amass.py --raw $X --out $C/amass_gmr --body-models $T/body_models --workers ${WORKERS:-12} --shard ${SHARD:-0/1}
