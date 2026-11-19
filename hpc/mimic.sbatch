#!/bin/bash
# Full-body motion tracker (cat_mjlab/mimic.py: mjlab's G1 tracking task, BeyondMimic-style PPO) on the policies' own
# clean motion: builds the clip libraries from a planner run's datasets (scripts/mimic_library.py), then trains.
# One RTX 5000 Ada 32 GB, ws-ia. Library stages are skipped when their files exist.
#
#   T=$HOME/cat_worktrees/<commit>; cd ~/jobs; RUN=mimic_v1 CLONE=$T sbatch --job-name=mimic_v1 --export=ALL $T/hpc/mimic.sbatch
#
# Knobs: DATA_RUN (planner run under runs/wbp, default planner_v1), MINUTES / HELD_MINUTES of motion per policy for the
# training / held-out library, ENVS, ITERS, RESUME_DIR (an earlier log dir to continue from). SMOKE=1: minutes.
#SBATCH --job-name=mimic
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=12
#SBATCH --mem=54000M
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
source $HOME/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1
set -euo pipefail
L=/l/users/$USER
CLONE=${CLONE:-$HOME/Click-and-Traverse-Mjlab}
export REPO=$TMPDIR/code
rsync -a --exclude .git $CLONE/ $REPO/
cd $REPO
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root WANDB_MODE=disabled
DATA=$L/runs/wbp/${DATA_RUN:-planner_v1}
if [[ -n "${SMOKE:-}" ]]; then
  D=$L/runs/mimic/smoke_${SLURM_JOB_ID}; MINUTES=1 HELD_MINUTES=0.5 ENVS=${ENVS:-512} ITERS=${ITERS:-20}
else
  D=$L/runs/mimic/${RUN:-mimic_v1}
fi
mkdir -p $D/library
{ echo "== job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname) at $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo "commit $(git -C $CLONE rev-parse HEAD); data $DATA"; } >> $D/run_info.txt
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 60 >> $D/gpu.csv &
SRC="$DATA/data_v7 $DATA/data_e2 $DATA/data_t2"
[[ -f $D/library/train.npz ]] || python scripts/mimic_library.py --data $SRC --split train --minutes ${MINUTES:-40} --out $D/library/train.npz
[[ -f $D/library/test.npz ]] || python scripts/mimic_library.py --data $SRC --split test --minutes ${HELD_MINUTES:-5} --out $D/library/test.npz
echo "== train at $(date -Is)"
python scripts/mimic_train.py Cat-Mimic-G1 --env.commands.motion.motion-file $D/library/train.npz \
    --env.scene.num-envs ${ENVS:-4096} --agent.max-iterations ${ITERS:-20000} --log-root $D/logs \
    ${RESUME_DIR:+--agent.resume True --agent.load-run $RESUME_DIR}
echo "== MIMIC JOB DONE at $(date -Is); peak GPU MiB $(cut -d, -f2 $D/gpu.csv | tr -dc '0-9\n' | sort -n | tail -1)" | tee -a $D/run_info.txt
