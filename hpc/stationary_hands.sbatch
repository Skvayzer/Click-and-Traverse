#!/bin/bash
# Stationary hand protection (configs/pilots/stationary_hands_20261113.sh) on one RTX 5000 Ada 32 GB, ws-ia.
#   sbatch --job-name=hands_S1 --export=ALL,VARIANT=S1 hpc/stationary_hands.sbatch
#   smoke: sbatch --job-name=hands_smoke --time=00:40:00 --export=ALL,VARIANT=S1,SMOKE=1 hpc/stationary_hands.sbatch
# VARIANT  S1 stationary: feet planted (<= 5 cm, <= 5 steps off the floor) after settling, pelvis <= 0.2 m, else failure
#          S2 energy: cost 0.02 per watt of joint power in standing episodes, old non-terminal 0.3 m spot rule
#          S3 both;  S0 control: old rules only
#          S1b: S1 plus a dense cost on foot travel from the settled spot (weight -10, (travel/5 cm)^2)
# Data: dep-1's tree is mirrored under /l/users/$USER/cat_root and reached through CAT_PATH_MAP (cat_mjlab/pathmap.py);
# data/furniture/<bank> and data/motion_library in the clone are symlinks into it. cat_root, runs/ and checkpoints/ are
# pinned to healthy OSTs (13/14/10): OSTs 7 and 8 hung on writes on 2026-11-13. W&B offline (wandb sync later).
# Resubmitting the same VARIANT resumes from its run directory.
#SBATCH --job-name=hands
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=12
#SBATCH --mem=54000M
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
source $HOME/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1
set -euo pipefail
L=/l/users/$USER
CLONE=$HOME/Click-and-Traverse-Mjlab
# Run from a private copy of the code (node-local disk): later pulls in the clone cannot change a running job.
# data/furniture/*, data/motion_library and outputs/saved_checkpoints are symlinks and stay symlinks.
export REPO=$TMPDIR/code
rsync -a --exclude .git $CLONE/ $REPO/
VARIANT=${VARIANT:?set VARIANT=S0, S1, S2 or S3}
case $VARIANT in
  S0) ;;
  S1) export STATIONARY="0.05 5 30 0.2" ;;
  S2) export POWER=-0.02 ;;
  S3) export STATIONARY="0.05 5 30 0.2" POWER=-0.02 ;;
  S1b) export STATIONARY="0.05 5 30 0.2" FEET_HOLD=-10 ;;
  *) echo "unknown VARIANT $VARIANT"; exit 2 ;;
esac
if [[ -n "${SMOKE:-}" ]]; then
  RUN=$L/runs/smoke_hands_${VARIANT}_${SLURM_JOB_ID}; export EXTRA="--max-updates ${SMOKE_UPDATES:-3}" WANDB_MODE=disabled
else
  RUN=$L/runs/stationary_hands_20261113_$VARIANT; export WANDB_MODE=${WANDB_MODE:-offline}
fi
mkdir -p $RUN
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
export STUDENT=$L/checkpoints/generalist_robust_delay_v1_u150.pt RUN_DIR=$RUN/run WANDB_DIR=$RUN
export NUM_ENVS=${NUM_ENVS:-12288} BATCH_SIZE=1024    # 16384 peaked at 31.2 of 32.8 GB (bank fields ~20 GB fixed)
export NUM_MINIBATCHES=$(( NUM_ENVS / 1024 ))          # 32768-transition minibatches, as 40960 envs / 40 on dep-1
if [[ -f $RUN/run/status.json ]]; then export EXTRA="--resume ${EXTRA:-}" FRESH=""; fi
{ echo "== job $SLURM_JOB_ID ($SLURM_JOB_NAME) variant $VARIANT on $(hostname) at $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo "commit $(git -C $CLONE rev-parse HEAD) $(git -C $CLONE status --porcelain --untracked-files=no | wc -l) modified tracked files (copied to $REPO)"
  echo "STATIONARY='${STATIONARY:-}' POWER='${POWER:-}' FEET_HOLD='${FEET_HOLD:-}' NUM_ENVS=$NUM_ENVS NUM_MINIBATCHES=$NUM_MINIBATCHES EXTRA='${EXTRA:-}'"
  echo "STUDENT=$STUDENT CAT_PATH_MAP=$CAT_PATH_MAP"
  echo "config: configs/pilots/stationary_hands_20261113.sh"; } >> $RUN/run_info.txt
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l ${GPU_LOG_SECONDS:-60} >> $RUN/gpu.csv &
( while sleep 60; do echo "$(date +%T) $(ps -u $USER -o rss= | awk '{s+=$1} END {printf "%.0f", s/1024}') MB"; done ) >> $RUN/host_mem.txt &
status=0
bash $REPO/configs/pilots/stationary_hands_20261113.sh || status=$?
echo "== exit $status at $(date -Is); peak GPU MiB $(cut -d, -f2 $RUN/gpu.csv | tr -dc '0-9\n' | sort -n | tail -1); peak host $(awk '{print $2}' $RUN/host_mem.txt | sort -n | tail -1) MB" | tee -a $RUN/run_info.txt
exit $status
