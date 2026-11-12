#!/bin/bash
# Smoke test: 3 training updates of the robust-v1 setup at NUM_ENVS (default 16384) on an RTX 5000 Ada 32 GB;
# logs GPU memory every 5 s. Does not touch existing runs (own run dir, W&B off).
#SBATCH --job-name=cat_smoke
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
source $HOME/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1
set -euo pipefail
source $HOME/Click-and-Traverse-Mjlab/hpc/stage.sh
NUM_ENVS=${NUM_ENVS:-16384}
RUN=/l/users/$USER/runs/smoke_${NUM_ENVS}_${SLURM_JOB_ID}
mkdir -p $RUN
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; } > $RUN/run_info.txt
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 5 > $RUN/gpu_mem.csv &
export NUM_ENVS NUM_MINIBATCHES=$(( NUM_ENVS / 1024 )) BATCH_SIZE=1024 RUN_DIR=$RUN/run WANDB_MODE=disabled EXTRA="--max-updates 3"
bash configs/pilots/finetune_robust_v1_20261020.sh 2>&1 | tail -40
echo "peak GPU MiB: $(cut -d, -f2 $RUN/gpu_mem.csv | tr -dc '0-9\n' | sort -n | tail -1)"
cat $RUN/run/status.json 2>/dev/null
