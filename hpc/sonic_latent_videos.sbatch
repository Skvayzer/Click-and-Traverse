#!/bin/bash
# Gate 5 videos: record SONIC + latent adaptor (CKPT) on the 10 obstacle scenes, render it and the pre-recorded plain
# SONIC-on-gait traces (GAIT_DIR/gait_*.npz) to OUT (1280x720 H.264). Grids are assembled afterwards on dep-1.
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root SONIC_DIR=$L/sonic_release
cd "${CLONE:?}"
export CAT_BANK=$PWD/data/furniture/tablegap_v1 COMBO_CKPT=$L/runs/stationary_hands_20261113_COMBO_FIX/run/resume.pt
export PLANNER_CKPT=$L/runs/wbp/planner_v1/train/planner.pt
P=.venv-mjlab/bin/python
OUT=$L/runs/sonic_latent_videos_$SLURM_JOB_ID; mkdir -p $OUT
"$P" -u scripts/sonic_obstacle_videos.py record latent:${CKPT:?} $OUT/latent.npz
"$P" -u scripts/sonic_obstacle_videos.py render $OUT/latent.npz $OUT
for f in ${GAIT_DIR:?}/gait_*.npz; do "$P" -u scripts/sonic_obstacle_videos.py render $f $OUT; done
ls $OUT
