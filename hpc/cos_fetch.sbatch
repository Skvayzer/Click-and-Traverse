#!/bin/bash
# COS data fetch onto Lustre (docs/cos_obstacles.md): Unitree LAFAN1 G1 retargeting, PHUMA data.zip, ember-lab AMASS->G1.
#   CODE=~/cat_worktrees/cos_<sha> sbatch --export=ALL $CODE/hpc/cos_fetch.sbatch
#SBATCH --job-name=cos_fetch
#SBATCH --output=/l/users/%u/cos/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=08:00:00
source $HOME/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1
set -euo pipefail
C=/l/users/$USER/cos
cd "${CODE:?}"
python -u scripts/cos_fetch.py --repo lvhaidong/LAFAN1_Retargeting_Dataset --out $C/lafan1_g1 --threads 8
python -u scripts/cos_fetch.py --repo DAVIAN-Robotics/PHUMA --out $C/phuma_raw --threads 1
python -u scripts/cos_fetch.py --repo ember-lab-berkeley/AMASS_Retargeted_for_G1 --out $C/ember_g1 --threads 16
