#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
export WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=/l/users/konstantin.smirnov
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
O=$L/runs/energy_round3_reference_$SLURM_JOB_ID
s2=$L/checkpoints/stationary_hands_S2_seed1_u180.pt
s2h=$L/runs/stationary_hands_20261113_S2H_peak/run/resume.pt
source_report=$L/runs/energy_round2_standing_315346/standing.json
for file in "$s2" "$s2h" "$source_report"; do test -s "$file"; done
echo "== job $SLURM_JOB_ID on $(hostname), CPU energy reference, $(git rev-parse HEAD), output $O"
.venv-mjlab/bin/python -u scripts/energy_optimal_reference.py --checkpoint "$s2" "$s2h" \
  --benchmark "$source_report" --out-dir "$O" --workers 10
