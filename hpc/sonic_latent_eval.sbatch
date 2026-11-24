#!/bin/bash
# SONIC latent adaptor evaluation. CKPT=zero (plain SONIC on the batched gait reference) or an adaptor checkpoint;
# PHASE=384 (narrow scenes, 0.15/0.30 m/s) | E4 (72 layouts, 0.15/0.30/0.50 m/s, certified starts).
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root SONIC_DIR=$L/sonic_release
cd "${CLONE:?}"
P=.venv-mjlab/bin/python
ON=$L/runs/stationary_hands_20261113_COMBO_FIX/run/resume.pt
CK=${CKPT:-zero}
O=$L/runs/sonic_latent_eval_${TAG:-zero}_${PHASE:?}_$SLURM_JOB_ID
mkdir "$O"
COMMON=("$CK" --sonic --sonic-residual-mode latent --nominal --tracker-mix field=1 --env-from "$ON")
echo "== SONIC latent eval $PHASE $CK, $(git rev-parse HEAD), $O, $(date -Is)"
case "$PHASE" in
 384) "$P" -u scripts/eval_per_scene.py "${COMMON[@]}" --bank data/furniture/tablegap_v1 \
        --scenes "@$HOME/jobs/retention_scenes_20261114.txt" --episodes 2 --num-envs 1536 --max-steps 21000 \
        --operator-speeds .15 .3 --out-dir "$O" ;;
 E4) B=$HOME/cat_worktrees/energy_overnight_38b62ce/inputs/planning_v1
     "$P" -u scripts/eval_per_scene.py "${COMMON[@]}" --bank "$B" --scenes "@${B}_packed/scene_ids.txt" \
        --goal-seed-cache "${B}_goal" --planning-benchmark --episodes 2 --num-envs 432 --max-steps 14500 \
        --operator-speeds .15 .3 .5 --out-dir "$O" ;;
 *) exit 2 ;;
esac
echo "== done $(date -Is)"
