#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1
L=/l/users/konstantin.smirnov
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
root=$HOME/cat_worktrees
eval_repo=$root/energy_price_3a6f7db
case "${ROUND4_CASE:?}" in
  planner)
    variants=(COMBO_FIX COMBO_NOREFLEX COMBO_FIX COMBO_NOREFLEX)
    seeds=(1 1 2 2)
    variant=${variants[SLURM_ARRAY_TASK_ID]}
    seed=${seeds[SLURM_ARRAY_TASK_ID]}
    tag=${variant}_seed$seed
    training=$eval_repo
    student=$L/checkpoints/plan_reflex_COMBO1b_u500.pt
    anchor=$L/checkpoints/stationary_hands_E2_u300.pt
    extra="--max-updates 150 --seed $seed"
    ;;
  standing)
    tags=(Flip_arm30_seed1 Flip_arm1_seed1 Heat250_seed1 Heat1000_seed1)
    alpha_values=(500 500 250 1000)
    arm_values=(30 1 1 1)
    tag=${tags[SLURM_ARRAY_TASK_ID]}
    variant=S2H
    seed=1
    training=$root/energy_price_f93f1c2
    student=$L/checkpoints/stationary_hands_S2_seed1_u180.pt
    anchor=$L/checkpoints/generalist_robust_delay_v1_u150.pt
    alpha=${alpha_values[SLURM_ARRAY_TASK_ID]}
    export ARM_MULT=${arm_values[SLURM_ARRAY_TASK_ID]} LEG_MULT=1
    extra="--max-updates 150 --seed $seed --energy-alpha $alpha"
    ;;
  *) exit 2 ;;
esac
for input in "$student" "$anchor"; do test -s "$input"; done
test "$(git -C "$eval_repo" rev-parse HEAD)" = 3a6f7dbab97f14198c50fdb2343488e09807d764
if [[ "$ROUND4_CASE" = standing ]]; then
  test "$(git -C "$training" rev-parse HEAD)" = f93f1c22d2c09f71477bde6246901938dc16bb38
fi
for repo in "$training" "$eval_repo"; do test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"; done
unset ENERGY_RESUME_FROM SMOKE
export CLONE=$training VARIANT=$variant STUDENT=$student ANCHOR=$anchor FRESH=--fresh-optimizer
export NUM_ENVS=12288 EXTRA=$extra ENERGY_RUN_TAG=R4_${tag}_${SLURM_ARRAY_JOB_ID}
run=$L/runs/stationary_hands_20261113_$ENERGY_RUN_TAG/run
out=$L/runs/energy_round4_${tag}_${SLURM_ARRAY_JOB_ID}
[[ ! -e "$out" && ! -e "$run" ]]
echo "== $SLURM_JOB_ID $ROUND4_CASE $tag seed $seed from $student; eval output $out"
bash "$training/hpc/stationary_hands.sbatch"
grep -qE '"phase":[[:space:]]*"completed_requested_updates"' "$run/status.json"
grep -qE '"updates":[[:space:]]*150,' "$run/status.json"
test -s "$run/resume.pt"
cd "$eval_repo"
PY=.venv-mjlab/bin/python
mkdir "$out"
if [[ "$ROUND4_CASE" = planner ]]; then
  scene_ids=$HOME/jobs/retention_scenes_20261114.txt
  sha256sum "$scene_ids"
  "$PY" -u scripts/eval_per_scene.py "$run/resume.pt" --bank data/furniture/tablegap_v1 --scenes "@$scene_ids" \
    --episodes 2 --num-envs 1536 --max-steps 21000 --nominal --operator-speeds 0.15 0.3 --tracker-mix plan=1 \
    --refused-rewards --out-dir "$out"
else
  "$PY" -u scripts/eval_standing_hands.py --checkpoint "$run/resume.pt" --rows 0:2000 --steps 800 --out "$out/standing.json"
fi
