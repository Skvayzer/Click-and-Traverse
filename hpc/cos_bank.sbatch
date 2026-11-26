#!/bin/bash
# C3/C4 harness banks from COS scenes (docs/cos_planner_reflex.md): fields -> collision -> resets -> goal-seed cache.
#   CLONE=~/cat_worktrees/cos_<sha> SCENES=<jsonl> O=/l/.../cos/c3/bank_train PLANS=<dir or empty> sbatch --export=ALL hpc/cos_bank.sbatch
#SBATCH --job-name=cos_bank
#SBATCH --output=/l/users/%u/cos/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=8:00:00
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 WANDB_MODE=offline JAX_PLATFORMS=cpu
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=/l/users/konstantin.smirnov/cat_root
cd "${CLONE:?}"
B=data/furniture/tablegap_v1; P=.venv-mjlab/bin/python
echo "== cos_bank $(git rev-parse --short HEAD) SCENES=$SCENES O=$O on $(hostname) $(date -Is)"
[[ -f ${O:?}_packed/manifest.json ]] || "$P" -u scripts/cos_bank.py build --base-manifest "${B}_packed/manifest.json" \
    --scenes "${SCENES:?}" --output "${O}_packed" ${PLANS:+--plans $PLANS} --workers 12
[[ -f ${O}_collision/manifest.json ]] || "$P" -u -c 'import cat_mjlab, runpy; runpy.run_path("scripts/build_body_collision_bank.py", run_name="__main__")' \
    --field-manifest "${O}_packed/manifest.json" --output "${O}_collision" --base-collision-manifest "${B}_collision/manifest.json" --workers 10
[[ -f ${O}_resets/manifest.json ]] || "$P" -u scripts/build_body_collision_resets.py --cpu-native --field-manifest "${O}_packed/manifest.json" \
    --collision-bank "${O}_collision/manifest.json" --output "${O}_resets"
[[ -f ${O}_goal/cache.json ]] || "$P" -u scripts/build_goal_seed_cache.py --bank "$O" --out "${O}_goal" \
    $([[ -n "${GOAL_ALL:-}" ]] && echo "--reuse ${B}_goal_column" || echo "--scenes ${O}_packed/scene_ids.txt") --workers 10
echo "== done $(date -Is)"
