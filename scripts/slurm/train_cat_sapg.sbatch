#!/usr/bin/env bash
#SBATCH --job-name=cat-sapg
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this recipe with sbatch; do not run training over SSH}"
: "${CAT_SOURCE_COMMIT:?Export the full reviewed commit hash before submitting}"
if [[ ! "$CAT_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo 'CAT_SOURCE_COMMIT must be an exact 40-character lowercase commit hash.' >&2
    exit 2
fi

CAT_REPO_ROOT="${CAT_REPO_ROOT:-/data1/users/konstantin.smirnov/Click-and-Traverse-WholeBody}"
CAT_REPO_ROOT="$(cd "$CAT_REPO_ROOT" && pwd -P)"
CAT_NUM_ENVS="${CAT_NUM_ENVS:-1536}"
CAT_BATCH_SIZE="${CAT_BATCH_SIZE:-192}"
for cat_count in "$CAT_NUM_ENVS" "$CAT_BATCH_SIZE"; do
    if [[ ! "$cat_count" =~ ^[1-9][0-9]*$ ]]; then
        echo 'CAT_NUM_ENVS and CAT_BATCH_SIZE must be positive integers.' >&2
        exit 2
    fi
done
if (( CAT_NUM_ENVS % 6 != 0 || CAT_BATCH_SIZE % 6 != 0 ||
      (CAT_BATCH_SIZE * 64) % CAT_NUM_ENVS != 0 )); then
    echo 'SAPG requires six equal groups and batch_size * 64 divisible by num_envs.' >&2
    exit 2
fi

cat_resolved_commit="$(git -C "$CAT_REPO_ROOT" rev-parse --verify "${CAT_SOURCE_COMMIT}^{commit}")"
if [[ "$cat_resolved_commit" != "$CAT_SOURCE_COMMIT" ]]; then
    echo 'The requested commit did not resolve exactly.' >&2
    exit 2
fi
cat_python="$CAT_REPO_ROOT/.venv/bin/python"
test -x "$cat_python"
cat_bank="$CAT_REPO_ROOT/data/furniture/cat_hand_protection_v1_20260917/manifest.json"
cat_collision="$CAT_REPO_ROOT/data/furniture/body_collision_hand_v1_20260917/manifest.json"
cat_resets="$CAT_REPO_ROOT/data/furniture/body_collision_hand_resets_v1_20260917/manifest.json"
for cat_manifest in "$cat_bank" "$cat_collision" "$cat_resets"; do
    test -f "$cat_manifest"
done

cat_run="$CAT_REPO_ROOT/outputs/cat_sapg_${SLURM_JOB_ID}"
cat_source="$CAT_REPO_ROOT/outputs/sources/cat_sapg_${SLURM_JOB_ID}_${CAT_SOURCE_COMMIT:0:12}"
if [[ -e "$cat_run" || -L "$cat_run" || -e "$cat_source" || -L "$cat_source" ]]; then
    echo 'This submission requires a fresh run directory and source worktree.' >&2
    exit 2
fi
mkdir -p "$CAT_REPO_ROOT/outputs/sources"
git -C "$CAT_REPO_ROOT" worktree add --detach "$cat_source" "$CAT_SOURCE_COMMIT"
if [[ -e "$cat_source/data" || -L "$cat_source/data" ]]; then
    mv "$cat_source/data" "$cat_source/data.git-checkout"
fi
ln -s "$CAT_REPO_ROOT/data" "$cat_source/data"
mkdir "$cat_run"

cat_request_stop() {
    # This job owns precisely this newly created directory. Do not forward
    # signals to Python during initialization or touch any other run.
    touch "$cat_run/STOP"
    printf 'Cooperative stop requested for %s\n' "$cat_run"
}
trap cat_request_stop USR1 TERM INT

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$cat_source"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${CAT_XLA_MEM_FRACTION:-0.90}"
# CUDA_VISIBLE_DEVICES belongs to Slurm; never override it here.

cd "$cat_source"
printf 'Pinned source: %s\nSource worktree: %s\nRun: %s\n' \
    "$CAT_SOURCE_COMMIT" "$cat_source" "$cat_run"
"$cat_python" train_cat_wholebody.py run \
    --algorithm sapg --finetuning cat_train_only --profile single_gpu_32gb \
    --sapg-num-policies 6 --sapg-embedding-dim 16 \
    --num-envs "$CAT_NUM_ENVS" --batch-size "$CAT_BATCH_SIZE" \
    --bank-manifest "$cat_bank" \
    --body-collision-bank "$cat_collision" --body-collision-resets "$cat_resets" \
    --run-dir "$cat_run" --wandb-mode online --wandb-project CAT-wholebody \
    --wandb-entity skvayzer &
cat_learner_pid=$!

# Bash interrupts wait when a trapped signal arrives. Keep waiting for the
# owned learner to finish its update and save the complete runtime snapshot.
while true; do
    if wait "$cat_learner_pid"; then
        cat_exit=0
        break
    else
        cat_exit=$?
    fi
    if ! kill -0 "$cat_learner_pid" 2>/dev/null; then
        break
    fi
done
exit "$cat_exit"
