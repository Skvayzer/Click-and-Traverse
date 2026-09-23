#!/usr/bin/env bash
# Prepared only: this script never stops the existing run.
set -euo pipefail
cd "$(dirname "$0")/../.."
num_envs="${1:-37632}"
case "$num_envs" in
  30720|33792|37632|38400) ;;
  *) echo 'Supported counts: 30720, 33792 (fallback), 37632 (initial), 38400 (measured step-up)' >&2; exit 2 ;;
esac
checkpoint="${CHECKPOINT_NATIVE:-outputs/cat_flat_balance_v3_30720/resume.pt}"
# Reject v3's update-0 startup snapshot; a graceful switchover saves current weights.
CUDA_VISIBLE_DEVICES='' .venv-mjlab/bin/python - "$checkpoint" <<'PY'
import sys
import torch
snapshot = torch.load(sys.argv[1], map_location='cpu', weights_only=True, mmap=True)
learner = snapshot['learner']
if learner['updates'] == 0:
    raise SystemExit('Source is still an update-0 snapshot. Wait for a durable v3 checkpoint or explicitly select v2 with CHECKPOINT_NATIVE.')
if learner['config']['entropy_cost'] != .003:
    raise SystemExit('Source entropy differs from the requested 0.003; review before launching.')
print(f"Warm start: {sys.argv[1]}, update {learner['updates']}, step {learner['env_steps']}", flush=True)
PY
export JAX_PLATFORMS=cpu MUJOCO_GL=egl PYTHONUNBUFFERED=1
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
exec .venv-mjlab/bin/python train_cat_mjlab.py run \
  --checkpoint-native "$checkpoint" --fresh-optimizer \
  --bank-manifest data/furniture/cat_flat_balance_v1_20260920/manifest.json \
  --body-collision-bank data/furniture/cat_flat_balance_v1_20260920_collision/manifest.json \
  --body-collision-resets data/furniture/cat_flat_balance_v1_20260920_resets/manifest.json \
  --run-dir "outputs/cat_flat_balance_ppo_${num_envs}_20260920" \
  --algorithm ppo --num-envs "$num_envs" --batch-size 768 --num-minibatches "$((num_envs / 768))" \
  --unroll-length 32 --discounting 0.98 --max-action-std 0.15 \
  --flat-bonus-scale 10.0 --flat-region-scale 0.40 \
  --nconmax 64 --njmax 256 --compile-task \
  --max-updates 0 --checkpoint-interval-updates 10 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer --seed 0
