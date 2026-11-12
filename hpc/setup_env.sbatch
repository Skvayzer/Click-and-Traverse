#!/bin/bash
# One-time: build the env (~/micromamba/envs/cat-mjlab) with the exact package versions of dep-1's .venv-mjlab
# (checkpoints record torch/mujoco/warp/mjlab versions and refuse to resume on others).
#SBATCH --job-name=setup_env
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
set -euo pipefail
export MAMBA_ROOT_PREFIX=$HOME/micromamba
export CONDA_PKGS_DIRS=$TMPDIR/pkgs PIP_CACHE_DIR=$TMPDIR/pip-cache
MM=$HOME/bin/micromamba
if [[ ! -x $MM ]]; then
  mkdir -p "$HOME/bin"
  curl -fsSL --retry 3 -o "$MM" https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
  chmod +x "$MM"
fi
"$MM" env list | grep -q '^ *cat-mjlab ' || "$MM" create -y -n cat-mjlab -c conda-forge python=3.12.9 pip
"$MM" run -n cat-mjlab pip install -r $HOME/Click-and-Traverse-Mjlab/hpc/requirements_cluster.txt
"$MM" run -n cat-mjlab python - <<'PY'
import torch, mujoco, warp, mujoco_warp, mjlab
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('mujoco', mujoco.__version__, 'warp', warp.__version__)
warp.init()
PY
