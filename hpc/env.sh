# Activate the project env inside a job:  source ~/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1   (before set -u)
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$("$HOME/bin/micromamba" shell hook -s bash)"
micromamba activate cat-mjlab
