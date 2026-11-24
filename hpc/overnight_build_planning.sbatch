#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 WANDB_MODE=offline JAX_PLATFORMS=cpu
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=/l/users/konstantin.smirnov/cat_root
cd "${CLONE:?}"
B=data/furniture/tablegap_v1
O=${BANK:?}
P=.venv-mjlab/bin/python
"$P" -u scripts/generate_planning_benchmark.py --base-manifest "${B}_packed/manifest.json" --output "${O}_packed" --workers 8
# Import the project path map before this standalone script reads the pinned bank.
"$P" -u -c 'import cat_mjlab, runpy; runpy.run_path("scripts/build_body_collision_bank.py", run_name="__main__")' --field-manifest "${O}_packed/manifest.json" --output "${O}_collision"   --base-collision-manifest "${B}_collision/manifest.json" --workers 6
"$P" -u scripts/build_body_collision_resets.py --cpu-native --field-manifest "${O}_packed/manifest.json"   --collision-bank "${O}_collision/manifest.json" --output "${O}_resets"
"$P" -u scripts/build_goal_seed_cache.py --bank "$O" --out "${O}_goal" --scenes "${O}_packed/scene_ids.txt" --workers 8
