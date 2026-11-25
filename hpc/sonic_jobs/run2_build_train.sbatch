#!/bin/bash
# Run 2: rebuild the missing collision bank / certified resets of tablegap_bb_v1 (tablegap_v1 + 600 beam/bar scenes,
# incremental on tablegap_v1), then warm-start the SONIC latent adaptor from u600 with the curricula.
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
L=/l/users/konstantin.smirnov
D=$HOME/cat_worktrees/sonic_aefa6cc
F=$L/cat_root/data/furniture
BB=$F/tablegap_bb_v1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd $D
P=.venv-mjlab/bin/python
# Builders read dep-1 absolute pins inside the base manifests: run them under the cluster path remap.
mapped() { "$P" -u -c 'import runpy, sys; sys.path.insert(0, "."); from cat_mjlab import pathmap; pathmap.install(); script = sys.argv[1]; sys.argv = sys.argv[1:]; runpy.run_path(script, run_name="__main__")' "$@"; }
echo "== build start $(date -Is)"
test -e ${BB}_collision/manifest.json || mapped scripts/build_body_collision_bank.py --field-manifest ${BB}_packed/manifest.json \
  --output ${BB}_collision --base-collision-manifest $F/tablegap_v1_collision/manifest.json --workers 8
test -e ${BB}_resets/manifest.json || mapped scripts/build_body_collision_resets.py --cpu-native \
  --field-manifest ${BB}_packed/manifest.json --collision-bank ${BB}_collision/manifest.json --output ${BB}_resets \
  --base-reset-manifest $F/tablegap_v1_resets/manifest.json
echo "== build done $(date -Is)"
export CLONE=$D PHASE=train TAG=run2_curric UPDATES=1400 SNAP=100 BANK=$BB \
  NATIVE=$L/runs/sonic_latent_seed0_322876/snapshots/update_0600.pt \
  EXTRA="--sonic-keep-sgs --sonic-soft-until-cat-clean 0.1 --sonic-height-curriculum 0.5 300 --sonic-goal-cache ${BB}_goal_column"
bash $D/hpc/sonic_latent.sbatch
