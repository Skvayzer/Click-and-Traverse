#!/bin/bash
# Whole-body planner v1 (cat_mjlab/planner): collect clean 2 s windows from V7, E2 and T2 on tablegap_v1, merge them,
# train the flow-matching planner, evaluate it on held-out scenes and save example plans for rendering.
# One RTX 5000 Ada 32 GB, ws-ia. Each stage is skipped when its output exists, so resubmitting continues a run.
#
#   T=$HOME/cat_worktrees/<commit>; cd ~/jobs; RUN=planner_v1 CLONE=$T sbatch --job-name=planner_v1 --export=ALL $T/hpc/planner.sbatch
#
# Knobs (environment): SRC_V7 / SRC_E2 / SRC_T2 checkpoints, MAX_V7 / MAX_E2 / MAX_T2 training examples per source
# (training holds every grid in host memory, ~46 KB each), MAX_TEST per source, ENVS worlds, STEPS training steps,
# SAVE_PLANS situations per scene type kept for videos. SMOKE=1: a few minutes end to end, into runs/wbp/smoke_<job>.
# Planner v2 (no standing windows in hard scenes): link data_v7/e2/t2 and all from planner_v1 into the new run directory,
# then TRAIN_EXTRA="--drop-stopped 0.05" HESITATION_REF=<v1 planner.pt> RUN=planner_v2_nostop.
#SBATCH --job-name=planner
#SBATCH --output=/l/users/%u/runs/logs/%x_%j.out
#SBATCH --partition=ws-ia
#SBATCH --cpus-per-task=12
#SBATCH --mem=54000M
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
source $HOME/Click-and-Traverse-Mjlab/hpc/env.sh || exit 1
set -euo pipefail
L=/l/users/$USER
CLONE=${CLONE:-$HOME/Click-and-Traverse-Mjlab}
export REPO=$TMPDIR/code
rsync -a --exclude .git $CLONE/ $REPO/
cd $REPO
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
BANK=data/furniture/tablegap_v1
if [[ -n "${SMOKE:-}" ]]; then
  D=$L/runs/wbp/smoke_${SLURM_JOB_ID}; ENVS=${ENVS:-256}; ROUNDS="--steps 300 --rounds 1 --test-rounds 1"
  MAX_V7=4000 MAX_E2=4000 MAX_T2=4000 MAX_TEST=1000 STEPS=200 SAVE_PLANS=2 TRAIN_EXTRA="--eval-every 100 --log-every 50"
else
  D=$L/runs/wbp/${RUN:-planner_v1}; ROUNDS="--steps 1500 --rounds 2 --test-rounds 1"; TRAIN_EXTRA="${TRAIN_EXTRA:-}"
fi
mkdir -p $D
{ echo "== job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname) at $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo "commit $(git -C $CLONE rev-parse HEAD) $(git -C $CLONE status --porcelain --untracked-files=no | wc -l) modified tracked files"; } >> $D/run_info.txt
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 60 >> $D/gpu.csv &
( while sleep 60; do echo "$(date +%T) $(ps -u $USER -o rss= | awk '{s+=$1} END {printf "%.0f", s/1024}') MB"; done ) >> $D/host_mem.txt &

collect() {   # name checkpoint source_id max_train
  local out=$D/data_$1
  if [[ -f $out/info.json ]]; then echo "== skip collect $1 (done)"; return; fi
  rm -rf $out
  echo "== collect $1 from $2 at $(date -Is)"
  python scripts/planner_collect.py --checkpoint $2 --bank $BANK --out $out --device cuda:0 --num-envs ${ENVS:-4096} $ROUNDS \
      --max-train $4 --max-test ${MAX_TEST:-20000} --source-id $3
}
collect v7 ${SRC_V7:-$L/runs/stationary_hands_20261113_V7/run/snapshots/update_0450.pt} 2 ${MAX_V7:-200000}
collect e2 ${SRC_E2:-$L/checkpoints/stationary_hands_E2_u300.pt} 1 ${MAX_E2:-150000}
collect t2 ${SRC_T2:-$L/checkpoints/stationary_hands_T2_u650.pt} 3 ${MAX_T2:-150000}
[[ -f $D/all/info.json ]] || python scripts/planner_merge.py --out $D/all $D/data_v7 $D/data_e2 $D/data_t2
if [[ ! -f $D/train/DONE ]]; then
  echo "== train at $(date -Is)"
  python scripts/planner_train.py --data $D/all --out $D/train --steps ${STEPS:-100000} --device cuda:0 $TRAIN_EXTRA
  touch $D/train/DONE
fi
echo "== eval at $(date -Is)"
python scripts/planner_eval.py --data $D/all --checkpoint $D/train/planner.pt --out $D/eval --save-plans ${SAVE_PLANS:-6}
if [[ -n "${HESITATION_REF:-}" ]]; then
  echo "== hesitation at $(date -Is)"
  python scripts/planner_hesitation.py --data $D/all --checkpoint $HESITATION_REF $D/train/planner.pt --out $D/hesitation.json | tail -40
fi
echo "== PLANNER JOB DONE at $(date -Is); peak GPU MiB $(cut -d, -f2 $D/gpu.csv | tr -dc '0-9\n' | sort -n | tail -1); peak host $(awk '{print $2}' $D/host_mem.txt | sort -n | tail -1) MB" | tee -a $D/run_info.txt
