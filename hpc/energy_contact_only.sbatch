#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
source "$HOME/Click-and-Traverse-Mjlab/hpc/env.sh" || exit 1
set -euo pipefail
export WANDB_MODE=offline MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
L=/l/users/konstantin.smirnov
export CAT_PATH_MAP=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab=$L/cat_root
cd "${CLONE:?}"
case "${ENERGY_ARM:?}" in
  ENERGY_ONLY) price=(--energy-model heat --energy-alpha 500 --reactive-power-weight -0.02) ;;
  CONTACT_ONLY) price=(--no-energy-price) ;;
  *) exit 2 ;;
esac
B=data/furniture/sidegap_v2
C=$L/checkpoints/generalist_robust_delay_v1_u150.pt
O=$L/runs/energy_contact_${ENERGY_ARM}_$SLURM_JOB_ID
test -s "$C"
mkdir "$O"
export WANDB_DIR=$O PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== job $SLURM_JOB_ID on $(hostname), $(git rev-parse HEAD), warm start $C, output $O"
# A second process using the assigned GPU left neither control room for the bank.
# Log the pre-launch usage and fail before importing CUDA if the card is occupied.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  used=$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  echo "== assigned GPU memory already in use: $used MiB; batch PID $$"
  if (( used > 1500 )); then exit 3; fi
fi
NUM_ENVS=8192
PY=.venv-mjlab/bin/python
"$PY" -u train_cat_mjlab.py run --checkpoint-native "$C" --fresh-optimizer --algorithm ppo \
  --standing-contact-control "${price[@]}" --num-envs "$NUM_ENVS" --batch-size 1024 --num-minibatches "$((NUM_ENVS / 1024))" --unroll-length 32 \
  --bank-manifest "${B}_packed/manifest.json" --body-collision-bank "${B}_collision/manifest.json" \
  --body-collision-resets "${B}_resets/manifest.json" --reactive-bank data/furniture/reactive_standing_v2_20260923/manifest.json \
  --reactive-episode-length 800 --reactive-walking-fraction 0 --sdf-rate-obs --disable-hand-contrast \
  --hand-body-contact-weight -10 --terminate-on-hand-self-contact --action-rate-weight -0.02 --joint-acc-weight -0.000005 \
  --upper-gravity-compensation --upper-target-average --init-action-std 0.10 --max-action-std 0.10 \
  --robust-action-latency-ms 20 --robust-pose-noise 0.01 0.005 0.6 --robust-pose-period 3 8 \
  --robust-mount 1.5 1.0 0.015 --robust-friction 0.5 1.2 --robust-mass 0.9 1.1 --robust-ramp-steps 0 \
  --run-dir "$O/run" --compile-task --device cuda:0 --seed 0 --checkpoint-interval-updates 10 \
  --max-updates 150 --wandb-mode offline --wandb-project CAT-wholebody --wandb-entity skvayzer
grep -q completed_requested_updates "$O/run/status.json"
grep -qE '"updates":[[:space:]]*150,' "$O/run/status.json"
"$PY" -u scripts/eval_standing_hands.py --checkpoint "$O/run/resume.pt" --rows 0:2000 --steps 800 --out "$O/standing.json"
