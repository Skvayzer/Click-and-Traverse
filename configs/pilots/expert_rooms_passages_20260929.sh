#!/usr/bin/env bash
# EXPERT: rooms + passages only (no CAT scenes), with the context-gated style prior.
# Part of the experts + distillation structure (see docs/obsidian/Style-Prior-Pipeline.md, "Experts").
# CAT navigation expert = the released CAT generalist (not retrained).
# Experience: groups 0,1 (CAT) = 0; furniture .15, clutter+passages+tables .45, narrow .22, flat .02,
# protected .08, transition .08. Same rewards and style prior as configs/pilots/style_20260929.sh.
# Style prior (as in the pilot), differences from bundle_20260924.sh:
#   + --style-library: 3 discriminators (locomotion / sidle / duck_step) from motion library v1,
#     gated by scene context; style reward in a separate critic; weight 0 for 20 updates, then a
#     50-update ramp to 0.3 with a success guard.
#   - approaching-object (reactive) episodes removed for now (user decision 2026-09-29):
#     no --reactive-bank, no reactive share/horizon/walking/hand-guidance/spot-hold flags.
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CHECKPOINT=${CHECKPOINT:-outputs/cat_style_20260929/resume.pt}   # style pilot, update 40
RUN_DIR=${RUN_DIR:-outputs/expert_rooms_passages_20260929}
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs ${NUM_ENVS:-40960} --batch-size ${BATCH_SIZE:-1024} \
  --num-minibatches ${NUM_MINIBATCHES:-40} --unroll-length 32 \
  --checkpoint-native "$CHECKPOINT" \
  --fresh-optimizer --max-action-std 0 \
  --bank-manifest data/furniture/table_edges_v1_packed/manifest.json \
  --body-collision-bank data/furniture/table_edges_v1_collision/manifest.json \
  --body-collision-resets data/furniture/table_edges_v1_resets/manifest.json \
  --experience-masses 0 0 0.15 0.45 0.22 0.02 0.08 0.08 \
  --experience-rebalance-every 5 \
  --narrow-sampling-group 4 \
  --standing-gf-bonus 0.5 --handsdf-weight 1 \
  --heading-align-weight 0.4 \
  --upright-weight 3.0 --stand-tall-weight 3.0 --torso-rate-weight -0.5 \
  --self-clearance-weight -10 --upper-posture-weight -0.5 --upper-home-shoulder-pitch -0.3 \
  --stand-still-weight -4 --standing-requires-stillness --standing-stillness-speed 0.05 \
  --sdf-rate-obs \
  --style-library data/motion_library/library_v1 --style-weight 0.3 \
  --style-warmup ${STYLE_WARMUP:-20} --style-ramp ${STYLE_RAMP:-50} --style-guard-tolerance 0.05 \
  --run-dir "$RUN_DIR" \
  --disable-hand-contrast --hand-clearance-weight -20 --arm-clearance-weight -8 \
  --tracking-root-field-weight 1 \
  --hand-clearance-target 0.09 --hand-clearance-anticipation 0.20 \
  --hand-clearance-near-weight 0.8 --hand-reward-soft-floor 0 \
  --hand-raised-reset-fraction 0 --upper-gravity-compensation \
  --compile-task --device cuda:0 --seed 0 \
  --checkpoint-interval-updates 10 ${EXTRA:-} \
  --wandb-mode ${WANDB_MODE:-online} --wandb-project CAT-wholebody --wandb-entity skvayzer
