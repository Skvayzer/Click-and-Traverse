#!/usr/bin/env bash
# Rebalanced launch. Warm-started from the pre-degradation checkpoint (505 updates,
# narrow 0.6577 and clutter 0.7975, both records) rather than the run that had been
# shedding both for 100+ updates.
#
# What changed versus every previous launch, and why:
#
#   --experience-masses     Reset mass is SOLVED from a target share of transitions as
#                           share/episode_length. Measured before: one empty 3x3x3 scene
#                           held 37.8% of experience while all CAT families shared 22.7%
#                           against 60% of declared mass, because CAT episodes are 1000
#                           steps and rooms are 4000. Episode lengths are left alone.
#   --narrow-sampling-group All 62 narrow scenes had been emitted into generic_clutter,
#                           drawing from a 0.034 budget shared with 461 rooms.
#   --standing-gf-bonus     gf_reward folded "standing still" into the same 4.0 constant as
#                           "reached the goal": +16/step that no action could change, measured
#                           at ~54% of mean per-step reward. Not removed outright, because
#                           standing would go net-negative and the reward floor would clamp
#                           its gradient to zero.
#   --reactive-hand-guidance Hand guidance was zeroed before the object merged, so handsgf --
#                           the only term anywhere depending on hand VELOCITY relative to an
#                           obstacle -- was identically zero in the scenes built to train
#                           hand retreat. Priority #1 was positional only.
#   --handsdf-weight 0      handsdf (+1) and hand_clearance (-20) are both functions of
#                           sdf[:,5:7]; handsdf carried 1.8x-5.4x more near-field gradient,
#                           so every sweep we ran on the -20 moved under a third of the
#                           real signal. One owner now, at -60.
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
# The previous run died at update 108 on fragmentation with 6 GB of headroom; the
# per-sample direction decode allocates transient tensors every call.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs 40960 --batch-size 1024 \
  --num-minibatches 40 --unroll-length 32 \
  --checkpoint-native outputs/cat_reactive_standing_20260923/resume.pt \
  --fresh-optimizer --max-action-std 0 \
  --bank-manifest data/furniture/procedural_rooms_v2_20260923/manifest.json \
  --body-collision-bank data/furniture/full_collision_20260924b/manifest.json \
  --body-collision-resets data/furniture/full_resets_20260924b/manifest.json \
  --reactive-bank data/furniture/reactive_standing_v2_20260923/manifest.json \
  --experience-masses 0.17 0.34 0.08 0.20 0.10 0.02 0.06 0.03 \
  --narrow-sampling-group 4 \
  --standing-gf-bonus 0.5 --reactive-hand-guidance --handsdf-weight 0 \
  --run-dir outputs/cat_rebalanced_20260924 \
  --disable-hand-contrast --hand-clearance-weight -60 --arm-clearance-weight -8 \
  --tracking-root-field-weight 1 \
  --hand-clearance-target 0.09 --hand-clearance-anticipation 0.20 \
  --hand-clearance-near-weight 0.8 --hand-reward-soft-floor 0 \
  --hand-raised-reset-fraction 0 --upper-gravity-compensation \
  --compile-task --device cuda:0 --seed 0 \
  --checkpoint-interval-updates 10 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
