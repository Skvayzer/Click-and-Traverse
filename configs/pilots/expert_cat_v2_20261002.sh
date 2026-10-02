#!/usr/bin/env bash
# EXPERT: CAT scenes (procedural CAT 2226 scenes + original/published CAT 64), in our setup.
# Warm start: the rooms+keyboard expert v9 final (update 492, checkpoints/cat_g1_rooms_keyboard_v9_u492),
# so both experts share one observation layout and command meaning for later distillation.
# Same reward/protection stack as v9 (hand clearance, self-collision, hand-body contact, posture with
# roll, smoothness, pose-based style prior with guard, gait-restart and facing fixes), with CAT changes:
#   * experience: procedural CAT .65, original/published CAT .30, flat .05; rooms/tables/passages 0
#     (joystick episodes and the room route/heading machinery only act in rooms, so they are idle here;
#     CAT worlds keep CAT's field-derived command, turn slot 0 = no orientation demand)
#   * hand clearance -20 (the CAT/passages expert's value; -40 was for tables)
#   * forward-facing (heading) weight 0.4 (v9 used 1.0 for the joystick heading)
#   * head-pull gate off (room-goal fix; CAT's head field must start ducking early under beams)
#   * style guard watches CAT success
# v2 (2026-10-02): the v9 warm start had forgotten CAT (3% success after 60 updates, 91-97% of episodes
# ending in contact; released CAT reaches 23-25% under the same collision rules). Warm start now from the
# RELEASED CAT generalist (converted to native with our environment config; actor input widened 222 -> 226
# with zero weights, identical actions), and the room-tuned body costs back to their earlier values:
# body_motion -0.5 (was -1.5), upright on pitch only (no roll).
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CHECKPOINT=${CHECKPOINT:-outputs/saved_checkpoints/released_cat_native.pt}   # released CAT generalist (scripts/convert_released_cat_to_native.py)
RUN_DIR=${RUN_DIR:-outputs/expert_cat_v2_20261002}
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs ${NUM_ENVS:-40960} --batch-size ${BATCH_SIZE:-1024} \
  --num-minibatches ${NUM_MINIBATCHES:-40} --unroll-length 32 \
  --checkpoint-native "$CHECKPOINT" \
  --fresh-optimizer --max-action-std 0.25 \
  --bank-manifest data/furniture/table_edges_v1_packed/manifest.json \
  --body-collision-bank data/furniture/table_edges_v1_collision/manifest.json \
  --body-collision-resets data/furniture/table_edges_v1_resets/manifest.json \
  --experience-masses 0.30 0.65 0 0 0 0.05 0 0 \
  --scene-group-override table_edges=6 \
  --scene-group-override protected_passage=7 --scene-group-override transition_passage=7 \
  --scene-group-override open_passage=7 --scene-group-override narrow_passage=7 \
  --experience-rebalance-every 5 \
  --narrow-sampling-group 4 \
  --standing-gf-bonus 0.5 --handsdf-weight 1 \
  --heading-align-weight 0.4 \
  --upright-weight 3.0 --stand-tall-weight 3.0 --torso-rate-weight -0.5 \
  --self-clearance-weight -10 --upper-posture-weight -0.5 --upper-home-shoulder-pitch -0.3 \
  --stand-still-weight -4 --standing-requires-stillness --standing-stillness-speed 0.05 \
  --goal-hold-seconds 1 \
  --action-rate-weight -0.02 --joint-acc-weight -0.000005 \
  --hand-body-contact-weight -10 --heading-hand-probes --head-guidance-near-sdf 0 \
  --teleop-fraction 0.5 --teleop-heading-commands --heading-track-weight 0.4 \
  --body-motion-weight -0.5 --style-free-near-obstacles \
  --sdf-rate-obs \
  --style-library data/motion_library/library_v1 --style-weight 0.1 \
  --style-stride 5 --style-features pose --style-grad-penalty 10 --style-disc-steps 4 \
  --style-warmup ${STYLE_WARMUP:-30} --style-ramp ${STYLE_RAMP:-50} --style-guard-tolerance 0.05 \
  --style-guard-window 10 --style-guard-watch scene/procedural_cat/success_rate scene/original_cat/success_rate \
  --run-dir "$RUN_DIR" \
  --disable-hand-contrast --hand-clearance-weight -20 --arm-clearance-weight -8 \
  --tracking-root-field-weight 1 \
  --hand-clearance-target 0.09 --hand-clearance-anticipation 0.20 \
  --hand-clearance-near-weight 0.8 --hand-reward-soft-floor 0 \
  --hand-raised-reset-fraction 0 --upper-gravity-compensation \
  --compile-task --device cuda:0 --seed 0 \
  --checkpoint-interval-updates 10 ${EXTRA:-} \
  --wandb-mode ${WANDB_MODE:-online} --wandb-project CAT-wholebody --wandb-entity skvayzer
