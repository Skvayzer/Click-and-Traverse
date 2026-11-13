#!/usr/bin/env bash
# STATIONARY HAND PROTECTION (2026-11-13): objects approach the standing robot's hands; the robot must keep them clear.
# Start: generalist_robust_delay_v1_u150 in its own setup (robust v1 randomisation, waist upper-target average), plus
# approaching-object standing episodes (reactive_standing_v2 object parameters, aimed at the live hand) as
# REACTIVE_SHARE of all transitions. Variants (env vars):
#   S1 stationary  STATIONARY="0.05 5 30 0.2": both feet stay on the floor (<= 5 steps off) and within 5 cm of where they
#                  settled (pelvis within 0.2 m), else the episode ends as a failure -> hands protected by the upper body.
#   S2 energy      POWER=-0.02: cost per watt of joint mechanical power in standing episodes (old 0.3 m spot rule) --
#                  does pricing energy alone make the robot move its light arms instead of stepping aside?
#   S0 control     neither (old 0.3 m spot rule): reproduces "the robot steps aside".
# Anchor to u150 everywhere except the flat background scene the standing episodes use (hpc/route_anchor_no_reactive.json),
# so new arm behaviour there is free while traversal skills stay anchored. 10 critic warm-up updates (new terms).
set -euo pipefail
cd "${REPO:-/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STUDENT=${STUDENT:-outputs/saved_checkpoints/generalist_robust_delay_v1_u150.pt}
TRAVERSAL=${TRAVERSAL:-outputs/traversal_expert_v1_20261012/snapshots/update_1764.pt}
NARROW=${NARROW:-outputs/saved_checkpoints/narrow_expert_v2_u1400.pt}
ROOMS=${ROOMS:-outputs/saved_checkpoints/cat_g1_rooms_keyboard_v9_u492_full.pt}
ROUTING=${ROUTING:-outputs/distill_v1_prep/routing_v2.json}
ZERO_WEIGHT=${ZERO_WEIGHT:-configs/pilots/finetune_infeasible_scenes.txt}
SGS_PRIOR=${SGS_PRIOR:-outputs/distill_v1_prep/eval_student_v1_cat.json outputs/distill_v1_prep/eval_rooms_v9.json}   # rooms: the teacher it copies
RUN_DIR=${RUN_DIR:-outputs/stationary_hands_20261113}
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs ${NUM_ENVS:-40960} --batch-size ${BATCH_SIZE:-1024} \
  --num-minibatches ${NUM_MINIBATCHES:-40} --unroll-length 32 \
  --checkpoint-native "$STUDENT" \
  ${FRESH---fresh-optimizer} --init-action-std 0.10 --max-action-std ${MAX_STD:-0.10} \
  --bank-manifest data/furniture/sidegap_v2_packed/manifest.json \
  --body-collision-bank data/furniture/sidegap_v2_collision/manifest.json \
  --body-collision-resets data/furniture/sidegap_v2_resets/manifest.json \
  --experience-masses ${MASSES:-0.15 0.35 0.08 0.14 0.12 0.03 0.10 0.03} \
  --scene-group-override table_edges=6 --scene-group-override side_gap=4 \
  --scene-group-override protected_passage=7 --scene-group-override transition_passage=7 \
  --scene-group-override open_passage=7 --scene-group-override narrow_passage=7 \
  --experience-rebalance-every 5 \
  --narrow-sampling-group 4 \
  --zero-weight-scenes @"$ZERO_WEIGHT" \
  --standing-gf-bonus 0.5 --handsdf-weight 1 \
  --heading-align-weight 0.4 \
  --upright-weight 3.0 --stand-tall-weight 3.0 --torso-rate-weight -0.5 \
  --self-clearance-weight -10 --upper-posture-weight -0.5 --upper-home-shoulder-pitch -0.3 \
  --stand-still-weight -4 --standing-requires-stillness --standing-stillness-speed 0.05 \
  --goal-hold-seconds 1 \
  --action-rate-weight -0.02 --joint-acc-weight -0.000005 \
  --hand-body-contact-weight -10 --heading-hand-probes --head-guidance-near-sdf 0.3 \
  --teleop-fraction 0.5 --teleop-heading-commands --heading-track-weight 0.4 \
  --body-motion-weight -0.5 --style-free-near-obstacles \
  --hand-clearance-tight 0.10 0.40 0.3 --hand-tight-walls-only --hand-tight-cat-only \
  --stall-weight -5 --goal-progress-weight 5 --goal-bonus-weight 2000 --timeout-failure \
  --heading-lookaheads 0.3 0.6 1.0 --heading-blocking-probes --sideways-bonus-weight 1.0 \
  --commanded-blocking-map off \
  --reward-group-room wholebody_hand_clearance=-40 heading_align=1.0 body_motion=-1.5 sideways_bonus=0 stall=0 goal_progress=0 goal_bonus=0 \
  --sdf-rate-obs \
  --distill-expert self="$STUDENT" --distill-route _default=self --distill-route-file "${ROUTE_FILE:-hpc/route_anchor_no_reactive.json}" \
  --distill-regularize ${REG:-0.5 0.3 100} --critic-warmup-updates ${WARMUP:-10} \
  --robust-action-latency-ms 20 --robust-pose-noise 0.01 0.005 0.6 --robust-pose-period 3 8 \
  --robust-mount 1.5 1.0 0.015 --robust-friction 0.5 1.2 --robust-mass 0.9 1.1 --robust-ramp-steps ${RAMP:-0} --upper-target-average \
  --reactive-bank data/furniture/reactive_standing_v2_20260923/manifest.json --experience-reactive-share ${REACTIVE_SHARE:-0.5} \
  --reactive-episode-length 800 --reactive-hand-guidance --reactive-event-weight ${EVENT_WEIGHT:-10} \
  ${STATIONARY:+--reactive-stationary $STATIONARY} ${POWER:+--reactive-power-weight $POWER} \
  ${FEET_HOLD:+--reactive-feet-hold-weight $FEET_HOLD} \
  ${HAND_FAR:+--reactive-hand-clearance $HAND_FAR} ${RETREAT_SUCCESS:+--reactive-retreat-success $RETREAT_SUCCESS} \
  ${TABLE_FACING:+--table-facing-weight $TABLE_FACING} ${TABLE_LEG_POWER:+--table-leg-power-weight $TABLE_LEG_POWER} \
  --sgs-slots ${SGS_SLOTS:-4} --sgs-history 100 --sgs-target 0.66 --sgs-kappa 5 --sgs-temperature 2 --sgs-epsilon 1e-8 \
  --sgs-uniform-mix ${SGS_MIX:-0.30} --sgs-prior $SGS_PRIOR --sgs-prior-count 4 \
  --style-library data/motion_library/library_v1 --style-weight 0.1 \
  --style-stride 5 --style-features pose --style-grad-penalty 10 --style-disc-steps 4 \
  --style-warmup ${STYLE_WARMUP:-30} --style-ramp ${STYLE_RAMP:-50} --style-guard-tolerance 0.05 \
  --style-guard-window 10 --style-guard-watch scene/procedural_cat/success_rate scene/original_cat/success_rate scene/clutter_dense/success_rate \
  --run-dir "$RUN_DIR" \
  --disable-hand-contrast --hand-clearance-weight -20 --arm-clearance-weight -8 \
  --tracking-root-field-weight 1 \
  --hand-clearance-target 0.09 --hand-clearance-anticipation 0.20 \
  --hand-clearance-near-weight 0.8 --hand-reward-soft-floor 0 \
  --hand-raised-reset-fraction ${RAISED:-0} --upper-gravity-compensation \
  ${COMPILE:---compile-task} --device ${DEVICE:-cuda:0} --seed 0 \
  --checkpoint-interval-updates 10 ${EXTRA:-} \
  --wandb-mode ${WANDB_MODE:-online} --wandb-project CAT-wholebody --wandb-entity skvayzer
