#!/usr/bin/env bash
# WAIST-DITHER FIX FINE-TUNE (2026-10-21) of generalist_robust_delay_v1_u150, in its own (robust v1) setup.
# On the real G1 the policy drove waist roll/pitch with a 25 Hz +-0.04 rad sawtooth: a learned period-2 loop through
# its own observed upper-body targets (loop gain about -4, inherited from all three teachers). Fix: upper-body motors
# (joints 12-28) track the mean of the last two upper targets (--upper-target-average), which removes the 25 Hz
# component exactly whatever the policy does; observations unchanged; the robot executor applies the same average.
# Everything else as robust v1: rewards, SGS, std 0.10, latency 0-20 ms, pose/mount error, friction x0.5-1.2,
# mass x0.9-1.1 -- at full strength from the start (u150 already trained at full strength; ramp 0).
# Anchor: u150 itself, lambda 0.5 -> 0.3 over 100 updates (v1 was at 0.49 at update 150); 5 critic warm-up updates.
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STUDENT=${STUDENT:-outputs/saved_checkpoints/generalist_robust_delay_v1_u150.pt}
TRAVERSAL=${TRAVERSAL:-outputs/traversal_expert_v1_20261012/snapshots/update_1764.pt}
NARROW=${NARROW:-outputs/saved_checkpoints/narrow_expert_v2_u1400.pt}
ROOMS=${ROOMS:-outputs/saved_checkpoints/cat_g1_rooms_keyboard_v9_u492_full.pt}
ROUTING=${ROUTING:-outputs/distill_v1_prep/routing_v2.json}
ZERO_WEIGHT=${ZERO_WEIGHT:-configs/pilots/finetune_infeasible_scenes.txt}
SGS_PRIOR=${SGS_PRIOR:-outputs/distill_v1_prep/eval_student_v1_cat.json outputs/distill_v1_prep/eval_rooms_v9.json}   # rooms: the teacher it copies
RUN_DIR=${RUN_DIR:-outputs/finetune_robust_v1f_20261021}
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
  --distill-expert self="$STUDENT" --distill-route _default=self \
  --distill-regularize ${REG:-0.5 0.3 100} --critic-warmup-updates ${WARMUP:-5} \
  --robust-action-latency-ms 20 --robust-pose-noise 0.01 0.005 0.6 --robust-pose-period 3 8 \
  --robust-mount 1.5 1.0 0.015 --robust-friction 0.5 1.2 --robust-mass 0.9 1.1 --robust-ramp-steps ${RAMP:-0} --upper-target-average \
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
  --hand-raised-reset-fraction 0 --upper-gravity-compensation \
  ${COMPILE:---compile-task} --device ${DEVICE:-cuda:0} --seed 0 \
  --checkpoint-interval-updates 10 ${EXTRA:-} \
  --wandb-mode ${WANDB_MODE:-online} --wandb-project CAT-wholebody --wandb-entity skvayzer
