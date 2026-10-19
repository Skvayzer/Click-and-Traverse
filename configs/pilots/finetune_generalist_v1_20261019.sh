#!/usr/bin/env bash
# RL FINE-TUNE of the three-teacher DAgger student (step 2 of the combine-skills plan, section 7k).
#   * PPO learns; the three teachers' labels stay as a fading regulariser (--distill-regularize 1 -> 0 over 300
#     updates; scenes without a competent teacher are unlabelled). First 20 updates: critic only (actor frozen):
#     the student's critic is the traversal expert's, never fitted to room rewards.
#   * Per-group rewards: CAT worlds use the CAT experts' weights (identical for traversal and narrow: hand clearance
#     -20, heading 0.4, body motion -0.5, sideways bonus, stall -5 / goal progress 5 / goal bonus 2000, failed
#     timeouts terminal, tight hand margins between full-height walls); ROOM worlds the rooms expert's
#     (--reward-group-room: hand clearance -40, heading 1.0, body motion -1.5, no sideways bonus / anti-refusal).
#   * Inputs as the teachers saw them: commanded worlds without the blocking map, head-guidance gate 0.3.
#   * Low action spread (Parkour-in-the-Wild recipe): initial std 0.10, ceiling 0.15.
#   * Success-Guided Sampling (arXiv 2610.12465, task._init_sgs): configurations = scenes x 4 fixed start states,
#     drawn at every reset with the paper's kernel (t 0.66, kappa 5, T 2, eps 1e-8, last 100 outcomes) within the
#     fixed group masses, 5% uniform per group; buffers primed with the student's measured per-scene success.
#     All scenes are in the pool -- also the 463 without a competent teacher (unlabelled, pure RL): SGS gives
#     hopeless configurations ~nothing and picks them up once some start succeeds. ZERO_WEIGHT: only the 11
#     geometrically infeasible scenes (scripts/narrow_scene_feasibility.py).
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STUDENT=${STUDENT:-outputs/distill_three_teachers_v1_20261018/resume.pt}
TRAVERSAL=${TRAVERSAL:-outputs/traversal_expert_v1_20261012/snapshots/update_1764.pt}
NARROW=${NARROW:-outputs/saved_checkpoints/narrow_expert_v2_u1400.pt}
ROOMS=${ROOMS:-outputs/saved_checkpoints/cat_g1_rooms_keyboard_v9_u492_full.pt}
ROUTING=${ROUTING:-outputs/distill_v1_prep/routing_v2.json}
ZERO_WEIGHT=${ZERO_WEIGHT:-configs/pilots/finetune_infeasible_scenes.txt}
SGS_PRIOR=${SGS_PRIOR:-outputs/distill_v1_prep/eval_student_v1_cat.json outputs/distill_v1_prep/eval_student_v1_room.json}
RUN_DIR=${RUN_DIR:-outputs/finetune_generalist_v1_20261019}
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs ${NUM_ENVS:-40960} --batch-size ${BATCH_SIZE:-1024} \
  --num-minibatches ${NUM_MINIBATCHES:-40} --unroll-length 32 \
  --checkpoint-native "$STUDENT" \
  ${FRESH---fresh-optimizer --init-action-std 0.10} --max-action-std 0.15 \
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
  --distill-expert traversal="$TRAVERSAL" --distill-expert narrow="$NARROW" --distill-expert rooms="$ROOMS" \
  --distill-route _default=rooms --distill-route-file "$ROUTING" \
  --distill-regularize ${REG:-1.0 0.0 300} --critic-warmup-updates ${WARMUP:-20} \
  --sgs-slots ${SGS_SLOTS:-4} --sgs-history 100 --sgs-target 0.66 --sgs-kappa 5 --sgs-temperature 2 --sgs-epsilon 1e-8 \
  --sgs-uniform-mix ${SGS_MIX:-0.05} --sgs-prior $SGS_PRIOR --sgs-prior-count 4 \
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
