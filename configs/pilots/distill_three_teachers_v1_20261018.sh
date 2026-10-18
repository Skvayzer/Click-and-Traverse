#!/usr/bin/env bash
# THREE-TEACHER DAgger (2026-10-18), field-observation student (step 1 of the combine-skills plan, see
#   docs/obsidian/Style-Prior-Pipeline.md section 7k). Teachers, frozen, queried at SAPG head 0 (the leader):
#   traversal = traversal expert v1 update 1764 (hurdles, crouch, open CAT, openings >= 0.45 m)
#   narrow    = narrow-passage expert v2 update 1400 (side gaps, narrow procedural openings)
#   rooms     = rooms+keyboard expert v9 update 492 (furniture, clutter, table edges, flat, keyboard driving)
# Routing PER SCENE (--distill-route-file, scripts/build_distill_routing.py): the expert that trained on the scene,
#   unless the other CAT expert measured > 15 points better (12 deterministic episodes per scene); scenes whose
#   owner is below 20% clean are dropped (--zero-weight-scenes) -- no competent teacher, learned in the RL stage.
# Environment: the rooms expert's command settings in commanded (room / keyboard) worlds -- no blocking map
#   (no --heading-blocking-probes / --hand-clearance-tight), head-guidance gate 0.3, no sideways joystick: both change
#   what the policy SEES there (heading command, guidance). CAT worlds are field-driven and identical for all three
#   teachers. Rewards are logged only (DAgger learns from labels); success metrics are the evaluation.
# Student: warm start from the traversal expert (largest share), standard actor 512-256-128-64.
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TRAVERSAL=${TRAVERSAL:-outputs/traversal_expert_v1_20261012/snapshots/update_1764.pt}
NARROW=${NARROW:-outputs/saved_checkpoints/narrow_expert_v2_u1400.pt}
ROOMS=${ROOMS:-outputs/saved_checkpoints/cat_g1_rooms_keyboard_v9_u492_full.pt}
STUDENT_INIT=${STUDENT_INIT:-$TRAVERSAL}
ROUTING=${ROUTING:-outputs/distill_v1_prep/routing_v1.json}
RUN_DIR=${RUN_DIR:-outputs/distill_three_teachers_v1_20261018}
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs ${NUM_ENVS:-40960} --batch-size ${BATCH_SIZE:-1024} \
  --num-minibatches ${NUM_MINIBATCHES:-40} --unroll-length 32 \
  --checkpoint-native "$STUDENT_INIT" \
  --fresh-optimizer --max-action-std 0 \
  --bank-manifest data/furniture/sidegap_v2_packed/manifest.json \
  --body-collision-bank data/furniture/sidegap_v2_collision/manifest.json \
  --body-collision-resets data/furniture/sidegap_v2_resets/manifest.json \
  --experience-masses ${MASSES:-0.15 0.35 0.08 0.14 0.12 0.03 0.10 0.03} \
  --scene-group-override table_edges=6 --scene-group-override side_gap=4 \
  --scene-group-override protected_passage=7 --scene-group-override transition_passage=7 \
  --scene-group-override open_passage=7 --scene-group-override narrow_passage=7 \
  --experience-rebalance-every 5 \
  --narrow-sampling-group 4 \
  --zero-weight-scenes @"$ROUTING.dropped.txt" \
  --standing-gf-bonus 0.5 --handsdf-weight 1 \
  --heading-align-weight 0.4 \
  --upright-weight 3.0 --stand-tall-weight 3.0 --torso-rate-weight -0.5 \
  --self-clearance-weight -10 --upper-posture-weight -0.5 --upper-home-shoulder-pitch -0.3 \
  --stand-still-weight -4 --standing-requires-stillness --standing-stillness-speed 0.05 \
  --goal-hold-seconds 1 \
  --action-rate-weight -0.02 --joint-acc-weight -0.000005 \
  --hand-body-contact-weight -10 --heading-hand-probes --head-guidance-near-sdf 0.3 \
  --teleop-fraction 0.5 --teleop-heading-commands --heading-track-weight 1.0 \
  --upright-roll --body-motion-weight -1.5 --style-free-near-obstacles \
  --sdf-rate-obs \
  --distill-expert traversal="$TRAVERSAL" --distill-expert narrow="$NARROW" --distill-expert rooms="$ROOMS" \
  --distill-route _default=rooms --distill-route-file "$ROUTING" \
  --distill-beta-decay ${BETA_DECAY:-60} --distill-epochs 2 --distill-lr 3e-4 \
  --run-dir "$RUN_DIR" \
  --disable-hand-contrast --hand-clearance-weight -40 --arm-clearance-weight -8 \
  --tracking-root-field-weight 1 \
  --hand-clearance-target 0.09 --hand-clearance-anticipation 0.20 \
  --hand-clearance-near-weight 0.8 --hand-reward-soft-floor 0 \
  --hand-raised-reset-fraction 0 --upper-gravity-compensation \
  ${COMPILE:---compile-task} --device ${DEVICE:-cuda:0} --seed 0 \
  --checkpoint-interval-updates 10 ${EXTRA:-} \
  --wandb-mode ${WANDB_MODE:-online} --wandb-project CAT-wholebody --wandb-entity skvayzer
