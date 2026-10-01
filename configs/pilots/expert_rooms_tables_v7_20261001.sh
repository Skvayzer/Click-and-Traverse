#!/usr/bin/env bash
# EXPERT: rooms + tables only (clutter, furniture, table edges). No CAT scenes, no passages.
# Split from the rooms+passages expert (2026-09-29): passages need tucked hands and mild hand clearance,
# rooms/tables need raised/kept-away hands and stronger clearance -- one weight set was a compromise
# (weakest skills: table edges 30%, narrow 41%). Differences from expert_rooms_passages_20260929.sh:
#   * sampling: furniture .25, clutter .43, table edges .30 (own group 6), flat .02; passages -> group 7 = 0
#   * hand clearance -40 (was -20)
#   * warm start: the combined expert at update 30
# v2 (2026-09-30): goal hold of 1 s. v1 never ended a room episode at the goal: robots that arrived
# stood parked until the 4000-step horizon (~2/3 of all experience was standing at goals, ~5 fresh
# clutter_dense attempts per update, and a synchronized timeout wave at update ~95 that crashed the
# success metric). Everything else identical to v1.
# v3 (2026-09-30): smoothness + style discriminator on pose. v2 analysis (update 380): robot joint
# accelerations 3-14x human (ankle roll, wrists, waist), the discriminator scored mostly that vibration
# (style reward stuck ~0.12-0.3, flat across speeds). Changes vs v2:
#   * max action std 0.25 (was uncapped, 0.46), action-rate cost -0.02, joint-acceleration cost -5e-6 (was -1e-6)
#   * discriminator: pose features (no joint velocities, no wrist angles), pairs 0.1 s apart (stride 5),
#     gradient penalty 10 (was 5), 4 steps per update (was 8)
# v4 (2026-09-30): v3 + hand vs own body contact (--hand-body-contact-weight -2: physics pairs for hand vs
# torso/pelvis/head/other arm/other hand, cost per touching pair, contact only). Warm start: the overnight
# v2 run at update 200 (best snapshot: rooms 76.7%, table edges 46%, 10-update means), not the combined
# expert. Style discriminators start fresh (pose features differ from v2's).
# v5 (2026-09-30): v4 plateaued; in v2/v3/v4 task success fell every time the style weight ramped
# (v4: rooms 81 -> 70%, tables 76 -> 42%) and the guard never fired. Changes vs v4:
#   * style weight 0.3 -> 0.1; guard watches table_edges / clutter_dense / furniture_dense success,
#     averaged over 10 updates; warmup 30 so the baseline is taken after episodes resolve
#   * hand-body contact -2 -> -10 per touching pair (hand-hand contact had risen to ~1% of steps)
#   * heading/style gates also probe hand height: 0.44 m table gaps now close the forward gate
#     (hand clearance -0.08 m) that shoulder probes left fully open (+0.42 m); 1.5 m gaps unchanged
#   * warm start: v4 update 20 (before the drop: rooms 81%, tables 76%)
# v6 (2026-10-01): v5 + --head-guidance-near-sdf 0.3. Hunching near room goals: room fields point at a goal
# at z=0.75 m, so within ~1.2 m of every goal the head field points down (z -0.4..-0.9) with nothing
# overhead -> crouch gate open (upright/stand_tall off on 100% of those points) and headgf paid the head
# for diving toward the goal; torso >15 deg in 54-60% of frames within 1 m of the goal, 0% beyond 2 m.
# The downward pull now needs geometry within 0.3 m of the head (gate open near goals: 100% -> 9%).
# Warm start: v5 final checkpoint, update 357 (rooms ~80%, tables ~53%).
# v7 (2026-10-01): joystick interface. A quarter of room episodes follow random body-frame joystick
# commands for 20 s (forward -0.5..0.8, sideways +-0.3 m/s, turn rate +-1 rad/s, 20% turn in place,
# 15% stops, 30% aimed at the nearest obstacle); the turn rate reaches the policy (it was always
# zeroed); turn-rate tracking reward 1.0; no forward-facing bonus in those episodes; contact ends them.
# Code fixes active from here: gait restart after standing (synchronized-hop bug) and the
# forward-facing reward now uses the body's world yaw (it compared the command with the world x axis).
# Warm start: v6 final checkpoint, update 98.
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CHECKPOINT=${CHECKPOINT:-outputs/expert_rooms_tables_v6_20261001/snapshots/update_0098_final.pt}   # v6, final update 98
RUN_DIR=${RUN_DIR:-outputs/expert_rooms_tables_v7_20261001}
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs ${NUM_ENVS:-40960} --batch-size ${BATCH_SIZE:-1024} \
  --num-minibatches ${NUM_MINIBATCHES:-40} --unroll-length 32 \
  --checkpoint-native "$CHECKPOINT" \
  --fresh-optimizer --max-action-std 0.25 \
  --bank-manifest data/furniture/table_edges_v1_packed/manifest.json \
  --body-collision-bank data/furniture/table_edges_v1_collision/manifest.json \
  --body-collision-resets data/furniture/table_edges_v1_resets/manifest.json \
  --experience-masses 0 0 0.25 0.43 0 0.02 0.30 0 \
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
  --hand-body-contact-weight -10 --heading-hand-probes --head-guidance-near-sdf 0.3 \
  --teleop-fraction 0.25 --teleop-body-commands --tracking-yaw-weight 1.0 \
  --sdf-rate-obs \
  --style-library data/motion_library/library_v1 --style-weight 0.1 \
  --style-stride 5 --style-features pose --style-grad-penalty 10 --style-disc-steps 4 \
  --style-warmup ${STYLE_WARMUP:-30} --style-ramp ${STYLE_RAMP:-50} --style-guard-tolerance 0.05 \
  --style-guard-window 10 --style-guard-watch scene/table_edges/success_rate scene/clutter_dense/success_rate scene/furniture_dense/success_rate \
  --run-dir "$RUN_DIR" \
  --disable-hand-contrast --hand-clearance-weight -40 --arm-clearance-weight -8 \
  --tracking-root-field-weight 1 \
  --hand-clearance-target 0.09 --hand-clearance-anticipation 0.20 \
  --hand-clearance-near-weight 0.8 --hand-reward-soft-floor 0 \
  --hand-raised-reset-fraction 0 --upper-gravity-compensation \
  --compile-task --device cuda:0 --seed 0 \
  --checkpoint-interval-updates 10 ${EXTRA:-} \
  --wandb-mode ${WANDB_MODE:-online} --wandb-project CAT-wholebody --wandb-entity skvayzer
