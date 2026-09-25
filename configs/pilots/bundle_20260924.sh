#!/usr/bin/env bash
# Restart bundle, 2026-09-24. Everything below was measured on the cat_rebalanced_20260924 run
# (66 updates) and its 19 recorded rollouts; see the session notes for the numbers.
#
#   --experience-rebalance-every / --experience-reactive-share
#       Reset masses re-solved from REALIZED episode lengths; the reactive standing coin
#       (fixed .25 of resets, 4000-step episodes that never end early) had grown to 64% of
#       all experience while CAT+rooms fell from 66% to 30% and every goal skill eroded.
#   --heading-align-weight 0.4
#       Face the guidance direction where a forward-facing body fits; gate closed in gaps
#       under ~0.42 m so sidling is never penalised.
#   --upright-weight / --stand-tall-weight / --torso-rate-weight
#       Straight back and full height unless the head field asks for a crouch. Measured
#       walking posture: 27-52 deg torso pitch, pelvis 10-14 cm low, head just under the
#       1.1 m line where tracking_orientation stops penalising pitch.
#   --self-clearance-weight / --upper-posture-weight / --upper-home-shoulder-pitch
#       Hand envelopes vs own thigh/shin capsules (fingers were within 2 cm of a leg in
#       40-64% of walking frames, penetrating in 8 of 17 walks); posture prior raised
#       from -0.05 and re-homed so the hands carry 24 cm in front of the hips.
#   --stand-still-weight / --standing-requires-stillness
#       Standing scenes: root motion at zero command was free and the standing bonus was
#       paid on the command, so the policy answered approaching objects by walking away
#       (0.10-0.12 m root motion per event, 3 cm hand retreat). reactive/left_spot_rate
#       reports how often an episode leaves its 0.3 m spot.
#   --hand-clearance-weight -20 --handsdf-weight 1
#       Back to the hand objective the warm start was trained with. The -60/0 setting of the
#       two later runs bought nothing on hand collisions (0.0% of passage outcomes in all
#       three runs) and doubled narrow non-entry (0.9% -> 1.8%): tripling the far-field
#       (0.09-0.20 m) pressure makes 0.40-0.74 m passages unenterable.
#   Physical hand-leg contact pairs are in the assembled model (cat_mjlab/model.py) and
#   feed scene/*/hand_self_contact_rate; termination on contact is opt-in
#   (--terminate-on-hand-self-contact) and deliberately off for this pilot.
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CHECKPOINT=${CHECKPOINT:-outputs/cat_bundle_20260924/resume.pt}   # continue from the first bundle pilot
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs 40960 --batch-size 1024 \
  --num-minibatches 40 --unroll-length 32 \
  --checkpoint-native "$CHECKPOINT" \
  --fresh-optimizer --max-action-std 0 \
  --bank-manifest data/furniture/table_edges_v1_packed/manifest.json \
  --body-collision-bank data/furniture/table_edges_v1_collision/manifest.json \
  --body-collision-resets data/furniture/table_edges_v1_resets/manifest.json \
  --reactive-bank data/furniture/reactive_standing_v2_20260923/manifest.json \
  --experience-masses 0.17 0.34 0.08 0.20 0.10 0.02 0.06 0.03 \
  --experience-rebalance-every 5 --experience-reactive-share 0.08 \
  --narrow-sampling-group 4 \
  --standing-gf-bonus 0.5 --reactive-hand-guidance --handsdf-weight 1 \
  --heading-align-weight 0.4 \
  --upright-weight 1.0 --stand-tall-weight 1.0 --torso-rate-weight -0.5 \
  --self-clearance-weight -10 --upper-posture-weight -0.5 --upper-home-shoulder-pitch -0.3 \
  --stand-still-weight -4 --standing-requires-stillness --standing-stillness-speed 0.05 --spot-hold-weight -3 \
  --reactive-episode-length 800 --reactive-walking-fraction 0.3 --sdf-rate-obs \
  --run-dir outputs/cat_bundle_20260924 \
  --disable-hand-contrast --hand-clearance-weight -20 --arm-clearance-weight -8 \
  --tracking-root-field-weight 1 \
  --hand-clearance-target 0.09 --hand-clearance-anticipation 0.20 \
  --hand-clearance-near-weight 0.8 --hand-reward-soft-floor 0 \
  --hand-raised-reset-fraction 0 --upper-gravity-compensation \
  --compile-task --device cuda:0 --seed 0 \
  --checkpoint-interval-updates 10 \
  --wandb-mode online --wandb-project CAT-wholebody --wandb-entity skvayzer
