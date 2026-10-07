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
# NARROW-PASSAGE SPECIALIST, phase 1 (2026-10-07). One expert for everything traded skills: as side gaps rose
#   (v10 corridor 35 -> 51%, gaps >= 0.50 m 36 -> 48%), crouch fell 62 -> 42% and hurdles 55 -> 49%; narrow
#   procedural openings (< 0.45 m) stayed at 1% (85-88% contact) for 515 updates although all are geometrically
#   passable sideways (scripts/narrow_scene_feasibility.py). Hurdles/crouch/open scenes stay with the traversal
#   expert; the voxel student learns from both.
#   * scenes: narrow only (narrowest opening < 0.77 m and passable: 1004 procedural, 630 side gaps, 30 original,
#     8 published) + flat; everything else zero weight (configs/pilots/narrow_specialist_excluded_scenes.txt)
#   * mix: original+published 0.10, procedural 0.45, side gaps 0.40, flat 0.05
#   * --contact-penalty 3: only HAND contact ends the episode, other touches cost 3 per step; success stays strict
#   * warm start v10 update 500 (best side-gap skill so far); curriculum, sideways starts/bonus kept
#   * phase 2: continue from phase 1 with every touch terminal again
# v10 (2026-10-06): hurdle skill eroded during v8 (deterministic published-hurdle success v7 79% -> v8 u100
#   80.5% -> u200 71% -> u355 63%, refusals 6% -> 20-25%; stochastic v2 71% / v9 53%) -- gradual forgetting
#   after procedural CAT (most hurdles) went 65% -> 50% of experience.
#   * mix: original+published 0.28, procedural 0.57, side gaps 0.10, flat 0.05
#   * warm start v8 update 100 (hurdles 80.5%, side gaps learning); v9 fixes kept
#   * skill/<tag>/{success,contact,refused}_rate in training (runner.scene_skill_tags): published types,
#     original, procedural by narrowest opening, side-gap layouts and widths, flat
#   * gate: deterministic hurdle check (scripts/diagnose_hurdles.py) at updates 100 / 200, stop below ~75%
# v9 (2026-10-06): v8 + --hand-tight-walls-only (hands touched the 0.60 m table gap 4/4: relax only between
#   full-height walls) + --side-gap-tumbling (v8 stuck at stage 1 for 200 updates on its all-time rate).
#   Warm start v8 final (update 355): side gaps placed forward 13% / free 28% / sideways 37% clean.
#   Stop-loss at ~100 updates: side-gap stage >= 2 and forward-start passes rising; course check vs v2.
# v8 (2026-10-05): side-gap CURRICULUM. v7 diagnostic: arrives facing forward (82%); placed sideways and lined
#   up 0.45 m before the gap it still passed 0% -- the sidestep-through skill is absent, RL had nothing to reinforce.
#   * --side-gap-curriculum: width levels >=0.50 / >=0.42 / >=0.36 / all (stage 0 samples only the widest), advance
#     when >= 30% of >= 300 resolved episodes at the current level are clean; sideways starts (+-90 deg, lined up
#     0.35-0.6 m before the gap) for 50 / 35 / 20 / 10% of side-gap episodes per stage
#   * forward-facing bonus released earlier: probes at 0.3 / 0.6 / 1.0 m ahead on the 2-D blocking footprint
#   * --sideways-bonus-weight 1.0: (1 - gate) * |sin(facing - travel)| where forward does not fit
#   * --teleop-side-prob 0.25: pure sidestep commands in the open-ground keyboard practice (flat scene)
#   * warm start v7 final (update 132; CAT success back at v2 level)
#   * stop-loss: scene/side_gap/success_rate and side_gap/side_gap_stage rising within ~100 updates
# v7 (2026-10-05): v6 at update 17 gave side gaps only 4.3% of experience (83 episodes/update, 5% success).
#   * bank sidegap_v2: 640 side-gap scenes (plain 166 incl. 49 bars / 22 beams, two walls 137, corridor 123
#     (narrow part 0.3-0.9 m long), angled +-15-30 deg 111, gap 0.40-0.65 m off the start line 103), gaps 0.32-0.60 m
#   * own sampling group (4, 'narrow' -- its room scenes stay moved to group 7) with 15% of experience, taken
#     from procedural CAT (0.65 -> 0.50)
#   * stop-loss: scene/side_gap/success_rate clearly rising by update ~100-150 (> 15%), else stop and look
# v6 (2026-10-05): v2 settings (field-driven CAT expert, no keyboard in CAT scenes) + sideways passages.
#   The CAT expert refuses side gaps: the robot is 0.764 m wide but 0.236 m deep (collision proxy), so any
#   opening < 0.77 m needs it sideways, yet only 7 of ~2300 CAT scenes need that and published-side1 (0.24 m,
#   0 cm spare) is unpassable without touching. In a 0.34 m gap the hand terms cost ~-12 per step.
#   * bank sidegap_v1 = table_edges_v1 + 160 procedural side gaps (0.32-0.60 m; second wall / bar / beam
#     variants), scripts/generate_side_gaps.py; own metrics bucket scene/side_gap/*
#   * published-side1 never sampled (--zero-weight-scenes)
#   * --hand-clearance-tight 0.10 0.40 0.3: hand target/anticipation, elbow margin and the handsdf knee scale
#     with pelvis room (0.34 m gap: ~-12 -> ~-2 per step; open space unchanged)
#   * warm start v2 update 697; regression: scripts/demo/course_check.py --per-obstacle --trials 4
# v2 (2026-10-02): the v9 warm start had forgotten CAT (3% success after 60 updates, 91-97% of episodes
# ending in contact; released CAT reaches 23-25% under the same collision rules). Warm start now from the
# RELEASED CAT generalist (converted to native with our environment config; actor input widened 222 -> 226
# with zero weights, identical actions), and the room-tuned body costs back to their earlier values:
# body_motion -0.5 (was -1.5), upright on pitch only (no roll).
set -euo pipefail
cd /home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CHECKPOINT=${CHECKPOINT:-outputs/saved_checkpoints/cat_expert_v10_u0500.pt}   # CAT expert v10, update 500
RUN_DIR=${RUN_DIR:-outputs/narrow_specialist_p1_20261007}
.venv-mjlab/bin/python train_cat_mjlab.py run \
  --algorithm ppo --num-envs ${NUM_ENVS:-40960} --batch-size ${BATCH_SIZE:-1024} \
  --num-minibatches ${NUM_MINIBATCHES:-40} --unroll-length 32 \
  --checkpoint-native "$CHECKPOINT" \
  --fresh-optimizer --max-action-std 0.25 \
  --bank-manifest data/furniture/sidegap_v2_packed/manifest.json \
  --body-collision-bank data/furniture/sidegap_v2_collision/manifest.json \
  --body-collision-resets data/furniture/sidegap_v2_resets/manifest.json \
  --experience-masses 0.10 0.45 0 0 0.40 0.05 0 0 \
  --scene-group-override table_edges=6 --scene-group-override side_gap=4 \
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
  --hand-clearance-tight 0.10 0.40 0.3 --zero-weight-scenes @configs/pilots/narrow_specialist_excluded_scenes.txt --contact-penalty 3 \
  --side-gap-curriculum --heading-lookaheads 0.3 0.6 1.0 --heading-blocking-probes --sideways-bonus-weight 1.0 --teleop-side-prob 0.25 \
  --hand-tight-walls-only --side-gap-tumbling \
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
